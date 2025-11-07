# Step Counting Quick Reference

**Last Updated**: 2025-10-28
**Status**: ✅ Fixed

---

## Two Types of Steps

### 1. Forward Pass Steps (`step`)
- **What**: Counts every forward pass through the model
- **Range**: 0 → `max_steps` (e.g., 0 → 100,000)
- **Increments**: Every training iteration
- **Used for**:
  - Logging frequency
  - Checkpoint frequency
  - Validation frequency
  - Sequence length curriculum
  - Global attention warmup

### 2. Optimizer Steps (`optimizer_step`)
- **What**: Counts optimizer weight updates
- **Range**: 0 → `max_steps // accum_steps` (e.g., 0 → 25,000 with accum=4)
- **Increments**: Only after gradient accumulation completes
- **Used for**:
  - Learning rate scheduler
  - Effective batch size computation

---

## Example with `accum_steps=4`

```
Forward Pass    Optimizer Step    Action
    0               0              Forward + backward
    1               0              Forward + backward
    2               0              Forward + backward
    3               0              Forward + backward + optimizer.step() + scheduler.step()
    4               1              Forward + backward
    5               1              Forward + backward
    ...
  100000          25000            Final step
```

---

## Config File Interpretation

All step counts in config files are **forward pass steps**:

```yaml
train:
  accum_steps: 4
  warmup_steps: 2000      # 2000 forward passes = 500 optimizer steps
  max_steps: 100000       # 100000 forward passes = 25000 optimizer steps
```

**Scheduler sees**:
- `num_warmup_steps = 2000 // 4 = 500`
- `num_training_steps = 100000 // 4 = 25000`

---

## When to Use Which Counter

| Operation | Use `step` | Use `optimizer_step` |
|-----------|------------|----------------------|
| Logging | ✅ | Show both for clarity |
| Checkpointing | ✅ | |
| Validation | ✅ | |
| Curriculum (seq_len) | ✅ | |
| Global warmup | ✅ | |
| LR scheduler | ❌ | ✅ |
| Effective batch calculation | ❌ | ✅ |

---

## Logging Best Practices

Always show **both** counters for clarity:

```python
print(f"Step {step:6d} (opt {optimizer_step:5d}) | Loss: {loss:.4f} | LR: {lr:.2e}")
# Output: Step   2000 (opt   500) | Loss: 7.234 | LR: 2.00e-04
```

**TensorBoard**:
```python
writer.add_scalar("train/loss", loss, step)           # Plot vs forward passes
writer.add_scalar("train/loss_opt", loss, optimizer_step)  # Plot vs optimizer steps
```

---

## Verification

At any point during training:

```python
assert optimizer_step == step // accum_steps
```

---

## LR Schedule Examples

### With `accum_steps=4`, `warmup_steps=2000`, `max_steps=100000`

| Forward Step | Optimizer Step | LR Phase |
|--------------|----------------|----------|
| 0 | 0 | Warmup start (LR = 0) |
| 500 | 125 | Warmup (LR = 0.25 × peak) |
| 1000 | 250 | Warmup (LR = 0.50 × peak) |
| 1500 | 375 | Warmup (LR = 0.75 × peak) |
| **2000** | **500** | **Warmup end (LR = peak)** |
| 10000 | 2500 | Cosine decay |
| 50000 | 12500 | Cosine decay (LR ≈ 0.5 × peak) |
| 100000 | 25000 | Training end (LR ≈ 0 × peak) |

---

## Common Mistakes

### ❌ Wrong: Use forward steps for scheduler
```python
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,      # WRONG!
    num_training_steps=total_steps,     # WRONG!
)
```

### ✅ Right: Divide by accum_steps
```python
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps // accum_steps,    # CORRECT
    num_training_steps=total_steps // accum_steps,   # CORRECT
)
```

---

## Summary

- **Config uses forward passes** (what users see)
- **Scheduler uses optimizer steps** (what scheduler sees)
- **Always log both** for clarity
- **Relationship**: `optimizer_step = step // accum_steps`
