# Step Counting Documentation Index

**Last Updated**: 2025-10-28
**Status**: ✅ Complete

---

## Quick Navigation

### 📚 Main Documents

1. **[STEP_COUNTING_QUICK_REFERENCE.md](./STEP_COUNTING_QUICK_REFERENCE.md)**
   - 🎯 **Start here** for quick lookup
   - Tables, examples, common mistakes
   - When to use which counter

2. **[SCHEDULER_STEP_COUNTING_FIX.md](./SCHEDULER_STEP_COUNTING_FIX.md)**
   - Detailed bug analysis
   - Before/after comparison
   - Config interpretation guide

3. **[STEP_COUNTING_FIX_SUMMARY.md](./STEP_COUNTING_FIX_SUMMARY.md)**
   - Implementation summary
   - Testing results
   - Migration guide

---

## Quick Answers

### "I just want to understand the basics"
→ Read **Section 1** of [STEP_COUNTING_QUICK_REFERENCE.md](./STEP_COUNTING_QUICK_REFERENCE.md)

### "Why was this needed?"
→ Read **"The Bug"** section of [SCHEDULER_STEP_COUNTING_FIX.md](./SCHEDULER_STEP_COUNTING_FIX.md)

### "What changed in the code?"
→ Read **"Changes Applied"** in [STEP_COUNTING_FIX_SUMMARY.md](./STEP_COUNTING_FIX_SUMMARY.md)

### "How do I use this in my config?"
→ See example in [config_3090.yaml](../config/config_3090.yaml) lines 37-45

### "How do I verify it's working?"
→ Run: `python tests/verify_step_counting.py`

---

## The TL;DR

### Two Counters

1. **`step`** = Forward passes (0 → 100,000)
2. **`optimizer_step`** = Weight updates (0 → 25,000 with accum=4)

### Key Formula

```python
optimizer_step = step // accum_steps
```

### What Changed

- ✅ Logs now show both counters
- ✅ Code has clear documentation
- ✅ Config files have clarifying comments

### What Stayed Same

- ✅ Training behavior identical
- ✅ Scheduler was already correct
- ✅ Checkpoint compatibility maintained

---

## Visual Guide

### Forward Passes vs Optimizer Steps

```
accum_steps = 4

Forward:     0   1   2   3 | 4   5   6   7 | 8   9  10  11 | ...
             ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
Optimizer:   0   0   0   0 | 1   1   1   1 | 2   2   2   2 | ...
                         ^                ^                ^
                    optimizer.step()  optimizer.step() optimizer.step()
```

### LR Schedule Timeline

```
Config: warmup_steps=2000, max_steps=100000, accum_steps=4

Forward Step:    0                2000              50000             100000
                 |__________________|___________________|__________________|

Optimizer Step:  0                 500               12500              25000

LR:             0%                100%               ~50%                0%
                (warmup start)  (peak LR)      (cosine decay)      (training end)
```

---

## Code Examples

### Logging Both Counters

```python
# Console output
print(f"Step {step:6d} (opt {optimizer_step:5d}) | Loss: {loss:.4f} | LR: {lr:.2e}")
# Output: Step   2000 (opt   500) | Loss: 7.234 | LR: 2.00e-04

# TensorBoard
writer.add_scalar("train/loss", loss, step)           # vs forward passes
writer.add_scalar("train/loss_opt", loss, optimizer_step)  # vs optimizer steps

# W&B
wandb.log({"step": step, "optimizer_step": optimizer_step, "loss": loss})
```

### Scheduler Setup

```python
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps // accum_steps,    # 2000 → 500
    num_training_steps=total_steps // accum_steps,    # 100000 → 25000
)
```

### Config File

```yaml
train:
  accum_steps: 4

  # All step counts are in FORWARD PASSES
  # Scheduler divides by accum_steps automatically
  warmup_steps: 2000      # → 500 optimizer steps
  max_steps: 100000       # → 25000 optimizer steps
```

---

## Files Reference

### Modified Files

| File | Changes | Lines |
|------|---------|-------|
| `scripts/train.py` | Enhanced docs, logging | 41-84, 486-509, 785, 875-879 |
| `config/config_3090.yaml` | Added comments | 37-45 |

### New Files

| File | Purpose |
|------|---------|
| `docs/STEP_COUNTING_QUICK_REFERENCE.md` | Quick lookup guide |
| `docs/SCHEDULER_STEP_COUNTING_FIX.md` | Detailed analysis |
| `docs/STEP_COUNTING_FIX_SUMMARY.md` | Implementation summary |
| `docs/STEP_COUNTING_INDEX.md` | This file |
| `tests/verify_step_counting.py` | Verification script |

---

## Testing

### Run Verification

```bash
python tests/verify_step_counting.py
```

Expected output:
```
✅ ALL TESTS PASSED!

Step counting is correctly implemented:
  - Forward passes counted separately from optimizer steps
  - Scheduler uses optimizer steps (step // accum_steps)
  - LR schedule matches expected behavior
```

### Manual Check

Start training and verify console output:

```bash
python scripts/train.py --config config/config_3090.yaml --max-steps 1000
```

Look for:
```
Step   2000 (opt   500) | Loss: 7.234 | ...
```
Both counters should be visible!

---

## Common Questions

### Q: Do I need to retrain from scratch?
**A**: No! The fix is backward compatible. Old checkpoints work fine.

### Q: Why do we use forward passes in config?
**A**: More intuitive for users. Scheduler converts automatically.

### Q: Which counter should I use for logging frequency?
**A**: Use `step` (forward passes) - it's what users expect to see.

### Q: Which counter does the scheduler use?
**A**: `optimizer_step` (weight updates) - automatic division by `accum_steps`.

### Q: How do I debug LR schedule issues?
**A**: Check both counters in logs. At step 2000 with accum=4, optimizer_step should be 500.

---

## Further Reading

### Related Documentation
- [TRAINING_FIXES_2025.md](./TRAINING_FIXES_2025.md) - Other training improvements
- [MEMORY_LEAK_FIXES_2025-10-28.md](./MEMORY_LEAK_FIXES_2025-10-28.md) - Memory management
- [PARAM_VALIDATION_SUMMARY.md](./PARAM_VALIDATION_SUMMARY.md) - Parameter validation

### External Resources
- [Transformers Scheduler Docs](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)
- [PyTorch LR Scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

---

## Contact & Support

If you encounter issues or have questions:
1. Check this documentation first
2. Run verification script: `python tests/verify_step_counting.py`
3. Review [SCHEDULER_STEP_COUNTING_FIX.md](./SCHEDULER_STEP_COUNTING_FIX.md) for details

---

**Summary**: Step counting is now clearly documented and logged. Training behavior unchanged, but transparency greatly improved!
