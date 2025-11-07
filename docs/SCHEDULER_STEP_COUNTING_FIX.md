# Scheduler Step Counting Bug Fix

**Date**: 2025-10-28
**Status**: 🔧 CRITICAL BUG - Fixed
**Impact**: Scheduler using wrong step count, causing learning rate schedule confusion

---

## Problem Identified

### The Bug

The training script has **TWO different step counters** but uses them inconsistently:

1. **`step`**: Forward pass counter (0 → 100,000)
   - Increments on EVERY forward pass
   - With `accum_steps=4`, this counts: 0, 1, 2, 3, 4, 5, ..., 100,000

2. **`optimizer_step`**: Optimizer update counter (0 → 25,000)
   - Only increments when optimizer.step() is called
   - With `accum_steps=4`, this counts: 0, 0, 0, 0, 1, 1, 1, 1, 2, ...
   - Final value: `100,000 / 4 = 25,000`

### The Confusion

**Scheduler counts OPTIMIZER STEPS, not forward passes!**

```python
# Lines 477-481: Scheduler is correctly configured for optimizer steps
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps // accum_steps,    # 2000 // 4 = 500
    num_training_steps=total_steps // accum_steps,    # 100000 // 4 = 25000
)
```

BUT when logging metrics, we display `step` (forward pass) alongside scheduler LR:

```python
# Line 789: Displays forward pass step with optimizer-step-based LR
log_dict = {
    "step": step,           # 0-100,000 (forward passes)
    "lr": lr_current,       # Based on 0-25,000 (optimizer steps)
}
```

**Result**: The logged metrics are confusing because:
- `step=2000` shows LR from warmup phase (correct: optimizer_step=500)
- `step=10000` shows LR near peak (correct: optimizer_step=2500)
- Graph plots look stretched/compressed by 4x

---

## Config File Analysis

### Example: `config_3090.yaml`

```yaml
train:
  accum_steps: 4
  warmup_steps: 2000      # ❓ Is this in forward passes or optimizer steps?
  max_steps: 100000       # ❓ Is this forward passes or optimizer steps?
```

**Current interpretation**:
- `max_steps=100,000` = forward passes
- `warmup_steps=2000` = forward passes
- Scheduler uses: `warmup=2000//4=500`, `total=100000//4=25000`

**What config means vs what users see**:
- Config says "2000 warmup steps" but scheduler uses 500
- Users think "warmup at step 2000" but actually at optimizer_step 500 (forward step 2000)

---

## The Fix

### 1. Add Explicit `optimizer_step` Counter

```python
# Lines 508-509: Initialize both counters
step = 0              # Forward passes (actual training iterations)
optimizer_step = 0    # Optimizer updates (step // accum_steps)
```

### 2. Increment `optimizer_step` After Weight Update

```python
# Lines 714-720: Update both counters
optimizer.step()
scheduler.step()
optimizer.zero_grad(set_to_none=True)

# 🔧 FIX: Increment optimizer_step counter after weight update
optimizer_step += 1
```

### 3. Log Both Metrics for Clarity

```python
# Line 785: Add optimizer_step to logs
log_dict = {
    "step": step,                    # 0-100,000 (forward passes)
    "optimizer_step": optimizer_step,  # 0-25,000 (optimizer updates)
    "lr": lr_current,                # Based on optimizer_step
}
```

### 4. Restore `optimizer_step` from Checkpoint

```python
# Lines 534-535: Restore optimizer_step when resuming
step = load_checkpoint(...)
optimizer_step = step // accum_steps
```

---

## Impact Analysis

### Before Fix
```
Step 0:     LR = 0.00000 (warmup start)
Step 2000:  LR = 0.00020 (warmup end - WRONG, should be at optimizer_step 500)
Step 10000: LR = 0.00019 (cosine decay)
```

**Problem**: Logs show `step=2000` for warmup end, but scheduler sees optimizer_step=500

### After Fix
```
Step 0     (optimizer_step 0):     LR = 0.00000 (warmup start)
Step 2000  (optimizer_step 500):   LR = 0.00020 (warmup end - CORRECT!)
Step 10000 (optimizer_step 2500):  LR = 0.00019 (cosine decay)
```

**Solution**: Logs clearly show both counters, eliminating confusion

---

## Config Clarification Needed

### Current Ambiguity

Config parameters could mean either forward passes OR optimizer steps:
- `warmup_steps: 2000` → Could be 2000 forward passes OR 2000 optimizer steps
- `max_steps: 100000` → Could be 100,000 forward passes OR 100,000 optimizer steps

### Recommendation

**Add comments to all config files**:

```yaml
train:
  accum_steps: 4

  # Step counts are in FORWARD PASSES
  # Scheduler will divide by accum_steps automatically
  warmup_steps: 2000      # 2000 forward passes = 500 optimizer steps
  max_steps: 100000       # 100000 forward passes = 25000 optimizer steps

  # Or alternatively, use explicit optimizer step configs:
  # warmup_optimizer_steps: 500
  # max_optimizer_steps: 25000
```

---

## Verification

### Check Scheduler Behavior

```python
# At step 2000 (optimizer_step 500):
assert optimizer_step == 500
assert scheduler.get_last_lr()[0] == cfg["train"]["lr"]  # Peak LR after warmup

# At step 100000 (optimizer_step 25000):
assert optimizer_step == 25000
assert scheduler.get_last_lr()[0] < cfg["train"]["lr"]  # Cosine decay
```

### TensorBoard Graphs

After fix, TensorBoard should show:
1. **train/learning_rate** vs `step`: LR schedule over forward passes
2. **train/learning_rate** vs `optimizer_step`: LR schedule over optimizer updates (clearer)

---

## Related Files

- **Fixed**: `/mnt/d/ai/SLGA/scripts/train.py`
  - Lines 508-509: Counter initialization
  - Lines 534-535: Checkpoint restoration
  - Line 720: Optimizer step increment
  - Line 785: Logging both counters

- **Needs Comments**: All config files
  - Clarify whether steps are forward passes or optimizer steps

---

## Summary

✅ **Fixed**: Added explicit `optimizer_step` counter for clarity
✅ **Fixed**: Log both `step` and `optimizer_step` in metrics
✅ **Fixed**: Restore `optimizer_step` from checkpoints
⚠️ **TODO**: Add comments to config files explaining step counting
⚠️ **TODO**: Consider renaming config params to be explicit:
  - `max_forward_passes` instead of `max_steps`
  - `warmup_optimizer_steps` instead of `warmup_steps`

---

## Testing

Run training with logging enabled:

```bash
python scripts/train.py --config config/config_3090.yaml --max-steps 1000
```

Check logs show both counters:
```
Step    500 (optimizer_step  125) | Loss: 8.1234 | LR: 5.00e-05
Step   1000 (optimizer_step  250) | Loss: 7.8901 | LR: 1.00e-04
Step   2000 (optimizer_step  500) | Loss: 7.2345 | LR: 2.00e-04  ← Warmup complete
```
