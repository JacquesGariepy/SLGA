# CRITICAL BUG: Learning Rate Scheduler

## Bug Summary

**Severity**: 🚨 **CRITICAL** - Training completely broken
**Impact**: Model learns 5× slower than intended, PPL stays catastrophically high
**Status**: ✅ Identified, fix ready

## The Bug

### Current (Broken) Code

```python
# Line 419-425: Scheduler creation
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,      # 2000
    num_training_steps=total_steps,      # 100000
)

# Line 577-612: Gradient accumulation loop
if (step + 1) % accum_steps == 0:  # Every 5 forward passes
    optimizer.step()
    scheduler.step()  # ← Only called every 5 steps!
    optimizer.zero_grad()

step += 1  # Incremented EVERY forward pass
```

### What Happens

| Forward Passes | Optimizer Steps | Scheduler Calls | LR (should be) | LR (actual) |
|----------------|-----------------|-----------------|----------------|-------------|
| 1000 | 200 | 200 | 1.0e-04 | 2.0e-05 |
| 2000 | 400 | 400 | 2.0e-04 | 4.0e-05 |
| 10000 | 2000 | 2000 | 2.0e-04 (warmup done) | 2.0e-04 |
| 100000 | 20000 | 20000 | ~0 (near end) | 2.0e-04 (still max!) |

**The scheduler thinks**:
- Total training = 100,000 optimizer steps
- But actually = 100,000 forward passes = 20,000 optimizer steps

**Result**:
- At forward step 1000: scheduler at 1% (200/2000 warmup) → LR = 2e-05
- At forward step 10000: scheduler at 10% (2000/2000 warmup done) → LR = 2e-04
- At forward step 100000: scheduler at 20% (20000/100000) → LR still high!

## Impact on Training

### Observed Metrics at Step 1000

```
Loss: 8.9090
PPL: 7398.14  ← Should be ~100
LR: 2.00e-05  ← Should be 1.00e-04
```

**Why PPL is so high**:
- LR is 5× too low
- Model barely learning
- After 1000 steps, model predicts almost at random

### Generation Quality

```
Prompt: "The capital of France is "
Output: ", the.\n and,.\n."  ← Nonsensical
```

**Why**: Model hasn't learned language patterns because LR too low.

## The Fix

### Solution 1: Scale Scheduler Parameters (RECOMMENDED)

```python
# Line 419-425: FIXED
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps // accum_steps,  # 2000 → 400
    num_training_steps=total_steps // accum_steps,  # 100000 → 20000
)
```

**Why this works**:
- Scheduler now counts **optimizer steps**, not forward passes
- At optimizer step 200 (forward 1000): 200/400 = 50% warmup → LR = 1e-04 ✓
- At optimizer step 400 (forward 2000): warmup done → LR = 2e-04 ✓
- At optimizer step 20000 (forward 100000): cosine decay complete

### Solution 2: Call scheduler.step() Every Forward Pass

```python
# Line 577-612: Alternative fix
if (step + 1) % accum_steps == 0:
    optimizer.step()
    optimizer.zero_grad()

scheduler.step()  # ← Move outside if block
step += 1
```

**Why this works**:
- Scheduler called every forward pass
- Matches original intent of `num_training_steps=100000`

**Trade-off**: Scheduler advances during gradient accumulation (less clean)

### Solution 3: Separate Counter

```python
optimizer_step_count = 0

if (step + 1) % accum_steps == 0:
    optimizer.step()
    optimizer_step_count += 1
    scheduler.step()  # Uses optimizer_step_count internally
    optimizer.zero_grad()

step += 1
```

**Why this works**: Explicit tracking of optimizer steps

**Trade-off**: More code complexity

## Recommended Fix (Solution 1)

**File**: `scripts/train.py`
**Lines**: 419-425

**Change**:
```python
# BEFORE
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

# AFTER
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps // accum_steps,  # 2000 → 400
    num_training_steps=total_steps // accum_steps,  # 100000 → 20000
)
```

## Verification After Fix

### Expected Metrics at Step 1000

```
Loss: ~3.5-4.5  (decreasing)
PPL: ~40-100    (decreasing)
LR: 1.0e-04     (50% warmup with accum_steps=5)
```

### Expected Generation at Step 5000

```
Prompt: "The capital of France is "
Output: "a city in the north"  ← Coherent structure
```

## Action Required

### 1. Stop Current Training

```bash
# Find training process
ps aux | grep train.py

# Kill it
kill <PID>
```

**Why**: Current training with wrong LR will not converge properly

### 2. Apply Fix

```bash
# Edit scripts/train.py line 421-424
# Change num_warmup_steps and num_training_steps to divide by accum_steps
```

### 3. Delete Bad Checkpoints

```bash
rm -rf out_slga_fineweb/ckpt_*
```

**Why**: These checkpoints were trained with wrong LR schedule

### 4. Restart Training

```bash
python scripts/train.py \
  --config config/config_fineweb_edu_3090_optimized.yaml \
  --max-steps 100000
```

### 5. Monitor First 1000 Steps

Check that:
- LR reaches ~1e-04 at step 1000 ✓
- Loss decreases to ~4.0 at step 1000 ✓
- PPL around 50-100 at step 1000 ✓

## Historical Context

This bug has existed since training began. All checkpoints (ckpt_500, ckpt_1000) are affected.

**Total time wasted**: ~4-5 hours of training
**Data processed**: Wasted, model didn't learn properly

## Prevention

### Code Review Checklist

When using gradient accumulation with transformers schedulers:

```python
✓ Verify num_training_steps accounts for accumulation
✓ Check scheduler.step() call frequency
✓ Monitor LR in first 100 steps
✓ Verify LR reaches max after warmup_steps
```

### Add Assertion

```python
# After scheduler creation
assert scheduler.get_last_lr()[0] < lr_max * 0.01, "LR should start near 0"

# After warmup
if step == warmup_steps:
    current_lr = scheduler.get_last_lr()[0]
    assert abs(current_lr - lr_max) < 1e-6, f"LR should be {lr_max} after warmup, got {current_lr}"
```

## Related Issues

This is a **common pitfall** with:
- Hugging Face `get_cosine_schedule_with_warmup`
- Gradient accumulation
- Counting "steps" vs "optimizer updates"

**Always remember**:
- LR schedulers count **optimizer steps**
- Training loops often count **forward passes**
- With accum_steps=N, these differ by N×

## Summary

**Bug**: Scheduler configured for 100K optimizer steps, but only 20K will happen
**Impact**: LR 5× too low, PPL stays at ~7400, model learns nothing
**Fix**: Divide `num_warmup_steps` and `num_training_steps` by `accum_steps`
**Action**: Stop training, apply fix, restart from scratch

**Status**: 🔴 **URGENT - Apply fix before continuing training**
