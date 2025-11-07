# Step Counting Fix - README

**Quick Start Guide for Understanding Step Counting in SLGA Training**

---

## 🎯 What You Need to Know

### The Two Counters

Your training uses **two different step counters**:

```python
step = 0              # Counts every forward pass
optimizer_step = 0    # Counts weight updates
```

### The Relationship

```python
optimizer_step = step // accum_steps
```

**Example with `accum_steps=4`**:
- Forward passes: 0, 1, 2, 3, **4**, 5, 6, 7, **8**, ...
- Optimizer steps: 0, 0, 0, 0, **1**, 1, 1, 1, **2**, ...

---

## 📚 Documentation Guide

### Start Here

1. **[STEP_COUNTING_INDEX.md](./STEP_COUNTING_INDEX.md)** ← Navigation hub
2. **[STEP_COUNTING_QUICK_REFERENCE.md](./STEP_COUNTING_QUICK_REFERENCE.md)** ← Quick lookup

### Deep Dive

3. **[SCHEDULER_STEP_COUNTING_FIX.md](./SCHEDULER_STEP_COUNTING_FIX.md)** ← Full analysis
4. **[STEP_COUNTING_FIX_SUMMARY.md](./STEP_COUNTING_FIX_SUMMARY.md)** ← Implementation details

---

## 🚀 Quick Examples

### Reading Logs

**Old format** (confusing):
```
Step   2000 | Loss: 7.234 | LR: 2.00e-04
```

**New format** (clear):
```
Step   2000 (opt   500) | Loss: 7.234 | LR: 2.00e-04
```

Now you can see:
- 2000 forward passes completed
- 500 optimizer updates completed
- LR at peak (warmup just finished!)

### Config File

```yaml
train:
  accum_steps: 4

  # Step counts are in FORWARD PASSES
  warmup_steps: 2000      # → 500 optimizer steps
  max_steps: 100000       # → 25000 optimizer steps
```

### Understanding LR Schedule

```
Config says: warmup_steps=2000

With accum_steps=4:
  Scheduler sees: 2000 // 4 = 500 optimizer steps

At forward step 2000:
  optimizer_step = 2000 // 4 = 500
  LR reaches peak (warmup complete!)
```

---

## ✅ Verification

Run the verification script:

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

---

## 🔍 What Changed?

### Code Changes
- ✅ Both counters now logged
- ✅ Console shows both in format: `Step X (opt Y)`
- ✅ TensorBoard/W&B log both
- ✅ Clear documentation in code

### Config Changes
- ✅ Comments explain forward passes vs optimizer steps
- ✅ Examples show conversion with `accum_steps`

### Documentation
- ✅ 4 comprehensive guides created
- ✅ Quick reference with tables
- ✅ Verification script included

### What Stayed Same
- ✅ Training behavior identical
- ✅ Old checkpoints still work
- ✅ No breaking changes

---

## 📊 Visual Guide

```
Timeline with accum_steps=4:

Forward:     0    500   1000   1500   2000         50000         100000
             |_____|_____|_____|_____|______________|_______________|

Optimizer:   0    125    250    375    500          12500          25000

LR:         0%    25%    50%    75%   100%           ~50%            0%
            └─────────── warmup ──────┘└────── cosine decay ───────┘
```

---

## 🎓 Key Concepts

### When to Use Which Counter?

| Operation | Use `step` | Use `optimizer_step` |
|-----------|------------|---------------------|
| Logging frequency | ✅ | Show both |
| Checkpointing | ✅ | |
| Validation | ✅ | |
| Curriculum | ✅ | |
| LR scheduler | ❌ | ✅ |

### Common Mistakes

❌ **Wrong**: Using forward steps for scheduler
```python
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,  # WRONG!
)
```

✅ **Right**: Divide by `accum_steps`
```python
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps // accum_steps,  # CORRECT!
)
```

---

## 📞 Need Help?

### Debugging Steps

1. **Check your logs**: Both counters should be visible
   ```
   Step   2000 (opt   500) | ...
   ```

2. **Verify relationship**:
   ```python
   assert optimizer_step == step // accum_steps
   ```

3. **Run verification**:
   ```bash
   python tests/verify_step_counting.py
   ```

4. **Check warmup timing**:
   - At step 2000 with accum=4: optimizer_step should be 500
   - LR should be at peak

### Still Confused?

- Read: [STEP_COUNTING_QUICK_REFERENCE.md](./STEP_COUNTING_QUICK_REFERENCE.md)
- Review: [STEP_COUNTING_INDEX.md](./STEP_COUNTING_INDEX.md)
- Check: Example in [config_3090.yaml](../config/config_3090.yaml)

---

## ✨ Benefits

### Before Fix
- 😕 Logs only showed forward passes
- 😕 Unclear when warmup ends
- 😕 Hard to debug LR schedule

### After Fix
- 😊 Both counters clearly displayed
- 😊 Easy to see scheduler progress
- 😊 Simple debugging with verification script

---

## 🎉 Summary

**The fix clarifies step counting without changing training behavior:**

- ✅ **Transparent**: See both forward passes and optimizer updates
- ✅ **Compatible**: Old checkpoints work fine
- ✅ **Verified**: Comprehensive tests confirm correctness
- ✅ **Documented**: Four guides + verification script

**Start training with confidence - you'll now see exactly what's happening!**

---

**Quick Links**:
- 📖 [Full Documentation Index](./STEP_COUNTING_INDEX.md)
- 🔍 [Quick Reference](./STEP_COUNTING_QUICK_REFERENCE.md)
- 🧪 [Verification Script](../tests/verify_step_counting.py)
