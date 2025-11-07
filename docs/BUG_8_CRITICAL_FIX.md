# 🚨 CRITICAL BUG FIX: cache_ids Shape Collapse

## ⚠️ Severity: HIGH

**Bug #8**: Troncation de `cache_ids` collapse la structure batch × landmarks.

---

## 🐛 The Bug

```python
# ❌ BUGGY CODE (train.py:668-669)
mask = cache_ids < current_seq_len
cache_ids = cache_ids[mask]  # SHAPE COLLAPSE!

# Before: (4, 8)  → 4 samples × 8 landmarks
# After:  (22,)   → flat vector (BROKEN!)
```

**Impact**:
- ❌ Forward pass crash or misalignment
- ❌ Accelerate distributed training fails
- ❌ Per-sample structure lost
- ❌ Can shrink to length < G (expected landmark count)

---

## ✅ The Fix

```python
# ✅ FIXED CODE (train.py:671)
cache_ids = torch.clamp(cache_ids, 0, current_seq_len - 1)

# Before: (4, 8)  → 4 samples × 8 landmarks
# After:  (4, 8)  → PRESERVED! ✅
```

**Benefits**:
- ✅ Shape (B, G) preserved
- ✅ Per-sample structure maintained
- ✅ Invalid indices clamped to max
- ✅ Valid indices unchanged
- ✅ Distributed training works

---

## 🧪 Test Results

```bash
$ python tests/test_cache_ids_shape_preservation.py

TEST: cache_ids Shape Preservation (Bug Fix)
================================================================================

Original cache_ids shape: (4, 8)

❌ OLD METHOD (buggy): cache_ids[mask]
  Result shape: (22,)  ← COLLAPSED!
  Lost batch structure!

✅ NEW METHOD (fixed): torch.clamp(cache_ids, 0, max)
  Result shape: (4, 8)  ← PRESERVED!
  Perfect structure maintained!

✓ PASS: Shape preserved correctly
✓ PASS: All values within valid range
✓ PASS: Valid values unchanged
✓ PASS: Out-of-bounds values clamped to max

Edge Cases:
✓ PASS: All landmarks within bounds
✓ PASS: All landmarks out of bounds
✓ PASS: Single sample (batch_size=1)
✓ PASS: Large batch (batch_size=32)

================================================================================
✓ ALL TESTS PASSED

The fix correctly:
  1. Preserves (batch, G) shape
  2. Clamps out-of-bounds indices
  3. Leaves valid indices unchanged
  4. Handles all edge cases

No more shape collapse bugs! ✅
```

---

## 🎯 Why This Was Critical

### Distributed Training
```python
# ❌ With bug: Different shapes per GPU
GPU 0: cache_ids shape (22,)
GPU 1: cache_ids shape (18,)
GPU 2: cache_ids shape (25,)
# Accelerate gather() → CRASH!

# ✅ With fix: Consistent shape
GPU 0: cache_ids shape (4, 8)
GPU 1: cache_ids shape (4, 8)
GPU 2: cache_ids shape (4, 8)
# Accelerate gather() → Works! ✅
```

### Forward Pass
```python
# model.forward() expects cache_global_ids: (B, G)

# ❌ With bug
cache_global_ids.shape: (22,)  # 1D!
# → Shape mismatch crash
# OR misinterprets as wrong dimensions
# → Wrong landmarks used

# ✅ With fix
cache_global_ids.shape: (4, 8)  # 2D correct!
# → Works as expected
# → Correct gather per sample
```

---

## 📋 Files Modified

- **scripts/train.py:665-671** - Fixed truncation logic
- **tests/test_cache_ids_shape_preservation.py** - Comprehensive test suite

---

## 🙏 Thanks

Special thanks to the user for:
- ✅ Identifying this subtle but critical bug
- ✅ Diagnosing the shape collapse issue
- ✅ Pointing out distributed training impact
- ✅ Suggesting shape-preserving solutions

**HIGH severity bug catch!** 🎯

---

## 📖 Documentation

- `docs/FIX_CACHE_IDS_SHAPE_COLLAPSE_BUG.md` - Complete technical analysis
- `docs/CHANGELOG_HIVE_MIND_2025-10-30.md` - Version 1.4 entry

---

**Status**: ✅ CRITICAL FIX APPLIED
**Date**: 2025-10-30
**Severity**: HIGH
**Bug #**: 8
**Test Status**: ALL PASSED ✅
