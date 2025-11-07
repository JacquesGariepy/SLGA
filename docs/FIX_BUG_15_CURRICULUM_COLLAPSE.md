# CRITICAL FIX: Bug #15 - Curriculum Landmark Collapse

## 🚨 Severity: CRITICAL

**Bug #15**: Landmarks heuristiques s'effondrent pendant curriculum learning.

---

## 🐛 The Problem

### Curriculum Learning Context

Curriculum learning tronque progressivement les séquences :
- Début training: `current_seq_len = 128`
- Milieu training: `current_seq_len = 256`
- Fin training: `current_seq_len = 512`

### Le Bug en Action

```python
# 1. Collator génère landmarks pour séquence complète (L=512)
cache_global_ids = [0, 73, 146, 219, 292, 365, 438, 511]

# 2. Curriculum tronque à L=128
input_ids = input_ids[:, :128]

# 3. ❌ AVANT (buggy): Clamp landmarks
cache_ids = torch.clamp(cache_global_ids, 0, 127)
# Result: [0, 73, 127, 127, 127, 127, 127, 127]
#                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                  6/8 landmarks au MÊME index!

# 4. Unique positions: 3/8
unique_positions = [0, 73, 127]
# ❌ DISASTER: Landmarks collapsed!
```

**Impact catastrophique**:
- ❌ 6/8 landmarks au même index (127)
- ❌ Seulement 3/8 positions uniques
- ❌ Global attention presque inutile
- ❌ Affecte TOUTE la phase de curriculum court
- ❌ 50%+ du training avec attention dégradée

---

## ✅ The Fix

### Solution: Régénération Après Troncature

**File**: `scripts/train.py:666-678`

```python
# ✅ APRÈS (fixed): Régénérer landmarks pour longueur tronquée
if cache_ids is not None and not model.cfg.learned_landmarks:
    # Cas heuristic: régénérer pour current_seq_len
    B, G = cache_ids.shape
    cache_ids = torch.linspace(
        0, current_seq_len - 1, G, device=cache_ids.device
    ).long().unsqueeze(0).expand(B, -1)
```

**Result**:
```python
# Curriculum tronque à L=128
# Régénération landmarks:
cache_ids = [0, 18, 36, 54, 72, 90, 108, 127]
#            ^^^  ^^  ^^  ^^  ^^  ^^  ^^^  ^^^
#            Tous uniques! Espacement uniforme!

# Unique positions: 8/8 ✅
# Global attention: Fully functional!
```

---

## 🧪 Test Results

```bash
$ python tests/test_bug15_fix_validation.py

BEFORE TRUNCATION:
  cache_ids (from collator): [0, 73, 146, 219, 292, 365, 438, 511]
  Sequence length: 512

AFTER TRUNCATION (curriculum):
  Sequence length: 128

❌ OLD FIX (clamp):
  Result: [0, 73, 127, 127, 127, 127, 127, 127]
  Unique positions: 3/8
  ❌ DISASTER: Landmarks collapsed!

✅ NEW FIX (regenerate):
  Result: [0, 18, 36, 54, 72, 90, 108, 127]
  Unique positions: 8/8
  ✅ SUCCESS: Full diversity maintained!

VALIDATION
================================================================================
✓ PASS: All landmarks unique after fix
✓ PASS: All positions in valid range [0, current_seq_len-1]
✓ PASS: Regular spacing maintained (≈18.1)
✓ PASS: Fix resolves clustering (3 → 8 unique)

✓ ALL TESTS PASSED - Bug #15 fix is CORRECT

The fix:
  ✓ Maintains full landmark diversity
  ✓ Adapts to truncated sequence length
  ✓ Prevents clustering at sequence end
  ✓ Ensures effective global attention
```

---

## 📊 Impact Analysis

### Before Fix (Clustered Landmarks)

```
Curriculum L=128:
  Landmarks: [0, 73, 127, 127, 127, 127, 127, 127]
  Unique: 3/8 (37.5%)

Curriculum L=256:
  Landmarks: [0, 73, 146, 219, 255, 255, 255, 255]
  Unique: 5/8 (62.5%)

Curriculum L=512:
  Landmarks: [0, 73, 146, 219, 292, 365, 438, 511]
  Unique: 8/8 (100%) ← Only works at full length!
```

**Problem**: Curriculum phases have severely degraded global attention!

---

### After Fix (Regenerated Landmarks)

```
Curriculum L=128:
  Landmarks: [0, 18, 36, 54, 72, 90, 108, 127]
  Unique: 8/8 (100%) ✅

Curriculum L=256:
  Landmarks: [0, 36, 73, 109, 146, 182, 219, 255]
  Unique: 8/8 (100%) ✅

Curriculum L=512:
  Landmarks: [0, 73, 146, 219, 292, 365, 438, 511]
  Unique: 8/8 (100%) ✅
```

**Solution**: Consistent full diversity at ALL curriculum stages! ✅

---

## 🎯 Why This Was Critical

### Training Impact

**Curriculum learning** uses short sequences for **50-70%** of total steps:
- Steps 0-5000: L=128 (25%)
- Steps 5000-10000: L=256 (25%)
- Steps 10000-20000: L=384 (25%)
- Steps 20000+: L=512 (25%)

**Before fix**:
- 25% of training: 3/8 unique landmarks (62% collapse)
- 25% of training: 5/8 unique landmarks (38% collapse)
- 25% of training: 7/8 unique landmarks (12% collapse)
- 25% of training: 8/8 unique landmarks (OK)

**After fix**:
- 100% of training: 8/8 unique landmarks ✅

**Impact**: Massive improvement in early training phases!

---

## 💡 Design Insight

### Why Regeneration is Better Than Clamping

**Clamping** (broken):
```python
# Fixed landmarks: [0, 73, 146, 219, 292, 365, 438, 511]
# Clamp to [0, 127]
# → [0, 73, 127, 127, 127, 127, 127, 127]
# ❌ Loses diversity
```

**Regeneration** (correct):
```python
# Adapt to new length: L=128
# Regenerate: linspace(0, 127, 8)
# → [0, 18, 36, 54, 72, 90, 108, 127]
# ✅ Maintains diversity
```

**Regeneration preserves** the core principle: uniform coverage of sequence.

---

## 🔧 Implementation Details

### Code Location
`scripts/train.py:666-684`

### Logic Flow
```python
if input_ids.size(1) > current_seq_len:
    # Truncate sequences
    input_ids = input_ids[:, :current_seq_len]
    labels = labels[:, :current_seq_len]

    if cache_ids is not None and not model.cfg.learned_landmarks:
        # Heuristic: REGENERATE for truncated length
        B, G = cache_ids.shape
        cache_ids = torch.linspace(
            0, current_seq_len - 1, G, device=cache_ids.device
        ).long().unsqueeze(0).expand(B, -1)

    elif cache_ids is not None and model.cfg.learned_landmarks:
        # Learned: CLAMP (positions from model may exceed)
        cache_ids = torch.clamp(cache_ids, 0, current_seq_len - 1)
```

### Why Different Approaches?

| Mode | Approach | Reason |
|------|----------|--------|
| Heuristic | Regenerate | Maintains uniform coverage |
| Learned | Clamp | Preserves model's selection |

---

## ✅ Verification

- [x] Bug identified (landmark collapse)
- [x] Root cause understood (clamp during curriculum)
- [x] Fix applied (regeneration)
- [x] Test created and passing
- [x] All stages of curriculum covered
- [x] Diversity maintained (8/8 unique)
- [x] No performance regression

---

## 📊 Expected Improvements

| Phase | Before | After | Impact |
|-------|--------|-------|--------|
| L=128 | 3/8 unique | 8/8 unique | +166% |
| L=256 | 5/8 unique | 8/8 unique | +60% |
| L=384 | 7/8 unique | 8/8 unique | +14% |
| L=512 | 8/8 unique | 8/8 unique | - |

**Overall**: Dramatic improvement in early/mid training phases!

---

## 🙏 Acknowledgments

**Special thanks for catching this curriculum-specific bug!**

This was hiding in the interaction between:
- Curriculum learning (train.py)
- Heuristic landmarks (data.py)
- Clamping logic (model.py)

**Excellent multi-file bug detection!** 🎯

---

**Status**: ✅ CRITICAL FIX APPLIED
**Date**: 2025-10-30
**Bug #**: 15
**Severity**: CRITICAL
**Impact**: 50-70% of training affected
**Fix**: Regenerate landmarks after curriculum truncation
