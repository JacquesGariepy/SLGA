# CRITICAL FIX: cache_ids Shape Collapse Bug (2025-10-30)

## 🚨 Severity: HIGH

**Bug #8**: Troncation de `cache_ids` avec masque booléen collapse la shape (B, G) en vecteur 1D.

---

## 🐛 Problem Description

### Symptôme
```python
# ❌ AVANT (buggy)
mask = cache_ids < current_seq_len
cache_ids = cache_ids[mask]  # Shape collapse!

# Original: (4, 8)  → batch × landmarks
# Result:   (22,)   → flat vector (BROKEN!)
```

**Conséquences**:
1. ❌ Perte de structure per-sample
2. ❌ Misalignment des landmarks entre samples
3. ❌ Forward pass échoue avec shape mismatch
4. ❌ Accelerate gather() crash
5. ❌ Peut réduire à length < G (nombre de landmarks attendu)

---

### Cause Racine

**File**: `scripts/train.py:666-669` (ancienne version)

```python
# ❌ CODE BUGUÉ
if cache_ids is not None:
    # Garder seulement landmarks dans la fenêtre tronquée
    mask = cache_ids < current_seq_len
    cache_ids = cache_ids[mask].unsqueeze(0) if mask.any() else None
```

**Problème avec `cache_ids[mask]`**:

L'indexation booléenne collapse TOUJOURS la tensor en 1D :
```python
cache_ids = torch.tensor([
    [100, 200, 300, 400, 500, 600, 700, 800],  # Sample 1
    [50, 150, 250, 350, 450, 550, 650, 750],   # Sample 2
    [10, 20, 30, 40, 50, 60, 70, 80],          # Sample 3
    [200, 300, 400, 500, 600, 700, 800, 900],  # Sample 4
])
# Shape: (4, 8) = 4 samples × 8 landmarks

mask = cache_ids < 512  # Boolean mask

cache_ids[mask]  # Boolean indexing
# Result: tensor([100, 200, 300, 400, 500, 50, 150, ...])
# Shape: (22,) ← COLLAPSED! Structure per-sample perdue!
```

**Impact catastrophique**:
- Impossible de mapper landmarks → samples
- Forward pass reçoit mauvaise shape
- Accelerate distributed training crash

---

## ✅ Solution Applied

### Nouveau Code

**File**: `scripts/train.py:665-671`

```python
# ✅ CODE CORRIGÉ
if cache_ids is not None:
    # Clamper les indices pour qu'ils restent dans [0, current_seq_len-1]
    # Préserve la shape (B, G) au lieu de la collapser
    cache_ids = torch.clamp(cache_ids, 0, current_seq_len - 1)
```

**Pourquoi `torch.clamp()` est correct**:

```python
cache_ids = torch.tensor([
    [100, 200, 300, 400, 500, 600, 700, 800],
    [50, 150, 250, 350, 450, 550, 650, 750],
    [10, 20, 30, 40, 50, 60, 70, 80],
    [200, 300, 400, 500, 600, 700, 800, 900],
])
# Shape: (4, 8)

torch.clamp(cache_ids, 0, 511)
# Result: tensor([
#     [100, 200, 300, 400, 500, 511, 511, 511],  # Out-of-bounds → 511
#     [50, 150, 250, 350, 450, 511, 511, 511],
#     [10, 20, 30, 40, 50, 60, 70, 80],          # Unchanged (all valid)
#     [200, 300, 400, 500, 511, 511, 511, 511],
# ])
# Shape: (4, 8) ← PRESERVED! ✅
```

**Avantages**:
1. ✅ Shape (B, G) préservée
2. ✅ Structure per-sample maintenue
3. ✅ Indices invalides clampés à `current_seq_len - 1` (dernière position valide)
4. ✅ Indices valides inchangés
5. ✅ Forward pass reçoit shape correcte
6. ✅ Accelerate gather() fonctionne

---

## 🧪 Tests Complets

**File**: `tests/test_cache_ids_shape_preservation.py`

### Test 1: Shape Preservation
```bash
Original cache_ids shape: (4, 8)

❌ OLD METHOD (buggy): cache_ids[mask]
  Result shape: (22,)  ← COLLAPSED!
  Lost batch structure!

✅ NEW METHOD (fixed): torch.clamp(cache_ids, 0, max)
  Result shape: (4, 8)  ← PRESERVED!
  Perfect structure maintained!

✓ PASS: Shape preserved correctly
```

### Test 2: Value Correctness
```bash
✓ PASS: All values within valid range [0, current_seq_len-1]
✓ PASS: Valid values unchanged
✓ PASS: Out-of-bounds values clamped to max
```

### Test 3: Edge Cases
```bash
Edge case 1: All landmarks within bounds
✓ PASS: No changes when all valid

Edge case 2: All landmarks out of bounds
✓ PASS: All clamped to max

Edge case 3: Single sample (batch_size=1)
✓ PASS: Shape preserved: (1, 4)

Edge case 4: Large batch (batch_size=32)
✓ PASS: Shape preserved: (32, 16)
```

**Résultat**: ALL TESTS PASSED ✅

---

## 📋 Impact Analysis

### Before Fix (Buggy Behavior)

```python
# Scenario: batch_size=4, num_landmarks=8, current_seq_len=512

cache_ids shape: (4, 8)

# After truncation with mask
cache_ids shape: (22,)  ← WRONG!

# Forward pass
model(input_ids, cache_global_ids=cache_ids)
# ❌ Shape mismatch: Expected (B, G), got (22,)
# ❌ Cannot map landmarks to samples
# ❌ Crash or incorrect attention computation
```

### After Fix (Correct Behavior)

```python
# Scenario: batch_size=4, num_landmarks=8, current_seq_len=512

cache_ids shape: (4, 8)

# After truncation with clamp
cache_ids shape: (4, 8)  ← CORRECT!

# Forward pass
model(input_ids, cache_global_ids=cache_ids)
# ✅ Shape correct: (B, G)
# ✅ Landmarks correctly mapped per sample
# ✅ Forward pass succeeds
# ✅ Accelerate gather() works
```

---

## 🎯 Why This Bug Was Critical

### Distributed Training Impact

**Avec Accelerate** (multi-GPU):
```python
# ❌ AVANT (buggy)
cache_ids = cache_ids[mask]  # Shape: (variable,) per GPU

# Accelerate gather() across GPUs
gathered = accelerator.gather(cache_ids)
# ❌ CRASH! Inconsistent shapes across GPUs
# GPU 0: (22,)
# GPU 1: (18,)
# GPU 2: (25,)
# Cannot concatenate!
```

**Après fix**:
```python
# ✅ APRÈS (fixed)
cache_ids = torch.clamp(cache_ids, 0, max)  # Shape: (B, G) per GPU

# Accelerate gather() across GPUs
gathered = accelerator.gather(cache_ids)
# ✅ Works! Consistent shape (B, G) across all GPUs
```

---

### Forward Pass Impact

**Fonction `model.forward()`** attend `cache_global_ids: (B, G)`:

```python
# src/model.py
def forward(self, input_ids, cache_global_ids=None, ...):
    if cache_global_ids is not None:
        # Attend: (B, G)
        B_cur, G = cache_global_ids.shape
        # ...
        landmark_indices_exp = cache_global_ids.unsqueeze(-1).expand(B_cur, G, D)
        landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)
```

**Avec le bug**:
```python
cache_global_ids shape: (22,)  ← 1D!
# ❌ cache_global_ids.shape crashes (expects 2D)
# OU misinterprets as (B=22, G=implicit?)
# → Incorrect gather operation
# → Wrong landmarks used
```

**Avec le fix**:
```python
cache_global_ids shape: (4, 8)  ← 2D correct!
# ✅ Works as expected
# ✅ Correct gather per sample
```

---

## 💡 Design Lessons

### DON'T: Boolean Mask Indexing for Batched Data

```python
# ❌ NEVER do this with batched tensors
mask = condition(tensor)
tensor = tensor[mask]  # Shape collapse!
```

### DO: Element-wise Operations

```python
# ✅ Use element-wise ops that preserve shape
tensor = torch.clamp(tensor, min_val, max_val)
tensor = torch.where(mask, replacement, tensor)
tensor = tensor.clamp_(min_val, max_val)  # In-place
```

---

## 🔧 Alternative Solutions (Not Used)

### Alternative 1: `torch.where()`
```python
# Also correct, but more verbose
mask = cache_ids >= current_seq_len
cache_ids = torch.where(mask,
                        torch.tensor(current_seq_len - 1, device=cache_ids.device),
                        cache_ids)
```
**Not chosen**: More complex, less readable than `clamp()`

### Alternative 2: Loop per sample
```python
# Correct but slow
for i in range(batch_size):
    mask = cache_ids[i] >= current_seq_len
    cache_ids[i][mask] = current_seq_len - 1
```
**Not chosen**: Much slower, not vectorized

### Alternative 3: Keep mask as separate tensor
```python
# Overly complex
valid_mask = cache_ids < current_seq_len
# Pass both cache_ids and mask to forward
```
**Not chosen**: Complicates API, unnecessary

---

## ✅ Verification Checklist

- [x] Bug identified and understood
- [x] Critical severity recognized (HIGH)
- [x] Fix applied with `torch.clamp()`
- [x] Shape preservation tested
- [x] Value correctness tested
- [x] Edge cases tested
- [x] Distributed training impact analyzed
- [x] No performance regression
- [x] Documentation complete

---

## 📊 Performance Impact

**Before vs After**:
- **Correctness**: ❌ Broken → ✅ Fixed
- **Performance**: Same (both O(1) operations)
- **Memory**: Same
- **Distributed**: ❌ Crash → ✅ Works

**No performance cost, only correctness gain!**

---

## 🙏 Acknowledgments

**Special thanks to the user for:**
- ✅ Identifying this subtle but critical shape bug
- ✅ Providing clear diagnosis ("collapses to flat vector")
- ✅ Pointing out distributed training impact
- ✅ Suggesting `torch.where` or per-row solution

**This is HIGH severity bug catching at its finest!** 🎯

---

## 📖 References

- `scripts/train.py:665-671` - Fixed code
- `tests/test_cache_ids_shape_preservation.py` - Comprehensive tests
- PyTorch docs: `torch.clamp()` - Element-wise clamping

---

**Status**: ✅ CRITICAL FIX APPLIED
**Date**: 2025-10-30
**Severity**: HIGH
**Bug #**: 8
**Impact**: Distributed training, forward pass correctness
**Fix Type**: Shape preservation with `torch.clamp()`
