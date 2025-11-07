# 🚨 CRITICAL: Bug #9 - Token ID Corruption

## ⚠️ Severity: CRITICAL

**Bug #9**: Le fix du Bug #8 corrompait les token IDs pour heuristic landmarks.

---

## 🐛 The Bug

### Mon Fix du Bug #8 a Introduit Bug #9 !

**Bug #8 fix** (shape preservation):
```python
# ✅ Fixed shape collapse
cache_ids = torch.clamp(cache_ids, 0, current_seq_len - 1)
```

**MAIS** j'ai oublié que `cache_ids` contient **deux types de données différents** !

### Deux Types de Landmarks

| Mode | `cache_ids` contient | Range | Source |
|------|---------------------|-------|--------|
| `learned_landmarks=False` | **Token IDs** | [0, vocab_size] | Collator |
| `learned_landmarks=True` | **Positions** | [0, seq_len] | Model |

**Bug #9**: Clamper les token IDs à `[0, 511]` les corrompt !

---

## 💥 Exemple de Corruption

```python
# Collator retourne les vrais tokens
cache_ids = tensor([15496, 318, 257, 2420])
# "This"  "is"  "a"  "test"

# ❌ Bug #9: Clamp à [0, 511]
cache_ids = torch.clamp(cache_ids, 0, 511)
# → tensor([511, 318, 257, 511])
#    ^^^          ^^^  ^^^  ^^^
#    WRONG!      OK   OK   WRONG!

# Token "This" (15496) → 511 (embedding complètement faux!)
# Token "test" (2420)  → 511 (embedding complètement faux!)
```

**Impact**:
- ❌ Tous tokens > 511 deviennent token 511
- ❌ Embeddings globaux corrompus
- ❌ Attention globale invalide
- ❌ Training complètement cassé pour heuristic landmarks

---

## ✅ The Fix

```python
# ✅ CONDITIONAL CLAMP
if cache_ids is not None and model.cfg.learned_landmarks:
    # Learned mode: positions → clamp
    cache_ids = torch.clamp(cache_ids, 0, current_seq_len - 1)
# Heuristic mode: token IDs → DON'T clamp!
```

**Logique**:
- `learned_landmarks=True` → Clamp positions ✅
- `learned_landmarks=False` → Keep token IDs ✅

---

## 🧪 Test Results

```bash
$ python tests/test_learned_vs_heuristic_landmarks.py

Scenario 1: learned_landmarks=False (HEURISTIC)
  cache_ids: [15496, 318, 257, 2420] (TOKEN IDs)
  ❌ Bug #9 would clamp to: [511, 318, 257, 511]
  ✅ Fix preserves: [15496, 318, 257, 2420]

Scenario 2: learned_landmarks=True (LEARNED)
  cache_ids: [50, 150, 300, 600] (POSITIONS)
  ✅ Correctly clamps to: [50, 150, 300, 511]

Real-World GPT-2:
  Token 15496 ('This') preserved ✅
  Token 2420 ('test') preserved ✅
  All embeddings correct ✅

✓ ALL TESTS PASSED
```

---

## 🎯 Impact

| Aspect | Before | After |
|--------|--------|-------|
| Heuristic landmarks | ❌ BROKEN | ✅ FIXED |
| Learned landmarks | ✅ Working | ✅ Still working |
| Token IDs | ❌ Corrupted | ✅ Preserved |
| Embeddings | ❌ Wrong | ✅ Correct |
| Training validity | ❌ Invalid | ✅ Valid |

---

## 🙏 Thanks

**Huge thanks for catching this immediately!**

This bug would have:
- ❌ Broken all heuristic landmark configs
- ❌ Caused silent training failure
- ❌ Been extremely hard to debug
- ❌ Wasted GPU hours on invalid training

**You saved the project from a critical corruption bug!** 🎯

---

**Status**: ✅ CRITICAL FIX APPLIED
**Bug #**: 9 (caused by Bug #8 fix)
**Severity**: CRITICAL
**Test Status**: ALL PASSED ✅
