# Quick Reference: Hive Mind Fixes (2025-10-30)

## 🎯 Three Critical Fixes Applied

### 1️⃣ Gumbel Training Activation
**File**: `src/model.py:255`

```python
# ✅ FIXED: Now uses Gumbel during training
landmark_indices, _, landmark_scores = self.landmark_selector(x, use_gumbel=self.training)
```

**Impact**: Proper gradient flow during landmark selection training

---

### 2️⃣ EOS Token Handling
**File**: `scripts/generate.py`

```bash
# Default: Stop on EOS (auto-detect from tokenizer)
python scripts/generate.py \
    --checkpoint out_slga/ckpt_11000 \
    --config config.yaml \
    --prompt "Your prompt"

# Explicit: Stop on EOS
python scripts/generate.py --stop-on-eos

# NEW: Continue after EOS
python scripts/generate.py --no-stop-on-eos

# Manual EOS token ID
python scripts/generate.py --eos-token-id 50256
```

**Impact**: Generation stops naturally at EOS token

---

### 3️⃣ Top-K + Top-P Warning
**Behavior**: Non-blocking warning when using both filters

```bash
# Will show warning (but still runs)
python scripts/generate.py \
    --top-k 50 \
    --top-p 0.95

# Output:
# ⚠️  Using both top_k=50 and top_p=0.95 simultaneously
# Recommendation: Use only one for most use cases
```

**Impact**: Better user understanding of sampling behavior

---

## 🧪 Quick Test

```bash
python tests/test_all_three_fixes.py
```

Expected output:
```
🎉 ALL TESTS PASSED! All three fixes are working correctly.
```

---

## 📋 Summary

| Fix | File | Line | Status |
|-----|------|------|--------|
| Gumbel Training | src/model.py | 255 | ✅ |
| EOS Handling | scripts/generate.py | Multiple | ✅ |
| Top-K+Top-P Warning | scripts/generate.py | Multiple | ✅ |

**All fixes validated and working!** ✅

See `docs/HIVE_MIND_FIXES_2025-10-30.md` for full documentation.
