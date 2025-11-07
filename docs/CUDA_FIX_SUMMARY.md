# 🔧 CUDA Assertion Error - Complete Fix Summary

**Date**: 2025-10-25
**Status**: ✅ FIXED - Ready for training restart

---

## 🐛 Root Causes Identified

### 1. **Padding Masking Bug** (Primary Issue)
- **Problem**: Collators were not masking padding tokens with `-100`
- **Impact**: Model learned to predict PAD tokens instead of language
- **Symptom**: Poor generation ("The\nThe\nThe"), high perplexity (601 at step 2,500)

### 2. **Ignore Index Mismatch** (CUDA Crash Cause)
- **Problem**: After fixing collators to use `-100`, loss function still used `ignore_index=pad_id` (50256)
- **Impact**: PyTorch received `-100` labels but tried to process them as valid class indices
- **Symptom**: `Assertion 't >= 0 && t < n_classes' failed` CUDA crash

### 3. **Token Counting Bug** (Validation Issue)
- **Problem**: Validation was counting tokens with `(labels != pad_id)` instead of `(labels != -100)`
- **Impact**: Incorrect validation metrics
- **Symptom**: Potential incorrect perplexity calculations

### 4. **Unicode Corruption** (Data Quality)
- **Problem**: FineWeb-Edu dataset contains `�` (U+FFFD) replacement characters
- **Impact**: Model learning to predict corruption artifacts
- **Fix**: Created `CleanedDataset` wrapper to filter Unicode errors

---

## ✅ Fixes Applied

### **File: `src/data.py`**

#### Fix 1: CollatorLocal (lines 113-125)
```python
# BEFORE:
input_ids = input_ids[:, :-1]
labels = input_ids[:, 1:].clone()
# ❌ No padding masking!

# AFTER:
input_ids = input_ids[:, :self.max_length + 1]  # Truncate first
labels = input_ids[:, 1:].clone()
input_ids = input_ids[:, :-1]

# 🔧 CRITICAL FIX: Mask padding with -100
pad_mask = (labels == self.tokenizer.pad_token_id)
labels[pad_mask] = -100
```

#### Fix 2: CollatorLocalGlobal (lines 215-224)
```python
# Same fix as CollatorLocal
pad_mask = (labels == self.tokenizer.pad_token_id)
labels[pad_mask] = -100
```

#### Fix 3: Unicode Cleaner Integration (lines 65-67)
```python
# Added CleanedDataset wrapper
if clean_unicode and not streaming:
    ds = CleanedDataset(ds, text_key="text")
```

### **File: `src/dataset_cleaner.py`** (NEW)
- Removes `�` replacement characters
- Filters control characters (except \n, \t, \r)
- Normalizes excessive whitespace
- Strips line whitespace

### **File: `scripts/train.py`**

#### Fix 4: cross_entropy_shifted (line 110)
```python
# BEFORE:
loss = F.cross_entropy(
    logits_shifted.view(-1, logits_shifted.size(-1)),
    labels_shifted.view(-1),
    ignore_index=pad_id,  # ❌ WRONG: was 50256
)

# AFTER:
loss = F.cross_entropy(
    logits_shifted.view(-1, logits_shifted.size(-1)),
    labels_shifted.view(-1),
    ignore_index=-100,  # ✅ CORRECT: matches masked labels
)
```

#### Fix 5: Validation Collators (lines 224-237, 277-289)
```python
# Both collate_val_reduced paths now include:
input_ids_final = input_ids[:, :-1]
labels = input_ids[:, 1:].clone()

# 🔧 CRITICAL FIX: Mask pads
pad_mask = (labels == pad_id)
labels[pad_mask] = -100
```

#### Fix 6: Validation Token Counting (line 372)
```python
# BEFORE:
num_tokens = (labels != pad_id).sum().item()  # ❌ WRONG

# AFTER:
num_tokens = (labels != -100).sum().item()  # ✅ CORRECT
```

#### Fix 7: Diagnostic Logging (lines 351-363)
```python
# Added pre-forward validation to catch invalid labels:
invalid_mask = (labels != -100) & ((labels < 0) | (labels >= 50257))
if invalid_mask.any():
    print(f"\n❌ VALIDATION BATCH {i} HAS INVALID LABELS!")
    # ... detailed diagnostics ...
    raise ValueError(f"Invalid labels detected in validation batch {i}")
```

---

## 🧪 Verification Status

### ✅ All Tests Pass:
1. **test_collator_fix.py** - ✅ Collator produces valid labels
2. **test_full_training_step.py** - ✅ Full forward/backward pass works
3. **test_validation_flow.py** - ✅ Validation flow works
4. **debug_labels.py** - ✅ Real batches have valid labels
5. **verify_final_fix.py** - ✅ Complete pipeline with diagnostics

### Example Output:
```
Batch 0: input_ids shape: torch.Size([7, 512]), labels shape: torch.Size([7, 512])
  Label stats:
    Masked (-100): 1081
    Valid tokens: 2503
    Unique values (first 10): [-100, 0, 1, 4, 7, 8, 9, 11, 12, 13]
  ✅ Loss: 10.8767
```

**No CUDA assertions, all labels valid!**

---

## 🚀 Training Restart Instructions

### 1. **Kill Any Running Training Processes**
```bash
# Check for running training
ps aux | grep train.py

# Kill if found
pkill -f train.py

# Verify Python cache cleared
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
```

### 2. **Optional: Start Fresh from Step 0**
```bash
# Archive old broken checkpoints
mv out_slga_fineweb out_slga_fineweb_broken_$(date +%Y%m%d_%H%M%S)

# Training will create new out_slga_fineweb/ directory
```

### 3. **Start Training with Diagnostics**
```bash
# The diagnostic logging is now built into train.py
# It will automatically catch any invalid labels before CUDA crash

python scripts/train.py \
    --config config/config_fineweb_edu_3090_optimized.yaml \
    --resume  # Add this if continuing from a checkpoint
```

### 4. **Monitor for Success**

Watch for these signs of success:

✅ **No CUDA assertion errors**
```
Step 250/40000 | Loss: 6.1234 | PPL: 456.78 | Val PPL: 512.34
```

✅ **Decreasing perplexity**
- Step 5,000: PPL should drop to ~200-300
- Step 10,000: PPL should be ~100-150

✅ **Better generation quality**
```bash
# Test generation at step 5000
python scripts/generate.py --checkpoint out_slga_fineweb/checkpoint_5000.pt

# Expected output (should be coherent):
"The capital of France is Paris. It is known for..."
```

---

## 🔍 If CUDA Error Still Occurs

The diagnostic logging will now catch the exact batch that causes the error:

```python
❌ VALIDATION BATCH 123 HAS INVALID LABELS!
   Invalid count: 45
   Invalid values: [50256, 50256, 50256, ...]
   First 10 positions (batch, seq): [(0, 234), (0, 235), ...]
```

This will tell us:
1. Which batch number failed
2. What the invalid values are
3. Where they appear in the batch

**Then we can debug further with that specific information.**

---

## 📊 Expected Training Behavior

### Before Fixes (Broken):
```
Step 2500: Loss 6.3990, PPL 601.23, Val PPL 846.45
Generation: "The capital of France is of\nThe in the\nThe U."
→ Model learning to predict PAD tokens
```

### After Fixes (Expected):
```
Step 2500: Loss ~5.5-6.0, PPL ~250-400, Val PPL ~300-500
Step 5000: Loss ~4.5-5.0, PPL ~100-150, Val PPL ~120-180
Step 10000: Loss ~3.8-4.5, PPL ~50-90, Val PPL ~60-100

Generation @ step 5000:
"The capital of France is Paris. Paris is the largest city..."
→ Coherent, grammatical text
```

---

## 🎯 Critical Changes Summary

| Component | Before | After | Impact |
|-----------|--------|-------|--------|
| Collators | No masking | Mask with -100 | Proper padding ignore |
| Loss function | ignore_index=50256 | ignore_index=-100 | Matches labels |
| Token counting | labels != 50256 | labels != -100 | Correct metrics |
| Validation | No diagnostics | Pre-check labels | Catch errors early |
| Dataset | Raw FineWeb | Unicode cleaned | Better data quality |

---

## ✅ Status: READY FOR TRAINING

All fixes verified, all tests passing. Training can be restarted with confidence.

**Expected first validation checkpoint (step 250)**: Should complete successfully with diagnostic confirmation.
