# Quick Fix: Validation KeyError

## Problem

```
KeyError: 'input_ids'
File "/mnt/d/ai/SLGA/scripts/train.py", line 209, in collate_val_reduced
    ids = ex["input_ids"]
```

## Root Cause

Python is using cached `.pyc` files with old buggy code, even though source was fixed.

## Solution

### Step 1: Clear Python Cache

```bash
cd /mnt/d/ai/SLGA

# Clear all Python cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

echo "✅ Cache cleared"
```

### Step 2: Verify Source Code is Fixed

```bash
# Should show: texts = [ex.get("text", "") for ex in examples]
grep -A 2 "Extraire textes" scripts/train.py
```

Expected output:
```python
# Extraire textes (dataset raw retourne ex["text"], pas ex["input_ids"])
texts = [ex.get("text", "") for ex in examples]
```

### Step 3: Restart Training

```bash
python scripts/train.py \
  --config config/config_fineweb_edu_3090_optimized.yaml \
  --max-steps 500
```

## Verification

Training should now show:
```
Validation config: batch_size=7, seq_len=512 (train: up to 2048)
=== Validation ===
  GPU Memory: 17-18 GB allocated, ...
  Validation: 10/10 batches...
✅ Validation Results: Loss: X.XX, PPL: XX.X
```

## If Still Fails

If KeyError persists after clearing cache:

1. **Check Python version**:
   ```bash
   python --version  # Should be 3.12
   ```

2. **Restart Python virtual environment**:
   ```bash
   deactivate
   source ~/.venvs/slga/bin/activate
   ```

3. **Force recompile**:
   ```bash
   python -B scripts/train.py --config config/config_fineweb_edu_3090_optimized.yaml --max-steps 500
   ```

   The `-B` flag prevents Python from writing .pyc files.

## Technical Details

### What Was Fixed

**Before (Buggy)**:
```python
def collate_val_reduced(examples):
    for ex in examples:
        ids = ex["input_ids"]  # ❌ KeyError: dataset has "text", not "input_ids"
        ...
```

**After (Fixed)**:
```python
def collate_val_reduced(examples):
    # Extract text from raw dataset
    texts = [ex.get("text", "") for ex in examples]

    # Tokenize to get input_ids
    encoded = tokenizer(
        texts,
        max_length=512 + 1,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    ...
```

### Why Cache Was the Problem

Python compiles `.py` files to `.pyc` bytecode for faster execution. When you modify source code, Python should detect changes and recompile, but sometimes:

1. File timestamps are not updated correctly (especially on WSL/network drives)
2. Import system caches module in memory
3. `.pyc` files have stale timestamps

**Solution**: Always clear cache after fixing bugs:
```bash
find . -name "*.pyc" -delete
find . -type d -name __pycache__ -rm -rf
```
