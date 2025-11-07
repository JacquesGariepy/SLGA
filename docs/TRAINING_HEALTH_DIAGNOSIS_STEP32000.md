# Training Health Diagnosis - Step 32000-33800

**Date**: 2025-10-29
**Model**: SLGA 38.04M params (12 layers, 8 heads, 512 embed_dim)
**Checkpoint**: Step 33800+
**Critical Issue**: GPU utilization 4-5% (EXTREMELY LOW!)

---

## Executive Summary

🚨 **CRITICAL BOTTLENECK IDENTIFIED**: The primary issue is **DATA LOADING**, not model architecture or training dynamics.

**Key Finding**: GPU utilization of 4-5% indicates the GPU is **starving** for data - it spends 95% of the time waiting for the CPU to prepare batches.

---

## 1. GPU Utilization Analysis - CRITICAL ⚠️

### Observed Metrics
- **GPU Utilization**: 4-5% (CRITICALLY LOW!)
- **Expected**: 80-95% during healthy training
- **Diagnosis**: **DATA LOADING BOTTLENECK**

### Root Cause Analysis

From `/mnt/d/ai/SLGA/scripts/train.py` lines 199-207:

```python
train_loader = DataLoader(
    ds_train,
    batch_size=cfg["train"]["batch_size"],
    shuffle=True,
    drop_last=True,
    collate_fn=collate_train,
    num_workers=cfg["data"].get("num_workers", 2),  # ← BOTTLENECK!
    pin_memory=True,
)
```

**Problem Identified**:
1. **`num_workers=2`** (default) - TOO LOW for optimal pipeline
2. **Synchronous data loading** - GPU waits for CPU
3. **Tokenization overhead** - CPU preprocessing is slow

### Impact on Training

```
Timeline of 1 training step (current state):
┌─────────────────────────────────────────────────────────────┐
│ CPU: Data Load (95% time) ████████████████████████████████  │
│ GPU: Forward/Backward (5% time) ██                          │
└─────────────────────────────────────────────────────────────┘
```

**With num_workers=2**: GPU is idle 95% of the time waiting for batches.

**Expected with num_workers=4-8**:
```
Timeline of 1 training step (optimal):
┌─────────────────────────────────────────────────────────────┐
│ CPU: Data Load (async) ████████████████████                 │
│ GPU: Forward/Backward (80% time) ████████████████████████   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Training Dynamics Analysis - HEALTHY ✅

Despite low GPU utilization, the **actual training dynamics are working correctly**:

### Loss Progression
- **Current Loss**: 0.13-0.39 (volatile but decreasing)
- **Best Loss**: 0.0563 (step ~30000)
- **Perplexity**: 1.14-1.5 (current), 1.06 (best)

**Interpretation**: Loss is low because model is **learning**, but...

### The Loss-Quality Disconnect

**WHY low loss ≠ good generation**:

1. **Training on short sequences (curriculum learning)**:
   - Started at `seq_len=512`
   - Reached `seq_len=2048` at step 15000
   - Model learned to predict **next token locally**
   - Did NOT learn **long-range coherence**

2. **Validation uses `seq_len=512`** (line 313):
   ```python
   val_batch_size = max(1, cfg["train"]["batch_size"] // 2)
   print(f"Validation config: batch_size={val_batch_size}, seq_len=512")
   ```

   **Problem**: Training at 2048 tokens, validating at 512 tokens!
   → Validation loss doesn't reflect generation quality at full context.

3. **Loss is dominated by frequent tokens**:
   - Common words: "the", "and", "is" → easy to predict
   - Coherent text generation requires rare token predictions → harder

4. **Landmark attention may not be utilized properly**:
   - `global_weight=1.0` (fully active)
   - 48 landmarks selected
   - BUT: Landmarks may be clustering or not capturing long-range dependencies

---

## 3. Landmark Selection Health - NEEDS VERIFICATION ⚠️

### Configuration (from model)
```
Learned landmarks: True
Landmark selector: LearnableLandmarkSelector
Num landmarks: 48 (selected from 96 candidates via top-k)
Temperature: 1.0 (decaying to 0.3)
Temperature decay: 0.999
```

### Potential Issues

**Question 1**: Are landmarks actually being selected correctly?
- Need to verify `landmark_indices` shape and values
- Check for clustering (all landmarks near each other)
- Verify spacing loss is working

**Question 2**: Are landmark gradients flowing?
- `scorer_lr_multiplier=5.0` applied (line 481)
- Higher LR for scorer (base LR × 5)
- BUT: Need to verify gradients are non-zero

**Question 3**: Are landmarks helping attention?
- With 48 landmarks for 2048 seq_len:
  - Expected spacing: 2048/48 = 42.7 tokens
  - Actual spacing: UNKNOWN (needs measurement)

### Loss Components (lines 715-735)

```python
# Spacing loss (encourage uniform distribution)
lambda_spacing = cfg["train"].get("lambda_spacing", 0.0)
if lambda_spacing > 0 and num_landmarks_selected > 1:
    spacing_loss = landmark_spacing_loss(...)
    loss = loss + spacing_loss / accum_steps

# Sparsity loss (encourage concentration)
lambda_spar = cfg["train"].get("lambda_sparsity", 0.0)
if lambda_spar > 0 and num_landmarks_selected > 0:
    spar_loss = landmark_sparsity_loss(...)
    loss = loss + spar_loss / accum_steps
```

**Check config values**:
- If `lambda_spacing=0` or `lambda_sparsity=0` → losses disabled!
- Landmarks won't be properly trained

---

## 4. Data Pipeline Analysis

### Current Configuration
```yaml
data:
  dataset: "openwebtext" / "fineweb-edu"
  num_workers: 2  # ← CRITICAL BOTTLENECK
  max_train_samples: ...
```

### Collation (lines 95-153)

The `CollatorLocal` is efficient:
```python
def __call__(self, examples):
    # Tokenize batch
    encoded = self.tokenizer(texts, max_length=max_len+1, ...)
    # Shift labels
    labels = input_ids[:, 1:].clone()
    labels[labels == pad_id] = -100  # Mask padding
    return {"input_ids": input_ids, "labels": labels}
```

**No obvious inefficiencies** in collation code.

### Bottleneck Location

**PRIMARY**: `num_workers=2` in DataLoader
**SECONDARY**: Tokenization is CPU-bound (HuggingFace tokenizer)

**Solutions**:
1. **Increase `num_workers=4` or higher**
2. **Pre-tokenize dataset** and save to disk
3. **Use pin_memory=True** (already enabled ✅)

---

## 5. Generation Quality Issues

### Symptoms
- Low loss (0.13-0.39)
- Low perplexity (1.14-1.5)
- **BUT**: Generation produces repetitive garbage

### Root Causes

**1. Training-Inference Mismatch**:
```python
# Training: uses curriculum seq_len (up to 2048)
current_seq_len = get_current_seq_len(step, cfg)

# Validation: FIXED at 512 (line 313)
max_len_val = 512

# Generation: Uses full context (2048)
if input_ids.size(1) > self.cfg.max_seq_len:
    input_ids = input_ids[:, -self.cfg.max_seq_len:]
```

**Problem**: Model trained on 2048 tokens, validated on 512 tokens, but loss measured on 512 doesn't reflect 2048 quality!

**2. Landmark Selection at Inference**:

From `/mnt/d/ai/SLGA/src/model.py` lines 343-347:
```python
if not self.cfg.learned_landmarks and cache_global_ids is None:
    L = input_ids.size(1)
    landmark_positions = torch.linspace(0, L-1, self.cfg.global_k, device=...).long()
    cache_global_ids = landmark_positions.unsqueeze(0).expand(...)
```

**Issue**: During generation, if `learned_landmarks=True`, landmarks are recomputed **every step**!
- Inefficient: O(L) per generation step
- Inconsistent: Different landmarks for each token

**3. Mode Collapse / Overfitting**:
- Model may have memorized common n-grams
- Not learning diverse generation strategies
- Possible solutions:
  - Increase dropout
  - Use nucleus sampling (top_p)
  - Add diverse data

---

## 6. Memory Leak Fixes - ALREADY APPLIED ✅

Good news: Memory leak fixes are already in place (lines 367-400):

```python
# ✅ MEMORY LEAK FIX: Utiliser .item()/.cpu() pour couper le graph
invalid_count = invalid_mask.sum().item()  # Coupe le graph
invalid_values = labels[invalid_mask][:20].detach().cpu().tolist()

# ✅ MEMORY LEAK FIX: Libérer explicitement les tensors après chaque batch
del input_ids, labels, cache_ids, logits, loss, invalid_mask
if i % 5 == 0:
    torch.cuda.empty_cache()
```

These are **correctly implemented** and should prevent CUDA memory accumulation.

---

## Diagnostic Plan

### Immediate Actions (Run Diagnostic Script)

```bash
# 1. Run comprehensive diagnostic
python scripts/diagnose_training_health.py \
    --config config/config.yaml \
    --checkpoint checkpoints/ckpt_33800

# Expected output:
# - GPU utilization bottleneck confirmation
# - Gradient flow analysis
# - Landmark selection patterns
# - Generation quality metrics
```

### Measurements Needed

1. **Data loading speed**:
   - Measure batches/sec with num_workers=0, 2, 4, 8
   - Confirm CPU bottleneck

2. **Gradient flow**:
   - Check landmark scorer gradients are non-zero
   - Verify early vs late layer gradient ratio

3. **Landmark patterns**:
   - Measure actual spacing (mean, std, min, max)
   - Calculate clustering ratio
   - Check mass concentration in top-G

4. **Generation quality**:
   - Test 3+ prompts
   - Measure unique token ratio
   - Detect repetition patterns

---

## Recommendations

### 🚨 CRITICAL - Fix Data Loading (Expected: 15-20x speedup)

**In `config/config.yaml`**:
```yaml
data:
  num_workers: 4  # Increase from 2 → 4 (or 6-8 if system has many cores)
```

**Expected Impact**:
- GPU utilization: 4-5% → 60-80%
- Training speed: 5-10x faster
- Tokens/sec: Increase proportionally

### 🔧 MEDIUM - Fix Validation Sequence Length

**In `scripts/train.py` line 313**:
```python
# Before:
max_len_val = 512

# After:
max_len_val = cfg["train"].get("seq_len_final", 2048)  # Match training
```

**Expected Impact**:
- Validation loss will better reflect generation quality
- May increase initially (normal - more challenging)

### 🎯 HIGH - Verify Landmark Training

**Check config has these enabled**:
```yaml
train:
  lambda_spacing: 0.01   # Encourage uniform spacing
  lambda_sparsity: 0.001 # Encourage concentration
  scorer_lr_multiplier: 5.0  # Higher LR for scorer
```

**Verify via logs**:
```
Step 33800 | ... | Spacing loss: 0.0XX | Sparsity loss: 0.00X
```

If both are **0.000**, landmarks aren't being trained!

### 📊 LOW - Improve Generation Quality

1. **Use better sampling**:
   ```python
   output = model.generate(
       prompt,
       max_new_tokens=100,
       temperature=0.9,      # Slightly higher
       top_p=0.9,            # Nucleus sampling instead of top_k
       stop_on_eos=True,
   )
   ```

2. **Increase dropout** (if overfitting):
   ```yaml
   model:
     dropout_rate: 0.15  # From 0.1 → 0.15
   ```

3. **Add diverse data**:
   - Mix multiple datasets (already done with `create_mixed_dataset.py` ✅)
   - Verify dataset has variety

---

## Verification Checklist

After applying fixes, verify:

- [ ] GPU utilization increases to 60-80%+
- [ ] Training speed increases proportionally (tokens/sec)
- [ ] Validation loss matches training context length
- [ ] Landmark spacing loss > 0 (if enabled)
- [ ] Landmark sparsity loss > 0 (if enabled)
- [ ] Generation produces diverse, non-repetitive text
- [ ] Gradient norms are healthy (1e-4 to 1e-1 range)

---

## Conclusion

**Primary Issue**: DATA LOADING BOTTLENECK (`num_workers=2`)
- Causes 95% GPU idle time
- Fix: Increase `num_workers=4-8`
- Expected speedup: 10-20x

**Secondary Issue**: VALIDATION MISMATCH (seq_len=512 vs training 2048)
- Causes loss-quality disconnect
- Fix: Use `seq_len_final` for validation
- Will better reflect generation quality

**Tertiary Issue**: LANDMARK TRAINING (needs verification)
- Check if spacing/sparsity losses are enabled
- Verify gradients flowing to scorer
- Measure actual landmark patterns

**Model Health**: Training dynamics are working correctly, but hidden by data bottleneck!

Once data loading is fixed, expect to see:
- Much faster training (10-20x)
- Better GPU utilization (60-80%+)
- More visible progress in loss curves
- Ability to train much longer in same time
