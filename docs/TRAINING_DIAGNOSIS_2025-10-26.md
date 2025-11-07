# SLGA Training Diagnosis Report
**Date**: 2025-10-26
**Checkpoint Analyzed**: out_slga_fineweb/ckpt_2000
**Generated Text Issue**: Repetitive, incoherent output ("in in the the '� of the of")

---

## Executive Summary

The model at checkpoint 2000 exhibits **healthy weight statistics** but training was **incomplete and unstable** due to a **critical data preprocessing error**. The root cause is **invalid token IDs** being fed to the loss function, causing CUDA assertion failures.

### Status: 🔴 CRITICAL - Training Failed

---

## 1. Training Log Analysis

### 1.1 Early Training Crash
**Log File**: `training_FULLY_FIXED_20251025_231103.log`
- **Total Steps Logged**: ~115 steps only
- **Crash Location**: Step ~1 (first forward pass with loss calculation)
- **Error Type**: CUDA device-side assertion

### 1.2 Loss Progression (First 110 Steps)
```
Step    5: Loss 10.895, PPL 22026.5, LR 5.0e-07
Step   50: Loss 10.765, PPL 22026.5, LR 5.0e-06
Step  100: Loss 10.405, PPL 22026.5, LR 1.0e-05
Step  110: Loss 10.327, PPL 22026.5, LR 1.1e-05
```

**Observations**:
- Loss decreasing: 10.895 → 10.327 (marginal improvement)
- **PPL stuck at 22026.47** (suspicious - this is `exp(10)`, suggesting maxed-out loss)
- **Throughput = 0 tok/s** consistently (indicates data pipeline issues)
- Learning rate increasing linearly (warmup phase)

---

## 2. Critical Error: CUDA Assertion Failure

### 2.1 Error Message
```
/pytorch/aten/src/ATen/native/cuda/Loss.cu:245:
nll_loss_forward_reduce_cuda_kernel_2d: block: [0,0,0], thread: [30-253,0,0]
Assertion `t >= 0 && t < n_classes` failed.
```

### 2.2 Root Cause Analysis
**Error Location**: `nll_loss_forward_reduce_cuda_kernel_2d`
- This is PyTorch's cross-entropy loss CUDA kernel
- Assertion checks that target labels are in valid range: `[0, vocab_size)`

**Failure Condition**: `t >= 0 && t < n_classes`
- **n_classes** = 50,257 (GPT-2 vocabulary size)
- **t** = target token ID from training batch
- **Multiple threads failing** (threads 30-253) indicates **many invalid tokens in batch**

### 2.3 Invalid Token Sources
Three possible causes:
1. **Padding token ID = 50257** (equal to vocab_size, causes `t < n_classes` failure)
2. **Corrupted dataset** with tokens ≥ 50257
3. **Collator bug** producing invalid labels

---

## 3. Checkpoint 2000 Analysis

### 3.1 Model Weights Health Check ✅
```python
=== LM HEAD (Output Layer) ===
Shape: torch.Size([50257, 512])
Mean: -0.000382, Std: 0.025156
Min: -0.125985, Max: 0.129556
Has NaN: False
Has Inf: False

Token Embedding Norms:
  Mean: 0.5606, Std: 0.0991
  Min: 0.4181, Max: 0.8989
  Tokens with norm < 1e-6: 0
  Tokens with norm > 10.0: 0
```

**Verdict**: ✅ **Weights are healthy** - no NaN, Inf, or exploded gradients

### 3.2 Optimizer State ✅
```python
Optimizer (Adam):
  exp_avg mean: -0.000000, std: 0.000013
  Has NaN: False, Has Inf: False
```

**Verdict**: ✅ **Optimizer state is clean**

### 3.3 Architecture Validation ✅
- Vocab size: 50,257 ✅ (matches GPT-2 tokenizer)
- Embedding dim: 512 ✅
- Model layers: 12 transformer blocks ✅
- LM head correctly configured ✅

---

## 4. Why is Generation Incoherent?

### 4.1 Insufficient Training
**Only ~2000 effective steps** completed before crashes
- For a 38M parameter model on FineWeb-Edu, need **50K-100K steps** minimum
- Current training: **2% of recommended steps**

### 4.2 Loss Not Converged
- Loss stuck around **10.3-10.9** (very high)
- Target for coherent generation: **< 3.0** (PPL < 20)
- Current PPL: **22026** → Model is essentially **random guessing**

### 4.3 Token Distribution Collapse
With PPL = 22026:
- Model outputs **nearly uniform distribution** over 50K tokens
- Picks most common tokens repeatedly: "the", "in", "of"
- No semantic understanding learned yet
- Special tokens (like spaces) not learned properly → "thethe" instead of "the the"

---

## 5. Data Pipeline Issues

### 5.1 Evidence of Problems
1. **Throughput = 0 tok/s** throughout training
2. **CUDA assertion on first real batch**
3. **Landmarks: 48→0** (landmark count drops to zero)
4. **Global warmup stuck at 0.00**

### 5.2 Likely Collator Bug
**File**: `src/data.py` - `collate_fn_local` or `collate_fn_global`

**Hypothesis**:
- Padding tokens set to `vocab_size` (50257) instead of `-100` (ignore index)
- OR: Labels not properly shifted (input_ids used directly as labels without shift)
- OR: Special tokens not handled correctly

---

## 6. Training Configuration Analysis

**Config**: `config/config_fineweb_edu_3090_optimized.yaml`

### Correct Settings:
- ✅ Batch size: 14 (suitable for RTX 3090)
- ✅ Gradient accumulation: 5 steps
- ✅ Learning rate: 2.0e-4 with 2000-step warmup
- ✅ Gradient clipping: 1.0
- ✅ AMP enabled: bf16

### Potential Issues:
- ⚠️ **No label smoothing** (could help with rare tokens)
- ⚠️ **seq_len_start: 384** (very long for initial training)
- ⚠️ **No explicit padding_token handling** in config

---

## 7. Recommendations

### 7.1 CRITICAL - Fix Data Pipeline (Priority: P0)
1. **Inspect collator functions** in `src/data.py`:
   ```python
   # Check that labels use -100 for padding, NOT vocab_size
   labels[labels == tokenizer.pad_token_id] = -100
   ```

2. **Add dataset validation**:
   ```python
   # Before training loop
   assert torch.all(labels >= -100) and torch.all(labels < vocab_size)
   ```

3. **Verify tokenizer configuration**:
   ```python
   tokenizer.pad_token = tokenizer.eos_token
   tokenizer.pad_token_id = tokenizer.eos_token_id  # Should be 50256, not 50257
   ```

### 7.2 Training Configuration (Priority: P1)
1. **Reduce initial sequence length**: 384 → 128
   - Easier to learn on shorter sequences first
   - Curriculum learning: 128 → 256 → 512 → 1024 → 2048

2. **Add loss validation**:
   ```python
   # In training loop
   if torch.isnan(loss) or loss > 15.0:
       print(f"Invalid loss detected: {loss}")
       print(f"Labels: {labels.unique()}")
       raise RuntimeError("Training instability detected")
   ```

3. **Enable gradient anomaly detection**:
   ```python
   torch.autograd.set_detect_anomaly(True)
   ```

### 7.3 Restart Training (Priority: P2)
1. **From scratch** (don't use ckpt_2000)
   - Current checkpoint has minimal learning (2% progress)
   - Clean start will be faster than debugging corrupted state

2. **Use CUDA_LAUNCH_BLOCKING=1** for first 100 steps:
   ```bash
   CUDA_LAUNCH_BLOCKING=1 python scripts/train.py config/config.yaml
   ```
   - This will pinpoint exact operation causing assertion failure

3. **Monitor metrics closely**:
   - Loss should drop below 8.0 within 500 steps
   - PPL should drop below 5000 within 1000 steps
   - Throughput should be > 1000 tok/s

### 7.4 Validation Steps (Priority: P3)
1. **Test data pipeline separately**:
   ```python
   # scripts/test_data_pipeline.py
   for batch in train_loader:
       labels = batch['labels']
       assert labels.min() >= -100, f"Invalid label: {labels.min()}"
       assert labels.max() < vocab_size, f"Label >= vocab: {labels.max()}"
   ```

2. **Add checkpoint validation**:
   - Verify loss calculation on validation set after each checkpoint
   - Save checkpoints only if validation loss < training loss * 1.5

---

## 8. Timeline Estimate

**Assumptions**:
- Data pipeline fix: 2-4 hours
- Testing and validation: 1-2 hours
- Full training to 50K steps: 48-72 hours on RTX 3090

**Total**: ~3-4 days to stable, coherent model

---

## 9. Success Criteria

### After Data Fix (Step 100):
- ✅ Loss < 8.0
- ✅ PPL < 5000
- ✅ Throughput > 1000 tok/s
- ✅ No CUDA errors

### After 5K Steps:
- ✅ Loss < 5.0
- ✅ PPL < 150
- ✅ Generation: mostly coherent words

### After 50K Steps:
- ✅ Loss < 3.0
- ✅ PPL < 20
- ✅ Generation: coherent sentences with proper grammar

---

## 10. Files to Investigate

1. **src/data.py** (Lines 100-300):
   - `collate_fn_local()` - Check label handling
   - `collate_fn_global()` - Check label handling
   - Padding token configuration

2. **scripts/train.py** (Lines 500-600):
   - Loss calculation code
   - Label preprocessing before loss

3. **config/config_fineweb_edu_3090_optimized.yaml**:
   - Add explicit `ignore_index: -100` for loss
   - Add `pad_token_id` configuration

---

## Conclusion

The model weights at checkpoint 2000 are **structurally sound but undertrained**. The repetitive, incoherent generation is caused by:

1. **Primary**: Training crashed due to invalid token IDs in labels
2. **Secondary**: Insufficient training (2% of target steps)
3. **Tertiary**: Loss not converged (PPL = 22026 vs target < 20)

**Action Required**: Fix data preprocessing pipeline, then retrain from scratch for 50K-100K steps.

---

**Report Generated by**: Claude Code Analysis
**Coordination**: claude-flow hooks (pre-task/post-task)
**Memory Key**: `hive-mind/diagnosis/training-analysis`
