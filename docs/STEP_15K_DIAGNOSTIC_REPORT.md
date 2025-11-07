# SLGA Training Diagnostic Report - Step 15000

**Date**: 2025-10-24
**Checkpoint**: `out_slga/ckpt_15000`
**Training Progress**: 15,000/100,000 steps (15%)

---

## 🔴 Critical Issues Identified

### 1. **Catastrophic Overfitting**

**Validation Perplexity**: 416-420 (Val Loss: 6.03-6.04)
**Training Perplexity**: 8-19 (Train Loss: 2.1-2.9)
**Gap**: ~3.0 loss units

The model shows severe overfitting with validation perplexity 20-50x higher than training perplexity.

### 2. **Non-Sensical Text Generation**

**Test Cases**:
```
Prompt: "The capital of France is"
Temperature 0.0: "the capital of 2004. It includes Spanish and capital"
Temperature 0.0: "the capital of three years. The capital of France"
Temperature 1.0: "what ?  of the capital of France. It is about the"
```

**Issues**:
- No semantic coherence
- Random token combinations
- No factual knowledge (should generate "Paris")
- Grammatically incoherent

### 3. **Training Instabilities**

**Observed anomalies**:
- **Step 14300**: Throughput drops to 927 tok/s (vs. normal 3400-6300 tok/s)
- **Step 14100**: Loss spike to 2.9447 (PPL: 19.00)
- **Step 14850**: Training time spikes to 6.17s/it (vs. 2.3-3.0s normal)

---

## 🔍 Root Cause Analysis

### Issue #1: Dataset Quality & Diversity

**Current Configuration**:
```yaml
data:
  dataset: "wikimedia/wikipedia"
  subset: "20231101.en"
  split_train: "train[:95%]"
  max_train_samples: null
```

**Problems**:
1. **Wikipedia is highly structured** → Model memorizes patterns rather than learning language
2. **95% train / 5% val split** → High overlap, validation not truly held-out
3. **No data augmentation** → Model sees same examples repeatedly
4. **Single domain** → Poor generalization

**Evidence**:
- Low training loss (2.1-2.9) → Model can memorize
- High validation loss (6.03) → Cannot generalize
- Nonsensical generation → No robust language understanding

### Issue #2: Learned Landmarks Instability

**Current Configuration**:
```yaml
model:
  learned_landmarks: true
  global_k: 24
  diverse_topk: true
train:
  lambda_diversity: 0.02
  lambda_sparsity: 0.001
  global_warmup_start: 1000
  global_warmup_end: 5000
```

**Problems**:
1. **Landmarks are learned** → Can converge to degenerate solutions
2. **Diversity penalty too weak** (0.02) → Landmarks may cluster
3. **Sparsity penalty too weak** (0.001) → All landmarks active, no selectivity
4. **Global warmup ended at step 5000** → Global attention fully active since then

**Evidence** (Code Review):
```python
# src/slga.py:326
if self.diverse_topk and self.training:
    topk_vals, topk_idxs = self._diverse_topk(scores_g, k=k_sel)
```
- Diversity only active during training, not inference
- May cause train/test mismatch

### Issue #3: Model Capacity vs. Task Mismatch

**Current Model**:
- **Parameters**: 38.04M
- **Layers**: 12
- **Embed dim**: 512
- **Heads**: 8
- **Sequence length**: 384 → 2048 (curriculum)

**Problems**:
1. **38M parameters** is relatively small for high-quality LLM
2. **Local window = 128** → 93% of context ignored for long sequences
3. **Global K = 24** → Only 24 tokens have global view
4. **Effective context**: 128 local + 24 global = 152 tokens (7.5% of 2048)

**Comparison**:
- GPT-2 Small (124M params): Can generate coherent text
- Your model (38M params): Cannot generate coherent text

### Issue #4: Optimization & Regularization

**Current Configuration**:
```yaml
train:
  lr: 2.0e-4
  weight_decay: 0.1
  dropout_rate: 0.1
  batch_size: 8
  accum_steps: 4
```

**Problems**:
1. **Weight decay = 0.1** → Very high, may over-regularize
2. **Dropout = 0.1** → Standard, but combined with high WD may be too much
3. **Effective batch = 32** → Small for 38M model
4. **No label smoothing** → Model overconfident on training data

---

## 📊 Training Metrics Analysis

### Loss Progression (Steps 14050-15000)

| Step  | Train Loss | Train PPL | Val Loss | Val PPL | Gap   |
|-------|-----------|-----------|----------|---------|-------|
| 14050 | 2.5213    | 12.45     | -        | -       | -     |
| 14100 | 2.9447    | 19.00     | -        | -       | -     |
| 14500 | 2.1679    | 8.74      | 6.0310   | 416.12  | 3.86  |
| 15000 | 2.5448    | 12.74     | 6.0425   | 420.94  | 3.50  |

**Observations**:
- Training loss oscillates (2.1 → 2.9 → 2.5)
- Validation loss static (~6.04)
- No improvement in validation from 14500 → 15000

### Throughput Issues

| Step  | Tok/s | It/s  | Notes                          |
|-------|-------|-------|--------------------------------|
| 14050 | 3422  | 2.48s | Normal                         |
| 14300 | 927   | 14.08s| **10x slowdown** ⚠️           |
| 14500 | 6022  | 2.39s | Recovered                      |
| 14850 | 3229  | 6.17s | **3x slowdown** ⚠️            |

**Possible causes**:
- Gradient explosion → Triggering gradient clipping
- Landmark selection instability → Expensive top-K operations
- CUDA memory issues → Garbage collection pauses

---

## ✅ Recommendations

### Priority 1: Fix Dataset & Validation Split

**Changes**:
```yaml
data:
  # Use diverse multi-domain dataset
  dataset: "HuggingFaceFW/fineweb-edu"  # High-quality, diverse
  # OR combine multiple sources:
  # - Wikipedia (structured knowledge)
  # - OpenWebText (natural language)
  # - BookCorpus (long-form text)

  # Better train/val split
  split_train: "train[:90%]"
  split_val: "train[90%:95%]"
  split_test: "train[95%:]"  # Add true test set

  # Data augmentation
  random_seed_per_epoch: true
  shuffle_buffer_size: 10000
```

### Priority 2: Stabilize Landmarks

**Option A: Disable learned landmarks temporarily**
```yaml
model:
  learned_landmarks: false  # Use heuristic landmarks
  global_k: 32              # Increase K (cheaper without learning)
```

**Option B: Strengthen landmark regularization**
```yaml
train:
  lambda_diversity: 0.1     # 0.02 → 0.1 (5x stronger)
  lambda_sparsity: 0.01     # 0.001 → 0.01 (10x stronger)

  # Slower global warmup
  global_warmup_start: 5000
  global_warmup_end: 20000  # Much longer ramp-up
```

**Option C: Fix diversity at inference**
```python
# src/slga.py:326
# Change to:
if self.diverse_topk:  # Remove "and self.training"
    topk_vals, topk_idxs = self._diverse_topk(scores_g, k=k_sel)
```

### Priority 3: Improve Regularization

**Changes**:
```yaml
train:
  weight_decay: 0.01        # 0.1 → 0.01 (reduce over-regularization)
  dropout_rate: 0.1         # Keep same

  # Add label smoothing
  label_smoothing: 0.1      # Reduce overconfidence

  # Increase effective batch size
  batch_size: 16            # 8 → 16 (if GPU allows)
  accum_steps: 4            # Keep same
  # Effective batch = 64 (was 32)
```

### Priority 4: Diagnostic Tools

**Create validation scripts**:
```bash
# 1. Test diverse prompts
python scripts/test_generation.py \
  --checkpoint out_slga/ckpt_15000 \
  --prompts-file data/test_prompts.txt

# 2. Analyze landmark behavior
python scripts/analyze_landmarks.py \
  --checkpoint out_slga/ckpt_15000 \
  --visualize

# 3. Compare with baseline
python scripts/compare_checkpoints.py \
  --ckpt1 out_slga/ckpt_11000 \
  --ckpt2 out_slga/ckpt_15000
```

### Priority 5: Model Architecture

**Considerations**:
1. **Increase capacity**: 38M → 100M+ parameters
   - More layers: 12 → 16-24
   - Wider embeddings: 512 → 768

2. **Adjust attention ratios**:
   ```yaml
   local_window: 256     # 128 → 256 (more local context)
   global_k: 32          # 24 → 32 (more global context)
   ```

3. **Add attention mechanisms**:
   - Rotary position embeddings (RoPE)
   - Flash attention for efficiency
   - Multi-query attention (MQA)

---

## 🎯 Immediate Action Plan

### Phase 1: Quick Fixes (1-2 hours)

1. **Test with learned_landmarks=false**
   ```bash
   # Modify config.yaml
   # learned_landmarks: false
   # Resume training from step 15000
   ```

2. **Reduce weight decay**
   ```yaml
   weight_decay: 0.01
   ```

3. **Add test prompts**
   - Create comprehensive prompt set
   - Include factual, creative, reasoning tasks

### Phase 2: Dataset Improvement (4-8 hours)

1. **Download better dataset**
   ```python
   # fineweb-edu: High-quality educational content
   # OR OpenWebText + Wikipedia mix
   ```

2. **Re-split data properly**
   - True held-out validation set
   - Separate test set

3. **Restart training from scratch or early checkpoint**

### Phase 3: Long-term Fixes (1-2 days)

1. **Increase model capacity**
   - Scale to 100M+ parameters
   - More layers and wider embeddings

2. **Implement advanced techniques**
   - RoPE embeddings
   - Flash attention
   - Better landmark selection

3. **Comprehensive evaluation**
   - Perplexity on multiple domains
   - Generation quality metrics
   - Attention analysis

---

## 📈 Success Metrics

**Target values for checkpoint 20K-30K**:

| Metric              | Current | Target   | Status |
|---------------------|---------|----------|--------|
| Validation PPL      | 420     | < 30     | ❌     |
| Train/Val Gap       | 3.0     | < 0.5    | ❌     |
| Generation Quality  | Incoherent | Coherent | ❌   |
| Throughput Stability| Unstable | Stable   | ⚠️     |
| Landmark Diversity  | Unknown | > 0.7    | ❓     |

---

## 🔗 Related Files

- Training logs: Terminal output (Steps 14050-15000)
- Model code: `src/model.py`, `src/slga.py`
- Config: `config.yaml`
- Inference: `scripts/generate_fixed.py`

---

## 📝 Notes

1. **The model is NOT learning language properly**
   - It's memorizing training patterns
   - Cannot generalize to validation data
   - Produces nonsensical text

2. **Main suspect: Dataset quality + Overfitting**
   - Wikipedia alone is insufficient
   - Need diverse, high-quality data

3. **Secondary issue: Landmark instability**
   - May be converging to bad local minima
   - Test with learned_landmarks=false first

4. **This requires intervention NOW**
   - Continuing training to 100K steps will NOT fix this
   - Model needs fundamental changes

---

**Recommendation**: Stop current training, fix dataset and landmarks, restart from checkpoint 5000-10000 or from scratch.
