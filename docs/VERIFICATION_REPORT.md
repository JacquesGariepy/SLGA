# 🔍 SLGA Training Verification Report

## Purpose
This document provides step-by-step verification that the dataset preprocessing, model architecture, and training pipeline are correctly configured before starting a full training run.

---

## Prerequisites

```bash
# Activate environment
conda activate slga

# Verify in correct directory
cd /mnt/d/ai/SLGA

# Verify config exists
ls -la config_3090.yaml
```

---

## Verification Steps

### Step 1: Inspect Training Batches

**Command:**
```bash
python scripts/inspect_training_batch.py --config config_3090.yaml --num-batches 3
```

**Expected Output:**

#### ✅ Dataset Loading
```
✓ Dataset loaded: 1000 training samples
  (Limited to 1000 samples for inspection)
✓ Dataloaders built
  Total batches: 125
```

#### ✅ Batch Shape Validation
```
📊 Shapes:
  input_ids: torch.Size([8, 512])
  labels: torch.Size([8, 512])
```
- Batch size: 8 (from config_3090.yaml)
- Sequence length: 512 (seq_len_start)

#### ✅ Token ID Validation
```
📈 Statistics:
  Batch size: 8
  Sequence length: 512
  Min token ID: 0
  Max token ID: 50256
  Vocab size: 50257

✅ Token IDs valid: All in range [0, 50257)
```
- **CRITICAL**: No invalid tokens (>= vocab_size or < 0)
- Max token should be 50256 (GPT2 has 50257 tokens, 0-indexed)

#### ✅ Padding Check
```
📝 Padding:
  Pad token ID: 50256
  Padding tokens: 245 / 4096 (6.0%)
```
- Reasonable padding percentage (< 20% is good)
- Pad token ID should be 50256 (GPT2 default)

#### ✅ Labels Validation
```
🎯 Labels:
  Min label: -100
  Max label: 50256
  Ignore index (-100): 253 tokens
```
- Labels should match input_ids range PLUS -100 for ignored positions
- Ignore index count should be similar to padding count + 1 per sequence (first token)

#### ✅ Text Decoding
```
📖 First Example (input_ids[0]):
  Length: 389 tokens
  Text (first 500 chars):
  Anarchism is a political philosophy and movement that is skeptical of all justifications for authority and seeks to abolish the institutions it claims maintain unnecessary coercion and hierarchy, including nation states, and capitalism. Anarchism advocates for the replacement of the state with stateless societies and voluntary free associations...
```
- **CRITICAL**: Text should be readable English (or target language)
- Should NOT be gibberish, repeated tokens, or corrupted
- Should be coherent Wikipedia-style text

#### ✅ Token Distribution
```
📊 Token Distribution:
  Unique tokens in first example: 187
  Most common tokens:
    ID   262: ' the' (23 times)
    ID   286: ' and' (18 times)
    ID   318: ' is' (12 times)
    ID   257: ' a' (11 times)
    ID   284: ' of' (10 times)
```
- Should show common English words (articles, conjunctions)
- Unique token count should be reasonable (not too low = repetitive, not too high = normal text)

#### ✅ Cache Global IDs (if present)
```
🎯 Cache Global IDs:
  Shape: torch.Size([8, 24])
  Min: 0
  Max: 511
  Should be < seq_len (512): 1
  ✅ All cache IDs valid
  First example cache IDs: [12, 45, 67, 89, ...]
```
- Only appears if `learned_landmarks: false` in config
- All IDs must be < sequence length

---

### Step 2: Quick Training Test (50 steps)

**Purpose**: Verify training loop runs without errors and metrics are sensible

**Command:**
```bash
python scripts/train.py --max-steps 50
```

**Expected Output:**

#### ✅ Initialization
```
✓ Config loaded from config.yaml
✓ Device: cuda (NVIDIA GeForce RTX 3090)
✓ AMP enabled with dtype=torch.bfloat16
✓ Model initialized: 61.3M parameters
✓ Optimizer: AdamW (lr=2.00e-04, betas=(0.9, 0.95))
✓ Scheduler: Warmup 2000 steps, then cosine decay
✓ Dataloader: 8 samples/batch, 8 accum steps = 64 effective batch
```

#### ✅ Training Metrics (First Steps)
```
Step     10 | Loss: 10.8234 | PPL: 50234.12 | LR: 1.00e-06 | GradNorm:  3.45
            | SeqLen:  512 | GW: 0.00 | Landmarks:  24 | GPU: 16.2GB | Tok/s:  4123

Step     20 | Loss:  9.9456 | PPL: 20567.89 | LR: 2.00e-06 | GradNorm:  2.87
            | SeqLen:  512 | GW: 0.00 | Landmarks:  24 | GPU: 16.3GB | Tok/s:  4256

Step     50 | Loss:  8.7234 | PPL:  6123.45 | LR: 5.00e-06 | GradNorm:  2.12
            | SeqLen:  512 | GW: 0.00 | Landmarks:  24 | GPU: 16.4GB | Tok/s:  4389
```

**Key Indicators**:
- ✅ **Loss decreasing**: 10.8 → 9.9 → 8.7 (even in first 50 steps)
- ✅ **Perplexity decreasing**: Should drop from ~50K to ~6K
- ✅ **GradNorm stable**: Between 1.0 and 5.0 is healthy
- ✅ **Global Weight (GW) = 0.00**: Correct (warmup starts at step 30000)
- ✅ **GPU Memory stable**: ~16GB (66% of 24GB), not increasing
- ✅ **Throughput consistent**: ~4000-5000 tokens/sec

**Red Flags**:
- ❌ Loss not decreasing or increasing
- ❌ GradNorm > 50 (exploding gradients)
- ❌ GradNorm < 0.001 (vanishing gradients)
- ❌ GPU memory increasing each step (memory leak)
- ❌ Perplexity staying > 50000 after 50 steps

#### ✅ TensorBoard Files Created
```
ls -la out_slga/tensorboard/
# Should show: events.out.tfevents.* file
```

---

### Step 3: Verify TensorBoard Logging

**Command (separate terminal):**
```bash
tensorboard --logdir out_slga/tensorboard --port 6006
```

**Then open**: http://localhost:6006

**Expected Graphs**:
- ✅ `train/loss` - Should show decreasing trend
- ✅ `train/perplexity` - Should show decreasing trend
- ✅ `train/learning_rate` - Should show warmup curve
- ✅ `train/grad_norm` - Should be stable ~1-5
- ✅ `landmarks/num_selected` - Should show ~24 landmarks
- ✅ `perf/steps_per_sec` - Should show ~1.0-1.5
- ✅ `perf/tokens_per_sec` - Should show ~4000-6000
- ✅ `perf/gpu_memory_allocated_gb` - Should show ~16GB

**If graphs are empty or missing**:
- ❌ TensorBoard logging not working (check writer.add_scalar calls)

---

### Step 4: Architecture Diagnostic

**Command:**
```bash
python scripts/diagnose.py
```

**Expected Output:**

```
================================================================================
SLGA Architecture Diagnostic
================================================================================
Config: config.yaml

🔍 Model Architecture:
  Total Parameters: 61.3M
  Embedding dim: 512
  Num layers: 12
  Num heads: 8
  Local window: 128
  Global top-k: 24
  Max sequence length: 2048

✅ Architecture Tests:

[1/6] Forward Pass (seq_len=512)
  ✅ Output shape: torch.Size([2, 512, 50257])
  ✅ No NaN/Inf in output
  ✅ Global weight scaling works

[2/6] Forward Pass (seq_len=1024)
  ✅ Output shape: torch.Size([2, 1024, 50257])
  ✅ Memory usage acceptable

[3/6] Landmark Selection
  ✅ Landmarks shape: torch.Size([2, 24, 512])
  ✅ Dynamic landmark extraction per layer
  ✅ Landmark scores reasonable

[4/6] Attention Mechanism
  ✅ Local attention working
  ✅ Global attention working
  ✅ Gated fusion working

[5/6] Generation Test
  ✅ Generated 20 tokens
  ✅ No NaN/Inf during generation
  ✅ Output: "The cat sat on the roof of the house and looked at the sky with a smile"

[6/6] Gradient Flow
  ✅ Gradients computed successfully
  ✅ Grad norms: min=0.001, max=2.345, mean=0.234
  ✅ No vanishing/exploding gradients

================================================================================
✅ ALL TESTS PASSED - Architecture is correct!
================================================================================
```

**Red Flags**:
- ❌ Any test fails
- ❌ NaN/Inf in outputs
- ❌ Generation produces gibberish (e.g., "the the the the...")
- ❌ Gradient vanishing (mean < 0.0001) or exploding (max > 100)

---

## Verification Checklist

### Before Starting Full Training

- [ ] **Dataset Inspection**: All 3 batches validated, no invalid tokens
- [ ] **Text Decoding**: Decoded text is coherent Wikipedia content
- [ ] **Quick Training**: 50 steps completed without errors
- [ ] **Loss Decreasing**: Loss drops within first 50 steps
- [ ] **TensorBoard Logging**: All metrics visible in TensorBoard
- [ ] **Architecture Diagnostic**: All 6 tests pass
- [ ] **GPU Memory Stable**: Memory not increasing over time
- [ ] **Global Weight Implementation**: GW=0.00 before step 30000
- [ ] **Config Correct**: Using config_3090.yaml (batch_size=8)
- [ ] **Old Checkpoints Cleaned**: Removed out_slga/ckpt_* from buggy run

---

## Expected Results After Fixes

With all corrections applied, training should show:

| Checkpoint | Steps | Time | Loss | Perplexity | Quality |
|------------|-------|------|------|------------|---------|
| Initial | 0 | 0h | ~11.0 | ~60000 | Random |
| Early | 100 | 5min | ~8.5 | ~5000 | Slightly less random |
| ckpt_2000 | 2000 | 1h | ~7.0-7.5 | ~800-2000 | Recognizable words |
| ckpt_10000 | 10000 | 5h | ~5.5-6.5 | ~150-400 | Partial sentences |
| ckpt_30000 | 30000 | 15h | ~4.5-5.5 | ~50-150 | Coherent text |
| ckpt_50000 | 50000 | 25h | ~3.5-4.5 | ~30-60 | Fluent text |

**With old buggy code**:
- All checkpoints: PPL ~10,000-15,000 (unusable)

---

## If Verification Fails

### Problem: Invalid Tokens Detected

**Cause**: Tokenizer mismatch or data corruption

**Fix**:
```bash
# Check tokenizer vocab size
python -c "from transformers import GPT2Tokenizer; t = GPT2Tokenizer.from_pretrained('gpt2'); print(f'Vocab size: {len(t)}')"
# Should output: Vocab size: 50257

# Verify config matches
grep "vocab_size" config_3090.yaml
# Should output: vocab_size: 50257
```

### Problem: Gibberish Text When Decoding

**Cause**: Tokenizer encoding/decoding issue

**Fix**:
```bash
# Test tokenizer
python -c "
from transformers import GPT2Tokenizer
t = GPT2Tokenizer.from_pretrained('gpt2')
text = 'The cat sat on the mat'
ids = t.encode(text)
decoded = t.decode(ids)
print(f'Original: {text}')
print(f'Decoded:  {decoded}')
"
```

### Problem: TensorBoard Graphs Empty

**Cause**: writer.add_scalar not called or wrong path

**Fix**:
```bash
# Check train.py has add_scalar calls
grep "writer.add_scalar" scripts/train.py | wc -l
# Should output: 15+ lines

# Check TensorBoard directory
ls -la out_slga/tensorboard/
# Should have events.out.tfevents.* file
```

### Problem: Loss Not Decreasing

**Cause**: Learning rate too low, or global_weight not passed

**Fix**:
```bash
# Verify global_weight is passed in train.py
grep "global_weight=global_weight" scripts/train.py
# Should find line in model forward call

# Check learning rate in logs
# LR should be > 0 and increasing during warmup
```

---

## Final Confirmation

**Before starting full 100K step training, confirm**:

✅ All verification steps completed
✅ All checklists marked complete
✅ No red flags encountered
✅ TensorBoard shows all metrics
✅ Quick test shows loss decreasing
✅ Architecture diagnostic passes all tests

**Then proceed with**:
```bash
# Clean old checkpoints
bash scripts/clean_restart.sh

# Start full training
python scripts/train.py

# (Separate terminal) Monitor
tensorboard --logdir out_slga/tensorboard --port 6006

# (Separate terminal - optional) Real-time dashboard
python scripts/monitor.py
```

---

## 🎉 Success Criteria

**Your dataset preprocessing is PERFECT if**:

✅ No invalid token IDs detected
✅ Decoded text is coherent Wikipedia content
✅ Padding percentage is reasonable (< 20%)
✅ Labels correctly aligned with input_ids
✅ Token distribution shows common English words
✅ All 3 batches pass validation
✅ Quick training test shows loss decreasing
✅ TensorBoard metrics all logging correctly
✅ Architecture diagnostic passes all 6 tests
✅ No NaN/Inf in any outputs
✅ GPU memory stable

**If all above are true**: 🚀 **READY FOR FULL TRAINING!**
