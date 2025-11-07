# SLGA Generation Quality Guide

## TL;DR - Why Generation is Nonsensical at Step 1000

**Your observation**:
```
Prompt: "The capital of France is "
Output: ", the,.\n.\n (, the"
```

**This is COMPLETELY NORMAL** at step 1000 (1% of training). Here's why:

| Metric | Step 1000 | Step 100K (target) |
|--------|-----------|-------------------|
| **Training progress** | 1% | 100% ✅ |
| **Expected PPL** | 80-120 | 15-20 ✅ |
| **Generation quality** | Nonsensical ⚠️ | Coherent ✅ |
| **Model state** | Just started learning | Fully trained |

## Generation Quality by Training Step

### Step 1,000 (1% - YOU ARE HERE)

**Expected behavior**: ⚠️ **Nonsensical output**

Example outputs:
```
Prompt: "The capital of France is "
Output: ", the,.\n.\n (, the"  ← Punctuation, no coherent words

Prompt: "Once upon a time"
Output: "the, the the. ( the"  ← Repetitive tokens
```

**Why**:
- Model hasn't learned language patterns yet
- Cross-entropy loss still very high (~4-5)
- Perplexity ~80-120 (random guessing is ~50,257)
- Global attention just warming up (1000-7500)
- Landmark selection unstable

**What the model knows**: Almost nothing
- Some punctuation patterns
- Some token frequency biases
- No semantic understanding

### Step 5,000 (5%)

**Expected behavior**: 🟡 **Beginning of coherence**

Example outputs:
```
Prompt: "The capital of France is "
Output: "a city of the United States"  ← Wrong but coherent structure

Prompt: "Once upon a time"
Output: "there was a man who was"  ← Simple sentences forming
```

**Why**:
- PPL ~40-60
- Basic sentence structure learned
- Some word associations
- Still factually incorrect

### Step 10,000 (10%)

**Expected behavior**: 🟢 **Simple coherent text**

Example outputs:
```
Prompt: "The capital of France is "
Output: "Paris, which is located in the north"  ← Factually correct!

Prompt: "Once upon a time"
Output: "there was a young girl who lived in a small village"
```

**Why**:
- PPL ~25-35
- Sentence structure solid
- Some factual knowledge
- Basic reasoning

### Step 25,000 (25%)

**Expected behavior**: ✅ **Quality text**

Example outputs:
```
Prompt: "The capital of France is "
Output: "Paris, the largest city in France and one of the most visited cities in Europe"

Prompt: "Once upon a time"
Output: "there was a magical kingdom where animals could talk and dreams came true"
```

**Why**:
- PPL ~18-25
- Coherent narratives
- Factual accuracy improving
- Context maintenance

### Step 100,000 (100% - TARGET)

**Expected behavior**: ✅ **Production quality**

Example outputs:
```
Prompt: "The capital of France is "
Output: "Paris, a global center for art, fashion, gastronomy and culture. The city is known for iconic landmarks like the Eiffel Tower"

Prompt: "Once upon a time"
Output: "there was a brave knight who embarked on a quest to save the kingdom from an ancient dragon that had awakened after centuries of slumber"
```

**Why**:
- PPL ~15-20 (FineWeb-Edu target)
- Consistent factual knowledge
- Long-range coherence
- Diverse vocabulary

## What to Test Now

### 1. Check Current Training Metrics

```bash
# View TensorBoard logs
tensorboard --logdir=out_slga_fineweb/tensorboard --port=6006

# Look for:
# - train/loss: Should be decreasing (4.5 → 3.5 → 2.5 over time)
# - train/perplexity: Should be decreasing (80 → 40 → 20)
# - val/perplexity: Should track train (with small gap)
```

### 2. Test Generation at Different Steps

**Don't test every checkpoint** - it wastes time. Test these milestones:

```bash
# Milestone 1: Step 5000 (expect simple coherence)
python scripts/generate.py \
  --checkpoint out_slga_fineweb/ckpt_5000 \
  --config config/config_fineweb_edu.yaml \
  --prompt "The capital of France is " \
  --max-tokens 50 \
  --temperature 0.8

# Milestone 2: Step 10000 (expect good sentences)
python scripts/generate.py \
  --checkpoint out_slga_fineweb/ckpt_10000 \
  --config config/config_fineweb_edu.yaml \
  --prompt "The capital of France is " \
  --max-tokens 50 \
  --temperature 0.8

# Milestone 3: Step 25000 (expect quality)
python scripts/generate.py \
  --checkpoint out_slga_fineweb/ckpt_25000 \
  --config config/config_fineweb_edu.yaml \
  --prompt "The capital of France is " \
  --max-tokens 50 \
  --temperature 0.8
```

### 3. Use Better Sampling Parameters

**Your current settings**:
```bash
--temperature 0.0  # Greedy (always most probable token)
--max-tokens 10    # Very short
```

**Recommended settings**:
```bash
--temperature 0.8  # More diverse, less repetitive
--max-tokens 50    # See longer patterns
--top-p 0.95       # Nucleus sampling (more natural)
```

**Why temperature 0.0 fails early in training**:
- Model's probability distribution is flat/noisy
- "Most probable" token might be punctuation
- No diversity to escape bad patterns

**Why temperature 0.8-1.0 is better**:
- Samples from top tokens with some randomness
- Can escape repetition loops
- More natural-looking text (even if incoherent)

## Diagnostic Commands

### Check Training Progress

```bash
# Find latest checkpoint
ls -lt out_slga_fineweb/ckpt_* | head -5

# Check step number
python3 -c "
import torch
state = torch.load('out_slga_fineweb/ckpt_XXXX/trainer_state.pt', map_location='cpu')
print(f'Step: {state[\"step\"]}')
print(f'LR: {state[\"scheduler\"][\"_last_lr\"][0]:.6f}')
"
```

### Monitor Training Live

```bash
# Watch training progress (if still running)
watch -n 5 'ls -lth out_slga_fineweb/ckpt_* | head -5'

# Or check GPU utilization
nvidia-smi -l 5
```

### Analyze Generation Issues

If generation stays nonsensical beyond step 10K:

```python
# scripts/debug_generation.py (create this)
import torch
from transformers import AutoTokenizer
from src.model import LLMTransformer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = LLMTransformer(config)
model.load_state_dict(torch.load("out_slga_fineweb/ckpt_XXXX/model.pt"))
model.eval()

# Generate with logit inspection
prompt = "The capital of France is "
input_ids = tokenizer.encode(prompt, return_tensors="pt")

with torch.no_grad():
    logits = model(input_ids)  # (1, seq_len, vocab_size)

# Check top-5 predictions
last_logits = logits[0, -1, :]  # Last position
top5 = torch.topk(last_logits, 5)
print("Top 5 predictions:")
for prob, idx in zip(top5.values, top5.indices):
    token = tokenizer.decode([idx])
    print(f"  {token:20s} prob={prob.item():.4f}")
```

If top predictions are mostly punctuation → model not learning properly.

## Red Flags (When to Worry)

### 🚩 Red Flag #1: PPL Not Decreasing

**Check at step 5000**:
- Train PPL should be < 60
- Val PPL should be < 80

If PPL is still > 100 at step 5000:
- ❌ Learning rate too low/high
- ❌ Gradient issues
- ❌ Dataset issues
- ❌ Landmark instability

**Action**: Check TensorBoard, examine loss curve

### 🚩 Red Flag #2: Repeating Same Token

**Example**:
```
Output: "the the the the the the the the"
```

**Causes**:
- Temperature too low (0.0)
- Model stuck in local minimum
- Attention mask issues

**Action**: Try temperature 0.8-1.0, check attention weights

### 🚩 Red Flag #3: Only Punctuation

**Example**:
```
Output: ", . , . , . , ."
```

**Causes**:
- Training loss not decreasing
- Vocabulary bias too strong
- Embedding matrix not learning

**Action**: Check train/loss in TensorBoard, verify tokenizer

### 🚩 Red Flag #4: Val PPL >> Train PPL

**Example**: Train PPL = 20, Val PPL = 100

**Causes**:
- Overfitting (but shouldn't happen on FineWeb-Edu)
- Val dataset different distribution
- Regularization too weak

**Action**: Check dataset splits, increase weight_decay

## Expected Timeline (100K steps)

Assuming ~4000 tokens/sec on RTX 3090:

| Step | Time Elapsed | PPL (expected) | Generation Quality |
|------|--------------|----------------|-------------------|
| 1K | ~2h | 80-120 | Nonsensical |
| 5K | ~10h | 40-60 | Word salad → simple phrases |
| 10K | ~20h | 25-35 | Coherent sentences |
| 25K | ~50h | 18-25 | Quality paragraphs |
| 50K | ~100h | 16-22 | Very good |
| 100K | ~200h (~8 days) | 15-20 | Production ✅ |

## Current Status (Step 1000)

**Progress**: 1,000 / 100,000 = **1%**
**Time invested**: ~2 hours
**Time remaining**: ~198 hours (~8 days)

**Recommendation**:
✅ **Continue training** - this is normal early behavior
✅ **Check again at step 5000** (~10h total)
✅ **Use TensorBoard** to monitor PPL decrease
❌ **Don't judge generation quality yet** - too early!

## Quick Test Script

Save as `scripts/quick_gen_test.py`:

```python
#!/usr/bin/env python3
"""Quick generation test at different temperatures"""
import sys
sys.path.insert(0, '.')

from scripts.generate import main
import argparse

checkpoints = [1000, 5000, 10000, 25000, 50000, 100000]
prompts = [
    "The capital of France is ",
    "Once upon a time",
    "In the year 2050,",
]
temperatures = [0.0, 0.5, 0.8, 1.0]

for ckpt in checkpoints:
    ckpt_path = f"out_slga_fineweb/ckpt_{ckpt}"
    # Check if exists
    import os
    if not os.path.exists(ckpt_path):
        continue

    print(f"\n{'='*80}")
    print(f"Testing checkpoint {ckpt}")
    print('='*80)

    for prompt in prompts:
        for temp in temperatures:
            print(f"\nPrompt: '{prompt}' @ temp={temp}")
            # Run generation
            # (implementation details...)
```

## Summary

**At step 1000**:
- ✅ Nonsensical output is **EXPECTED**
- ✅ Model is just 1% through training
- ✅ Continue training to step 5K-10K minimum

**First meaningful test**: Step 5,000 (~10h training time)

**Production quality**: Step 100,000 (~8 days training time)

**Don't panic** - your model is learning normally! 🎯
