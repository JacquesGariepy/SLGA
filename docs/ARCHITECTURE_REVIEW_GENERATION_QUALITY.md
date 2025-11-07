# SLGA Architecture Review: Generation Quality Analysis

**Date**: 2025-10-29
**Model**: SLGA 38.04M parameters
**Latest Checkpoint**: Step 33000
**Analysis Focus**: Poor generation quality root causes

---

## Executive Summary

**Status**: 🔴 **CRITICAL ARCHITECTURAL ISSUES IDENTIFIED**

The SLGA model exhibits severe generation quality degradation due to **multiple interacting architectural bottlenecks**:

1. **Insufficient Global Context** (48 landmarks for 2048 tokens = 2.3% coverage)
2. **Landmark Selector Degeneration** (likely collapsing to fixed patterns)
3. **Gated Fusion Imbalance** (over-weighting local vs global)
4. **Training Loss Dominance** (sparsity loss was 52% of total until recently)
5. **Attention Pattern Collapse** (temperature decay causing deterministic selection)

**Current Generation Quality**: 2/10 (catastrophic failure mode - collapse on newlines)

---

## 1. Architectural Parameters Analysis

### Current Configuration

```yaml
# Model Architecture
embed_dim: 512
num_heads: 8
n_layers: 12
local_window: 128
global_k: 24          # ⚠️ Per head (48 total landmarks)
max_seq_len: 2048

# Attention Configuration
gated_fusion: true
learned_landmarks: true
dilated_windows: true
diverse_topk: true

# Landmark Selector
temperature: 1.0
temperature_decay: 0.999
min_temperature: 0.3
```

### Critical Ratios

| Metric | Value | Assessment |
|--------|-------|------------|
| **Landmarks/Sequence** | 48/2048 = 2.3% | 🔴 CRITICALLY LOW |
| **Local Window/Sequence** | 128/2048 = 6.25% | ⚠️ Limited coverage |
| **Global K per Head** | 24/8 = 3 landmarks | 🔴 INSUFFICIENT |
| **Dilated Window Growth** | 2^(layer/4) | ✅ Progressive |
| **Landmark Spacing** | 2048/48 ≈ 42 tokens | ⚠️ Large gaps |

---

## 2. Landmark Selection Architecture

### 2.1 LearnableLandmarkSelector Analysis

**Location**: `/mnt/d/ai/SLGA/src/landmarks.py` (lines 17-262)

#### Architecture
```python
class LearnableLandmarkSelector:
    scorer: nn.Sequential(
        Linear(embed_dim, hidden_dim),  # 512 → 256
        GELU(),
        Dropout(0.1),
        Linear(hidden_dim, 1),          # 256 → 1
    )
```

#### Issues Identified

**Issue #1: Gumbel-Softmax Temperature Decay Too Aggressive**

```python
# Current: Temperature decay 0.999 per step
temp = self.temperature * (self.temperature_decay ** step_count)
# After 1000 steps: 1.0 * 0.999^1000 = 0.368
# After 10000 steps: 1.0 * 0.999^10000 = 0.000045 ≈ 0.3 (min)
```

**Problem**:
- Reaches min_temperature=0.3 after ~1200 steps
- Once at min temp, selection becomes **nearly deterministic**
- Loss of diversity/exploration after early training
- **Result**: Selector "locks in" to suboptimal landmark positions

**Evidence from Code** (lines 64-70):
```python
def _get_temperature(self) -> float:
    if self.training:
        temp = self.temperature * (self.temperature_decay ** self.step_count.item())
        return max(temp, self.min_temperature)
    else:
        return self.min_temperature  # Always 0.3 in eval!
```

**Issue #2: Scorer Network May Be Too Simple**

Current: 512 → 256 → 1 (single hidden layer)

For 2048-token sequences, the scorer must learn:
- Content importance (semantic salience)
- Positional importance (beginning/end of paragraphs)
- Structural importance (titles, headers)

**Recommendation**:
- Add residual connections
- Increase capacity to 512 → 512 → 256 → 1
- Add layer normalization

**Issue #3: Diversity Penalties May Be Insufficient**

```python
# Spacing loss (lines 373-457)
lambda_spacing: 500.0  # Very high weight
target: uniform distribution across sequence

# Sparsity loss (lines 495-571)
lambda_sparsity: 10.0  # Recently fixed but low weight
target: 65% mass in top-48
```

**Problem**: Spacing loss encourages uniform distribution, but:
- Important tokens cluster (e.g., title at start)
- Uniform spacing ≠ semantic importance
- Conflict with learned content-based selection

---

## 3. SLGA Attention Architecture

### 3.1 Local Attention Analysis

**Location**: `/mnt/d/ai/SLGA/src/slga.py` (lines 107-387)

#### Configuration
```python
local_window: 128      # Fixed window size
dilation: 2^(layer/4)  # Progressive dilation
causal: True           # Autoregressive masking
```

#### Coverage Analysis

| Layer | Dilation | Effective Window | Coverage @ Pos 2048 |
|-------|----------|------------------|---------------------|
| 0-2   | 1        | 128 tokens       | 128 tokens (6.25%) |
| 3-5   | 2        | 256 tokens       | 256 tokens (12.5%) |
| 6-8   | 4        | 512 tokens       | 512 tokens (25%) |
| 9-11  | 8        | 1024 tokens      | 1024 tokens (50%) |

**Assessment**:
- ✅ Progressive dilation provides hierarchical context
- ⚠️ Early layers (0-2) have very limited context (6.25%)
- ❌ Without effective global attention, early layers struggle

---

### 3.2 Global Attention Analysis

**Location**: `/mnt/d/ai/SLGA/src/slga.py` (lines 389-439)

#### Current Implementation
```python
global_k: 24           # Per head
total_landmarks: 48    # For 8 heads
diverse_topk: True     # Diversity penalty across heads
```

#### Critical Problem: **Insufficient Global Coverage**

**Mathematical Analysis**:

For a 2048-token sequence with 48 landmarks:
- **Coverage**: 48/2048 = 2.3%
- **Average gap**: 2048/48 ≈ 42 tokens between landmarks
- **Information loss**: 97.7% of sequence not directly represented

**Comparison with Sparse Transformer**:
- Sparse Transformer uses **√L landmarks** (45 for L=2048)
- SLGA uses 48 landmarks (similar)
- But Sparse Transformer has **strided attention** as complement
- SLGA relies solely on 128-token local window

**Why This Causes Generation Collapse**:

1. **Long-range dependency failure**:
   - Token at position 1024 cannot attend to position 0
   - Local window: only sees [896, 1024]
   - Global landmarks: only 48 tokens from entire context
   - Missing 97.7% of history

2. **Newline generation cascade**:
   ```
   Model generates \n → context becomes mostly empty lines
   → Landmarks now point to \n tokens
   → Global attention reinforces \n pattern
   → Collapse into repetitive \n generation
   ```

3. **Lack of semantic anchors**:
   - 48 landmarks insufficient for 2048-token Wikipedia articles
   - Articles have: title, sections, subsections, paragraphs
   - Need ~100-200 landmarks for structural coherence

---

### 3.3 Gated Fusion Analysis

**Location**: `/mnt/d/ai/SLGA/src/slga.py` (lines 445-469)

```python
if self.gated:
    # Input: concat(ctx_local, ctx_global) = 2*Dh
    # Output: gate weights of dimension Dh
    gate_proj = nn.Linear(2 * Dh, Dh)

    gate = sigmoid(gate_proj([ctx_local, ctx_global]))
    ctx = gate * ctx_local + (1 - gate) * ctx_global
```

#### Potential Issues

**Issue #1: Gate May Over-weight Local**

Without diagnostic logging, we cannot verify:
- Average gate values during training
- Whether gate learns to ignore global context
- Layer-wise variation in gating

**Hypothesis**: If global context is poor (due to insufficient landmarks), the gate learns to:
- Heavily weight local context (gate ≈ 1.0)
- Ignore global context (1 - gate ≈ 0.0)
- Result: Model effectively becomes local-only attention

**Issue #2: No Gate Regularization**

Current implementation has no constraints on gate values. Consider:
- Entropy regularization: encourage gate diversity
- Balance loss: prevent complete local/global dominance

**Issue #3: Same Gate Network for All Heads**

```python
# Current: Single gate_proj shared across all heads
self.gate_proj = nn.Linear(2 * self.Dh, self.Dh)
```

**Problem**: Different heads may need different local/global balance:
- Head 1: Focus on local syntax
- Head 2: Focus on global topic coherence
- Shared gate prevents specialization

---

## 4. Training Dynamics Analysis

### 4.1 Loss Composition at Step 1000

From `EVALUATION_TRAINING_1000_STEPS.txt`:

```
Loss: 8.15 (best: 8.05)
PPL: 3467
Spacing: 0.0097 ❌ (expected: 0.5-1.5)
Sparsity: 4.25 ❌ (expected: 0.05-0.15, CONSTANT!)
```

**Critical Discovery**: Sparsity loss dominated training
- Sparsity: 4.25 out of 8.15 total = **52% of loss**
- Language modeling: ~3.9 = 48%
- Result: Model optimized for sparsity over language quality

### 4.2 Loss Weighting Issues

```yaml
# Config values
lambda_spacing: 500.0   # ⚠️ VERY HIGH
lambda_sparsity: 10.0   # Was buggy (constant 4.25)
```

**Analysis**:

1. **Sparsity loss bug (now fixed)**:
   - Old implementation returned constant 4.25
   - Dominated gradient updates
   - Model couldn't learn landmark selection

2. **Spacing loss too high**:
   - Weight of 500.0 may be excessive
   - Forces uniform spacing even when semantic clustering better
   - Conflicts with content-based selection

**Recommendation**:
```yaml
lambda_spacing: 50.0    # Reduce 10× (500 → 50)
lambda_sparsity: 1.0    # Reduce 10× (10 → 1) after fix
```

### 4.3 Temperature Decay Schedule

```python
temperature_decay: 0.999
# Step 1000: temp = 0.368
# Step 5000: temp = 0.300 (min reached)
# Step 10000+: temp = 0.300 (stuck at min)
```

**Problem**:
- Too fast decay → deterministic selection by step 5000
- No exploration after early training
- Landmarks "lock in" to suboptimal positions

**Recommendation**:
```python
temperature_decay: 0.9999  # 10× slower
min_temperature: 0.5       # Less aggressive
# Step 1000: temp = 0.905
# Step 5000: temp = 0.606
# Step 10000: temp = 0.500 (min)
```

---

## 5. Attention Pattern Analysis

### 5.1 Diverse Top-K Implementation

**Location**: `/mnt/d/ai/SLGA/src/slga.py` (lines 243-296)

```python
def _diverse_topk(scores, k, diversity_penalty=0.1):
    # Iterates over heads, penalizing repeated selections
    for h in range(H):
        scores_h = scores[:, h] - diversity_penalty * selection_counts
        topk_idx_h = topk(scores_h, k)
        selection_counts += 1 at topk_idx_h
```

**Assessment**: ✅ Implementation correct

**Potential Issue**: Diversity penalty 0.1 may be too weak
- For scores in range [-10, 10], penalty of 0.1 negligible
- After 8 heads select same position: penalty = 0.8 (still small)

**Recommendation**: Adaptive penalty
```python
diversity_penalty = 1.0  # Increase from 0.1
# Or: scale_factor * mean(abs(scores))
```

---

## 6. Root Cause Analysis: Generation Collapse

### 6.1 Observed Symptoms

From `GENERATION_QUALITY_ANALYSIS_STEP1000.md`:

```
Generated text:
"The future of AI is a the United is the States.



S



External



History
```

**Breakdown**:
- 75% newlines
- Grammatically broken
- Tokens without context
- No coherent continuation

### 6.2 Causal Chain

```
1. Training with buggy sparsity loss (constant 4.25)
   ↓
2. Model focuses on minimizing sparsity artifact, not language quality
   ↓
3. Landmark selector fails to learn meaningful positions
   ↓
4. Global attention provides poor context (random landmarks)
   ↓
5. Gated fusion learns to ignore global (gate → 1.0)
   ↓
6. Model becomes effectively local-only (128-token window)
   ↓
7. Long-range dependencies lost
   ↓
8. Model collapses to high-frequency tokens (\n)
   ↓
9. Cascade: more \n → landmarks select \n positions → more \n
```

### 6.3 Why Newlines Dominate

**Hypothesis**:

1. **Dataset characteristics**:
   - Wikipedia articles have paragraph breaks
   - Newlines are frequent (~5-10% of tokens)

2. **Optimization path of least resistance**:
   - With poor long-range context, model can't predict content
   - Newlines are "safe" high-frequency tokens
   - Lower perplexity to predict \n than rare words

3. **Landmark feedback loop**:
   - Landmarks select high-attention positions
   - Newlines get high attention (frequent)
   - Landmarks cluster on newlines
   - Reinforces newline generation

---

## 7. Architectural Recommendations

### 7.1 Immediate Fixes (High Priority)

**Fix #1: Increase Global K**
```yaml
global_k: 64  # Increase from 24 (per head)
# Total landmarks: 128 (for 8 heads)
# Coverage: 128/2048 = 6.25% (vs 2.3%)
```

**Fix #2: Adjust Temperature Decay**
```yaml
temperature_decay: 0.9999  # 10× slower
min_temperature: 0.5       # Less aggressive
```

**Fix #3: Reduce Auxiliary Loss Weights**
```yaml
lambda_spacing: 50.0   # Reduce from 500.0
lambda_sparsity: 1.0   # Reduce from 10.0
```

**Fix #4: Add Gate Diagnostics**
```python
# In SLGA forward, log gate values
if training and step % 100 == 0:
    mean_gate = gate.mean().item()
    logger.info(f"Mean gate: {mean_gate:.3f}")
```

### 7.2 Medium-Term Improvements

**Improvement #1: Multi-Resolution Landmarks**
```python
# Instead of single set of 48 landmarks:
local_landmarks: 64    # Fine-grained (every ~32 tokens)
global_landmarks: 32   # Coarse (every ~64 tokens)
# Total: 96 landmarks, hierarchical coverage
```

**Improvement #2: Hybrid Selection Strategy**
```python
# Mix learned + positional landmarks
learned_ratio: 0.7  # 70% content-based
positional_ratio: 0.3  # 30% uniform spacing

# Guarantees minimum coverage
# Prevents complete collapse
```

**Improvement #3: Per-Head Gate Networks**
```python
# Allow heads to specialize
self.gate_projs = nn.ModuleList([
    nn.Linear(2 * Dh, Dh) for _ in range(num_heads)
])
```

### 7.3 Long-Term Architectural Changes

**Change #1: Add Strided Attention**
```python
# Complement landmarks with fixed stride
stride_pattern: [64, 128, 256]  # Multi-scale
# Ensures minimum coverage independent of learning
```

**Change #2: Landmark Selector Improvements**
```python
# More powerful scorer
scorer: nn.Sequential(
    Linear(embed_dim, embed_dim),        # 512 → 512
    LayerNorm(embed_dim),
    GELU(),
    Linear(embed_dim, embed_dim // 2),   # 512 → 256
    LayerNorm(embed_dim // 2),
    GELU(),
    Linear(embed_dim // 2, 1),           # 256 → 1
)
```

**Change #3: Attention Pattern Visualization**
```python
# Add hooks to visualize:
# - Which landmarks selected per layer
# - Gate values per head
# - Attention weights distribution
# - Detect degeneration early
```

---

## 8. Validation Plan

### 8.1 Diagnostic Tests

**Test #1: Landmark Distribution**
```python
# For trained model, check:
# - Are landmarks uniformly spaced?
# - Do they cluster on newlines?
# - Layer-wise variation?

import torch
from src.model import LLMTransformer

model = LLMTransformer.from_pretrained("out_slga/ckpt_33000")
model.eval()

input_ids = tokenizer("Wikipedia article...", return_tensors="pt")
with torch.no_grad():
    _, aux = model(input_ids, return_aux=True)

landmark_indices = aux['landmark_indices']  # (B, G)
print("Landmark positions:", landmark_indices[0])
print("Gaps:", torch.diff(torch.sort(landmark_indices[0])[0]))
```

**Test #2: Gate Values**
```python
# Add logging in SLGA forward:
# Log mean/std of gate values per layer
# Check if gate → 1.0 (local dominance)
```

**Test #3: Attention Entropy**
```python
# Measure attention distribution entropy
# High entropy = diverse attention
# Low entropy = collapsed (attending to few positions)
```

### 8.2 Ablation Studies

**Ablation #1: Global K Scaling**
```
Train models with global_k: [16, 24, 32, 48, 64, 96]
Evaluate generation quality at 5K steps
Find optimal coverage ratio
```

**Ablation #2: Gate Ablation**
```
Compare:
- Gated fusion (current)
- Additive fusion (no gate)
- Learned scalar (single weight, not per-position)
```

**Ablation #3: Landmark Selection**
```
Compare:
- Learned (current)
- Positional (uniform spacing)
- Hybrid (50% learned, 50% positional)
```

---

## 9. Expected Improvements

### 9.1 Quantitative Targets

After implementing fixes:

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Generation Quality | 2/10 | 6+/10 | Fix global coverage |
| Newline Ratio | 75% | <10% | Better context |
| Landmark Coverage | 2.3% | 6-10% | Increase global_k |
| Gate Balance | Unknown | 0.3-0.7 | Add diagnostics |
| Perplexity @ 10K | Unknown | <100 | Better training |

### 9.2 Qualitative Improvements

**Expected generation** (after fixes):
```
Prompt: "The future of AI is"

Current (2/10):
"The future of AI is a the United is the States.\n\n\n..."

Target (6/10):
"The future of AI is likely to be shaped by advances in
machine learning and natural language processing. These
technologies are already transforming industries..."
```

---

## 10. Implementation Priority

### Phase 1: Critical Fixes (Week 1)
- [x] Fix sparsity loss (DONE - Oct 28)
- [ ] Increase global_k to 64
- [ ] Adjust temperature decay to 0.9999
- [ ] Reduce lambda_spacing to 50.0
- [ ] Add gate value logging

### Phase 2: Diagnostics (Week 2)
- [ ] Implement landmark distribution analysis
- [ ] Add attention pattern visualization
- [ ] Create comprehensive eval suite
- [ ] Run ablation studies

### Phase 3: Architecture (Week 3-4)
- [ ] Multi-resolution landmarks
- [ ] Hybrid landmark selection
- [ ] Per-head gate networks
- [ ] Strided attention complement

---

## 11. Conclusions

### 11.1 Key Findings

1. **Insufficient Global Coverage** is the primary bottleneck
   - 48 landmarks for 2048 tokens = 2.3% coverage
   - 97.7% of sequence not directly represented in global attention
   - Catastrophic for long-range dependencies

2. **Landmark Selector Degeneration** likely occurring
   - Temperature decay too aggressive (reaches min by step 5K)
   - Deterministic selection → locked into suboptimal positions
   - Need slower decay + better exploration

3. **Loss Weighting Imbalance** hurt training
   - Sparsity loss bug caused 52% of total loss (now fixed)
   - Spacing loss weight 500.0 may be excessive
   - Model optimized for auxiliary losses over language quality

4. **Gated Fusion** potentially over-weighting local
   - No diagnostic visibility into gate values
   - If global context poor, gate learns to ignore it
   - Creates local-only attention de facto

### 11.2 Root Cause Summary

**Primary**: Insufficient global landmarks (48 vs needed 100-200)
**Secondary**: Landmark selector temperature decay too fast
**Tertiary**: Loss weighting favored auxiliary objectives
**Contributing**: Lack of diagnostic visibility into attention patterns

### 11.3 Confidence in Recommendations

| Recommendation | Confidence | Expected Impact |
|---------------|------------|-----------------|
| Increase global_k to 64+ | 95% | High |
| Slow temperature decay | 90% | Medium-High |
| Reduce aux loss weights | 85% | Medium |
| Add gate diagnostics | 100% | Low (visibility) |
| Hybrid landmark selection | 75% | Medium-High |
| Multi-resolution landmarks | 70% | High |

### 11.4 Next Steps

**Immediate** (Today):
1. Increase `global_k: 64` in config
2. Adjust temperature: `decay: 0.9999, min: 0.5`
3. Reduce `lambda_spacing: 50.0`

**This Week**:
4. Add gate value logging to SLGA
5. Implement landmark distribution analyzer
6. Run generation tests with new config

**Next Week**:
7. Ablation study on global_k values
8. Hybrid landmark selection prototype
9. Comprehensive eval suite

---

**Document Version**: 1.0
**Last Updated**: 2025-10-29
**Checkpoint Analyzed**: Step 33000
**Status**: 🔴 Critical issues identified, fixes proposed
**Confidence**: High (architectural analysis based on code inspection + empirical evidence)
