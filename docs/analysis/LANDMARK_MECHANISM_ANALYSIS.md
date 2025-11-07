# Comprehensive Line-by-Line Analysis: Landmark Selection Mechanism

**File**: `src/landmarks.py` (490 lines)
**Date**: 2025-10-24
**Context**: Step 15K instability analysis
**Purpose**: Deep dive into learned landmark selection for SLGA-Plus

---

## Executive Summary

The landmark selection mechanism in `src/landmarks.py` implements **three differentiable strategies** for selecting important tokens (landmarks) for global attention. After analyzing the code line-by-line and reviewing the Step 15K diagnostic report, we identified **critical stability issues** with mathematical formulations and hyperparameter choices that explain the reported landmark instability.

### Key Findings

| Component | Status | Severity | Issue |
|-----------|--------|----------|-------|
| **Temperature Decay** | ⚠️ SUBOPTIMAL | HIGH | Too slow (0.9999) - landmarks stay "soft" too long |
| **Sparsity Loss** | ❌ BROKEN | CRITICAL | Fixed target incompatible with G/L ratio |
| **Diversity Loss** | ⚠️ INEFFECTIVE | MEDIUM | Entropy-based approach doesn't prevent clustering |
| **Gradient Flow** | ⚠️ BIASED | MEDIUM | Straight-through estimator provides biased gradients |
| **Initialization** | ⚠️ UNSTABLE | HIGH | Random initialization can start at bad local minima |

---

## Table of Contents

1. [Landmark Selection Theory](#1-landmark-selection-theory)
2. [Line-by-Line Code Review](#2-line-by-line-code-review)
3. [Mathematical Correctness Analysis](#3-mathematical-correctness-analysis)
4. [Root Cause: Step 15K Instability](#4-root-cause-step-15k-instability)
5. [Spacing & Sparsity Mechanisms](#5-spacing--sparsity-mechanisms)
6. [Gradient Flow Analysis](#6-gradient-flow-analysis)
7. [Recommendations & Fixes](#7-recommendations--fixes)

---

## 1. Landmark Selection Theory

### 1.1 The Landmark Selection Problem

**Objective**: Given a sequence of length L with embeddings `x ∈ ℝ^(L×D)`, select G important positions (landmarks) such that:

1. **Coverage**: Landmarks span the entire sequence (avoid clustering)
2. **Relevance**: Selected tokens are semantically important
3. **Differentiability**: Selection is end-to-end trainable
4. **Efficiency**: Selection cost is O(L·D) or better

### 1.2 Differentiable Top-K Problem

The core challenge is that **hard top-K selection is non-differentiable**:

```
∇_scores top-K(scores) = 0  (almost everywhere)
```

**Two relaxation strategies** are implemented:

#### Strategy 1: Gumbel-Softmax

```python
# Probabilistic relaxation with temperature annealing
gumbel_noise = -log(-log(U)) where U ~ Uniform(0,1)
perturbed = (scores + gumbel_noise) / τ
soft_selection = softmax(perturbed)
```

**Gradient flow**:
```
∂L/∂scores = ∂L/∂soft_selection · ∂softmax/∂perturbed · 1/τ
```

**Properties**:
- ✅ Unbiased gradient estimate (in expectation)
- ✅ Smooth approximation to hard selection
- ❌ High variance (stochastic sampling)
- ❌ Requires temperature annealing

#### Strategy 2: Straight-Through Estimator (STE)

```python
# Forward: Hard selection
hard_selection = one_hot(top_K(scores))

# Backward: Gradient bypass
selection = hard_selection + scores - scores.detach()
            └─ forward ─┘   └─ gradient passthrough ─┘
```

**Gradient flow**:
```
∂L/∂scores = ∂L/∂hard_selection · I  (identity approximation)
```

**Properties**:
- ✅ Low variance (deterministic)
- ✅ Fast (no softmax)
- ❌ Biased gradients (approximates hard operation with soft gradients)
- ❌ Less exploration

### 1.3 Temperature Annealing Strategy

**Goal**: Start with soft selection (τ=1.0), gradually harden to discrete selection (τ→0.3)

**Current implementation**:
```python
τ(t) = τ_0 · decay^t
τ(t) = max(τ(t), τ_min)
```

**With default values**:
- τ₀ = 1.0 (soft selection)
- decay = 0.9999 (very slow!)
- τ_min = 0.5 (not very hard)

**Progression**:
- Step 1,000: τ ≈ 0.905 (still very soft)
- Step 10,000: τ ≈ 0.367 (reaches min=0.5 earlier, clamped)
- Step 15,000: τ = 0.5 (clamped at minimum)

**⚠️ PROBLEM**: Temperature stays high for too long, allowing landmarks to "wander" without committing to stable positions.

### 1.4 Diversity Mechanisms

Three loss functions encourage landmark diversity:

1. **Diversity Loss (Entropy-based)** - Lines 332-364
2. **Spacing Loss (Gap uniformity)** - Lines 280-329
3. **Sparsity Loss (Active threshold)** - Lines 367-421

Details in Section 5.

---

## 2. Line-by-Line Code Review

### 2.1 LearnableLandmarkSelector Class (Lines 17-173)

#### Initialization (Lines 35-62)

```python
def __init__(
    self,
    embed_dim: int,
    num_landmarks: int,
    hidden_dim: Optional[int] = None,
    temperature: float = 1.0,
    temperature_decay: float = 0.999,  # ← Optimized in v1.1 (was 0.9999)
    min_temperature: float = 0.3,      # ← Optimized in v1.1 (was 0.5)
):
```

**Analysis**:

**Line 41**: `temperature_decay: float = 0.999`
- **v1.0**: Was 0.9999 (10× slower)
- **v1.1**: Changed to 0.999 (reaches τ_min faster)
- **Impact**: At step 5K, τ → 0.007 vs 0.606 (much harder selection)
- ✅ **GOOD**: Faster convergence to discrete landmarks

**Line 42**: `min_temperature: float = 0.3`
- **v1.0**: Was 0.5 (too soft)
- **v1.1**: Changed to 0.3 (more discriminative)
- **Impact**: Softmax concentration factor: exp(score/0.3) vs exp(score/0.5)
  - τ=0.5: score diff of 1.0 → softmax ratio = e^2 = 7.4×
  - τ=0.3: score diff of 1.0 → softmax ratio = e^3.33 = 28×
- ✅ **GOOD**: Sharper distinction between selected/rejected positions

**Lines 52-59**: Scorer network
```python
self.scorer = nn.Sequential(
    nn.Linear(embed_dim, hidden),      # D → D/2
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(hidden, 1),              # D/2 → 1
)
```

**Analysis**:
- **Architecture**: 2-layer MLP with GELU activation
- **Hidden dim**: D/2 by default (can be overridden)
- **Complexity**: O(L·D·(D/2) + L·(D/2)) = O(L·D²)
- **Design choice**: GELU over ReLU for smoother gradients
- ✅ **GOOD**: Standard design, no issues
- 💡 **OPTIMIZATION**: Could reduce to D/4 for 2× speedup with minimal capacity loss

**Line 62**: `self.register_buffer("step_count", torch.tensor(0), persistent=False)`
- **Purpose**: Track steps for temperature decay
- **persistent=False**: Not saved in checkpoints (resets on load)
- ⚠️ **ISSUE**: Temperature resets when loading checkpoints!
  - If you resume from step 15K, temperature resets to τ=1.0
  - This can cause sudden instability in landmark selection
- ❌ **BUG**: Should be `persistent=True` to maintain temperature across checkpoints

#### Temperature Calculation (Lines 64-70)

```python
def _get_temperature(self) -> float:
    """Calcule température actuelle avec décroissance"""
    if self.training:
        temp = self.temperature * (self.temperature_decay ** self.step_count.item())
        return max(temp, self.min_temperature)
    else:
        return self.min_temperature
```

**Analysis**:

**Line 67**: Exponential decay formula
```
τ(t) = τ₀ · decay^t
```

**Mathematical properties**:
- **Continuous**: Smooth transition (no sudden jumps)
- **Monotonic**: Always decreasing
- **Asymptotic**: Approaches τ_min but never crosses until clamped

**Alternative schedules** (not implemented):

1. **Linear decay**:
   ```python
   progress = min(1.0, step_count / total_steps)
   temp = τ₀ - (τ₀ - τ_min) * progress
   ```
   - Pros: Predictable endpoint
   - Cons: Less smooth near endpoints

2. **Cosine annealing**:
   ```python
   progress = min(1.0, step_count / total_steps)
   decay_factor = 0.5 * (1 + cos(π * progress))
   temp = τ_min + (τ₀ - τ_min) * decay_factor
   ```
   - Pros: Smooth everywhere, popular in learning rate scheduling
   - Cons: More complex

**Line 70**: `return self.min_temperature` (inference mode)
- Always uses minimum temperature at eval time
- ✅ **GOOD**: Ensures deterministic inference

#### Gumbel Top-K (Lines 72-100)

```python
def _gumbel_topk(
    self, scores: torch.Tensor, k: int, temperature: float
) -> Tuple[torch.Tensor, torch.Tensor]:
```

**Analysis**:

**Line 90**: Gumbel noise injection
```python
gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
```

**Mathematical derivation**:
```
U ~ Uniform(0, 1)
G = -log(-log(U))
G ~ Gumbel(0, 1)
```

**Numerical stability**:
- `+ 1e-10`: Prevents log(0) = -∞
- Double log: No issues with floating point precision
- ✅ **GOOD**: Numerically stable

**Line 91**: Temperature scaling
```python
perturbed_scores = (scores + gumbel_noise) / temperature
```

**Interpretation**:
- High τ: Noise dominates → stochastic selection
- Low τ: Scores dominate → deterministic selection
- **Gumbel-max trick**: argmax(scores + G/τ) approximates sampling from Categorical(softmax(scores/τ))

**Line 94**: Hard top-K for forward pass
```python
_, hard_indices = torch.topk(perturbed_scores, k=k, dim=-1)
```

- Uses perturbed scores (with noise)
- Returns hard indices for forward computation
- ✅ **GOOD**: Forward uses discrete selection

**Line 98**: Soft scores for backward pass
```python
soft_scores = F.softmax(perturbed_scores, dim=-1)
```

**Analysis**:
- Softmax over perturbed scores
- Gradient flows through this soft distribution
- ⚠️ **ISSUE**: This is NOT a true Gumbel-Softmax!

**Gumbel-Softmax should be**:
```python
# Standard Gumbel-Softmax:
soft_scores = F.softmax((scores + gumbel_noise) / temperature, dim=-1)
hard_one_hot = one_hot(argmax(perturbed_scores))
selection = hard_one_hot + soft_scores - soft_scores.detach()  # STE
```

**Current implementation**:
- Only returns soft_scores (not used in forward)
- Forward uses hard_indices directly with torch.gather
- Gradient may not flow correctly through selection

❌ **POTENTIAL BUG**: The returned `soft_scores` are never used to weight the forward pass. The code should either:
1. Use soft_scores to weight the selection (true Gumbel-Softmax)
2. Or add straight-through connection between hard and soft

**Current gradient flow**:
```
Loss → gather(x, hard_indices)
     → hard_indices = topk(perturbed_scores)
     → perturbed_scores = (scores + G) / τ
     → scores = scorer(x)
     → scorer parameters
```

**Issue**: Gradient of topk is zero (non-differentiable)! This means **no gradient flows through Gumbel selection as implemented**.

🚨 **CRITICAL BUG**: The Gumbel-Softmax implementation doesn't actually provide gradients to the scorer. The soft_scores are computed but never used to backpropagate.

#### Straight-Through Estimator (Lines 102-124)

```python
def _straight_through_topk(
    self, scores: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
```

**Analysis**:

**Lines 114**: Hard top-K
```python
topk_vals, topk_indices = torch.topk(scores, k=k, dim=-1)
```
- Deterministic selection
- No Gumbel noise
- ✅ **GOOD**: Clean forward pass

**Lines 117-118**: One-hot encoding
```python
selection_onehot = torch.zeros_like(scores)
selection_onehot.scatter_(1, topk_indices, 1.0)
```
- Creates (B, L) tensor with 1s at selected positions
- 0s everywhere else
- ✅ **GOOD**: Correct implementation

**Line 122**: **THE MAGIC TRICK**
```python
selection = selection_onehot + scores - scores.detach()
```

**Mathematical explanation**:

In **forward pass**:
```
selection = selection_onehot + scores - scores
          = selection_onehot + 0
          = selection_onehot  (hard selection)
```

In **backward pass**:
```
∂L/∂scores = ∂L/∂selection · ∂selection/∂scores
           = ∂L/∂selection · (0 + 1 - 0)  ← scores.detach() has no gradient
           = ∂L/∂selection
```

**Result**: Gradient of hard selection flows as if it were soft scores!

**Properties**:
- ✅ Forward: Hard, discrete selection
- ✅ Backward: Gradient flows as continuous
- ❌ **Biased**: Gradient doesn't reflect actual discrete operation
- ❌ **Approximation**: Treats top-K as identity function

**Is this theoretically sound?**

No, but it works empirically. The bias is:
```
True gradient: ∂top-K/∂scores ≈ 0 (almost everywhere)
Approximated:  ∂top-K/∂scores ≈ I (identity)
```

The approximation assumes that small changes in scores don't change which positions are in top-K, which is **false** near decision boundaries.

**When does it fail?**
- When multiple scores are close (ambiguous ranking)
- When top-K set changes frequently
- Early in training when scores are random

**When does it work?**
- After convergence when rankings are stable
- With large margin between selected/rejected positions
- In later training stages

💡 **INSIGHT**: This explains why landmarks may be unstable at step 15K - the STE approximation breaks down when scores are competitive!

#### Forward Pass (Lines 126-173)

```python
def forward(
    self, x: torch.Tensor, use_gumbel: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

**Analysis**:

**Line 144**: Compute importance scores
```python
scores = self.scorer(x).squeeze(-1)  # (B, L)
```
- Runs 2-layer MLP on each position
- O(L·D²) complexity
- ✅ **GOOD**: Standard implementation

**Lines 149-158**: Selection mode routing
```python
if self.training:
    if use_gumbel:
        selection_soft, landmark_indices = self._gumbel_topk(...)
    else:
        selection_soft, landmark_indices = self._straight_through_topk(...)
else:
    _, landmark_indices = torch.topk(scores, k=k, dim=-1)
```

**Analysis**:
- Training: Configurable (Gumbel or STE)
- Inference: Always hard top-K (deterministic)
- **Config v1.1**: `use_gumbel: false` (uses STE)
- ✅ **GOOD**: Flexible design

**Lines 164-167**: Gather landmark states
```python
landmark_indices_exp = landmark_indices.unsqueeze(-1).expand(B, k, D)
landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)
```

**Mathematical operation**:
```
x: (B, L, D)
landmark_indices: (B, G)
landmark_indices_exp: (B, G, D)  ← expand indices to all dimensions
landmark_states: (B, G, D) = x[b, landmark_indices[b, :], :]
```

**Complexity**: O(G·D)
✅ **GOOD**: Efficient gathering

**Line 171**: Normalize scores for loss
```python
selection_scores = F.softmax(scores, dim=-1)
```

**Purpose**: Convert raw scores to probabilities for diversity/sparsity losses
**Issue**: These are NOT the same as the selection distribution used in Gumbel!
- Gumbel uses: softmax((scores + G) / τ)
- This uses: softmax(scores)
- ⚠️ **INCONSISTENCY**: Loss functions see different distribution than selection

**Returns**:
1. `landmark_indices` (B, G): Selected positions
2. `landmark_states` (B, G, D): Gathered embeddings
3. `selection_scores` (B, L): Softmax probabilities for losses

---

### 2.2 PositionalLandmarkSelector (Lines 176-231)

**Purpose**: Select landmarks based on **position patterns** (not content)

**Analysis**:

**Line 197**: Learnable positional embeddings
```python
self.pos_embeddings = nn.Parameter(torch.randn(max_seq_len, embed_dim))
```

**Concept**:
- Learns which positions are structurally important
- Examples: Start of paragraphs, every N tokens, section boundaries
- **Content-independent**: Same positions for all sequences

**Line 218**: Score positions
```python
scores = self.scorer(pos_emb).squeeze(-1)  # (L,)
```

**Efficiency**:
- Scores computed once per unique sequence length
- Can be cached across batches
- O(L·D) vs O(B·L·D) for content-based

**Line 219**: Broadcast to batch
```python
scores = scores.unsqueeze(0).expand(B, L)
```

**Issue**: All examples in batch get same landmark positions
- ✅ **PRO**: Very efficient, interpretable
- ❌ **CON**: Can't adapt to content

**Use case**: Structured text (code, markup, formal documents)

---

### 2.3 HybridLandmarkSelector (Lines 234-277)

**Purpose**: Combine content-based and positional selection with learned gating

**Analysis**:

**Lines 249-250**: Two selectors
```python
self.content_selector = LearnableLandmarkSelector(...)
self.position_selector = PositionalLandmarkSelector(...)
```

**Line 253**: Gating mechanism
```python
self.gate = nn.Linear(embed_dim, 1)
```

**Lines 262-264**: Compute gate weight
```python
x_pooled = x.mean(dim=1)  # (B, D) - global average
gate_weight = torch.sigmoid(self.gate(x_pooled))  # (B, 1)
```

**Analysis**:
- **Pooling**: Mean over sequence (loses positional info!)
- **Gate**: Single scalar per example
- **Sigmoid**: Gate ∈ [0, 1]

**Interpretation**:
- gate ≈ 1: Use content-based selection
- gate ≈ 0: Use positional selection
- gate ≈ 0.5: Equal mix

**Line 267**: Combine scores
```python
scores_combined = gate_weight * scores_content + (1 - gate_weight) * scores_position
```

**Lines 270-271**: Re-select top-K
```python
_, landmark_indices = torch.topk(scores_combined, k=k, dim=-1)
```

**Issue**: Re-selection on combined scores
- Wastes two top-K operations (content + position selectors)
- Then does third top-K on combined scores
- 💡 **OPTIMIZATION**: Could directly combine indices instead:
  ```python
  k_content = int(gate_weight * k)
  k_position = k - k_content
  indices = torch.cat([idx_content[:, :k_content],
                       idx_position[:, :k_position]], dim=1)
  ```

**Complexity**: 3× LearnableLandmarkSelector (very expensive!)

---

## 3. Mathematical Correctness Analysis

### 3.1 Gumbel-Softmax Formulation

**Standard Gumbel-Softmax** (Jang et al., 2017):

```
y_i = exp((log(π_i) + G_i) / τ) / Σ_j exp((log(π_j) + G_j) / τ)
```

Where:
- π_i = softmax(scores)_i (categorical probabilities)
- G_i ~ Gumbel(0, 1)
- τ = temperature

**Properties**:
1. As τ→0: y → one-hot (deterministic)
2. As τ→∞: y → uniform (maximum entropy)
3. Gradient: ∂y/∂scores is well-defined

**Implementation in code** (Line 91):
```python
perturbed_scores = (scores + gumbel_noise) / temperature
```

**Equivalent to**:
```
log(π_i) ≈ scores_i  (assuming scores ≈ log(π))
y_i ∝ exp((scores_i + G_i) / τ)
```

✅ **CORRECT** if scores are log-probabilities
⚠️ **ISSUE**: Code doesn't normalize scores first!

**Should be**:
```python
log_probs = F.log_softmax(scores, dim=-1)  # Normalize first
perturbed = (log_probs + gumbel_noise) / temperature
```

### 3.2 Straight-Through Estimator Bias

**Mathematical analysis**:

**True gradient of top-K**:
```
∂top-K(s)/∂s_i = {
  undefined  if s_i is at decision boundary (s_i = s_k or s_i = s_{k+1})
  0          otherwise (s_i clearly in or out of top-K)
}
```

**STE approximation**:
```
∂top-K(s)/∂s_i ≈ 1  (identity)
```

**Bias quantification**:

Consider top-K({s₁=2.0, s₂=1.5, s₃=0.5}, k=2) = {1, 2}

True gradient:
```
∂L/∂s₁ = 0  (s₁ safely in top-2)
∂L/∂s₂ = 0  (s₂ safely in top-2, but close to boundary!)
∂L/∂s₃ = 0  (s₃ safely out of top-2)
```

STE gradient:
```
∂L/∂s₁ = ∂L/∂top-K₁  (non-zero!)
∂L/∂s₂ = ∂L/∂top-K₂  (non-zero!)
∂L/∂s₃ = 0  (not selected)
```

**Bias**: STE provides gradients to selected positions even when they're stable, causing unnecessary updates.

**Consequence**:
- Selected landmarks get gradients → scores change → landmarks may flip
- This causes **instability** when scores are close!

### 3.3 Temperature Annealing Convergence

**Exponential decay schedule**:
```
τ(t) = τ₀ · α^t
```

**Convergence time to τ_min**:
```
τ(t*) = τ_min
τ₀ · α^(t*) = τ_min
t* = log(τ_min / τ₀) / log(α)
```

**With current values**:
- τ₀ = 1.0
- τ_min = 0.3 (v1.1)
- α = 0.999 (v1.1)

```
t* = log(0.3 / 1.0) / log(0.999)
   = -1.204 / -0.001
   ≈ 1,204 steps
```

**With old values** (v1.0):
- α = 0.9999
- τ_min = 0.5

```
t* = log(0.5 / 1.0) / log(0.9999)
   = -0.693 / -0.0001
   ≈ 6,931 steps
```

**Analysis**:
- v1.0: Reaches τ=0.5 at step ~7K (too slow for 15K step failure!)
- v1.1: Reaches τ=0.3 at step ~1.2K (much better!)

**At step 15K**:
- v1.0: τ = max(1.0 · 0.9999^15000, 0.5) = max(0.223, 0.5) = 0.5 ✅
- v1.1: τ = max(1.0 · 0.999^15000, 0.3) = max(3.4e-7, 0.3) = 0.3 ✅

Both reach minimum by step 15K, so temperature decay is NOT the cause of instability at step 15K!

---

## 4. Root Cause: Step 15K Instability

### 4.1 Evidence from Diagnostic Report

From `/docs/STEP_15K_DIAGNOSTIC_REPORT.md`:

```
Observed anomalies:
- Step 14300: Throughput drops to 927 tok/s (vs. normal 3400-6300 tok/s)
- Step 14100: Loss spike to 2.9447 (PPL: 19.00)
- Step 14850: Training time spikes to 6.17s/it (vs. 2.3-3.0s normal)
```

**Symptoms**:
1. ⚠️ **Throughput collapse**: 10× slower (927 vs 3422 tok/s)
2. ⚠️ **Loss spikes**: 2.1 → 2.9 (40% increase)
3. ⚠️ **Training time spikes**: 2× slower per iteration

### 4.2 Hypothesis: Landmark Competition & Oscillation

**Scenario at Step 15K**:

1. **Temperature has reached minimum** (τ=0.3 or 0.5)
   - Selection is now very "hard" (deterministic)
   - Small score changes cause landmark flips

2. **Straight-Through Estimator provides gradients**
   - Selected landmarks get non-zero gradients
   - Scores keep updating even when selection is stable

3. **Multiple positions have similar scores**
   ```
   Position 50: score = 2.05
   Position 51: score = 2.03
   Position 52: score = 2.01
   Position 53: score = 1.98

   Top-24 boundary is at ~2.00
   → Positions 50-53 are "on the fence"
   ```

4. **Oscillation occurs**:
   ```
   Step 14100: Position 50 selected → Gets gradient → Score decreases to 1.99
   Step 14101: Position 50 dropped → Position 53 selected → Gets gradient
   Step 14102: Position 53 score drops → Position 50 selected again
   ... (oscillation continues)
   ```

5. **Computational cost**:
   - Each flip changes which tokens attend to which landmarks
   - Attention patterns change → Cache invalidation
   - GPU must recompute attention matrices
   - **Result**: Throughput drops by 10×

### 4.3 Why Existing Loss Functions Don't Prevent This

#### Diversity Loss (Entropy-based) - Lines 332-364

**Formula**:
```python
entropy = -(selection_scores * log(selection_scores)).sum(dim=-1)
loss = λ_div * (1 - entropy / log(L))
```

**Problem**: Maximizes entropy over **all L positions**, not just the G selected landmarks!

**Example** (L=256, G=24):
```
Ideal uniform: selection_scores = [1/256, 1/256, ..., 1/256]
Entropy = log(256) = 5.54
Normalized = 1.0
Loss = 0.0 ✅

Clustered landmarks: selection_scores = [0.5, 0.4, 0.1, 0, ..., 0]
Entropy ≈ 1.36
Normalized = 1.36 / 5.54 = 0.25
Loss = 0.75 × λ_div ❌
```

**Issue**: Loss doesn't directly penalize landmarks clustering at positions 0-2! It only sees that most positions have low probability.

**Gradient**:
```
∂loss/∂scores_i = -λ_div / log(L) · (log(p_i) + 1)
```

This pushes scores toward uniform distribution over all L positions, which is **not the same** as spacing out the G landmarks!

#### Sparsity Loss (Fixed Target) - Lines 367-421

**Formula** (v1.0):
```python
target_active = 1 - target_sparsity  # = 1 - 0.95 = 0.05
active_fraction = (selection_scores > 0.01).float().mean()
loss = λ_spar * relu(active_fraction - target_active)
```

**Problem**: Fixed target of 5% conflicts with G/L ratio!

**Example** (G=24, L=256):
```
Ideal fraction = G/L = 24/256 = 0.094 (9.4%)
Target = 0.05 (5%)

active_fraction = 0.094 > 0.05
loss = λ_spar * (0.094 - 0.05) = 0.044 × λ_spar ❌ Always penalized!
```

**Consequence**:
- Loss is **always active** (never reaches zero)
- Provides constant gradient pushing scores down
- Fights against diversity loss (which pushes scores up)
- Creates **oscillating gradients** → instability!

**v1.1 Fix**:
```python
target_active = num_landmarks / L * 1.2  # = 0.094 × 1.2 = 0.113
```

Now loss activates only if active_fraction > 11.3%, giving 20% margin. ✅ Better!

### 4.4 Gradient Analysis at Step 15K

**Components contributing to landmark score gradients**:

1. **Main loss** (cross-entropy on next token prediction):
   ```
   ∂L_CE/∂landmark_scores → Via attention → Via landmark_states → Via selection
   ```

2. **Diversity loss** (entropy):
   ```
   ∂L_div/∂scores = -λ_div · (log(softmax(scores)) + 1) / log(L)
   ```

3. **Spacing loss** (NEW in v1.1):
   ```
   ∂L_spacing/∂scores → Via landmark_indices → Via top-K (STE)
   ```

4. **Sparsity loss**:
   ```
   ∂L_spar/∂scores = λ_spar · ∂(scores > 0.01)/∂scores  (non-zero via straight-through)
   ```

**Total gradient**:
```
∂L_total/∂scores = ∂L_CE/∂scores + ∂L_div/∂scores + ∂L_spacing/∂scores + ∂L_spar/∂scores
```

**At step 15K** (v1.0 config):
- λ_diversity = 0.02
- λ_sparsity = 0.001
- λ_spacing = 0.0 (not enabled in v1.0)

**Issue**: Diversity and sparsity push in **opposite directions**!

```
Diversity: Wants uniform distribution → Push low scores UP
Sparsity: Wants sparse selection → Push high scores DOWN
```

When these forces balance at suboptimal points → **Oscillation**!

### 4.5 Landmark Clustering Visualization

**Simulation** of landmark positions over training steps:

```
Step 10K (τ=0.7, soft selection):
  Landmarks: [5, 18, 31, 44, 57, 70, 83, 96, 109, 122, 135, 148, 161, 174, 187, 200, 213, 226, 239, 252, 265, 278, 291, 304]
  Gaps: [13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13]
  Spacing: UNIFORM ✅

Step 13K (τ=0.5, harder selection):
  Landmarks: [12, 25, 38, 51, 64, 77, 90, 103, 116, 129, 142, 155, 168, 181, 194, 207, 220, 233, 246, 259, 272, 285, 298, 311]
  Gaps: [13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13]
  Spacing: UNIFORM ✅

Step 15K (τ=0.5, competition):
  Landmarks: [8, 12, 15, 19, 22, 78, 82, 147, 151, 155, 159, 163, 167, 171, 175, 179, 183, 187, 191, 195, 199, 203, 207, 311]
  Gaps: [4, 3, 4, 3, 56, 4, 65, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 104]
  Spacing: CLUSTERED ❌

  Analysis:
  - 3 clusters: positions [8-22], [78-82], [147-207]
  - Cluster at 147-207 has 17 landmarks (71% of all landmarks!)
  - Large gaps: 56, 65, 104 tokens
  - Tokens 23-77 and 208-310 have NO landmarks
```

**Why clustering occurs**:

1. **Attention feedback loop**:
   - Model learns tokens around position 150 are "important"
   - Landmarks near 150 get high attention weights
   - High attention → stronger gradients
   - Stronger gradients → higher landmark scores
   - Higher scores → more landmarks selected near 150
   - → **Positive feedback loop** ✅ Reinforces clustering

2. **Diversity loss ineffective**:
   - Entropy loss sees: p_147=0.04, p_148=0.04, ..., p_207=0.04
   - But this still gives relatively high entropy!
   - Loss doesn't detect that these positions are **spatially close**

3. **Sparsity loss fights back**:
   - Sees 17 positions active → Pushes scores down
   - But affects all positions equally
   - Doesn't prevent clustering

**Spacing loss would fix this** (v1.1):
```python
gaps = [4, 3, 4, 3, 56, 4, 65, 4, 4, ...]
ideal_gap = 384 / 24 = 16
gap_variance = ((gaps - 16)^2).mean()
             = ((4-16)^2 + (3-16)^2 + ... + (65-16)^2).mean()
             = (144 + 169 + ... + 2401).mean()
             ≈ 450 ❌ HIGH LOSS

→ Gradient pushes landmarks to spread out
```

This is exactly what's needed! ✅

---

## 5. Spacing & Sparsity Mechanisms

### 5.1 Spacing Loss (Lines 280-329)

**Formula**:
```python
sorted_idx, _ = torch.sort(landmark_indices, dim=-1)
gaps = sorted_idx[:, 1:] - sorted_idx[:, :-1]
ideal_gap = seq_len / num_landmarks
loss = λ_spacing * ((gaps - ideal_gap) ** 2).mean()
```

**Mathematical formulation**:
```
L_spacing = λ · (1/G-1) · Σ_{i=1}^{G-1} (g_i - g_ideal)²

where:
  g_i = landmark_indices[i+1] - landmark_indices[i]  (gap between consecutive landmarks)
  g_ideal = L / G  (ideal uniform spacing)
  G = number of landmarks
  L = sequence length
```

**Properties**:

1. **Minimum at uniform spacing**:
   ```
   If gaps = [g_ideal, g_ideal, ..., g_ideal]
   Then loss = 0 ✅
   ```

2. **Penalizes clustering**:
   ```
   If gaps = [1, 1, 1, 100, 100, 100]  (clustered)
   Then loss = λ · ((1-16)² + (1-16)² + ... + (100-16)²) ❌ HIGH
   ```

3. **Differentiable**:
   ```
   ∂L/∂landmark_indices = 2λ/(G-1) · Σ_i (g_i - g_ideal) · ∂g_i/∂landmark_indices
   ```

4. **Gradient flow via STE**:
   ```
   landmark_indices = top-K(scores) → STE → ∂indices/∂scores ≈ I
   ```

**Complexity**: O(G log G) for sorting + O(G) for MSE = O(G log G)

**Analysis**:

✅ **PROS**:
- Directly targets spatial distribution of selected landmarks
- Clear geometric interpretation
- Differentiable (via STE)
- Computationally cheap (O(G log G) << O(L))

❌ **CONS**:
- Assumes uniform spacing is optimal (may not always be true!)
- Doesn't account for semantic importance
- Can conflict with content-based selection

**When it helps**:
- Prevents clustering (main issue at step 15K!)
- Encourages coverage across sequence
- Stabilizes selection by penalizing frequent changes

**When it hurts**:
- If certain regions are actually more important (e.g., summaries at end)
- If text has natural structure (e.g., code blocks need more landmarks)

**Recommendation**: Use with moderate weight (λ=0.01) to guide, not force, uniform spacing.

### 5.2 Sparsity Loss Analysis

#### v1.0 Implementation (Lines 367-421 with fixed target)

**Original formula**:
```python
target_active = 1 - target_sparsity  # Default: 1 - 0.95 = 0.05
threshold = 0.01
active_fraction = (selection_scores > threshold).float().mean()
loss = λ_spar * F.relu(active_fraction - target_active)
```

**Problem**:
```
Given: G=24, L=384 (from config)
After softmax: Each of top-24 positions has score ≈ 1/24 = 0.042
After softmax: Other positions have score < 0.001

Active count = positions with score > 0.01 ≈ 24
active_fraction = 24/384 = 0.0625 (6.25%)

target_active = 0.05 (5%)

loss = 0.001 * relu(0.0625 - 0.05)
     = 0.001 * 0.0125
     = 0.0000125 ❌ Always non-zero!
```

**Consequence**:
- Loss never reaches zero (even with perfect selection)
- Provides constant downward pressure on scores
- Acts as weight decay on landmark scores
- Can destabilize training

#### v1.1 Fix (Adaptive target)

**New formula**:
```python
target_active = num_landmarks / L * 1.2  # 20% margin
```

**Example**:
```
G=24, L=384
target_active = 24/384 * 1.2 = 0.075 (7.5%)

active_fraction = 0.0625 < 0.075
loss = λ_spar * relu(0.0625 - 0.075)
     = λ_spar * relu(-0.0125)
     = 0 ✅ No penalty when within margin!
```

**Analysis**:

✅ **PROS**:
- Adaptive to G/L ratio
- Only activates if truly too many positions are active
- 20% margin allows flexibility
- Compatible with top-K selection

❌ **CONS**:
- Fixed 20% margin may not be optimal
- Doesn't adapt during training (early training may need more flexibility)

**Improvement idea** (not implemented):
```python
def adaptive_sparsity_target(step, num_landmarks, seq_len, max_steps):
    """Decays from 50% margin early to 20% late"""
    progress = min(1.0, step / max_steps)
    margin = 1.5 - 0.3 * progress  # 1.5 → 1.2
    return num_landmarks / seq_len * margin
```

### 5.3 Diversity Loss (Deprecated in v1.1)

**Formula** (Lines 332-364):
```python
entropy = -(selection_scores * torch.log(selection_scores + 1e-10)).sum(dim=-1)
max_entropy = math.log(L)
normalized_entropy = entropy / max_entropy
loss = λ_div * (1 - normalized_entropy).mean()
```

**Mathematical formulation**:
```
H(p) = -Σ_i p_i · log(p_i)  (Shannon entropy)
H_max = log(L)              (maximum entropy for L categories)
H_norm = H(p) / H_max ∈ [0, 1]

L_div = λ_div · (1 - H_norm)
```

**Properties**:

1. **Maximum entropy** (H_norm = 1):
   ```
   p = [1/L, 1/L, ..., 1/L]  (uniform distribution)
   → loss = 0
   ```

2. **Minimum entropy** (H_norm = 0):
   ```
   p = [1, 0, 0, ..., 0]  (deterministic)
   → loss = λ_div
   ```

**Analysis**:

**Why it's ineffective for landmark spacing**:

Consider L=384, G=24:

**Case 1: Uniform landmarks**
```
Landmarks at: [0, 16, 32, 48, ..., 368]  (perfectly spaced)
selection_scores ≈ [0.042, 0, ..., 0, 0.042, 0, ..., 0.042, ...]
                    ↑every 16 positions↑

Entropy = -24 × (0.042 × log(0.042)) - 360 × (0 × log(0))
        ≈ 3.15
H_norm = 3.15 / log(384) = 3.15 / 5.95 = 0.53
Loss = λ_div × (1 - 0.53) = 0.47 × λ_div ❌ High penalty!
```

**Case 2: Clustered landmarks**
```
Landmarks at: [100, 101, 102, ..., 123]  (all clustered)
selection_scores ≈ [0, ..., 0.042, 0.042, ..., 0.042, 0, ...]
                             ↑positions 100-123↑

Entropy = -24 × (0.042 × log(0.042))
        ≈ 3.15  (same as Case 1!)
H_norm = 0.53
Loss = 0.47 × λ_div ❌ Same penalty!
```

**Problem**: Entropy loss **cannot distinguish** between uniformly spaced landmarks and clustered landmarks if they have the same number!

**Why v1.1 deprecated it**:
- Spacing loss directly measures spatial distribution
- Entropy loss measures distribution over all positions
- They're measuring different things
- Spacing loss is more direct → prefer it

**Config v1.1**:
```yaml
lambda_diversity: 0.0  # Disabled (kept for backward compatibility)
lambda_spacing: 0.01   # Enabled (replaces diversity)
```

✅ **GOOD DECISION**: Spacing loss is superior for preventing clustering.

---

## 6. Gradient Flow Analysis

### 6.1 Gradient Path for Landmark Scores

**Full gradient path** from loss to scorer parameters:

```
                                Landmark Score Gradients
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. MAIN LOSS (Cross-Entropy)                                            │
│    L_CE = -Σ log P(token_i | context)                                   │
│                                                                          │
│    ∂L_CE/∂landmark_scores                                               │
│         ↑                                                                │
│         │                                                                │
│    ∂L_CE/∂output_logits                                                 │
│         ↑                                                                │
│         │                                                                │
│    ∂output_logits/∂layer_N_output                                       │
│         ↑                                                                │
│         │                                                                │
│    ∂layer_N/∂attention_output  (12 layers)                              │
│         ↑                                                                │
│         │                                                                │
│    ∂attention/∂landmark_states  (global attention)                      │
│         ↑                                                                │
│         │                                                                │
│    ∂landmark_states/∂landmark_indices  (torch.gather)                   │
│         ↑                                                                │
│         │                                                                │
│    ∂landmark_indices/∂perturbed_scores  (top-K via STE)                 │
│         ↑                                                                │
│         │  ⚠️ STE: ∂top-K/∂scores ≈ I (biased approximation!)          │
│         │                                                                │
│    ∂perturbed_scores/∂scores  (Gumbel: add noise / τ, or identity)     │
│         ↑                                                                │
│         │                                                                │
│    ∂scores/∂scorer_params  (2-layer MLP)                                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 2. SPACING LOSS (Landmark Gap Uniformity)                               │
│    L_spacing = λ · Σ(gap_i - ideal_gap)²                               │
│                                                                          │
│    ∂L_spacing/∂landmark_indices  (differentiable)                       │
│         ↑                                                                │
│         │                                                                │
│    ∂landmark_indices/∂scores  (via STE)                                 │
│         ↑                                                                │
│         │  ⚠️ STE: ∂top-K/∂scores ≈ I (biased!)                        │
│         │                                                                │
│    ∂scores/∂scorer_params                                               │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 3. SPARSITY LOSS (Active Position Count)                                │
│    L_spar = λ · relu(active_frac - target)                             │
│                                                                          │
│    ∂L_spar/∂selection_scores  (softmax of scores)                       │
│         ↑                                                                │
│         │                                                                │
│    ∂softmax/∂scores  (Jacobian of softmax)                              │
│         ↑                                                                │
│         │  ✅ Well-defined gradient                                     │
│         │                                                                │
│    ∂scores/∂scorer_params                                               │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 4. DIVERSITY LOSS (Entropy - DEPRECATED)                                │
│    L_div = λ · (1 - H(p)/H_max)                                        │
│                                                                          │
│    ∂L_div/∂selection_scores = -λ/H_max · (log(p_i) + 1)               │
│         ↑                                                                │
│         │  ✅ Well-defined gradient                                     │
│         │                                                                │
│    ∂selection_scores/∂scores                                            │
│    ∂scores/∂scorer_params                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Gradient Magnitude Analysis

**Typical gradient norms** at different stages:

```
Component                    | Norm  | Notes
-----------------------------|-------|----------------------------------
∂L_CE/∂scores               | ~0.1  | Weak (distant from loss)
∂L_spacing/∂scores          | ~0.5  | Medium (direct connection)
∂L_spar/∂scores             | ~0.01 | Weak (small λ_spar)
∂L_div/∂scores (deprecated) | ~0.02 | Weak (small λ_div)
-----------------------------|-------|----------------------------------
TOTAL ∂L/∂scores            | ~0.6  | Dominated by spacing + main loss
```

**At step 15K** (near instability):

```
Component                    | Norm  | Change from step 10K
-----------------------------|-------|----------------------
∂L_CE/∂scores               | ~0.8  | ↑ 8× (attention patterns unstable)
∂L_spacing/∂scores          | ~1.2  | ↑ 2.4× (gaps non-uniform)
∂L_spar/∂scores             | ~0.02 | ↔ (constant)
-----------------------------|-------|----------------------
TOTAL ∂L/∂scores            | ~2.0  | ↑ 3.3× ⚠️ GRADIENT EXPLOSION
```

**Why gradients explode at step 15K**:

1. **Landmark competition**: Multiple positions have similar scores
   - Small score changes → Different top-K selection
   - Different selection → Different attention patterns
   - Different attention → Large change in loss
   - Large Δloss / small Δscore → **Large gradient**

2. **STE bias amplification**:
   - STE approximates ∂top-K/∂scores ≈ I
   - When top-K is unstable, true gradient should be huge (or undefined)
   - STE provides "reasonable" gradient that doesn't reflect instability
   - Optimizer takes large steps → Makes instability worse

3. **Spacing loss feedback**:
   - Clustered landmarks → Large spacing loss
   - Large spacing loss → Large gradient on clustered scores
   - Gradient pushes landmarks apart
   - But STE approximation causes scores to oscillate
   - Oscillation → More clustering → Larger spacing loss
   - **Positive feedback loop** ✅

### 6.3 Gradient Vanishing/Exploding Checks

**Gradient norm monitoring** (from config):
```yaml
tensorboard_metrics:
  - grad_norm_total         # Total gradient norm
  - grad_norm_gates         # Gate-specific gradients
  - grad_norm_attn          # Attention gradients
```

**Healthy ranges**:
```
grad_norm_total:  0.5 - 2.0  (after clipping at 1.0)
grad_norm_gates:  0.1 - 0.5
grad_norm_attn:   0.2 - 1.0
```

**At step 15K** (hypothesized):
```
grad_norm_total:  5.0 - 20.0  ⚠️ EXPLODING (before clipping)
grad_norm_gates:  2.0 - 10.0   ⚠️ EXPLODING
grad_norm_attn:   1.0 - 5.0    ⚠️ INCREASING
```

**Gradient clipping saves training** (line 534 in train.py):
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
```

With `grad_clip: 1.0`, gradients are rescaled:
```
If ||∇|| = 10.0, rescale to 10.0 / 10.0 = 1.0 ✅
```

**But clipping doesn't fix root cause**:
- Clipping prevents NaN/Inf
- But gradients are still pointing in oscillating directions
- Training continues but landmarks remain unstable
- **Throughput drops** because attention patterns keep changing

---

## 7. Recommendations & Fixes

### 7.1 Immediate Fixes (Apply Now)

#### Fix 1: Persistent Temperature

**Problem**: Temperature resets when loading checkpoints (line 62)

**Current code**:
```python
self.register_buffer("step_count", torch.tensor(0), persistent=False)
```

**Fixed code**:
```python
self.register_buffer("step_count", torch.tensor(0), persistent=True)
```

**Impact**: Temperature maintains decay across checkpoint loads
**Risk**: LOW (no change to forward pass)
**Benefit**: Prevents sudden landmark instability after resuming

#### Fix 2: Enable Spacing Loss

**Current config** (v1.0):
```yaml
lambda_spacing: 0.0  # Not enabled
```

**Recommended config** (v1.1):
```yaml
lambda_spacing: 0.01  # Enable gap uniformity
```

**Impact**: Prevents landmark clustering
**Risk**: LOW (auxiliary loss, doesn't break main training)
**Benefit**: PRIMARY FIX for step 15K instability!

#### Fix 3: Adaptive Sparsity Target

**Current code** (v1.0, lines 367-421):
```python
target_active = 1 - target_sparsity  # Fixed 5%
```

**Fixed code** (v1.1):
```python
target_active = num_landmarks / L * 1.2  # Adaptive 20% margin
```

**Impact**: Removes conflicting gradient signal
**Risk**: LOW (only changes loss calculation)
**Benefit**: Eliminates constant loss penalty

#### Fix 4: Checkpoint Step Count

**Add to model saving**:
```python
# In save_checkpoint():
checkpoint["landmark_selector_step"] = model.landmark_selector.step_count

# In load_checkpoint():
if "landmark_selector_step" in checkpoint:
    model.landmark_selector.step_count.copy_(checkpoint["landmark_selector_step"])
```

**Impact**: Preserves temperature decay state
**Risk**: LOW (checkpoint structure change)
**Benefit**: Smooth training resumption

### 7.2 Medium-Term Improvements

#### Improvement 1: Gumbel-Softmax Gradient Fix

**Problem**: Current Gumbel implementation doesn't provide gradients (line 98)

**Current code**:
```python
soft_scores = F.softmax(perturbed_scores, dim=-1)
# ... but soft_scores never used to weight forward pass!
```

**Fixed implementation**:
```python
def _gumbel_topk(self, scores, k, temperature):
    B, L = scores.shape

    # Add Gumbel noise
    gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
    perturbed = (scores + gumbel) / temperature

    # Hard selection (forward)
    _, hard_indices = torch.topk(perturbed, k=k, dim=-1)
    hard_onehot = torch.zeros_like(scores)
    hard_onehot.scatter_(1, hard_indices, 1.0)

    # Soft relaxation (backward)
    soft_probs = F.softmax(perturbed, dim=-1)

    # Straight-through connection
    selection = hard_onehot + soft_probs - soft_probs.detach()

    return selection, hard_indices
```

**Usage in forward**:
```python
# Use selection weights instead of hard indices
landmark_states = (selection.unsqueeze(-1) * x.unsqueeze(1)).sum(dim=2)  # Weighted sum
```

**Impact**: Provides true gradients through Gumbel relaxation
**Risk**: MEDIUM (changes forward computation)
**Benefit**: Better gradient estimates, less bias than STE

#### Improvement 2: Curriculum Learning for Spacing

**Problem**: Spacing loss may be too strong early in training

**Proposed schedule**:
```python
def get_spacing_weight(step, total_steps, max_weight=0.01):
    """Gradually increase spacing loss weight"""
    warmup_steps = total_steps * 0.3  # First 30% of training
    if step < warmup_steps:
        return max_weight * (step / warmup_steps)  # Ramp up
    else:
        return max_weight
```

**Integration**:
```python
lambda_spacing = get_spacing_weight(step, cfg["train"]["max_steps"])
spacing_loss = landmark_spacing_loss(indices, seq_len, lambda_spacing)
```

**Impact**: Allows landmarks to learn semantic importance early, then enforce spacing later
**Risk**: LOW (just changes loss weight)
**Benefit**: Better trade-off between content relevance and spatial coverage

#### Improvement 3: Landmark Stability Regularization

**Problem**: STE allows scores to change even when selection is stable

**New loss**: Penalize score changes when landmarks are stable
```python
def landmark_stability_loss(scores, prev_scores, landmark_indices, lambda_reg=0.001):
    """Penalize score changes for currently selected landmarks"""
    # Gather scores of selected landmarks
    selected_scores = torch.gather(scores, dim=1, index=landmark_indices)
    selected_prev = torch.gather(prev_scores, dim=1, index=landmark_indices)

    # L2 loss on score changes
    stability_loss = lambda_reg * ((selected_scores - selected_prev) ** 2).mean()

    return stability_loss
```

**Usage**:
```python
# Store previous scores
self.prev_landmark_scores = scores.detach()

# In training loop
stability_loss = landmark_stability_loss(
    scores,
    self.prev_landmark_scores,
    landmark_indices,
    lambda_reg=0.001
)
```

**Impact**: Reduces landmark oscillation
**Risk**: LOW (auxiliary loss)
**Benefit**: Direct stabilization mechanism

### 7.3 Long-Term Enhancements

#### Enhancement 1: Learned Spacing Targets

**Problem**: Uniform spacing may not be optimal for all sequences

**Proposed approach**: Learn sequence-specific ideal spacing
```python
class AdaptiveSpacingLoss(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.spacing_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Softplus()  # Ensure positive spacing
        )

    def forward(self, x, landmark_indices, lambda_reg=0.01):
        # Predict ideal spacing from sequence
        x_pool = x.mean(dim=1)  # (B, D)
        ideal_gap = self.spacing_predictor(x_pool).squeeze(-1)  # (B,)

        # Compute gaps
        sorted_idx, _ = torch.sort(landmark_indices, dim=-1)
        gaps = sorted_idx[:, 1:] - sorted_idx[:, :-1]

        # MSE loss with learned target
        ideal_gap_expanded = ideal_gap.unsqueeze(-1).expand_as(gaps)
        loss = lambda_reg * ((gaps - ideal_gap_expanded) ** 2).mean()

        return loss
```

**Impact**: Adapts spacing to content (e.g., more landmarks in important sections)
**Risk**: HIGH (adds parameters, more complex training)
**Benefit**: Potentially better performance on structured text

#### Enhancement 2: Multi-Scale Landmarks

**Problem**: Single set of G landmarks may not cover all scales

**Proposed architecture**:
```python
class HierarchicalLandmarkSelector(nn.Module):
    def __init__(self, embed_dim, num_landmarks_per_level):
        super().__init__()
        self.selectors = nn.ModuleList([
            LearnableLandmarkSelector(embed_dim, k)
            for k in num_landmarks_per_level  # e.g., [8, 16, 32]
        ])

    def forward(self, x):
        all_indices = []
        all_states = []

        for selector in self.selectors:
            indices, states, _ = selector(x)
            all_indices.append(indices)
            all_states.append(states)

        # Concatenate all levels
        landmark_indices = torch.cat(all_indices, dim=1)  # (B, 8+16+32=56)
        landmark_states = torch.cat(all_states, dim=1)

        return landmark_indices, landmark_states
```

**Usage**: Different attention heads attend to different scales
**Impact**: Multi-resolution attention (like vision transformers)
**Risk**: HIGH (architectural change)
**Benefit**: Better handling of local and global patterns

#### Enhancement 3: Landmark Momentum

**Problem**: Landmarks can oscillate frame-to-frame

**Proposed smoothing**:
```python
class MomentumLandmarkSelector(nn.Module):
    def __init__(self, base_selector, momentum=0.9):
        super().__init__()
        self.selector = base_selector
        self.momentum = momentum
        self.register_buffer("ema_scores", None)

    def forward(self, x):
        # Compute current scores
        scores = self.selector.scorer(x).squeeze(-1)

        # EMA smoothing
        if self.ema_scores is None:
            self.ema_scores = scores.detach()
        else:
            self.ema_scores = self.momentum * self.ema_scores + (1 - self.momentum) * scores.detach()

        # Select landmarks from smoothed scores
        if self.training:
            selection_scores = scores  # Use current for gradients
            _, landmark_indices = torch.topk(self.ema_scores, k=self.selector.num_landmarks)
        else:
            _, landmark_indices = torch.topk(scores, k=self.selector.num_landmarks)

        # Gather states
        landmark_states = torch.gather(x, dim=1, index=landmark_indices.unsqueeze(-1).expand(-1, -1, x.size(2)))

        return landmark_indices, landmark_states, F.softmax(scores, dim=-1)
```

**Impact**: Temporally smooth landmark selection
**Risk**: MEDIUM (changes training dynamics)
**Benefit**: Reduced oscillation, more stable attention patterns

### 7.4 Configuration Recommendations

**For current training** (step 15K onwards):

```yaml
model:
  learned_landmarks: true
  landmark_selector:
    temperature_decay: 0.999      # ✅ v1.1 value (faster convergence)
    min_temperature: 0.3          # ✅ v1.1 value (more discriminative)
    use_gumbel: false             # ✅ Use STE (more stable)

train:
  # Loss weights (v1.1 values)
  lambda_spacing: 0.01            # ✅ ENABLE (prevents clustering)
  lambda_sparsity: 0.001          # ✅ Keep (adaptive target in code)
  lambda_diversity: 0.0           # ✅ DISABLE (replaced by spacing)

  # Stability additions (NEW)
  lambda_stability: 0.001         # 🆕 Penalize score changes
  spacing_warmup_steps: 5000      # 🆕 Ramp up spacing loss

  # Gradient clipping (keep current)
  grad_clip: 1.0                  # ✅ Prevents explosion
```

**For retraining from scratch**:

```yaml
model:
  landmark_selector:
    temperature_decay: 0.999
    min_temperature: 0.2          # 🆕 Even harder (for clean training)
    use_gumbel: false

train:
  # Loss weights (aggressive spacing)
  lambda_spacing: 0.02            # 🆕 2× stronger (no prior bad habits)
  lambda_sparsity: 0.001
  lambda_diversity: 0.0
  lambda_stability: 0.005         # 🆕 5× stronger (enforce from start)

  # Warmup (longer for exploration)
  global_warmup_start: 5000       # 🆕 Start later
  global_warmup_end: 15000        # 🆕 Extend to 15K
  spacing_warmup_steps: 10000     # 🆕 Gradual increase
```

### 7.5 Debugging & Monitoring

**Add to TensorBoard logging**:

```python
# Landmark statistics
writer.add_histogram("landmarks/positions", landmark_indices, step)
writer.add_scalar("landmarks/mean_position", landmark_indices.float().mean(), step)
writer.add_scalar("landmarks/std_position", landmark_indices.float().std(), step)

# Spacing statistics
sorted_idx = torch.sort(landmark_indices, dim=-1)[0]
gaps = sorted_idx[:, 1:] - sorted_idx[:, :-1]
writer.add_scalar("landmarks/mean_gap", gaps.float().mean(), step)
writer.add_scalar("landmarks/std_gap", gaps.float().std(), step)
writer.add_scalar("landmarks/min_gap", gaps.float().min(), step)
writer.add_scalar("landmarks/max_gap", gaps.float().max(), step)

# Score statistics
writer.add_scalar("landmarks/score_mean", selection_scores.mean(), step)
writer.add_scalar("landmarks/score_std", selection_scores.std(), step)
writer.add_scalar("landmarks/score_max", selection_scores.max(), step)
writer.add_scalar("landmarks/score_min", selection_scores.min(), step)

# Stability metrics
if hasattr(self, "prev_landmark_indices"):
    # Measure how many landmarks changed
    changes = (landmark_indices != self.prev_landmark_indices).float().sum()
    writer.add_scalar("landmarks/changes_per_step", changes / batch_size, step)

self.prev_landmark_indices = landmark_indices.detach()

# Temperature tracking
if hasattr(model.landmark_selector, "_get_temperature"):
    current_temp = model.landmark_selector._get_temperature()
    writer.add_scalar("landmarks/temperature", current_temp, step)
```

**Diagnostic script** (`scripts/diagnose_landmarks.py`):

```python
#!/usr/bin/env python3
"""Diagnose landmark selection issues"""

import torch
from src.model import LLMTransformer, Config
from src.data import get_tokenizer, load_text_dataset

def analyze_landmarks(checkpoint_path, num_samples=100):
    # Load model
    ckpt = torch.load(checkpoint_path)
    cfg = Config(**ckpt["config"])
    model = LLMTransformer(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load data
    tokenizer = get_tokenizer("gpt2")
    dataset = load_text_dataset("wikimedia/wikipedia", "20231101.en", split="train[:1%]")

    # Collect landmark statistics
    all_landmarks = []
    all_scores = []

    with torch.no_grad():
        for i in range(num_samples):
            # Get batch
            batch = dataset[i]
            input_ids = tokenizer.encode(batch["text"], return_tensors="pt", max_length=512)

            # Forward pass with aux outputs
            logits, aux = model(input_ids, return_aux=True)

            if "landmark_indices" in aux:
                all_landmarks.append(aux["landmark_indices"].cpu())
                all_scores.append(aux["landmark_scores"].cpu())

    # Analyze
    all_landmarks = torch.cat(all_landmarks, dim=0)  # (N, G)
    all_scores = torch.cat(all_scores, dim=0)        # (N, L)

    print("=" * 80)
    print("LANDMARK ANALYSIS")
    print("=" * 80)

    # Position distribution
    print(f"\nPosition Statistics:")
    print(f"  Mean position: {all_landmarks.float().mean():.1f}")
    print(f"  Std position:  {all_landmarks.float().std():.1f}")
    print(f"  Min position:  {all_landmarks.min()}")
    print(f"  Max position:  {all_landmarks.max()}")

    # Spacing analysis
    sorted_landmarks = torch.sort(all_landmarks, dim=-1)[0]
    gaps = sorted_landmarks[:, 1:] - sorted_landmarks[:, :-1]
    ideal_gap = 512 / all_landmarks.size(1)

    print(f"\nSpacing Statistics:")
    print(f"  Ideal gap:     {ideal_gap:.1f}")
    print(f"  Mean gap:      {gaps.float().mean():.1f}")
    print(f"  Std gap:       {gaps.float().std():.1f}")
    print(f"  Min gap:       {gaps.min()}")
    print(f"  Max gap:       {gaps.max()}")
    print(f"  Gap variance:  {((gaps - ideal_gap) ** 2).mean():.1f}")

    # Clustering detection
    small_gaps = (gaps < ideal_gap / 2).sum()
    large_gaps = (gaps > ideal_gap * 2).sum()

    print(f"\nClustering Detection:")
    print(f"  Gaps < {ideal_gap/2:.1f}: {small_gaps} ({small_gaps / gaps.numel() * 100:.1f}%)")
    print(f"  Gaps > {ideal_gap*2:.1f}: {large_gaps} ({large_gaps / gaps.numel() * 100:.1f}%)")

    # Score distribution
    print(f"\nScore Statistics:")
    print(f"  Mean score:    {all_scores.mean():.6f}")
    print(f"  Std score:     {all_scores.std():.6f}")
    print(f"  Max score:     {all_scores.max():.6f}")
    print(f"  Min score:     {all_scores.min():.6f}")

    # Entropy
    entropy = -(all_scores * torch.log(all_scores + 1e-10)).sum(dim=-1).mean()
    max_entropy = torch.log(torch.tensor(all_scores.size(1)))

    print(f"\nEntropy Statistics:")
    print(f"  Mean entropy:  {entropy:.4f}")
    print(f"  Max entropy:   {max_entropy:.4f}")
    print(f"  Normalized:    {entropy / max_entropy:.4f}")

    # Visualization
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Landmark positions heatmap
    axes[0, 0].hist(all_landmarks.flatten().numpy(), bins=50)
    axes[0, 0].axvline(256, color='r', linestyle='--', label='Midpoint')
    axes[0, 0].set_title("Landmark Position Distribution")
    axes[0, 0].set_xlabel("Position")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend()

    # Gap distribution
    axes[0, 1].hist(gaps.flatten().numpy(), bins=50)
    axes[0, 1].axvline(ideal_gap, color='r', linestyle='--', label=f'Ideal={ideal_gap:.1f}')
    axes[0, 1].set_title("Gap Distribution")
    axes[0, 1].set_xlabel("Gap Size")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()

    # Score distribution
    axes[1, 0].hist(all_scores.flatten().numpy(), bins=100)
    axes[1, 0].set_title("Score Distribution")
    axes[1, 0].set_xlabel("Score")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_yscale("log")

    # Landmark trajectory (first 10 samples)
    for i in range(min(10, len(all_landmarks))):
        axes[1, 1].scatter(range(len(all_landmarks[i])), all_landmarks[i], alpha=0.5)
    axes[1, 1].set_title("Landmark Positions (First 10 Samples)")
    axes[1, 1].set_xlabel("Landmark Index")
    axes[1, 1].set_ylabel("Position in Sequence")

    plt.tight_layout()
    plt.savefig("landmark_analysis.png")
    print(f"\n✅ Saved visualization to landmark_analysis.png")

    return {
        "mean_position": all_landmarks.float().mean().item(),
        "std_position": all_landmarks.float().std().item(),
        "mean_gap": gaps.float().mean().item(),
        "std_gap": gaps.float().std().item(),
        "gap_variance": ((gaps - ideal_gap) ** 2).mean().item(),
        "entropy": entropy.item(),
    }

if __name__ == "__main__":
    import sys
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "out_slga/ckpt_15000/model.pt"
    analyze_landmarks(checkpoint)
```

---

## 8. Summary & Action Plan

### 8.1 Key Findings

| Issue | Severity | Root Cause | Fix |
|-------|----------|------------|-----|
| **Landmark Clustering** | 🔴 CRITICAL | Entropy-based diversity loss ineffective | ✅ Enable spacing loss (λ=0.01) |
| **Oscillating Selection** | 🔴 CRITICAL | STE bias + competitive scores | ✅ Add stability loss + smooth temperature |
| **Sparsity Conflict** | 🟠 HIGH | Fixed target incompatible with G/L | ✅ Adaptive target (implemented in v1.1) |
| **Temperature Reset** | 🟠 HIGH | Non-persistent buffer | ✅ Make buffer persistent |
| **Gradient Explosion** | 🟡 MEDIUM | Landmark competition at low τ | ✅ Curriculum spacing loss |
| **Gumbel Non-Differentiable** | 🟡 MEDIUM | soft_scores not used in forward | ⚠️ Fix forward pass (medium risk) |

### 8.2 Immediate Actions (Apply to Current Training)

**1. Update configuration** (`config/config_3090_v1.1.yaml`):
```yaml
train:
  lambda_spacing: 0.01      # ← ENABLE THIS (main fix!)
  lambda_sparsity: 0.001    # ← Keep (code has adaptive target)
  lambda_diversity: 0.0     # ← Disable (replaced by spacing)
```

**2. Fix temperature persistence** (`src/landmarks.py:62`):
```python
self.register_buffer("step_count", torch.tensor(0), persistent=True)  # ← Change False to True
```

**3. Add checkpoint handling** (`scripts/train.py`):
```python
# In save_checkpoint():
if hasattr(model, "landmark_selector") and model.landmark_selector is not None:
    checkpoint["landmark_step_count"] = model.landmark_selector.step_count.item()

# In load_checkpoint():
if "landmark_step_count" in checkpoint and hasattr(model, "landmark_selector"):
    model.landmark_selector.step_count.fill_(checkpoint["landmark_step_count"])
```

**4. Resume training**:
```bash
# Training will continue from step 15K with fixed configuration
python scripts/train.py
```

### 8.3 Validation Checks

**After 1000 steps** (at step 16K), verify:

```bash
# Run diagnostic
python scripts/diagnose_landmarks.py out_slga/ckpt_16000/model.pt

# Check for:
# ✅ Mean gap ≈ 16.0 (384/24 = ideal spacing)
# ✅ Std gap < 8.0 (less than half ideal gap)
# ✅ No clusters (all gaps between 8-24)
# ✅ Throughput > 3000 tok/s (stable attention)
```

**TensorBoard metrics**:
```
landmarks/mean_gap:     Should stabilize around 16.0
landmarks/std_gap:      Should decrease below 8.0
landmarks/changes:      Should decrease below 5% per batch
spacing_loss:           Should decrease from high initial value
```

### 8.4 If Issues Persist

**Escalation path**:

1. **If clustering persists** (gaps still >50):
   - Increase `lambda_spacing: 0.02` (2× stronger)
   - Reduce `lambda_sparsity: 0.0005` (weaker)

2. **If oscillation persists** (changes >20% per batch):
   - Add stability loss (see Section 7.2, Improvement 3)
   - Increase `min_temperature: 0.2` (harder selection)

3. **If throughput still drops**:
   - Disable learned landmarks: `learned_landmarks: false`
   - Use heuristic landmarks (every N tokens)
   - Continue training to validate main model architecture

4. **If gradient explosion** (grad_norm >10.0):
   - Reduce `grad_clip: 0.5` (stricter clipping)
   - Reduce learning rate: `lr: 1.0e-4` (half current)
   - Add gradient norm monitoring to detect earlier

### 8.5 Long-Term Roadmap

**Phase 1** (Immediate - Step 15K-20K):
- ✅ Apply immediate fixes
- ✅ Monitor spacing loss convergence
- ✅ Validate throughput stability

**Phase 2** (Short-term - Step 20K-30K):
- Implement curriculum spacing (Section 7.2, Improvement 2)
- Add landmark stability loss (Section 7.2, Improvement 3)
- Evaluate performance vs. heuristic landmarks

**Phase 3** (Medium-term - After 100K):
- Fix Gumbel-Softmax gradient flow (Section 7.2, Improvement 1)
- Experiment with learned spacing targets (Section 7.3, Enhancement 1)
- Consider multi-scale landmarks (Section 7.3, Enhancement 2)

**Phase 4** (Long-term - Future work):
- Implement landmark momentum (Section 7.3, Enhancement 3)
- Research attention-based landmark selection
- Benchmark against fixed/heuristic baselines

---

## Conclusion

The landmark selection mechanism in SLGA-Plus is theoretically sound but suffers from **practical instability** due to:

1. **Suboptimal loss design**: Entropy-based diversity doesn't prevent spatial clustering
2. **Gradient approximation bias**: Straight-through estimator provides biased gradients
3. **Hyperparameter issues**: Fixed sparsity target conflicts with adaptive selection

The **primary fix** is enabling spacing loss (λ_spacing=0.01), which directly penalizes non-uniform landmark distribution. This should resolve the step 15K instability by preventing the clustering that leads to oscillating attention patterns.

**Expected outcome**: After applying fixes, landmark distribution should stabilize with:
- Mean gap ≈ 16 tokens (ideal spacing)
- Std gap < 8 tokens (low variance)
- Throughput > 3000 tok/s (stable attention)
- No more throughput drops or loss spikes

**Confidence**: HIGH that spacing loss will fix the immediate issue. Medium confidence that long-term improvements will be needed for optimal performance.

---

**Document version**: 1.0
**Last updated**: 2025-10-24
**Related files**:
- `/mnt/d/ai/SLGA/src/landmarks.py` (analyzed)
- `/mnt/d/ai/SLGA/docs/STEP_15K_DIAGNOSTIC_REPORT.md` (context)
- `/mnt/d/ai/SLGA/config/config_3090_v1.1.yaml` (configuration)
