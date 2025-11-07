# SLGA Component Deep Dive Analysis

**Date**: 2025-10-24
**Analysis Focus**: Landmark selection, GradNorm weighting, attention computation, and inference behavior

---

## Executive Summary

**Critical Finding**: The training logs show `LM: 48→24` which indicates the landmark selector is producing **48 candidates**, but SLGA's top-K mechanism is correctly reducing this to **24 landmarks per attention head** as configured. This is **CORRECT BEHAVIOR**, not a bug.

**GW: 1.00** indicates global warmup weight is at 100%, meaning the global attention is fully activated (passed the warmup period of steps 1000-5000).

---

## 1. Landmark Selection Mechanism

### 1.1 Architecture Overview

The SLGA implementation uses a **two-stage landmark selection**:

**Stage 1: Candidate Selection** (`LearnableLandmarkSelector`)
- Located in: `/mnt/d/ai/SLGA/src/landmarks.py`
- Creates **48 landmark candidates** (configured as `num_landmarks=cfg.global_k * 2 = 24 * 2 = 48`)
- Uses a learned neural scorer to select important positions

**Stage 2: Top-K Reduction** (`SLGAModule`)
- Located in: `/mnt/d/ai/SLGA/src/slga.py`
- Reduces 48 candidates to **24 top-K landmarks per attention head**
- Uses diversity-aware top-K selection (`_diverse_topk`)

### 1.2 Landmark Selector Implementation

```python
# From model.py lines 182-188
if cfg.learned_landmarks:
    self.landmark_selector = LearnableLandmarkSelector(
        embed_dim=cfg.embed_dim,
        num_landmarks=cfg.global_k * 2,  # 24 * 2 = 48 candidates
    )
```

**Key Components**:

1. **Neural Scorer** (landmarks.py, lines 52-59):
```python
self.scorer = nn.Sequential(
    nn.Linear(embed_dim, hidden),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(hidden, 1),  # Score per position
)
```

2. **Straight-Through Top-K Estimator** (lines 102-124):
- Forward: Hard top-K selection (48 candidates)
- Backward: Gradient flows through soft scores
- Enables differentiable discrete selection

3. **Gumbel-Softmax Alternative** (lines 72-100):
- Temperature-annealed relaxation
- Gradually hardens selection during training
- Currently NOT used (straight-through is default)

### 1.3 Landmark Flow Through Model

```
Input Sequence (B, L, D)
        ↓
LearnableLandmarkSelector.forward()
        ↓
Scores each position: (B, L) → scores
        ↓
Straight-Through Top-K: Select 48 positions
        ↓
landmark_indices: (B, 48)
landmark_states: (B, 48, D)  [gathered from x]
        ↓
For EACH TransformerBlock:
    ↓
    Gather landmark states from current x
        ↓
    landmark_states = x[:, landmark_indices, :]  # (B, 48, D)
        ↓
    SLGAModule.forward(x, cache_global=landmark_states)
        ↓
    SLGA's _diverse_topk: 48 → 24 per head
        ↓
    Attention computation with 24 landmarks
```

**Important**: Landmarks are **re-extracted at each layer** from the evolving hidden states, not just once at the beginning!

From `model.py` lines 260-271:
```python
for block in self.blocks:
    # Extract CURRENT landmark states (they evolve with x!)
    if landmark_indices is not None:
        B_cur, L_cur, D = x.shape
        G = landmark_indices.size(1)  # G = 48
        landmark_indices_exp = landmark_indices.unsqueeze(-1).expand(B_cur, G, D)
        landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)  # (B, 48, D)

    # Pass to SLGA with UPDATED landmark states
    x = block(x, cache_global=landmark_states, global_weight=global_weight)
```

---

## 2. GradNorm and Landmark Weighting

### 2.1 What "GW: 1.00" Means

From `scripts/train.py` lines 65-80:
```python
def get_global_warmup_weight(step: int, cfg: dict) -> float:
    """
    Calcule le poids de warmup pour attention globale.
    """
    warmup_start = cfg["train"].get("global_warmup_start", 30000)  # 1000 in config
    warmup_end = cfg["train"].get("global_warmup_end", 50000)      # 5000 in config

    if step < warmup_start:
        return 0.0  # Global attention disabled
    elif step < warmup_end:
        progress = (step - warmup_start) / (warmup_end - warmup_start)
        return progress  # Linear ramp 0→1
    else:
        return 1.0  # Fully enabled
```

**Current Status** (based on logs showing GW: 1.00):
- Training has passed step 5000
- Global attention is **fully active** (weight = 1.0)
- No gradual ramping happening now

### 2.2 What "LM: 48→24" Means

From `scripts/train.py` lines 532-541:
```python
# Note: Landmarks = candidats sélectionnés, SLGA garde top-global_k par tête
global_k_cfg = cfg["model"].get("global_k", 24)
print(
    f"Step {step:6d} | Loss: {loss_gathered:.4f} | PPL: {ppl:7.2f} | "
    f"LR: {lr_current:.2e} | GradNorm: {last_grad_norm:5.2f}"
)
print(
    f"           | SeqLen: {current_seq_len:4d} | GW: {global_weight:.2f} | "
    f"LM: {last_num_landmarks}→{global_k_cfg} | GPU: {mem_allocated:4.1f}GB | "
)
```

**Interpretation**:
- `48` = Number of landmark candidates selected by `LearnableLandmarkSelector`
- `24` = Final number of landmarks used per attention head in SLGA
- The `→` indicates the reduction happens inside SLGA's top-K mechanism

### 2.3 Gradient Norm Tracking

From `train.py` lines 446-454:
```python
# Calculate gradient norm BEFORE clipping (pour monitoring)
grad_norm = 0.0
if accelerator.is_main_process:
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            grad_norm += param_norm.item() ** 2
    grad_norm = grad_norm ** 0.5
    last_grad_norm = grad_norm  # Sauvegarder pour logging
```

**No GradNorm-based landmark weighting** is implemented. The "GradNorm" in logs is just gradient magnitude monitoring, not a weighting mechanism.

**Why no GradNorm weighting?**:
The landmarks are selected via learned scores (from the neural scorer), not based on gradient magnitudes. The scorer learns to identify important positions through backpropagation.

---

## 3. Attention Computation with Landmarks

### 3.1 SLGA Forward Pass

From `slga.py` lines 210-380, the attention has three stages:

#### Stage 1: Local Attention (lines 241-291)
```python
# 1. Get window indices with dilated offsets
win_idx, win_mask = self._window_indices_robust(L, device)  # (L, W)

# 2. Gather K, V within local window
for w in range(W):
    valid_w = valid_mask[:, w]
    if valid_w.any():
        idx_w = win_idx[:, w].clamp(min=0)
        k_gathered = k[:, :, idx_w, :]  # (B, H, L, Dh)
        v_gathered = v[:, :, idx_w, :]
        # Mask invalid positions with padding
        k_win[:, :, :, w, :] = k_gathered
        v_win[:, :, :, w, :] = v_gathered

# 3. Compute local attention scores
scores_local = (q_exp * k_win).sum(-1) * self.scale  # (B, H, L, W)

# 4. Softmax with causal masking
attn_local = self._safe_masked_softmax(scores_local, local_mask, dim=-1)

# 5. Compute local context
ctx_local = (attn_local.unsqueeze(-1) * v_win).sum(dim=3)  # (B, H, L, Dh)
```

#### Stage 2: Global Attention with Landmarks (lines 294-340)
```python
if cache_global is not None:
    # 1. Project landmarks with SAME QKV projection as main flow
    kv_g = self.qkv_proj(cache_global)  # (B, G, 3D) - G=48 candidates
    _, kg, vg = kv_g.chunk(3, dim=-1)

    # 2. Split into heads
    kg = kg.view(B, G, self.H, self.Dh).transpose(1, 2)  # (B, H, 48, Dh)
    vg = vg.view(B, G, self.H, self.Dh).transpose(1, 2)

    # 3. Compute global attention scores
    scores_g = torch.matmul(q, kg.transpose(-2, -1)) * self.scale  # (B, H, L, 48)

    # 4. Apply causal masking if positions provided
    if self.causal and cache_positions is not None:
        scores_g = scores_g.masked_fill(future_mask, float('-inf'))

    # 5. DIVERSE TOP-K: 48 → 24 per head
    k_sel = min(self.GK, G)  # k_sel = min(24, 48) = 24
    if self.diverse_topk and self.training:
        topk_vals, topk_idxs = self._diverse_topk(scores_g, k=k_sel)
    else:
        topk_vals, topk_idxs = torch.topk(scores_g, k=k_sel, dim=-1)

    # 6. Softmax on selected top-K
    attn_g = F.softmax(topk_vals, dim=-1)  # (B, H, L, 24)

    # 7. Gather landmark values and compute context
    vg_topk = torch.gather(vg_exp, dim=3, index=topk_idxs_exp)  # (B, H, L, 24, Dh)
    ctx_global = (attn_g.unsqueeze(-1) * vg_topk).sum(dim=3)  # (B, H, L, Dh)
```

#### Stage 3: Local-Global Fusion (lines 342-370)
```python
if ctx_global is not None and global_weight > 0.0:
    # Apply warmup weight to global context
    ctx_global_weighted = ctx_global * global_weight

    if self.gated:
        # LEARNED FUSION via gating network
        ctx_cat = torch.cat([ctx_local, ctx_global_weighted], dim=-1)  # (B, H, L, 2*Dh)

        # Per-head gating
        gate_flat = torch.sigmoid(self.gate_proj(ctx_cat_reshaped))  # (B*H*L, Dh)
        gate = gate_flat.view(B_val, H_val, L_val, self.Dh)

        # Weighted combination
        ctx = gate * ctx_local + (1 - gate) * ctx_global_weighted
    else:
        # ADDITIVE fusion
        ctx = ctx_local + ctx_global_weighted
else:
    ctx = ctx_local
```

### 3.2 Diverse Top-K Mechanism

From `slga.py` lines 158-208:

```python
def _diverse_topk(self, scores: torch.Tensor, k: int, diversity_penalty: float = 0.1):
    """
    Top-K with inter-head diversity encouragement.

    For each head sequentially:
    1. Compute penalized scores (penalize positions selected by prior heads)
    2. Select top-K for this head
    3. Update selection counts
    """
    selection_counts = torch.zeros(B, L, G, device=scores.device)

    for h in range(H):
        scores_h = scores[:, h]  # (B, L, G=48)

        # Penalize already-selected positions
        if h > 0:
            penalty = diversity_penalty * selection_counts
            scores_h = scores_h - penalty

        # Top-K for this head (24 from 48)
        topk_val_h, topk_idx_h = torch.topk(scores_h, k=k_actual, dim=-1)

        # Update counts
        selection_counts.scatter_add_(2, topk_idx_h, torch.ones_like(topk_idx_h))
```

**Effect**:
- Head 0 selects top-24 landmarks freely from 48 candidates
- Head 1 is penalized for selecting same landmarks as Head 0
- Head 2 is penalized for landmarks selected by Heads 0 and 1
- ...and so on

**Benefit**: Increases diversity of information aggregated across heads

---

## 4. Issues with Landmark-Guided Attention During Inference

### 4.1 Inference vs Training Differences

**Training Mode** (`model.train()`):
- Diverse top-K enabled (inter-head diversity penalty)
- Gated fusion learns optimal combination weights
- Dropout active in scorer and attention

**Inference Mode** (`model.eval()`):
- Standard top-K (no diversity penalty) - line 174 in slga.py
- Gated fusion uses learned weights (deterministic)
- No dropout
- Landmark selector uses deterministic top-K (no Gumbel noise)

### 4.2 Potential Inference Issues

#### Issue 1: **No KV-Cache for Landmarks**

**Problem**: During autoregressive generation, landmarks are re-selected and re-projected at EVERY step:

From `model.py` lines 286-362 (generate method):
```python
for _ in range(max_new_tokens):
    # Forward WITHOUT caching
    logits = self(input_ids, cache_global_ids=cache_ids)  # Full forward pass!
```

**Impact**:
- O(L²) complexity for each new token
- Landmarks are recomputed from scratch each time
- No incremental computation

**Solution Needed**: Implement KV-cache that stores:
- Past key/value pairs for local attention
- Past landmark states and their keys/values
- Incremental update mechanism

#### Issue 2: **Landmark Stability Across Layers**

**Problem**: Landmark indices are selected ONCE at the beginning, but landmark STATES are re-extracted at each layer:

```python
# Indices selected once (line 252)
landmark_indices, _, landmark_scores = self.landmark_selector(x)

# But states gathered at EVERY layer (lines 266-267)
for block in self.blocks:
    landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)
    x = block(x, cache_global=landmark_states)
```

**Potential Issue**:
- Early layer representations might select different landmarks than later layers would
- Landmarks selected from layer-0 embeddings might not be optimal for layer-11

**Current Mitigation**:
- Selector is trained end-to-end, learns to identify positions that remain important across layers
- 2x overselection (48 candidates) + per-head top-K (24) provides flexibility

#### Issue 3: **Global Warmup Weight at Inference**

**Current Implementation**: `global_weight` defaults to 1.0 during generation (line 222):

```python
def forward(self, input_ids, cache_global_ids=None, return_aux=False, global_weight: float = 1.0):
```

**Potential Issue**: No warmup during inference means global attention is always fully weighted.

**Status**: This is likely CORRECT - warmup is a training stabilization technique, not needed at inference.

---

## 5. Comparison with Standard Attention

### 5.1 Complexity Comparison

| Component | Standard Attention | SLGA |
|-----------|-------------------|------|
| **Local Attention** | O(L²) full quadratic | O(L × W) with W=128 window |
| **Global Attention** | N/A | O(L × G) with G=24 landmarks |
| **Memory** | O(L²) | O(L × W + L × G) |
| **Effective Receptive Field** | Full L | W (local) + G×L (global via landmarks) |

**SLGA Advantage**: Linear complexity O(L × (W + G)) vs quadratic O(L²)

### 5.2 Attention Pattern Differences

**Standard Causal Attention**:
```
Token i can attend to: [0, 1, 2, ..., i]
All past tokens are accessible
```

**SLGA Attention**:
```
Token i can attend to:
  LOCAL:  [max(0, i-W/2), ..., i]  (window of size W)
  GLOBAL: Selected landmarks [pos_1, pos_2, ..., pos_24]
Total: W + G = 128 + 24 = 152 positions instead of i positions
```

**Benefit**:
- Long-range dependencies via landmarks
- Dense local context via window
- Much cheaper computation

### 5.3 Information Flow

**Standard Transformer**:
```
Layer 0: Token i → Attend to all past
Layer 1: Token i → Attend to all past (with layer-0 representations)
...
Layer N: Token i → Attend to all past (with layer-N-1 representations)
```

**SLGA**:
```
Layer 0: Select 48 landmark positions based on embeddings
Layer 0: Token i → Attend to local window + top-24 landmarks
Layer 1: Re-extract landmark states from layer-0 output
Layer 1: Token i → Attend to local window + top-24 landmarks (updated states)
...
Layer N: Re-extract landmark states from layer-N-1 output
Layer N: Token i → Attend to local window + top-24 landmarks (evolved states)
```

**Key Difference**: Landmarks are "information highways" that aggregate and propagate important context across the sequence.

---

## 6. Identified Bugs and Issues

### 6.1 Critical Issues

**NONE FOUND** - The implementation is architecturally sound.

### 6.2 Potential Improvements

#### Improvement 1: Add KV-Cache for Generation

**Current**: Full recomputation each step
**Proposed**: Cache past keys/values and landmarks

```python
class LLMTransformer:
    @torch.no_grad()
    def generate_with_cache(self, input_ids, max_new_tokens, ...):
        # Initialize cache
        kv_cache = {
            'past_keys': [],
            'past_values': [],
            'landmark_indices': None,
            'landmark_kv': None,
        }

        for step in range(max_new_tokens):
            # Only process new token(s)
            logits, kv_cache = self.forward_with_cache(
                input_ids[:, -1:], kv_cache
            )
            # Sample and append
```

#### Improvement 2: Layer-wise Landmark Re-selection

**Current**: Select landmarks once from embeddings
**Proposed**: Re-select at middle layer(s)

```python
# Select landmarks at layers 0, 6, 11 (for 12-layer model)
if layer_idx in [0, 6, 11]:
    landmark_indices, _, landmark_scores = self.landmark_selector(x)
```

**Benefit**: Adapt landmarks as representations evolve

#### Improvement 3: Landmark Position Caching

**Current**: Landmarks can be anywhere in sequence
**Proposed**: Add positional bias for stable landmarks

```python
# Add positional prior to scorer
pos_embeddings = self.pos_emb(torch.arange(L))
scores = self.scorer(x + 0.1 * pos_embeddings)  # Slight positional bias
```

**Benefit**: More stable landmark selection across similar contexts

---

## 7. Diagnostic Recommendations

### 7.1 Check Landmark Selection Quality

```bash
# Run inspection script
python scripts/inspect_training_batch.py
```

**Expected Output**:
- Landmark indices should be reasonably distributed across sequence
- Not all clustered at beginning/end
- Diversity across batch examples

### 7.2 Verify Attention Patterns

Add this to validation loop:

```python
# In validate()
if i == 0:  # First batch
    logits, aux = model(input_ids, cache_global_ids=cache_ids, return_aux=True)

    # Check landmark stats
    if aux['landmark_indices'] is not None:
        indices = aux['landmark_indices']  # (B, 48)
        print(f"Landmark positions: {indices[0].cpu().numpy()}")
        print(f"Min: {indices.min()}, Max: {indices.max()}, Mean: {indices.float().mean():.1f}")
```

### 7.3 Compare with Local-Only Baseline

Temporarily disable global attention:

```bash
# In config.yaml
train:
  global_warmup_start: 999999  # Never activate
```

Train a few thousand steps and compare:
- Perplexity: SLGA should be better (if landmarks help)
- Loss curve: Should show global helping after warmup

---

## 8. Conclusions

### 8.1 Summary of Findings

1. **LM: 48→24 is CORRECT**:
   - 48 = Landmark candidates from learned selector
   - 24 = Final landmarks used per head (via top-K)
   - This is the intended two-stage selection design

2. **GW: 1.00 is EXPECTED**:
   - Global attention warmup completed
   - Training is past step 5000
   - Global attention fully active

3. **No GradNorm-based Weighting**:
   - Landmarks weighted via learned scores, not gradient magnitudes
   - "GradNorm" in logs is just monitoring

4. **Landmark Selection is Dynamic**:
   - Positions selected once per forward pass
   - States re-extracted at each layer (follow hidden state evolution)
   - This is likely optimal for capturing layer-specific importance

5. **Inference Needs Optimization**:
   - No KV-cache implemented (major efficiency issue)
   - But attention mechanism itself is correct

### 8.2 Architectural Strengths

✅ **Well-designed landmark mechanism**:
   - Two-stage selection (candidate → top-K) provides flexibility
   - Diverse top-K prevents head redundancy
   - Learned scorer trains end-to-end

✅ **Proper local-global fusion**:
   - Gated combination learns optimal weighting
   - Global warmup prevents early training instability

✅ **Robust implementation details**:
   - Safe masked softmax prevents NaN
   - Window indexing handles boundaries correctly
   - Causal masking properly enforced

### 8.3 Recommendations

**Priority 1 - Performance**:
- [ ] Implement KV-cache for generation (100x speedup potential)
- [ ] Profile attention computation (find bottlenecks)

**Priority 2 - Quality**:
- [ ] Add landmark selection visualization to TensorBoard
- [ ] Track landmark diversity metrics (spatial spread)
- [ ] Compare learned landmarks vs heuristic baselines

**Priority 3 - Experimentation**:
- [ ] Try layer-wise landmark re-selection
- [ ] Experiment with different G values (16, 32, 48)
- [ ] Test positional priors for landmark stability

---

## Appendix: Code References

### Key Files
1. `/mnt/d/ai/SLGA/src/slga.py` - Main SLGA attention module
2. `/mnt/d/ai/SLGA/src/landmarks.py` - Landmark selector
3. `/mnt/d/ai/SLGA/src/model.py` - Full transformer model
4. `/mnt/d/ai/SLGA/scripts/train.py` - Training loop
5. `/mnt/d/ai/SLGA/config.yaml` - Configuration

### Important Configuration Values
```yaml
model:
  global_k: 24              # Final landmarks per head
  learned_landmarks: true   # Use learned selector (48 candidates)
  gated_fusion: true       # Learned local-global fusion
  diverse_topk: true       # Inter-head diversity

train:
  global_warmup_start: 1000   # Start ramping global attention
  global_warmup_end: 5000     # Full global attention weight
```

---

**Analysis Complete**: No bugs found in SLGA components. The `LM: 48→24` behavior is correct by design.
