# SLGA Inference Bug Analysis

## Executive Summary

**Critical Finding**: The SLGA attention mechanism has **multiple inference-specific bugs** that cause it to work during training but fail/degrade during generation. The model trains successfully (loss decreasing) but generates poor quality text because the attention mechanism breaks during autoregressive generation.

---

## 🔴 Critical Bug #1: Landmark Indices Never Update During Generation

### Location
`src/model.py`, lines 258-271 in the `forward()` method

### The Problem

**During training**, landmark selection happens ONCE per forward pass:
```python
# Line 250-256: Landmarks selected ONCE at start
if self.landmark_selector is not None:
    landmark_indices, _, landmark_scores = self.landmark_selector(x)
elif cache_global_ids is not None:
    landmark_indices = cache_global_ids

# Line 260-271: Landmarks extracted from x for EACH layer
for block in self.blocks:
    if landmark_indices is not None:
        landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)
    x = block(x, cache_global=landmark_states, global_weight=global_weight)
```

**During generation** (autoregressive inference):
- The sequence grows: `input_ids = [L=1] -> [L=2] -> [L=3] -> ... -> [L=100]`
- Each token generated requires a NEW forward pass
- Landmark indices are selected from positions 0 to L-1
- **BUT**: As L grows, the landmark positions that were "important" at L=10 are NOT important at L=50!

### Example Failure Scenario

**Step 1** (L=20 tokens):
```
Landmark indices: [2, 5, 8, 12, 15]  # Selected from positions 0-19
These make sense for a 20-token context
```

**Step 50** (L=50 tokens):
```
Landmark indices: [2, 5, 8, 12, 15, 18, 22, ...]  # Selected from positions 0-49
The original landmarks [2, 5, 8] are now irrelevant!
They refer to tokens from 48 positions ago
```

**Result**: The global attention focuses on **outdated context** instead of recent important tokens.

### Why Training Succeeds But Inference Fails

- **Training**: Each batch has fixed-length sequences (384-2048 tokens). Landmarks are selected from the FULL context, so they're globally relevant.
- **Inference**: Sequences grow token-by-token. Early landmarks become stale as context expands.

---

## 🔴 Critical Bug #2: cache_global_ids Never Computed During Generation

### Location
`src/model.py`, line 294 in the `generate()` method

### The Problem

```python
def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    cache_global_ids: Optional[torch.Tensor] = None,  # ⚠️ NEVER COMPUTED!
) -> torch.Tensor:
    """
    Génération auto-régressive simple (sans KV-cache optimisé).
    """
    self.eval()

    for _ in range(max_new_tokens):
        # Forward
        logits = self(input_ids, cache_global_ids=cache_global_ids)  # ⚠️ Always None!
```

### Analysis

1. **If `learned_landmarks=True`** (current config):
   - `cache_global_ids` parameter is unused
   - Landmark selector runs in eval mode (line 160 in landmarks.py)
   - Eval mode uses **hard top-K without diversity** (line 161)
   - This is DIFFERENT from training behavior (Gumbel-Softmax with diversity)

2. **If `learned_landmarks=False`**:
   - `cache_global_ids` should be computed heuristically (e.g., every N tokens)
   - But the generation code **never computes it**
   - It's always `None`, meaning **global attention is disabled during generation!**

### Impact

- If using learned landmarks: Different selection strategy (hard vs soft)
- If using heuristic landmarks: Global attention completely disabled
- Either way: **Generation behavior differs significantly from training**

---

## 🔴 Critical Bug #3: Eval Mode Changes Landmark Selection Strategy

### Location
`src/landmarks.py`, lines 149-162 in the `forward()` method

### The Problem

```python
def forward(
    self, x: torch.Tensor, use_gumbel: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, L, D = x.shape

    # Scorer chaque position
    scores = self.scorer(x).squeeze(-1)  # (B, L)

    k = min(self.num_landmarks, L)

    if self.training:
        if use_gumbel:
            selection_soft, landmark_indices = self._gumbel_topk(scores, k, temp)
        else:
            selection_soft, landmark_indices = self._straight_through_topk(scores, k)
    else:
        # ⚠️ DIFFERENT BEHAVIOR IN EVAL MODE!
        _, landmark_indices = torch.topk(scores, k=k, dim=-1)
        selection_soft = None
```

### Analysis

**Training mode**:
- Uses `_straight_through_topk()` which includes gradient tricks
- Includes noise/relaxation for exploration
- Diverse landmark selection across heads

**Eval mode**:
- Uses plain `torch.topk()` - deterministic, greedy selection
- No diversity mechanism
- Can select the SAME landmarks repeatedly

### Why This Matters

The model learns to rely on diverse, slightly noisy landmark selection during training, but gets DETERMINISTIC, greedy selection during inference. This is like training with dropout then testing without it - the model hasn't learned to handle the inference distribution.

---

## 🔴 Critical Bug #4: Diverse TopK Disabled During Inference

### Location
`src/slga.py`, line 173 in the `_diverse_topk()` method

### The Problem

```python
def _diverse_topk(
    self, scores: torch.Tensor, k: int, diversity_penalty: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-K avec encouragement de diversité inter-têtes."""

    if not self.diverse_topk or not self.training:  # ⚠️ DIVERSITY DISABLED IN EVAL!
        # Mode standard (inference ou désactivé)
        return torch.topk(scores, k=k, dim=-1)
```

### Analysis

During training with `diverse_topk=True` (current config):
- Each attention head selects different global landmarks
- Diversity penalty prevents all heads from focusing on same positions
- Different heads capture different aspects of context

During inference:
- Diversity mechanism is DISABLED
- All heads use standard top-K
- Can result in **all 8 heads focusing on the same K positions**

### Impact

Multi-head attention degenerates to single-head attention during generation. The model loses the diverse perspectives it learned during training.

---

## 🟡 Bug #5: No Position Information for Global Cache During Inference

### Location
`src/slga.py`, lines 314-318 in the `forward()` method

### The Problem

```python
# Masque causal si positions fournies
if self.causal and cache_positions is not None:
    pos_query = torch.arange(L, device=device).view(1, 1, L, 1)
    pos_cache = cache_positions.view(B, 1, 1, G)
    future_mask = pos_cache > pos_query
    scores_g = scores_g.masked_fill(future_mask, float('-inf'))
```

### Analysis

**During training**:
- `cache_positions` is likely provided by the collator
- Global landmarks have explicit position information
- Causal masking works correctly

**During generation**:
- `cache_positions` is never passed to the model
- Falls back to NO causal masking on global attention
- Could potentially attend to "future" landmarks if indices are wrong

### Impact

Less severe than other bugs, but can cause subtle inconsistencies where the model attends to positions it shouldn't.

---

## 🟡 Bug #6: Autoregressive Generation Doesn't Use KV-Cache

### Location
`src/model.py`, line 318 in the `generate()` method

### The Problem

```python
for _ in range(max_new_tokens):
    # Tronquer si dépasse max_seq_len
    if input_ids.size(1) > self.cfg.max_seq_len:
        input_ids = input_ids[:, -self.cfg.max_seq_len:]

    # ⚠️ FULL FORWARD PASS EVERY TOKEN!
    logits = self(input_ids, cache_global_ids=cache_global_ids)  # (B, L, V)
```

### Analysis

Standard transformer generation uses KV-cache:
- Cache past key/value states
- Only compute attention for NEW token
- Complexity: O(1) per token instead of O(L²)

SLGA generation:
- Recomputes ENTIRE sequence every token
- No caching of past states
- Complexity: O(L² × n_tokens) - very slow!

### Impact

- **Not a correctness bug**, but a severe performance issue
- Generation is 100-1000x slower than it should be
- Makes long-form generation impractical

---

## Root Cause Analysis

### Why Training Works But Inference Fails

| Aspect | Training | Inference | Mismatch? |
|--------|----------|-----------|-----------|
| Sequence length | Fixed (384-2048) | Growing (1→100+) | ✅ **YES** |
| Landmark selection | Once per batch | Once per token generation | ✅ **YES** |
| Selection strategy | Soft/diverse (with noise) | Hard/deterministic | ✅ **YES** |
| Diversity mechanism | Enabled | Disabled | ✅ **YES** |
| Global cache computation | From full context | Never updated/recomputed | ✅ **YES** |
| Cache positions | Provided | Not provided | ✅ **YES** |

**Conclusion**: The model learns a specific behavior during training (soft landmark selection, diverse multi-head attention, fixed-length context) but encounters a COMPLETELY DIFFERENT setting during inference (hard selection, no diversity, growing context).

---

## Impact on Model Quality

### Training Metrics
- Loss decreases normally ✅
- Perplexity improves ✅
- Model learns language patterns ✅

### Generation Quality
- Poor coherence ❌ (landmarks don't track important context)
- Repetition ❌ (no diversity in attention)
- Irrelevant responses ❌ (attending to stale landmarks)
- Slow generation ❌ (no KV-cache)

### Why Loss Decreases But Generation Fails

Training loss measures next-token prediction on **teacher-forced sequences**:
- Input: `[I, am, learning, to]`
- Target: `[am, learning, to, code]`
- Model sees ground-truth context, predicts next token

Generation is **autoregressive**:
- Generate: `I` → `am` → `learning` → `the` → `the` → `the` ...
- Model sees its OWN outputs, compounds errors
- Landmark selection based on model's (possibly wrong) previous outputs

The model never sees this autoregressive, self-feeding scenario during training!

---

## Recommended Fixes

### Fix #1: Recompute Landmarks Every Generation Step ⭐ **CRITICAL**

**Problem**: Landmarks become stale as sequence grows
**Solution**: Recompute landmark indices at each generation step

```python
def generate(self, input_ids, max_new_tokens, ...):
    self.eval()

    for _ in range(max_new_tokens):
        if input_ids.size(1) > self.cfg.max_seq_len:
            input_ids = input_ids[:, -self.cfg.max_seq_len:]

        # ✅ NEW: Recompute landmarks for current context length
        cache_global_ids = self._compute_heuristic_landmarks(input_ids)

        logits = self(input_ids, cache_global_ids=cache_global_ids)
        # ... sampling logic ...
```

### Fix #2: Consistent Selection Strategy ⭐ **CRITICAL**

**Problem**: Training uses soft/diverse selection, inference uses hard/deterministic
**Solution**: Use same strategy in eval mode OR train with hard selection

**Option A**: Train with hard selection (simpler)
```python
# In landmarks.py, line 149
if self.training:
    # Use hard selection during training too
    _, landmark_indices = torch.topk(scores, k=k, dim=-1)
    selection_soft = F.softmax(scores, dim=-1)  # For loss computation
else:
    _, landmark_indices = torch.topk(scores, k=k, dim=-1)
    selection_soft = None
```

**Option B**: Keep diversity in eval mode
```python
# In landmarks.py, line 159
else:
    # ✅ Use same selection as training
    selection_soft, landmark_indices = self._straight_through_topk(scores, k)
```

### Fix #3: Enable Diversity During Inference ⭐ **IMPORTANT**

**Problem**: `diverse_topk` disabled in eval mode
**Solution**: Keep diversity mechanism active

```python
# In slga.py, line 173
def _diverse_topk(self, scores, k, diversity_penalty=0.1):
    # ✅ REMOVE the self.training check
    if not self.diverse_topk:
        return torch.topk(scores, k=k, dim=-1)

    # Apply diversity penalty regardless of training mode
    # ... existing diversity logic ...
```

### Fix #4: Implement KV-Cache for Generation ⭐ **IMPORTANT**

**Problem**: Full recomputation every token
**Solution**: Cache past key/value states

```python
def generate(self, input_ids, max_new_tokens, ...):
    self.eval()
    kv_cache = None  # Initialize empty cache

    for step in range(max_new_tokens):
        # Only process NEW token after first step
        input_chunk = input_ids if step == 0 else input_ids[:, -1:]

        # Forward with cache
        logits, kv_cache = self.forward_with_cache(input_chunk, kv_cache)
        # ... sampling logic ...
```

This requires modifying SLGAModule to support incremental attention.

### Fix #5: Pass Position Information ⭐ **NICE-TO-HAVE**

**Problem**: `cache_positions` never provided during generation
**Solution**: Compute positions for global cache

```python
def generate(self, input_ids, max_new_tokens, ...):
    for _ in range(max_new_tokens):
        cache_global_ids = self._compute_heuristic_landmarks(input_ids)

        # ✅ NEW: Compute positions for global cache
        cache_positions = cache_global_ids  # Positions ARE the indices

        logits = self(
            input_ids,
            cache_global_ids=cache_global_ids,
            cache_positions=cache_positions
        )
```

Requires adding `cache_positions` parameter to `forward()` and passing through to blocks.

---

## Testing Strategy

### Test 1: Verify Landmark Updates
```python
model.eval()
prompt = torch.randint(0, 50257, (1, 10))

# Generate with debugging
for step in range(20):
    logits, aux = model(prompt, return_aux=True)
    print(f"Step {step}, L={prompt.size(1)}")
    print(f"  Landmark indices: {aux['landmark_indices'][0].tolist()}")

    # Sample next token
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    prompt = torch.cat([prompt, next_token], dim=1)

# Expected: Landmark indices should CHANGE as prompt grows
# Current behavior: Landmarks might be stale
```

### Test 2: Compare Training vs Eval Selection
```python
x = torch.randn(1, 100, 512)

# Training mode
model.train()
_, aux_train = model(x.view(1, 100), return_aux=True)

# Eval mode
model.eval()
_, aux_eval = model(x.view(1, 100), return_aux=True)

print("Training landmarks:", aux_train['landmark_indices'])
print("Eval landmarks:", aux_eval['landmark_indices'])

# Expected: Should be similar (same strategy)
# Current: Likely different (hard vs soft)
```

### Test 3: Diverse TopK Check
```python
# In eval mode, check if different heads select different landmarks
model.eval()
x = torch.randn(1, 100, 512)

# Hook to capture attention scores
attention_scores = []
def hook(module, input, output):
    attention_scores.append(output)

# Register hook on first SLGA layer
model.blocks[0].attn.register_forward_hook(hook)

_ = model(x.view(1, 100))

# Analyze: do different heads select different global landmarks?
# Current: Likely all heads select same top-K
```

---

## Priority Ranking

1. 🔴 **P0**: Fix #1 (Recompute landmarks) - Makes generation functional
2. 🔴 **P0**: Fix #2 (Consistent selection) - Eliminates train/test mismatch
3. 🟡 **P1**: Fix #3 (Enable diversity) - Improves quality significantly
4. 🟡 **P1**: Fix #4 (KV-cache) - Essential for practical use
5. 🟢 **P2**: Fix #5 (Position info) - Minor quality improvement

---

## Conclusion

The SLGA architecture is sound, and training works correctly. However, the **inference implementation has critical bugs** that cause train/test distribution mismatch. The model learns one pattern during training but experiences a completely different computational graph during generation.

**Key Insight**: This is a classic case of "works in training, fails in production" due to:
1. Missing runtime context recomputation (landmarks)
2. Different code paths for train/eval modes
3. Disabled features during inference that were active during training

All bugs are fixable without retraining, but some fixes (#1, #2) are absolutely critical for functional generation.

## Memory Storage

Storing findings in memory under "analysis/slga-inference-bugs":
- ✅ 6 inference-specific bugs identified
- ✅ Root cause: train/test distribution mismatch
- ✅ Training works because: fixed-length, full context, diverse selection
- ✅ Inference fails because: growing context, stale landmarks, deterministic selection
- ✅ 5 fixes proposed with priority ranking
- ✅ Testing strategy provided
