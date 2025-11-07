# SLGA Inference Bugs - Quick Summary

## 🔴 Critical Finding

**The SLGA attention mechanism has 6 inference-specific bugs that cause it to work during training but fail during generation.**

## Why Training Works But Generation Fails

- **Training**: Fixed-length sequences, landmarks selected from full context once per batch, soft/diverse selection
- **Generation**: Growing sequences, landmarks never updated, hard/deterministic selection, diversity disabled
- **Result**: Train/test distribution mismatch - model learns pattern A, executes pattern B during inference

## The 6 Bugs

### 🔴 P0 - CRITICAL (Must Fix)

1. **Stale Landmarks** (`model.py:250-271`)
   - Landmarks selected once at start, never updated as sequence grows
   - At L=50 tokens, landmarks from L=20 are outdated
   - Global attention focuses on irrelevant old context

2. **Missing cache_global_ids** (`model.py:294`)
   - `cache_global_ids` never computed in `generate()` method
   - If `learned_landmarks=False`: global attention completely disabled
   - If `learned_landmarks=True`: uses wrong selection strategy

3. **Eval Strategy Mismatch** (`landmarks.py:159`)
   - Training: soft/relaxed selection with gradients
   - Inference: hard/deterministic greedy selection
   - Model never learned to handle deterministic distribution

### 🟡 P1 - IMPORTANT (Major Quality Impact)

4. **Disabled Diversity** (`slga.py:173`)
   - Training: 8 heads select different landmarks
   - Inference: all 8 heads use same top-K
   - Multi-head attention degenerates to single-head

5. **No KV-Cache** (`model.py:318`)
   - Recomputes full forward pass every token
   - 100-1000x slower than standard transformers
   - Makes long generation impractical

### 🟢 P2 - NICE TO HAVE

6. **Missing Position Info** (`slga.py:314-318`)
   - `cache_positions` never passed during generation
   - Global attention missing causal masking

## Quick Fixes

### Fix #1: Recompute Landmarks (CRITICAL)
```python
def generate(self, input_ids, max_new_tokens, ...):
    for _ in range(max_new_tokens):
        # ✅ Recompute landmarks for current context
        cache_global_ids = self._compute_heuristic_landmarks(input_ids)
        logits = self(input_ids, cache_global_ids=cache_global_ids)
```

### Fix #2: Consistent Selection (CRITICAL)
```python
# Option A: Train with hard selection (simpler)
if self.training:
    _, landmark_indices = torch.topk(scores, k=k, dim=-1)
else:
    _, landmark_indices = torch.topk(scores, k=k, dim=-1)
```

### Fix #3: Enable Diversity (IMPORTANT)
```python
# Remove self.training check
if not self.diverse_topk:  # Changed from: not self.diverse_topk or not self.training
    return torch.topk(scores, k=k, dim=-1)
```

## Testing

```python
# Test landmark updates during generation
model.eval()
prompt = torch.randint(0, 50257, (1, 10))

for step in range(20):
    logits, aux = model(prompt, return_aux=True)
    print(f"Step {step}: landmarks = {aux['landmark_indices'][0].tolist()}")
    # Landmarks should CHANGE as prompt grows
```

## Next Steps

1. ✅ Analysis complete - stored in memory
2. 🔄 Implement Fix #1 and #2 (critical for functional generation)
3. 🔄 Test with generation script
4. 🔄 Implement Fix #3 and #4 (for quality)
5. 🔄 Benchmark performance

## Files

- **Full Analysis**: `/mnt/d/ai/SLGA/docs/SLGA_INFERENCE_BUGS_ANALYSIS.md`
- **Key Code**:
  - `/mnt/d/ai/SLGA/src/model.py` (forward, generate methods)
  - `/mnt/d/ai/SLGA/src/slga.py` (_diverse_topk method)
  - `/mnt/d/ai/SLGA/src/landmarks.py` (forward method)
  - `/mnt/d/ai/SLGA/scripts/generate.py` (generation script)

---

**Memory Key**: `analysis/slga-inference-bugs/summary`
