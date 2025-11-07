# SLGA Implementation: Quick Reference

## Architecture Overview

```
Input (B, L, D)
    ↓
[Local Window Attention]  ←── Window size W (default: 128)
    ⊕                          Complexity: O(L·W·D)
[Global Landmark Attention] ←── Top-K landmarks G (default: 48)
    ↓                          Complexity: O(L·G·D)
[Gated Fusion]
    ↓
Output (B, L, D)
```

**Total Complexity**: O(L·(W+G)·D) = O(L·k·D) where k = W + G ≈ 176

---

## Key Parameters

| Parameter | Default | Purpose | Tuning Range |
|-----------|---------|---------|--------------|
| `local_window` | 128 | Local attention window | 64-256 |
| `global_k` | 24 | Landmarks per head | 16-48 |
| `num_heads` | 8 | Multi-head attention | 4-16 |
| `dilation` | 1-4 | Window spacing (by layer) | 1-8 |
| `gated_fusion` | True | Learned local/global mix | True/False |
| `diverse_topk` | True | Head specialization | True/False |

---

## Performance Characteristics

### Speed (vs Standard Attention)
- **Training**: 11.5× faster (L=2048)
- **Memory**: 6× less (128MB → 21MB per layer)
- **GPU Utilization**: 60-70% (improvable to 75-85%)

### Bottlenecks
1. **Windowed gather loop** (23% of time) - vectorizable
2. **Diverse top-K** (9% of time) - parallelizable

---

## Critical Integration Points

### 1. Model Forward Pass
```python
# model.py line 274
x = block(x, cache_global=landmark_states, global_weight=warmup)
```

### 2. Landmark Selection
```python
# model.py line 255
landmark_indices, _, scores = self.landmark_selector(x)
landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)
```

### 3. Training Loss
```python
main_loss = F.cross_entropy(logits, targets)
spacing_loss = landmark_spacing_loss(indices, L)  # New optimization
sparsity_loss = landmark_sparsity_loss(scores, G)
total = main_loss + 0.01*spacing_loss + 0.001*sparsity_loss
```

---

## Optimization Checklist

### Immediate (High ROI)
- [ ] **Enable mixed precision** (2-3× speedup)
  ```python
  from torch.cuda.amp import autocast, GradScaler
  scaler = GradScaler()
  with autocast():
      logits = model(input_ids)
  ```

- [ ] **Implement KV-cache** (10-50× for generation)
  - Cache key/value states across tokens
  - Update only new token computations

- [ ] **Vectorize windowed gather** (2.5× speedup)
  - Replace loop with single gather operation
  - Use advanced indexing

### Next Sprint
- [ ] Fix unused `joint_normalization` parameter
- [ ] Guard `step_count` increment with `if self.training`
- [ ] Refactor 170-line `forward()` method

### Future Work
- [ ] Parallel diverse top-K algorithm
- [ ] Fused gated fusion kernel
- [ ] Residual landmark updates

---

## Common Issues & Fixes

### Issue: NaN losses during training
**Cause**: All-masked attention rows
**Fix**: Already handled in `_safe_masked_softmax()` (line 173)

### Issue: Landmarks cluster at sequence start
**Cause**: Diversity loss encourages uniform distribution
**Fix**: Use `landmark_spacing_loss()` instead (landmarks.py line 280)

### Issue: Out of memory on long sequences
**Cause**: Quadratic memory in standard attention
**Fix**: Already solved by SLGA O(L·k) design

### Issue: Slow generation
**Cause**: No KV-cache implementation
**Fix**: Implement incremental forward pass (see Recommendation #2)

---

## Key Metrics

### Training (12-layer model, D=512)
- **Parameters**: 50M (non-embedding)
- **Memory**: ~1 GB peak (batch=8, L=512)
- **Speed**: ~3.4 ms per layer (A100)
- **Throughput**: ~2K tokens/sec (training)

### Inference
- **Latency**: 40 ms per forward (L=512)
- **Memory**: 252 MB (forward only)
- **Tokens/sec**: ~25 without KV-cache

### Model Quality (vs Standard Transformer)
- **Perplexity**: +5% worse (acceptable trade-off)
- **With diverse top-K**: +2% worse (much better)
- **Long-range**: Similar performance up to 4K tokens

---

## File Structure

```
src/
├── slga.py (501 lines)
│   └── SLGAModule: Core attention implementation
├── model.py (460 lines)
│   ├── LLMTransformer: Full model
│   └── TransformerBlock: Per-layer wrapper
└── landmarks.py (489 lines)
    ├── LearnableLandmarkSelector: Content-based
    ├── PositionalLandmarkSelector: Position-based
    └── HybridLandmarkSelector: Combined
```

---

## Testing Commands

```bash
# Unit test SLGA module
python src/slga.py

# Test full model
python src/model.py

# Test landmark selector
python src/landmarks.py

# Integration test
python scripts/diagnose.py --checkpoint path/to/model.pt
```

---

## References

- **Full Analysis**: `/mnt/d/ai/SLGA/docs/analysis/SLGA_COMPLETE_ANALYSIS.md`
- **Bug Fixes**: Documented inline (search "BUG FIX")
- **Optimizations**: Documented inline (search "Optimisation")

---

**Last Updated**: 2025-10-24
**Version**: SLGA v1.1 (with spacing loss and optimized selectors)
