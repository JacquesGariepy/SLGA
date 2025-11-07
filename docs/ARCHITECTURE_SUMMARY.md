# SLGA Architecture Summary - Quick Reference

**Full Analysis:** See [ARCHITECTURE_SYNTHESIS.md](/mnt/d/ai/SLGA/docs/ARCHITECTURE_SYNTHESIS.md)

---

## System at a Glance

**What:** Custom Transformer LLM with sparse local-global attention for efficient long-sequence processing
**Hardware:** RTX 3090 24GB (consumer GPU)
**Parameters:** 124M (comparable to GPT-2 large)
**Innovation:** O(L·W + L·G) attention complexity vs O(L²) standard transformers

---

## Architecture Stack

```
Input Tokens (B, L)
    ↓
Embeddings (Token + Position)
    ↓
Learnable Landmark Selector → (B, G) landmark positions
    ↓
N × Transformer Blocks:
    - SLGA Attention (Local W=128 + Global G=24 + Gated Fusion)
    - Feed-Forward Network (4x expansion)
    ↓
LM Head → Logits (B, L, V)
```

---

## Key Differences from HuggingFace Transformers

| Aspect | HuggingFace | SLGA |
|--------|-------------|------|
| Attention | O(L²) dense | O(L·W + L·G) sparse |
| Landmarks | None | Learnable via Gumbel-Softmax |
| Fusion | Single stream | Gated local+global |
| Memory | Cloud-optimized | RTX 3090 optimized |
| Windowing | None | Dilated per layer |

**Why Custom?** HF doesn't support windowed+landmark attention patterns natively.

---

## Top 5 Strengths

1. ✅ **Memory Efficient:** Trains 2048-token sequences on 24GB VRAM
2. ✅ **Differentiable Landmarks:** End-to-end learnable landmark selection
3. ✅ **Training Stability:** Dual warmup (seq_len + global) prevents collapse
4. ✅ **Novel Losses:** Spacing loss for uniform landmark distribution
5. ✅ **Comprehensive Logging:** Real-time metrics + TensorBoard

---

## Top 5 Weaknesses

1. 🔴 **No KV-Cache:** Inference is 10-20x slower than possible
2. 🔴 **Limited Tests:** No comprehensive unit test suite
3. 🔴 **Memory Leak:** Mask cache grows unbounded
4. 🟡 **Inefficient Loops:** Landmark gathering should be vectorized (20% speedup)
5. 🟡 **No Multi-GPU:** Can't scale beyond single device

---

## v2.0 Top Priorities

### Week 1 (Stability)
1. Fix memory leak in mask cache
2. Add checkpoint versioning
3. Vectorize landmark gathering
4. Start pytest suite

**Impact:** +20% speed, prevent OOM crashes

### Month 1 (Production-Ready)
5. Implement KV-cache for inference
6. Integrate Flash Attention
7. Full unit test coverage
8. CI/CD pipeline

**Impact:** 10x faster inference, safe refactoring

### Quarter 1 (Scale)
9. Multi-GPU support
10. Dynamic batching
11. Trainer class refactor
12. Documentation

**Impact:** Scale to 1B+ params, better maintainability

---

## Performance Metrics

**Current (RTX 3090):**
- Training: ~4000 tokens/sec
- Inference: ~200 tokens/sec (no KV-cache)
- Memory: 18GB / 24GB (seq_len=2048, batch=4)

**v2.0 Target:**
- Training: ~6000 tokens/sec (Flash Attention)
- Inference: ~2000 tokens/sec (with KV-cache)
- Memory: 12GB / 24GB (optimizations)

---

## File Locations

**Core Implementation:**
- `/mnt/d/ai/SLGA/src/model.py` - Main LLMTransformer
- `/mnt/d/ai/SLGA/src/slga.py` - SLGA attention module
- `/mnt/d/ai/SLGA/src/landmarks.py` - Landmark selector + losses
- `/mnt/d/ai/SLGA/src/data.py` - Data collators

**Training:**
- `/mnt/d/ai/SLGA/scripts/train.py` - Main training loop
- `/mnt/d/ai/SLGA/scripts/eval_perplexity.py` - Evaluation
- `/mnt/d/ai/SLGA/scripts/generate.py` - Text generation

**Config:**
- `/mnt/d/ai/SLGA/config.yaml` - Hyperparameters

---

## Quick Command Reference

```bash
# Train model
python scripts/train.py --config config.yaml

# Evaluate perplexity
python scripts/eval_perplexity.py --checkpoint out_slga/ckpt_50000

# Generate text
python scripts/generate.py --checkpoint out_slga/ckpt_50000 --prompt "Hello"

# Run tests
pytest tests/

# Monitor training
tensorboard --logdir out_slga/tensorboard
```

---

## Code Quality Checklist for v2.0

- [ ] Unit tests (>80% coverage)
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Type hints (mypy strict)
- [ ] Docstrings (Google style)
- [ ] Pre-commit hooks (black, isort)
- [ ] Checkpoint versioning
- [ ] Memory leak fixes
- [ ] Vectorized operations (no loops)
- [ ] Multi-GPU support
- [ ] Production logging

---

## Research Opportunities

**Low-Hanging Fruit:**
1. KV-cache implementation (3 days)
2. Flash Attention integration (4 days)
3. Vectorize landmark gathering (1 day)

**Novel Contributions:**
1. Hierarchical landmarks (multi-scale)
2. Adaptive window sizes (content-based)
3. Per-layer landmark selection
4. Attention-based fusion (vs gated)

---

## Contact & Resources

**Documentation:**
- Full Analysis: `/mnt/d/ai/SLGA/docs/ARCHITECTURE_SYNTHESIS.md`
- README: `/mnt/d/ai/SLGA/README.md`

**Key Papers:**
- Longformer: https://arxiv.org/abs/2004.05150
- BigBird: https://arxiv.org/abs/2007.14062
- Gumbel-Softmax: https://arxiv.org/abs/1611.01144

**Dependencies:**
- PyTorch 2.0+
- HuggingFace Transformers 4.30+
- Accelerate 0.20+

---

**Last Updated:** 2025-10-24
**Version:** 1.0
**Codebase:** git commit e02fde0
