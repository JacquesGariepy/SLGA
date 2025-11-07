# SLGA Quick Reference Card

## Training Logs Interpretation

```
Step   1234 | Loss: 3.4567 | PPL:  31.82 | LR: 2.00e-4 | GradNorm:  1.23
           | SeqLen:  512 | GW: 1.00 | LM: 48→24 | GPU:  8.5GB | Tok/s:  4200
```

| Metric | Value | Meaning | Status |
|--------|-------|---------|--------|
| **LM: 48→24** | 48 candidates → 24 per head | Two-stage landmark selection | ✅ Correct |
| **GW: 1.00** | 100% global weight | Global attention fully active | ✅ Expected |
| **GradNorm** | 1.23 | Gradient magnitude | ℹ️ Monitoring only |

---

## Architecture Quick Facts

### Landmark Selection
- **Stage 1**: Neural scorer → 48 candidates (configured as `global_k * 2`)
- **Stage 2**: Diverse top-K → 24 per head (configured as `global_k`)
- **Heads**: 8 heads × 24 landmarks = 192 total landmark accesses per position

### Attention
- **Local**: Window size 128, dilated by layer
- **Global**: 24 landmarks per head from 48 candidates
- **Complexity**: O(L × (W + G)) = O(L × 152) instead of O(L²)

### Fusion
- **Type**: Gated learning
- **Formula**: `output = gate × local + (1-gate) × global`
- **Per**: Position and head

---

## Key Configuration

```yaml
# config.yaml
model:
  embed_dim: 512
  num_heads: 8
  n_layers: 12
  local_window: 128          # Local attention window
  global_k: 24               # Landmarks per head (48 candidates selected)
  gated_fusion: true         # Learned fusion weights
  learned_landmarks: true    # Use LearnableLandmarkSelector
  diverse_topk: true         # Inter-head diversity

train:
  global_warmup_start: 1000  # Start global attention ramp
  global_warmup_end: 5000    # Full global attention
```

---

## Code Locations

### Main Components
| Component | File | Lines |
|-----------|------|-------|
| SLGA Attention | `src/slga.py` | 22-381 |
| Landmark Selector | `src/landmarks.py` | 17-174 |
| Transformer Model | `src/model.py` | 155-408 |
| Training Loop | `scripts/train.py` | 266-600+ |

### Key Functions
```python
# Landmark selection
LearnableLandmarkSelector.forward()  # landmarks.py:126-173
  → Returns: (indices, states, scores)
  → indices: (B, 48)  # Candidate positions

# SLGA attention
SLGAModule.forward()  # slga.py:210-380
  → Local attention: lines 241-291
  → Global attention: lines 294-340
  → Fusion: lines 342-370

# Diverse top-K
SLGAModule._diverse_topk()  # slga.py:158-208
  → Reduces 48 candidates to 24 per head
  → Applies inter-head diversity penalty
```

---

## Common Questions

### Q: Why 48→24 instead of directly selecting 24?

**A**: Flexibility and diversity
- 48 candidates give each head choice
- Diversity penalty prevents heads from selecting same landmarks
- Allows specialization across heads

### Q: Are landmarks fixed or dynamic?

**A**: Hybrid approach
- **Indices**: Selected once per forward pass (from embeddings)
- **States**: Re-extracted at each layer (from evolved representations)
- Best of both worlds: stable positions, evolving content

### Q: When is global attention used?

**A**: After warmup
- Steps 0-1000: GW=0.0 (local only)
- Steps 1000-5000: GW ramps 0.0→1.0 (gradual)
- Steps 5000+: GW=1.0 (full global)

### Q: What if I see LM: 0→24?

**A**: Either:
1. Very early training (landmarks not yet selected)
2. Using heuristic landmarks (learned_landmarks=false)
3. Validation mode with no aux output

---

## Debugging Commands

```bash
# Inspect landmark selection
python scripts/inspect_training_batch.py

# Check attention patterns
python -c "
import torch
from src.model import LLMTransformer, Config
cfg = Config()
model = LLMTransformer(cfg)
x = torch.randint(0, 50257, (2, 128))
logits, aux = model(x, return_aux=True)
print('Landmark indices:', aux['landmark_indices'][0])
print('Shape:', aux['landmark_indices'].shape)
"

# Monitor training
tensorboard --logdir out_slga/tensorboard
# Look for: landmarks/num_selected
```

---

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Perplexity | <30 after 10K steps | Track in logs |
| Tokens/sec | >4000 on RTX 3090 | See "Tok/s" |
| GPU Memory | <10GB @ seq_len=512 | See "GPU" |
| Generation Speed | TBD (need KV-cache) | Currently slow |

---

## Next Steps Checklist

- [ ] **Implement KV-cache** for generation (Priority 1)
  - File: `src/model.py`, method: `generate()`
  - Expected: 50-100x speedup

- [ ] **Add landmark visualization** (Priority 2)
  - TensorBoard: Show position distributions
  - Track diversity metrics

- [ ] **Profile attention** (Priority 2)
  - Use `torch.profiler` or `nvprof`
  - Find bottlenecks in SLGA module

- [ ] **Experiment with G values** (Priority 3)
  - Try global_k in [16, 32, 48]
  - Measure perplexity vs compute trade-off

---

## Contact / Issues

- Full analysis: `docs/SLGA_COMPONENT_ANALYSIS.md`
- Summary: `docs/ANALYSIS_SUMMARY.md`
- This reference: `docs/QUICK_REFERENCE_SLGA.md`

**Status**: Implementation verified correct. No bugs found. ✅
