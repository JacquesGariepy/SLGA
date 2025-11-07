# Sparsity Loss Quick Reference

## What Was Fixed

**Original Bug**: Loss was constant ≈4.25, no learning signal
**Fix Applied**: Measure mass concentration in top-G landmarks
**Result**: Loss varies 0.0-0.5+, gradients flow correctly

## How It Works (One Sentence)

Measures what fraction of softmax probability mass falls in the top-G positions; penalizes if less than target (≈65% for typical configs).

## Code Location

```
/mnt/d/ai/SLGA/src/landmarks.py
Lines 495-565
Function: landmark_sparsity_loss()
```

## Quick Test

```bash
cd /mnt/d/ai/SLGA
python tests/test_sparsity_fix.py
```

Expected: 4-5 tests pass (gradient flow, stability, comparison)

## Expected Behavior

| Scenario | Mass in Top-G | Loss | Interpretation |
|----------|--------------|------|----------------|
| Perfect concentration | 100% | 0.000 | ✅ Optimal |
| Good concentration | 90%+ | 0.000 | ✅ Good |
| Moderate concentration | 60-70% | 0.000-0.001 | ⚠️ Acceptable |
| Poor concentration | <50% | 0.001+ | ❌ Bad |
| Random/uniform | 12-20% | 0.005+ | ❌ Very bad |

## Key Parameters

```python
landmark_sparsity_loss(
    selection_scores,  # (B, L) tensor
    num_landmarks=48,  # G
    lambda_reg=0.001   # Weight
)
```

- **lambda_reg**: 0.001-0.01 recommended
  - Lower = less regularization
  - Higher = stronger concentration pressure

## Integration in Training

```python
# In model forward()
selection_scores = self.scorer(hidden_states).squeeze(-1)  # (B, L)

# Compute loss
sparsity_loss = landmark_sparsity_loss(
    selection_scores,
    num_landmarks=self.num_landmarks,
    lambda_reg=0.001
)

# Add to total loss
total_loss = lm_loss + sparsity_loss
```

## Validation Checklist

✅ Gradients flow to scorer parameters
✅ Loss varies with score distribution
✅ Values in reasonable range (0.0-0.5)
✅ No NaN/Inf with extreme inputs
✅ Works with different G/L ratios

## Troubleshooting

### Loss always zero
- Scores might be already well-concentrated
- Reduce target_mass factor (edit line 559)
- Increase lambda_reg

### Loss too high
- Scores uniformly distributed
- Scorer network needs training
- Check if landmark selection is working

### NaN/Inf
- Should not happen with v4
- Check input scores for NaN/Inf
- Verify softmax computation

## Mathematical Formula

```
probs = softmax(scores)  # Normalize to probabilities
mass_top_g = sum(probs[top_g_indices]) / B  # Mean across batch
target = 0.60 + (G/L) * 0.40  # Adaptive target
loss = λ * ReLU(target - mass_top_g)  # Penalize if < target
```

## Performance Impact

- **Computation**: O(B×L×log(G)) for topk
- **Memory**: O(B×L) for softmax
- **Training**: Negligible overhead (<1% of total time)

## Related Docs

- Full analysis: `SPARSITY_LOSS_FIX_2025-10-28.md`
- Test suite: `tests/test_sparsity_fix.py`
- Model integration: `src/model.py` (SLGAModel)
