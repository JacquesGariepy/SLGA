# SLGA Logits Diagnostic Report

**Date**: 2025-10-24
**Checkpoint**: `out_slga/ckpt_11000/model.pt`
**Diagnostic Script**: `/mnt/d/ai/SLGA/scripts/diagnose_logits.py`

---

## Executive Summary

✅ **Sampling code is CORRECT** - it correctly selects argmax token
❌ **Model has insufficient factual knowledge** - Paris ranks #796 out of 50,257 tokens

---

## Test Case

**Prompt**: `"The capital of France is"`
**Expected Answer**: `" Paris"`
**Model Prediction**: `" the"` (38.19% probability)

---

## Detailed Analysis

### Logits Statistics

| Metric | Value |
|--------|-------|
| Min    | -15.29 |
| Max    | 10.22 |
| Mean   | -6.87 |
| Std    | 4.24 |

### Top-20 Token Predictions

| Rank | Token | Token ID | Logit | Probability |
|------|-------|----------|-------|-------------|
| 1 | ' the' | 262 | 10.22 | 38.19% |
| 2 | ' a' | 257 | 9.10 | 12.47% |
| 3 | ' located' | 5140 | 8.09 | 4.55% |
| 4 | ' situated' | 22765 | 7.51 | 2.55% |
| 5 | ' in' | 287 | 7.51 | 2.54% |
| 6 | ' one' | 530 | 7.51 | 2.53% |
| 7 | ' at' | 379 | 7.49 | 2.48% |
| 8 | ':' | 25 | 7.14 | 1.75% |
| 9 | ' between' | 1022 | 6.59 | 1.01% |
| 10 | ',' | 11 | 6.44 | 0.87% |
| 11 | ' part' | 636 | 6.21 | 0.69% |
| 12 | ' called' | 1444 | 6.19 | 0.68% |
| 13 | ' present' | 1944 | 6.18 | 0.67% |
| 14 | ' of' | 286 | 6.14 | 0.65% |
| 15 | ' named' | 3706 | 6.10 | 0.62% |
| 16 | ' composed' | 13160 | 6.00 | 0.56% |
| 17 | ' based' | 1912 | 6.00 | 0.56% |
| 18 | ' within' | 1626 | 5.97 | 0.54% |
| 19 | ' known' | 1900 | 5.78 | 0.45% |
| 20 | ' held' | 2714 | 5.77 | 0.45% |

### Paris Token Analysis

| Property | Value |
|----------|-------|
| Token | ' Paris' |
| Token ID | 6342 |
| **Rank** | **#796** |
| Logit | 0.92 |
| Probability | 0.0035% |

**Critical Finding**: Paris ranks #796 out of 50,257 tokens, meaning the model assigns it extremely low probability (0.0035%).

---

## Root Cause Analysis

### 1. Pattern Learning vs Factual Knowledge

The model has successfully learned:
- ✅ **Grammatical patterns**: Articles ("the", "a"), prepositions ("in", "at")
- ✅ **Sentence structures**: "X is located...", "X is situated..."
- ❌ **Factual knowledge**: Specific facts like "The capital of France is Paris"

### 2. Training Data Issues

**WikiText-103 Limitations**:
- Relatively small dataset (~100M tokens)
- Limited exposure to factual statements
- "X is the Y" pattern dominates over specific facts
- May not contain enough instances of "The capital of France is Paris"

### 3. Training Duration

**Current Status**: 11,000 steps
- Model is still learning grammatical patterns
- Hasn't converged to factual knowledge yet
- Needs more training epochs

---

## Temperature Sampling Verification

| Temperature | Selected Token | Token ID |
|-------------|----------------|----------|
| 0.0 (greedy) | ' the' | 262 |
| 0.5 | ' the' | 262 |
| 1.0 | ' the' | 262 |

**Conclusion**: All temperature settings correctly select the highest probability token (" the"). The sampling code is working as intended.

---

## Recommendations

### Immediate Actions

1. **Continue Training**
   - Train for at least 50,000-100,000 more steps
   - Monitor perplexity on validation set
   - Check if Paris rank improves over time

2. **Evaluate on More Test Cases**
   - Test other factual knowledge questions
   - Check if this is a systemic issue or specific to this prompt

3. **Consider Data Augmentation**
   - Add factual knowledge datasets (e.g., SQUAD, Natural Questions)
   - Mix with WikiText for balanced learning
   - Use knowledge-intensive pretraining

### Long-term Improvements

1. **Better Training Objectives**
   - Add factual QA auxiliary task
   - Use knowledge-grounded pretraining
   - Implement curriculum learning (grammar → facts)

2. **Architecture Enhancements**
   - Consider retrieval-augmented generation (RAG)
   - Add explicit knowledge memory modules
   - Larger model capacity (more parameters)

3. **Evaluation Framework**
   - Create factual knowledge benchmark
   - Track knowledge retention metrics
   - Monitor fact vs grammar learning balance

---

## Conclusion

The diagnostic confirms that:

1. ✅ **Sampling implementation is correct** - greedy sampling correctly selects argmax
2. ✅ **Model training is progressing** - learned grammatical patterns
3. ❌ **Factual knowledge is insufficient** - Paris ranks #796
4. 🔄 **More training needed** - model is still early in training

**Next Steps**: Continue training and re-evaluate at checkpoints 20k, 30k, 50k to track Paris rank improvement.

---

## Appendix: Running the Diagnostic

To reproduce this analysis:

```bash
python scripts/diagnose_logits.py
```

The script:
1. Loads checkpoint from `out_slga/ckpt_11000/model.pt`
2. Runs forward pass on "The capital of France is"
3. Extracts raw logits and computes probabilities
4. Shows top-20 predictions
5. Checks Paris token rank
6. Verifies temperature sampling behavior
