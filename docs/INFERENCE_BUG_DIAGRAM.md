# SLGA Inference Bug Visualization

## Bug #1: Stale Landmarks During Generation

### How It Should Work (Fixed Behavior)

```
Step 0: L=10 tokens
┌─────────────────────────────────┐
│ Tokens: [0 1 2 3 4 5 6 7 8 9]  │
│ Landmarks: [2, 5, 8]            │ ← Selected from 0-9
└─────────────────────────────────┘
         ↓ Generate token 10

Step 1: L=11 tokens
┌──────────────────────────────────────┐
│ Tokens: [0 1 2 3 4 5 6 7 8 9 10]   │
│ Landmarks: [2, 6, 9]                │ ← RECOMPUTED from 0-10
└──────────────────────────────────────┘
         ↓ Generate token 11

Step 2: L=12 tokens
┌───────────────────────────────────────────┐
│ Tokens: [0 1 2 3 4 5 6 7 8 9 10 11]     │
│ Landmarks: [3, 7, 10]                    │ ← RECOMPUTED from 0-11
└───────────────────────────────────────────┘
```

**Key**: Landmarks TRACK important context as sequence grows

---

### How It Actually Works (Buggy Behavior)

```
Step 0: L=10 tokens
┌─────────────────────────────────┐
│ Tokens: [0 1 2 3 4 5 6 7 8 9]  │
│ Landmarks: [2, 5, 8]            │ ← Selected from 0-9
└─────────────────────────────────┘
         ↓ Generate token 10

Step 1: L=11 tokens
┌──────────────────────────────────────┐
│ Tokens: [0 1 2 3 4 5 6 7 8 9 10]   │
│ Landmarks: [2, 5, 8]                │ ← STILL using old landmarks!
└──────────────────────────────────────┘
         ↓ Generate token 11

Step 2: L=12 tokens
┌───────────────────────────────────────────┐
│ Tokens: [0 1 2 3 4 5 6 7 8 9 10 11]     │
│ Landmarks: [2, 5, 8]                     │ ← STILL old! (10 steps behind)
└───────────────────────────────────────────┘

         ↓ ... continues ...

Step 40: L=50 tokens
┌─────────────────────────────────────────────────────────────────────┐
│ Tokens: [0 1 2 ... 47 48 49]                                       │
│ Landmarks: [2, 5, 8]  ← STALE! Attending 40-48 positions behind!  │
└─────────────────────────────────────────────────────────────────────┘
```

**Problem**: Global attention focuses on IRRELEVANT old context

---

## Bug #3: Training vs Inference Selection Strategy Mismatch

### Training Mode

```python
# landmarks.py:156-158
if self.training:
    # Soft selection with gradient tricks
    selection_soft, landmark_indices = self._straight_through_topk(scores, k)

    # Creates smooth gradients:
    # ┌───────────────────┐
    # │  Position Scores  │
    # ├───────────────────┤
    # │  0: 0.95 ✓ TOP    │ ← Selected
    # │  1: 0.92 ✓ TOP    │ ← Selected
    # │  2: 0.88 ✓ TOP    │ ← Selected
    # │  3: 0.45 (near)   │ ← Gets gradient signal
    # │  4: 0.42 (near)   │ ← Gets gradient signal
    # │  5: 0.01          │
    # └───────────────────┘
    #
    # Model learns: "Top 3 AND near-misses are important"
```

### Inference Mode

```python
# landmarks.py:160-162
else:
    # Hard selection, no gradients
    _, landmark_indices = torch.topk(scores, k=k, dim=-1)

    # Deterministic greedy:
    # ┌───────────────────┐
    # │  Position Scores  │
    # ├───────────────────┤
    # │  0: 0.95 ✓ TOP    │ ← Selected
    # │  1: 0.92 ✓ TOP    │ ← Selected
    # │  2: 0.88 ✓ TOP    │ ← Selected
    # │  3: 0.45 ✗ OUT    │ ← IGNORED (no gradient info during training)
    # │  4: 0.42 ✗ OUT    │ ← IGNORED
    # │  5: 0.01 ✗ OUT    │
    # └───────────────────┘
    #
    # Model encounters: "Only strict top-3, nothing else"
```

**Result**: Model never learned to handle this deterministic regime!

---

## Bug #4: Multi-Head Diversity Loss

### Training Mode (with diversity)

```
Head 0: Selects landmarks [2, 8, 15]    ← Different!
Head 1: Selects landmarks [5, 12, 20]   ← Different!
Head 2: Selects landmarks [1, 10, 18]   ← Different!
...
Head 7: Selects landmarks [4, 14, 22]   ← Different!

┌─────────────────────────────────────────┐
│  Context Coverage (8 heads)            │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓              │
│  Most positions attended by some head  │
└─────────────────────────────────────────┘

Result: Rich, diverse context representation
```

### Inference Mode (no diversity)

```
Head 0: Selects landmarks [2, 5, 8]    ← SAME!
Head 1: Selects landmarks [2, 5, 8]    ← SAME!
Head 2: Selects landmarks [2, 5, 8]    ← SAME!
...
Head 7: Selects landmarks [2, 5, 8]    ← SAME!

┌─────────────────────────────────────────┐
│  Context Coverage (8 heads)            │
│  ▓▓▓   ▓   ▓▓                          │
│  All heads focus on same 3 positions!  │
└─────────────────────────────────────────┘

Result: Degenerate single-head attention
```

---

## Architectural Comparison: Training vs Inference

### Training Pipeline

```
Input: [Fixed-length batch, L=384-2048]
  ↓
Token Embeddings
  ↓
Landmark Selector (ONCE per batch)
  ├─ Soft selection (straight-through estimator)
  ├─ With noise/relaxation
  └─ Landmarks: [positions from full context]
  ↓
For each transformer layer:
  ├─ Local Attention (window=128)
  │   └─ Causal masking ✓
  └─ Global Attention
      ├─ Diverse top-K (different per head) ✓
      ├─ Attend to CURRENT landmark states ✓
      └─ Causal masking via positions ✓
  ↓
Output: Next-token logits
  ↓
Loss: Cross-entropy (teacher-forced)
```

### Inference Pipeline (BUGGY!)

```
Input: [Growing sequence, L=1→100]
  ↓
Token Embeddings
  ↓
Landmark Selector (ONCE per forward pass)
  ├─ Hard selection (deterministic) ⚠️ MISMATCH!
  ├─ No noise/relaxation ⚠️ DIFFERENT!
  └─ Landmarks: [STALE positions] ⚠️ NEVER UPDATED!
  ↓
For each transformer layer:
  ├─ Local Attention (window=128)
  │   └─ Causal masking ✓
  └─ Global Attention
      ├─ No diversity (all heads same) ⚠️ DISABLED!
      ├─ Attend to OUTDATED landmark states ⚠️ STALE!
      └─ Causal masking: missing positions ⚠️ INCOMPLETE!
  ↓
Output: Next-token logits
  ↓
Sample: Autoregressive (compounds errors)
```

---

## Impact Visualization

### Training Loss Curve

```
Training Loss (Teacher-Forced)
6.0 ┤                              ✓ Works fine!
5.0 ┤╮
4.0 ┤ ╲                            Model learns patterns
3.0 ┤  ╲___                        from ground-truth context
2.0 ┤      ────___
1.0 ┤             ────____         Loss decreases normally
0.0 └────────────────────────────
    0      20k     40k     60k   steps
```

### Generation Quality

```
Generation Perplexity (Autoregressive)
∞   ┤     ⚠️ Gets WORSE!
100 ┤    /
 50 ┤   /                          Landmarks become stale
 20 ┤  /                           Attention focuses on
 10 ┤ /                            irrelevant old context
  5 ┤/____
  1 └────────────────────────────
    0    20    40    60   tokens generated
```

**Why**: Model sees its OWN wrong outputs, with wrong attention patterns!

---

## The Fix: Aligned Training and Inference

```
┌────────────────────────────────────────────────────────────┐
│                    TRAINING REGIME                         │
│  ✓ Recompute landmarks every forward pass                  │
│  ✓ Hard deterministic selection                            │
│  ✓ Keep diversity mechanism                                │
│  ✓ Full position information                               │
├────────────────────────────────────────────────────────────┤
│                   INFERENCE REGIME                         │
│  ✓ Recompute landmarks every generation step (same!)       │
│  ✓ Hard deterministic selection (same!)                    │
│  ✓ Keep diversity mechanism (same!)                        │
│  ✓ Full position information (same!)                       │
└────────────────────────────────────────────────────────────┘
                    ↓
            CONSISTENT BEHAVIOR
                    ↓
           WORKING GENERATION!
```

---

## Summary Table

| Component | Training | Inference (Buggy) | Impact |
|-----------|----------|-------------------|---------|
| Landmark selection frequency | Once per batch | Once per forward | ⚠️ Stale |
| Selection strategy | Soft/relaxed | Hard/greedy | ⚠️ Mismatch |
| Diversity mechanism | Enabled | Disabled | ⚠️ Degenerate |
| Context length | Fixed | Growing | ⚠️ Never adapted |
| Position info | Provided | Missing | ⚠️ Incomplete |
| KV caching | Not needed | Not implemented | ⚠️ 100x slower |

**All fixable without retraining!**
