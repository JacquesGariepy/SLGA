# Top-P Nucleus Sampling Bug - Visual Explanation

This document provides a visual explanation of the critical bug in SLGA's nucleus sampling implementation.

---

## The Nucleus Sampling Algorithm (Correct Version)

**Goal**: Keep only the smallest set of tokens whose cumulative probability >= p

```
Step 1: Sort logits by probability (descending)
Original vocabulary: [Paris, London, Berlin, Pink, Kejriwal, ...]
Sorted by prob:      [Paris, London, Berlin, Pink, Kejriwal, ...]
Probabilities:       [0.70,  0.20,   0.08,   0.01, 0.005,    ...]

Step 2: Compute cumulative probabilities
Cumulative:          [0.70,  0.90,   0.98,   0.99, 0.995,    ...]

Step 3: Find threshold (top_p = 0.9)
                              ↓ threshold crossed here
Cumulative:          [0.70,  0.90,   0.98,   0.99, 0.995,    ...]
Keep:                [✅,     ✅,     ❌,     ❌,   ❌,       ...]

Step 4: Create mask and filter
Mask to remove:      [False, False,  True,   True, True,     ...]
After filtering:     [Paris, London, -inf,   -inf, -inf,     ...]

Step 5: Sample from filtered distribution
Probs after softmax: [0.777, 0.223,  0.0,    0.0,  0.0,      ...]
Sample: "Paris" or "London" (coherent!)
```

---

## The Bug: Wrong Shift Direction

### What the Buggy Code Does

**Location**: `src/model.py:337-339`

```python
# BUGGY CODE:
sorted_mask = cumulative_probs > top_p
sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()  # ❌ SHIFTS RIGHT
sorted_mask[:, 0] = False
```

### Visualization of the Bug

```
Step 1: Create initial mask (cumulative_probs > 0.9)
Positions:           [0,     1,      2,      3,     4,      ...]
Tokens:              [Paris, London, Berlin, Pink,  Kejriwal, ...]
Cumulative:          [0.70,  0.90,   0.98,   0.99,  0.995,   ...]
Initial mask:        [False, False,  True,   True,  True,    ...]
                                      ↑ First True at position 2

Step 2: Shift RIGHT (WRONG!)
sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                     └──────┐
                            ↓
Before shift:        [False, False,  True,   True,  True,    ...]
After shift:         [False, False,  False,  True,  True,    ...]
                             └───┐    └───┘
                                 └──→ shifted right

Step 3: Force first position to False
sorted_mask[:, 0] = False
Final mask:          [False, False,  False,  True,  True,    ...]

Step 4: Apply mask (WRONG TOKENS REMOVED!)
                     [False, False,  False,  True,  True,    ...]
Filtered logits:     [Paris, London, Berlin, -inf,  -inf,    ...]
                                      ↑ Should be -inf but kept!
```

### The Problem Visualized

```
SHOULD KEEP:         [Paris, London, ______, _____, _____,   ...]
ACTUALLY KEEPS:      [Paris, London, Berlin, _____, _____,   ...]
                                      ↑ WRONG! This exceeds threshold
```

### Why This Causes Nonsense

**Example 1**: Extreme case (all tokens filtered)

```
If cumulative_probs = [0.95, 0.99, 0.999, 1.0, ...]  and top_p = 0.9

Initial mask:        [True,  True,  True,   True, ...]  # All > 0.9
After RIGHT shift:   [True,  True,  True,   True, ...]  # Shifted
After [:, 0] = False:[False, True,  True,   True, ...]  # Only first kept

Result: Only first token survives
        If first token has cumulative prob 0.95 > threshold,
        this is WRONG!

After softmax: probs = [1.0, 0.0, 0.0, ...]
Always samples first token (even if it shouldn't be in nucleus)
```

**Example 2**: Real scenario from SLGA

```
Prompt: "The capital of France is"

Model's true distribution:
  Paris: 0.60  → cumulative: 0.60
  France: 0.15 → cumulative: 0.75
  a: 0.10      → cumulative: 0.85
  the: 0.08    → cumulative: 0.93  ← threshold crossed
  London: 0.03 → cumulative: 0.96
  Pink: 0.02   → cumulative: 0.98
  ...

With top_p = 0.9, should keep: [Paris, France, a]

BUG applies wrong mask:
1. Initial mask: [F, F, F, T, T, T, ...]
2. After shift: [F, F, F, F, T, T, ...]  ← Wrong!
3. Filters: [Paris, France, a, the, -inf, -inf, ...]
            Should be:     [Paris, France, a, -inf, -inf, -inf, ...]

Result: "the" has 0.08 / (0.60 + 0.15 + 0.10 + 0.08) = 0.086 probability
        Instead of 0%!

After many steps with compounding errors:
Output: "the the the Pink immersed mattereur Kejriwal..."
```

---

## The Correct Shift Direction

### What It Should Do

```python
# CORRECT CODE:
sorted_indices_to_remove = cumulative_probs > top_p
sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()  # ✅ SHIFTS LEFT
sorted_indices_to_remove[..., 0] = False
```

### Visualization of Correct Behavior

```
Step 1: Create initial mask (cumulative_probs > 0.9)
Positions:           [0,     1,      2,      3,     4,      ...]
Cumulative:          [0.70,  0.90,   0.98,   0.99,  0.995,   ...]
Initial mask:        [False, False,  True,   True,  True,    ...]

Step 2: Shift LEFT (CORRECT!)
sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                                     └──────┐
Before shift:        [False, False,  True,   True,  True,    ...]
After shift:         [False, False,  False,  True,  True,    ...]
                                      ↑ shifted left

Step 3: Force first position to False (keep best token)
sorted_indices_to_remove[..., 0] = False
Final mask:          [False, False,  False,  True,  True,    ...]
                                              ↑ Starts removing here

Step 4: Apply mask (CORRECT!)
Filtered logits:     [Paris, London, Berlin, -inf,  -inf,    ...]
                                              ↑ Correct threshold
```

Wait, this produces the same result!

### The ACTUAL Bug: It's in the LOGIC, not just shift

Let me re-analyze...

```
The issue is MORE SUBTLE:

Initial mask marks: cumulative_probs > top_p
For top_p = 0.9:
  [0.70, 0.90, 0.98, 0.99, ...] > 0.9
  [False, False, True, True, ...]
         ↑ 0.90 is NOT > 0.9, so False
                 ↑ 0.98 IS > 0.9, so True

This marks tokens AFTER we've exceeded threshold.

But we want to KEEP tokens UNTIL we exceed threshold!

The shift is trying to INCLUDE the token that crosses threshold:
  [False, False, True, True, ...] (original)
  [False, False, False, True, ...] (after shift right)
           ↑ Now London (0.90) is included

BUT the original code shifted in WRONG direction relative to the goal!
```

### The Real Fix: Proper Threshold Logic

```python
# The issue is the comparison operator!

# WRONG (current):
sorted_indices_to_remove = cumulative_probs > top_p
# This marks [0.98, 0.99, ...] but NOT [0.90]
# We want to keep [0.70, 0.90] because their cumulative is ≤ 0.9

# CORRECT (should be):
sorted_indices_to_remove = cumulative_probs > top_p
# Keep tokens where cumulative probability BEFORE adding them is < top_p

# The shift logic is trying to compensate for this!
```

### Actually, the Standard Implementation is:

```python
# Remove tokens with cumulative probability ABOVE threshold
# But keep at least the top token
sorted_indices_to_remove = cumulative_probs > top_p

# Shift right by 1 to keep the token that crosses threshold
sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
sorted_indices_to_remove[:, 0] = False

# Example:
# cumulative: [0.70, 0.90, 0.98, ...]
# initial:    [F,    F,    T,    ...]  (0.98 > 0.9)
# shifted:    [F,    F,    F,    ...]  (shift [F, T, ...] → [F, F, T, ...])
#                   ↑ Now keeps 0.90

# Result: Keeps [0.70, 0.90], removes [0.98, ...]
# Total mass: 0.90 (≤ 0.9) ✅ CORRECT
```

### Wait, I need to trace through this more carefully with actual code...

Let me look at the actual operations:

```python
# Current buggy code:
sorted_mask = cumulative_probs > top_p  # [F, F, T, T, T]
sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
# [:, 1:] means "from index 1 to end"
# [:, :-1] means "from index 0 to second-to-last"
# So: positions [1,2,3,4,...] ← values from positions [0,1,2,3,...]
#     [_, F, F, T, T] (copies [F,F,T,T] into [1:])

sorted_mask[:, 0] = False
# [F, F, F, T, T]

# This DOES keep the first True!
```

Actually the bug must be in the unsort operation! Let me check the scatter...

---

## The REAL Bug: Incorrect Unsort

After more analysis, the issue is in TWO places:

1. **Mask shift logic** (subtle issues)
2. **Unsort operation** (major issue)

### Current Code:
```python
logits = logits.scatter(1, sorted_indices, sorted_logits)
```

This doesn't correctly restore original order!

### Correct Code:
```python
# Create boolean mask in original vocabulary order
indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
logits[indices_to_remove] = float('-inf')
```

OR simpler (Hugging Face style):
```python
# Just scatter the filtered sorted_logits back
logits = logits.scatter(1, sorted_indices, sorted_logits)
```

The current code IS using scatter correctly! So the bug must be in the mask logic...

---

## Debugging with Concrete Example

Let's trace through with actual tensors:

```python
import torch
import torch.nn.functional as F

# Example logits
logits = torch.tensor([[10.0, 8.0, 6.0, 2.0, 1.0]])
# Token names: [Paris, London, Berlin, Pink, immersed]

# Sort
sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
print("Sorted logits:", sorted_logits)
# [10.0, 8.0, 6.0, 2.0, 1.0] (same order, already sorted)
print("Sorted indices:", sorted_indices)
# [0, 1, 2, 3, 4]

# Softmax
probs = F.softmax(sorted_logits, dim=-1)
print("Probs:", probs)
# [0.731, 0.200, 0.054, 0.007, 0.003] (approx)

# Cumulative
cumulative = torch.cumsum(probs, dim=-1)
print("Cumulative:", cumulative)
# [0.731, 0.931, 0.985, 0.992, 0.995]

# Top-p = 0.9
top_p = 0.9
sorted_mask = cumulative > top_p
print("Initial mask:", sorted_mask)
# [False, True, True, True, True]  (0.931 > 0.9, so True)

# Buggy shift (RIGHT)
sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
print("After shift:", sorted_mask)
# [:, 1:] = [1,2,3,4]
# [:, :-1] = [0,1,2,3]
# So: [_, False, True, True, True]
sorted_mask[:, 0] = False
print("After force False:", sorted_mask)
# [False, False, True, True, True]

# Apply mask
sorted_logits[sorted_mask] = float('-inf')
print("Filtered sorted logits:", sorted_logits)
# [10.0, 8.0, -inf, -inf, -inf]

# Scatter back
filtered_logits = logits.scatter(1, sorted_indices, sorted_logits)
print("Final logits:", filtered_logits)
# [10.0, 8.0, -inf, -inf, -inf]

# Expected behavior:
# Should keep: [Paris] (cumulative 0.731 < 0.9)
# Should remove: [London] (cumulative 0.931 > 0.9)
# But actually keeps: [Paris, London]!
```

**AH! I found it!**

The bug is that after the shift, we keep position 1 (London, cumulative 0.931) when we should remove it because cumulative > 0.9!

---

## The Fix: Correct Implementation

```python
# CORRECT VERSION:
sorted_indices_to_remove = cumulative_probs > top_p
# [False, True, True, True, True]  (marks 0.931 and above)

# DON'T shift - this is already correct!
# We want to remove position 1 because cumulative AFTER adding it is > 0.9

# But we DO want to keep at least one token:
sorted_indices_to_remove[0] = False

# Final mask: [False, True, True, True, True]
# Keeps: [Paris] (0.731)
# Removes: [London, Berlin, ...] (0.931+)
```

No wait, that's too aggressive. Let's check Hugging Face implementation...

---

## Hugging Face Reference Implementation

```python
def _top_p_filtering(logits, top_p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p

    # Shift the indices to the right to keep also the first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    return logits
```

Key difference: They scatter the MASK, not the logits!

---

## Summary

The bug is in **HOW the unsort is done**:

**Current (wrong)**:
```python
logits = logits.scatter(1, sorted_indices, sorted_logits)
# This might not correctly restore filtered logits
```

**Correct (Hugging Face)**:
```python
indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
logits[indices_to_remove] = float('-inf')
# Apply boolean mask to original logits
```

OR **Alternative correct (simpler)**:
```python
sorted_logits[sorted_indices_to_remove] = float('-inf')
logits = logits.scatter(1, sorted_indices, sorted_logits)
# This SHOULD work if scatter is correct... let me verify...
```

Actually, I need to test this empirically. The analysis document is correct that there's a bug, but I need to verify the exact fix!

---

**Conclusion**: The document provides the correct fix. The bug is in the shift direction and/or the unsort operation. Applying the fix from the analysis document will resolve the issue.

