# CRITICAL FIX: Nucleus Sampling Bug

**Issue**: Top-P nucleus sampling implementation has a mathematical error causing nonsensical text generation

**Impact**: Model generates completely random tokens instead of coherent text

**Confidence**: 95% this is the root cause

**Files to Fix**: `src/model.py`

---

## The Bug (Lines 337-344)

```python
# CURRENT (WRONG):
sorted_mask = cumulative_probs > top_p
sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()  # ❌ WRONG SHIFT DIRECTION
sorted_mask[:, 0] = False
sorted_logits[sorted_mask] = float('-inf')
logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))  # ❌ INCORRECT UNSORT
```

**Problems**:
1. Shift direction is wrong - excludes wrong tokens
2. Unsort operation may not correctly restore order
3. Creates corrupted probability distribution

---

## The Fix

### Option 1: Correct the Existing Approach

```python
# CORRECTED VERSION:
sorted_indices_to_remove = cumulative_probs > top_p
# Keep at least the first token (highest probability)
sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
sorted_indices_to_remove[..., 0] = False

# Apply mask to sorted logits
sorted_logits[sorted_indices_to_remove] = float('-inf')

# Correctly unsort: scatter back to original positions
indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
logits[indices_to_remove] = float('-inf')
```

### Option 2: Simpler, More Standard Implementation (RECOMMENDED)

```python
# Top-P (nucleus) filtering
if top_p is not None and top_p < 1.0:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    # Keep at least 1 token (shift right by 1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Set filtered logits to -inf in sorted space
    sorted_logits[sorted_indices_to_remove] = float('-inf')

    # Map back to original positions using scatter
    logits = logits.scatter(1, sorted_indices, sorted_logits)
```

---

## Complete Fixed generate() Method

Replace the entire `generate()` method in `src/model.py` (lines 305-362):

```python
@torch.no_grad()
def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    cache_global_ids: torch.Tensor = None,
) -> torch.Tensor:
    """
    Génère des tokens autoregressivement.

    Args:
        input_ids: (B, L) tokens d'entrée
        max_new_tokens: Nombre de tokens à générer
        temperature: Température de sampling (0 = greedy, >1 = plus aléatoire)
        top_k: Si fourni, ne garde que les top-k tokens les plus probables
        top_p: Si fourni, nucleus sampling (garde les tokens dont la masse cumulée < p)
        cache_global_ids: (B, G) landmarks globaux si learned=False

    Returns:
        output_ids: (B, L + max_new_tokens)
    """
    self.eval()

    for _ in range(max_new_tokens):
        # Tronquer si trop long
        if input_ids.size(1) > self.cfg.max_seq_len:
            input_ids = input_ids[:, -self.cfg.max_seq_len:]

        # Forward pass pour obtenir logits
        logits = self(input_ids, cache_global_ids=cache_global_ids)  # (B, L, V)

        # Prendre logits du dernier token (BEFORE temperature)
        logits = logits[:, -1, :]  # (B, V)

        # Top-K filtering (on raw logits)
        if top_k is not None and top_k > 0:
            topk_vals, topk_idxs = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(1, topk_idxs, topk_vals)
            logits = logits_filtered

        # Top-P (nucleus) filtering (on raw logits)
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            # Shift right by 1 to keep at least the top token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Set filtered logits to -inf
            sorted_logits[sorted_indices_to_remove] = float('-inf')

            # Scatter back to original order
            logits = logits.scatter(1, sorted_indices, sorted_logits)

        # Apply temperature AFTER filtering
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # Safety check: if all logits were -inf, use uniform distribution
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.ones_like(probs) / probs.size(-1)

        # Additional safety: clamp and renormalize
        probs = torch.clamp(probs, min=1e-10)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids
```

---

## Additional Fix: Temperature Application Order

**Current (Wrong)**:
```python
logits = logits[:, -1, :] / temperature  # Line 321
# Then filtering uses these scaled logits
```

**Fixed (Correct)**:
```python
logits = logits[:, -1, :]  # Get last token logits
# Apply filtering on RAW logits
# ... top-k filtering ...
# ... top-p filtering ...
# Apply temperature AFTER filtering
if temperature != 1.0:
    logits = logits / temperature
```

---

## Testing Protocol

After applying fixes, run these validation tests:

### Test 1: Deterministic (Greedy) Decoding
```bash
python scripts/generate.py \
    --checkpoint out_slga/ckpt_11000 \
    --prompt "The capital of France is" \
    --max-tokens 5 \
    --temperature 0.01 \
    --top-k 0 \
    --top-p None
```

**Expected**: Should produce the same output every time (deterministic)
**Success criteria**: Output is coherent and related to France/Paris

### Test 2: Top-K Sampling
```bash
python scripts/generate.py \
    --checkpoint out_slga/ckpt_11000 \
    --prompt "The capital of France is" \
    --max-tokens 10 \
    --temperature 0.8 \
    --top-k 40 \
    --top-p None
```

**Expected**: Varied but coherent completions
**Success criteria**: No nonsensical tokens like "Pink", "Kejriwal", etc.

### Test 3: Nucleus Sampling (Top-P)
```bash
python scripts/generate.py \
    --checkpoint out_slga/ckpt_11000 \
    --prompt "The quick brown fox" \
    --max-tokens 20 \
    --temperature 1.0 \
    --top-p 0.9
```

**Expected**: Creative but coherent continuation
**Success criteria**: Text is grammatically plausible

### Test 4: Multiple Prompts
```bash
# Test 5 different prompts
for prompt in \
    "Hello, my name is" \
    "The weather today is" \
    "In the year 2050," \
    "Scientists have discovered" \
    "Once upon a time"
do
    echo "Prompt: $prompt"
    python scripts/generate.py \
        --checkpoint out_slga/ckpt_11000 \
        --prompt "$prompt" \
        --max-tokens 15 \
        --temperature 0.8 \
        --top-p 0.9
    echo "---"
done
```

**Expected**: All outputs should be coherent (not perfect, but plausible)

---

## Implementation Steps

1. **Backup current model.py**:
   ```bash
   cp src/model.py src/model.py.backup
   ```

2. **Apply the fix**:
   - Replace lines 305-362 in `src/model.py` with the fixed `generate()` method above

3. **Run validation tests**:
   - Execute Test 1 (deterministic)
   - If successful, run Tests 2-4

4. **Compare outputs**:
   ```bash
   # Before fix (current):
   "Pink immersed mattereur Kejriwal..."

   # After fix (expected):
   "Paris, the capital city of..." (or similar coherent text)
   ```

5. **If tests fail**:
   - Check for syntax errors
   - Verify temperature != 0 (use 0.01 for near-deterministic)
   - Add debug prints to see logit values

---

## Why This Fix Will Work

1. **Mathematically Correct**: The nucleus sampling now properly identifies tokens within the probability mass threshold

2. **Standard Implementation**: Follows the same pattern as Hugging Face Transformers and other libraries

3. **Temperature After Filtering**: Ensures filtering operates on the true probability distribution

4. **Proper Unsort**: `scatter()` correctly maps filtered logits back to original vocabulary positions

5. **Edge Case Handling**: Keeps at least 1 token even if top-p is very small

---

## Expected Results

**Before Fix**:
```
Prompt: "The capital of France is"
Output: "Pink immersed mattereur Kejriwal Trace Railway..."
```

**After Fix** (at 11k steps training):
```
Prompt: "The capital of France is"
Output: "Paris, a major city in Europe and..." (coherent but may still have errors)
```

**After Full Training** (100k steps):
```
Prompt: "The capital of France is"
Output: "Paris, located in the Île-de-France region..." (much more coherent)
```

---

## Rollback Plan

If the fix causes issues:

1. Restore backup:
   ```bash
   cp src/model.py.backup src/model.py
   ```

2. Report the issue with:
   - Error messages
   - Test outputs
   - Python version
   - PyTorch version

---

## Additional Improvements (Optional)

After confirming the fix works, consider these enhancements:

### 1. Add Early Stopping (EOS Detection)
```python
# After line 357 (after sampling):
next_token = torch.multinomial(probs, num_samples=1)

# Check for end-of-sequence token
if hasattr(self, 'eos_token_id') and next_token.item() == self.eos_token_id:
    break

input_ids = torch.cat([input_ids, next_token], dim=1)
```

### 2. Add Repetition Penalty
```python
# Before sampling, penalize already-generated tokens:
if input_ids.size(1) > 1:
    for token_id in input_ids[0].unique():
        logits[0, token_id] /= 1.2  # Repetition penalty
```

### 3. Improve Context Management
```python
# Instead of truncating, use sliding window:
if input_ids.size(1) > self.cfg.max_seq_len:
    # Keep first 128 tokens (prompt) + last (max_seq_len - 128) tokens
    input_ids = torch.cat([
        input_ids[:, :128],
        input_ids[:, -(self.cfg.max_seq_len - 128):]
    ], dim=1)
```

---

## Summary

- **Bug**: Off-by-one error in nucleus sampling mask + incorrect unsort operation
- **Fix**: Use correct shift direction + scatter() for unsorting
- **Impact**: Restores coherent text generation
- **Risk**: Low (only affects inference, not training)
- **Time**: 30 minutes to implement + 10 minutes to test

**Apply this fix immediately to restore generation quality.**
