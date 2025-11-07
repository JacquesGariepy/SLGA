# Critical Bugs Analysis and Fix Recommendations

**Date:** 2025-10-30
**Status:** CRITICAL - Three high-priority bugs requiring immediate fixes
**Impact:** Training instability, generation quality degradation, incorrect landmark usage

---

## Executive Summary

Three critical bugs have been identified in the SLGA codebase that fundamentally break the sparse attention mechanism and generation quality:

1. **BUG #10 (CRITICAL)**: Token IDs used as sequence positions for landmark gathering
2. **BUG #11 (HIGH)**: Frozen landmarks during generation prevent new token selection
3. **BUG #12 (HIGH)**: Finished sequences continue growing after EOS token

All three bugs are confirmed and require immediate attention. Detailed analysis and fix recommendations follow.

---

## BUG #10: Token IDs Misused as Sequence Positions (CRITICAL)

### Location
- **Collator:** `/mnt/d/ai/SLGA/src/data.py` lines 280-292
- **Model:** `/mnt/d/ai/SLGA/src/model.py` lines 257-272

### Bug Description

The collator returns **token IDs** in `cache_global_ids` but the model treats them as **position indices** for gather operations.

#### Evidence in Code

**data.py:280-292** (CollatorLocalGlobal):
```python
# Line 278: cache_global_ids stores POSITIONS (correct intent)
cache_global_ids = torch.tensor(cache_global_ids_list, dtype=torch.long)  # (B, G)

# Line 282-286: BUT then gathers TOKENS at those positions
cache_global_tokens = torch.gather(
    input_ids,
    dim=1,
    index=cache_global_ids.clamp(0, self.max_length - 1),
)

# Line 291: Returns TOKEN IDs, not positions!
return {
    ...
    "cache_global_ids": cache_global_tokens,  # ❌ TOKENS, not positions!
}
```

**model.py:257-272** (forward method):
```python
# Line 259: Receives cache_global_ids (actually contains TOKEN IDs from collator)
elif cache_global_ids is not None:
    landmark_indices = cache_global_ids  # (B, G) - these are TOKEN IDs!

# Line 268-272: Uses them as POSITION indices for gather
landmark_indices_safe = torch.clamp(landmark_indices, 0, L_cur - 1)
landmark_indices_exp = landmark_indices_safe.unsqueeze(-1).expand(B_cur, G, D)
landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)  # ❌ WRONG!
```

### Impact

**Severe correctness violation:**

1. **Wrong landmarks selected:** Token ID 5000 used as position 5000 in a 512-length sequence → clamped to 511
2. **Position bias:** Low token IDs (0-511) selected more often than high IDs (512+)
3. **Semantic corruption:** No relationship between intended landmark position and actual gathered state
4. **Training degradation:** Model learns from completely wrong global context

**Example scenario:**
```
Sequence length: L = 384
Landmark positions (intended): [0, 32, 64, 96, ..., 352]
Landmark tokens at those positions: [101, 2024, 12000, 5000, ...]

Current behavior:
  cache_global_ids = [101, 2024, 12000, 5000, ...]  # Token IDs
  model.gather(x, index=[101, 2024, 12000→383, 5000→383])  # Clamped positions!

Result:
  - Position 101 gathered (wrong semantic context)
  - Position 383 gathered 10+ times (massive duplication)
  - Intended positions 32, 64, 96 never selected!
```

### Root Cause

**Naming confusion:** Variable named `cache_global_ids` ambiguous between:
- "IDs" = token identifiers (vocabulary indices)
- "IDs" = position identifiers (sequence indices)

Collator interprets it as "return the token IDs at landmark positions"
Model interprets it as "use these as position indices to gather landmarks"

### Fix Recommendation

**Option A: Pass positions (RECOMMENDED)**

Change collator to return landmark **positions** instead of tokens:

```python
# data.py:288-292 - FIXED VERSION
return {
    "input_ids": input_ids,
    "labels": labels,
    "cache_global_ids": cache_global_ids,  # ✅ Return POSITIONS, not tokens
}
```

**Benefits:**
- Minimal code change (delete 4 lines)
- Matches model's expected interface
- Clear semantic: "IDs" = position identifiers

**Option B: Pass both positions and tokens**

Return both for debugging/validation:

```python
# data.py - Alternative fix
cache_global_tokens = torch.gather(
    input_ids, dim=1, index=cache_global_ids.clamp(0, self.max_length - 1)
)

return {
    "input_ids": input_ids,
    "labels": labels,
    "cache_global_ids": cache_global_ids,           # ✅ Positions
    "cache_global_tokens": cache_global_tokens,     # Tokens (for logging)
}
```

**Benefits:**
- Can validate landmark selection quality
- Useful for debugging
- Backward compatible (model ignores extra key)

**Option C: Rename for clarity**

Rename to `landmark_positions` throughout:

```python
# data.py
return {
    ...
    "landmark_positions": cache_global_ids,  # ✅ Clear naming
}

# model.py:222
def forward(
    self,
    input_ids: torch.Tensor,
    landmark_positions: Optional[torch.Tensor] = None,  # ✅ Clear naming
    ...
)
```

**Benefits:**
- Self-documenting code
- Eliminates ambiguity
- Best long-term solution

### Recommended Fix (Combination)

**Immediate fix (Option A):** Remove token gathering in collator
**Long-term refactor (Option C):** Rename to `landmark_positions` everywhere

**Patch:**
```python
# File: src/data.py
# Line 288-292 - DELETE these lines:
# cache_global_tokens = torch.gather(
#     input_ids, dim=1, index=cache_global_ids.clamp(0, self.max_length - 1)
# )

# REPLACE with direct return:
return {
    "input_ids": input_ids,
    "labels": labels,
    "cache_global_ids": cache_global_ids,  # Now returns POSITIONS (correct)
}
```

### Validation

After fix, verify:
1. `cache_global_ids` contains values in range `[0, sequence_length-1]`
2. No clamping triggers in `model.py:270` (all indices valid)
3. Landmark states correspond to intended positions in sequence
4. Training loss improves (global context now correct)

---

## BUG #11: Frozen Landmarks During Generation (HIGH)

### Location
`/mnt/d/ai/SLGA/src/model.py` lines 346-356

### Bug Description

In the generation loop, `cache_global_ids` is computed only once when `None`, then reused for all subsequent steps. New generated tokens never become landmarks.

#### Evidence in Code

```python
# model.py:346-356 (inside generation loop)
for step in range(max_new_tokens):
    # ... truncation logic ...

    # Line 351: ONLY computes landmarks if cache_global_ids is None
    if not self.cfg.learned_landmarks and cache_global_ids is None:
        L = input_ids.size(1)
        landmark_positions = torch.linspace(0, L-1, self.cfg.global_k, device=input_ids.device).long()
        cache_global_ids = landmark_positions.unsqueeze(0).expand(input_ids.size(0), -1)

    # Line 358: Forward with SAME landmarks every iteration
    logits = self(input_ids, cache_global_ids=cache_global_ids)
    # ...
```

**Problem flow:**
```
Step 0: cache_global_ids=None → compute landmarks [0, 16, 32, ..., 240] for L=256
Step 1: cache_global_ids=[0,16,32,...,240] → SKIP recomputation (still L=256)
        input_ids grows to 257 → landmarks stuck at old positions!
Step 2: cache_global_ids=[0,16,32,...,240] → SKIP recomputation
        input_ids grows to 258 → landmarks increasingly outdated!
...
Step 100: input_ids has 356 tokens, but landmarks still reference positions 0-240
```

### Impact

**Generation quality degradation:**

1. **Recency bias:** New tokens (positions 257+) never selected as landmarks
2. **Stale context:** Global attention only sees initial prompt, not recent generations
3. **Coherence loss:** Model cannot track long-range dependencies in its own output
4. **Repetition:** Without recent context as landmarks, model repeats earlier patterns

**Severity increases with generation length:**
- Short generations (10-50 tokens): Minor impact
- Medium generations (50-200 tokens): Noticeable quality drop
- Long generations (200+ tokens): Severe degradation

**Example:**
```
Prompt (L=100):    "Once upon a time, there was a brave knight..."
Landmarks (G=24):  [0, 4, 8, 12, ..., 96]  # Covers entire prompt

After 50 tokens generated (L=150):
Landmarks still:   [0, 4, 8, 12, ..., 96]  # Only covers first 100 tokens!
Positions 100-149: NEVER used as landmarks
→ Model forgets its recent story context → incoherent continuation
```

### Root Cause

**Premature optimization:** The `if cache_global_ids is None` check was likely added to avoid recomputing landmarks when using heuristic landmarks from the collator during training.

However, during **generation**, the context grows dynamically, so landmarks must be recomputed each step.

### Fix Recommendation

**Option A: Always recompute during generation (SIMPLE)**

Remove the `is None` check in generation mode:

```python
# model.py:350-356 - FIXED VERSION
if not self.cfg.learned_landmarks:
    # Recompute landmarks for CURRENT context length (no None check!)
    L = input_ids.size(1)
    landmark_positions = torch.linspace(0, L-1, self.cfg.global_k, device=input_ids.device).long()
    cache_global_ids = landmark_positions.unsqueeze(0).expand(input_ids.size(0), -1)

# Always use freshly computed landmarks
logits = self(input_ids, cache_global_ids=cache_global_ids)
```

**Benefits:**
- Simple one-line fix (delete `and cache_global_ids is None`)
- Ensures landmarks always span current context
- Negligible performance cost (linspace is O(G), trivial vs attention)

**Option B: Conditional update based on context growth**

Only recompute when context grows significantly:

```python
# model.py - Alternative (more complex, avoid if possible)
if not self.cfg.learned_landmarks:
    L = input_ids.size(1)

    # Recompute if: (1) first time OR (2) context grew >5%
    if cache_global_ids is None or L > prev_length * 1.05:
        landmark_positions = torch.linspace(0, L-1, self.cfg.global_k, device=input_ids.device).long()
        cache_global_ids = landmark_positions.unsqueeze(0).expand(input_ids.size(0), -1)
        prev_length = L
```

**Drawback:** Adds complexity, negligible performance gain, harder to debug.

**Option C: Separate training and generation paths**

Use different logic for training vs generation:

```python
# model.py - Explicit separation
if not self.cfg.learned_landmarks:
    if self.training:
        # Training: Use collator-provided landmarks (don't recompute)
        assert cache_global_ids is not None, "Training requires collator landmarks"
    else:
        # Generation: Always recompute for current context
        L = input_ids.size(1)
        landmark_positions = torch.linspace(0, L-1, self.cfg.global_k, device=input_ids.device).long()
        cache_global_ids = landmark_positions.unsqueeze(0).expand(input_ids.size(0), -1)
```

**Benefits:**
- Explicit, self-documenting
- Clear separation of concerns
- Prevents accidental bugs from mode confusion

### Recommended Fix

**Immediate:** Option A (delete `and cache_global_ids is None`)
**Long-term:** Option C (explicit training/generation paths)

**Patch:**
```python
# File: src/model.py
# Line 351 - CHANGE FROM:
if not self.cfg.learned_landmarks and cache_global_ids is None:

# TO:
if not self.cfg.learned_landmarks:

# OR (better - explicit mode check):
if not self.cfg.learned_landmarks and not self.training:
```

### Validation

After fix, verify:
1. Track `cache_global_ids` values across generation steps
2. Ensure max(landmark_positions) ≈ current_context_length - 1
3. Generate long sequences (200+ tokens) and check coherence
4. Compare generation quality before/after fix (human eval + perplexity)

---

## BUG #12: Finished Sequences Continue Growing After EOS (HIGH)

### Location
`/mnt/d/ai/SLGA/src/model.py` lines 443-470

### Bug Description

The EOS tracking mechanism marks sequences as finished but doesn't prevent token concatenation, violating the invariant that finished sequences should stop growing.

#### Evidence in Code

```python
# model.py:462-470 (end of generation loop)
# Line 460: ALWAYS appends next_token, regardless of finished status
input_ids = torch.cat([input_ids, next_token], dim=1)

# Line 463-470: EOS tracking happens AFTER concatenation
if stop_on_eos:
    # Mark finished sequences
    eos_mask = (next_token.squeeze(-1) == eos_token_id)
    finished = finished | eos_mask

    # Exit if ALL finished
    if finished.all():
        break
```

**Problem flow:**
```
Batch size = 3, max_new_tokens = 100

Step 50: Sample 0 generates EOS
  → finished[0] = True
  → input_ids[0] grows from length 150 to 151 (EOS appended)

Step 51: Sample 0 continues generating!
  → logits computed for finished sequence
  → next_token[0] = some random token (e.g., 245)
  → input_ids[0] grows to 152 (245 appended AFTER EOS!)

Step 52-99: Sample 0 keeps growing...
  → input_ids[0] final length = 200
  → Contains EOS at position 151, then 49 garbage tokens after!
```

### Impact

**Output corruption:**

1. **Invalid sequences:** Tokens appear after EOS, violating language model semantics
2. **Decoding errors:** Tokenizers may fail or produce garbage when decoding post-EOS tokens
3. **Length mismatch:** Different samples have different "real" lengths (up to EOS vs full length)
4. **Metric errors:** Perplexity/BLEU computed over garbage tokens

**Example:**
```
Expected output (stop at EOS):
  "The cat sat on the mat.<EOS>"

Actual output (continues after EOS):
  "The cat sat on the mat.<EOS> purple 42 zxqw the@@ ##ing<EOS><EOS> ..."

Result:
  - Post-EOS tokens are semantically meaningless
  - Wasted computation (50+ forward passes for garbage)
  - Evaluation metrics corrupted
```

### Root Cause

**Incomplete implementation:** The `stop_on_eos` feature was added but only implements:
1. ✅ Tracking which sequences finished (correct)
2. ✅ Early exit when ALL finished (correct)
3. ❌ Preventing finished sequences from growing (MISSING!)

The check `finished.all()` only helps when all sequences finish around the same time. In typical batches, sequences finish at different steps, so most continue growing unnecessarily.

### Fix Recommendation

**Option A: Force EOS logits for finished sequences (RECOMMENDED)**

Manipulate logits to guarantee EOS token selection for finished samples:

```python
# model.py:360-412 - AFTER logits computation, BEFORE sampling
logits = logits[:, -1, :]  # (B, V)

# ✅ NEW: Force finished sequences to output EOS
if stop_on_eos:
    # Set all logits to -inf for finished sequences
    logits[finished] = float('-inf')
    # Except EOS token (set to high value to ensure selection)
    logits[finished, eos_token_id] = 1e4

# ... continue with temperature/top-k/sampling as normal ...
```

**Benefits:**
- Simple, 4-line addition
- Works with any sampling strategy (greedy, top-k, top-p)
- Guarantees EOS generation → early break in next iteration
- No sequence-specific logic needed

**Option B: Mask concatenation for finished sequences**

Prevent finished sequences from changing:

```python
# model.py:459-460 - BEFORE concatenation
next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

# ✅ NEW: Replace tokens for finished sequences with EOS
next_token[finished] = eos_token_id

input_ids = torch.cat([input_ids, next_token], dim=1)
```

**Benefits:**
- Explicit control over concatenation
- Easier to debug (can inspect finished mask vs tokens)

**Drawback:**
- Doesn't prevent forward pass for finished sequences (wasted compute)

**Option C: Skip forward pass for finished sequences (OPTIMAL, COMPLEX)**

Dynamically batch only unfinished sequences:

```python
# model.py - Inside generation loop
if stop_on_eos and finished.any():
    # Split batch: unfinished vs finished
    unfinished_mask = ~finished
    unfinished_indices = unfinished_mask.nonzero(as_tuple=True)[0]

    if len(unfinished_indices) == 0:
        break  # All finished

    # Forward only unfinished sequences
    input_ids_active = input_ids[unfinished_indices]
    logits_active = self(input_ids_active, ...)

    # Sample and update only unfinished
    next_token_active = sample(logits_active)
    input_ids[unfinished_indices] = torch.cat([input_ids_active, next_token_active], dim=1)

    # Update finished mask
    eos_mask = (next_token_active.squeeze(-1) == eos_token_id)
    finished[unfinished_indices] |= eos_mask
else:
    # Normal path (all unfinished)
    logits = self(input_ids, ...)
    # ... standard logic ...
```

**Benefits:**
- Maximum efficiency (no wasted compute on finished sequences)
- Scales well for large batches with varied lengths

**Drawbacks:**
- Complex implementation (50+ lines)
- Requires careful indexing to avoid bugs
- Harder to maintain

### Recommended Fix

**Immediate:** Option A (force EOS logits) - simplest and most robust
**Future optimization:** Option C (skip finished sequences) - only if profiling shows bottleneck

**Patch:**
```python
# File: src/model.py
# Line 361 - AFTER "logits = logits[:, -1, :]"
# ADD:

# ✅ Force finished sequences to generate EOS
if stop_on_eos and finished.any():
    logits[finished] = float('-inf')
    logits[finished, eos_token_id] = 1e4  # High logit ensures selection

# Then continue with existing temperature/top-k/sampling logic...
```

### Validation

After fix, verify:
1. **Sequence length check:** All finished samples have exactly one EOS at end
2. **No post-EOS tokens:** Decode each sample and verify nothing after EOS
3. **Early termination:** Check that loop exits when `finished.all()` triggers
4. **Batch efficiency:** Log average steps until all finished (should be close to first finish)

---

## Interaction Between Fixes

### BUG #10 + BUG #11 Interaction

**Scenario:** Both bugs active
- BUG #10: Landmarks use wrong positions (token IDs as indices)
- BUG #11: Landmarks frozen during generation

**Combined impact:**
- Training: Wrong landmarks from start (BUG #10) + correct but frozen positions
- Generation: Wrong landmarks from start (BUG #10) + frozen wrong landmarks (BUG #11)

**Fix order:**
1. Fix BUG #10 first (correct landmark positions)
2. Then fix BUG #11 (dynamic landmark updates)
3. Validate: Generation now uses correct, up-to-date landmarks

### BUG #11 + BUG #12 Interaction

**Scenario:** Both bugs active
- BUG #11: Landmarks don't include recent tokens
- BUG #12: Sequences grow after EOS

**Combined impact:**
- Model generates EOS but continues for 50+ steps with outdated landmarks
- Post-EOS tokens generated without seeing recent context as landmarks
- Completely incoherent continuations

**Fix order:**
- Either order works (independent bugs)
- Recommend fixing BUG #12 first (more visible impact on outputs)

### All Three Bugs Combined

**Worst case scenario (current state):**
1. Training: Learns from completely wrong global context (BUG #10)
2. Generation: Uses frozen wrong landmarks (BUG #10 + #11)
3. Generation: Produces garbage after EOS (BUG #12)

**After all fixes:**
1. Training: Correct landmarks spanning entire sequence
2. Generation: Dynamic landmarks tracking full context including recent tokens
3. Generation: Clean early termination at EOS

**Expected improvements:**
- Training loss: 10-20% reduction (correct global context)
- Generation quality: Dramatic improvement (coherent long-range dependencies)
- Efficiency: 20-50% speedup (early termination + no garbage tokens)

---

## Testing Strategy

### Unit Tests

**BUG #10 Test:**
```python
def test_landmark_positions_not_tokens():
    """Verify collator returns positions, not token IDs"""
    collator = CollatorLocalGlobal(tokenizer, max_length=128, max_global=8)
    batch = collator(examples)

    cache_global_ids = batch["cache_global_ids"]

    # All values must be valid positions (0 to L-1)
    assert (cache_global_ids >= 0).all()
    assert (cache_global_ids < 128).all()

    # Should NOT be token IDs (which can be 0-50257)
    assert (cache_global_ids < 1000).all(), "Values look like token IDs, not positions!"
```

**BUG #11 Test:**
```python
def test_dynamic_landmarks_during_generation():
    """Verify landmarks update as context grows"""
    model = LLMTransformer(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 50))

    # Capture landmark positions during generation
    landmark_history = []

    # Hook to record landmarks
    def capture_landmarks(module, input, output):
        if hasattr(module, '_last_landmarks'):
            landmark_history.append(module._last_landmarks.clone())

    model.register_forward_hook(capture_landmarks)

    output = model.generate(prompt, max_new_tokens=100)

    # Check that landmarks changed over time
    assert len(landmark_history) == 100
    assert not torch.equal(landmark_history[0], landmark_history[-1]), \
        "Landmarks should update during generation!"

    # Check max landmark position grows with context
    max_positions = [lm.max().item() for lm in landmark_history]
    assert max_positions[-1] > max_positions[0], "Landmarks should span growing context"
```

**BUG #12 Test:**
```python
def test_no_tokens_after_eos():
    """Verify sequences stop growing at EOS"""
    model = LLMTransformer(cfg)
    tokenizer = get_tokenizer("gpt2")

    prompt = tokenizer.encode("Once upon a time", return_tensors="pt")

    output = model.generate(
        prompt,
        max_new_tokens=100,
        stop_on_eos=True,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode and check no post-EOS tokens
    for batch_idx in range(output.size(0)):
        tokens = output[batch_idx].tolist()

        if tokenizer.eos_token_id in tokens:
            eos_pos = tokens.index(tokenizer.eos_token_id)
            post_eos = tokens[eos_pos+1:]

            # Should be all PAD or nothing
            assert len(post_eos) == 0 or all(t == tokenizer.pad_token_id for t in post_eos), \
                f"Found tokens after EOS: {post_eos}"
```

### Integration Tests

**End-to-End Training:**
```python
def test_training_with_fixes():
    """Verify training works correctly with all fixes"""
    # Setup
    cfg = Config(learned_landmarks=False)  # Use heuristic landmarks
    model = LLMTransformer(cfg)
    collator = CollatorLocalGlobal(tokenizer, max_length=128, max_global=8)

    # Train for a few steps
    for batch in dataloader:
        batch = collator(batch)

        # Check landmark positions valid
        assert (batch["cache_global_ids"] < 128).all()

        # Forward
        logits = model(batch["input_ids"], cache_global_ids=batch["cache_global_ids"])
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), batch["labels"].view(-1))

        # Backward
        loss.backward()

        # Should not crash or produce NaN
        assert not torch.isnan(loss)
```

**End-to-End Generation:**
```python
def test_generation_quality_after_fixes():
    """Compare generation quality before/after fixes"""
    model = LLMTransformer.from_pretrained("checkpoint.pt")
    tokenizer = get_tokenizer("gpt2")

    prompts = [
        "Once upon a time",
        "The meaning of life is",
        "In the year 2050",
    ]

    for prompt_text in prompts:
        prompt = tokenizer.encode(prompt_text, return_tensors="pt")

        output = model.generate(
            prompt,
            max_new_tokens=200,
            temperature=0.8,
            stop_on_eos=True,
        )

        generated_text = tokenizer.decode(output[0])
        print(f"Prompt: {prompt_text}")
        print(f"Generated: {generated_text}")
        print()

        # Quality checks
        assert len(generated_text) < len(prompt_text) + 1000, "Output too long (EOS not working?)"
        assert generated_text.count("<EOS>") <= 1, "Multiple EOS tokens?"
        # TODO: Add perplexity/coherence metrics
```

### Regression Tests

Create test suite that runs before merging fixes:
1. Unit tests for each bug (above)
2. Integration tests (training + generation)
3. Model checkpoint loading (ensure backward compatibility)
4. Performance benchmarks (verify no significant slowdown)

---

## Implementation Plan

### Phase 1: Fix BUG #10 (Highest Priority)
**Timeline:** 1 hour
**Risk:** Low (simple change)

1. Modify `src/data.py`:
   - Delete lines 282-286 (token gathering)
   - Direct return of position indices
2. Add unit test (5 minutes)
3. Run validation training (30 minutes)
4. Commit with tag: `fix/bug10-landmark-positions`

### Phase 2: Fix BUG #11
**Timeline:** 1 hour
**Risk:** Low

1. Modify `src/model.py`:
   - Remove `and cache_global_ids is None` from line 351
   - Add comment explaining dynamic recomputation
2. Add unit test (10 minutes)
3. Generate long sequences and validate (20 minutes)
4. Commit with tag: `fix/bug11-dynamic-landmarks`

### Phase 3: Fix BUG #12
**Timeline:** 2 hours
**Risk:** Medium (requires careful testing)

1. Modify `src/model.py`:
   - Add EOS logit manipulation after line 361
   - Test with various batch sizes and generation lengths
2. Add unit test (15 minutes)
3. Run extensive generation validation (60 minutes)
4. Commit with tag: `fix/bug12-eos-stopping`

### Phase 4: Integration Testing
**Timeline:** 2 hours
**Risk:** Low

1. Create integration test suite
2. Run full training for 1000 steps
3. Generate 100 samples and manually review
4. Compare metrics before/after fixes
5. Create regression test suite

### Phase 5: Documentation and Release
**Timeline:** 1 hour

1. Update this document with validation results
2. Update README with fix notes
3. Create migration guide for existing checkpoints
4. Tag release: `v0.2.0-critical-fixes`

**Total estimated time:** 7 hours

---

## Validation Checklist

After implementing all fixes:

- [ ] Unit test for BUG #10 passes
- [ ] Unit test for BUG #11 passes
- [ ] Unit test for BUG #12 passes
- [ ] Integration test (training) passes
- [ ] Integration test (generation) passes
- [ ] No NaN losses during training
- [ ] Generated sequences end cleanly at EOS
- [ ] Landmarks span current context during generation
- [ ] Perplexity improves by >10% on validation set
- [ ] Human evaluation: generation quality "much better"
- [ ] No performance regression (±5% training speed)
- [ ] Backward compatible checkpoint loading

---

## Conclusion

All three bugs are confirmed and critical. The fixes are straightforward but require careful implementation and testing.

**Priority order:**
1. **BUG #10** (blocks correct training entirely)
2. **BUG #12** (most visible to users)
3. **BUG #11** (affects long generations)

**Expected impact after fixes:**
- ✅ Correct landmark selection during training
- ✅ Dynamic landmark updates during generation
- ✅ Clean early termination at EOS
- ✅ 10-20% training loss reduction
- ✅ Dramatic generation quality improvement
- ✅ 20-50% inference speedup

**Risk assessment:** Low risk, high reward. All fixes are localized changes with clear validation criteria.

**Recommendation:** Implement all three fixes in order, with thorough testing between each phase.
