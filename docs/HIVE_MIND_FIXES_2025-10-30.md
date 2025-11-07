# Hive Mind Critical Fixes - 2025-10-30

## Overview
Three critical fixes applied using hive-mind collective intelligence coordination:

1. **Gumbel Activation in Training** (model.py)
2. **EOS Token ID Handling** (generate.py)
3. **Top-K + Top-P Warning System** (generate.py)

## Fix 1: Gumbel-Softmax Training Mode

### Problem
The landmark selector was not using Gumbel-Softmax during training, preventing proper gradient flow through discrete landmark selection.

### Solution
**File**: `src/model.py:255`

```python
# BEFORE
landmark_indices, _, landmark_scores = self.landmark_selector(x)

# AFTER
landmark_indices, _, landmark_scores = self.landmark_selector(x, use_gumbel=self.training)
```

### Impact
- ✅ Enables gradient flow during training
- ✅ Proper backpropagation through landmark selection
- ✅ Better landmark learning convergence
- ✅ Maintains deterministic behavior in eval mode

### Why This Matters
Gumbel-Softmax allows differentiable sampling from discrete distributions. During training, we need gradients to flow through the landmark selection process to learn which positions are most informative. Without this, the landmark selector cannot improve during training.

## Fix 2: EOS Token ID Parameter

### Problem
The `generate()` method in model.py supports EOS stopping, but generate.py wasn't passing `eos_token_id` from the tokenizer.

### Solution
**Files**: `scripts/generate.py:43-44, 87-88, 101-102, 357-368, 536-537`

#### Added CLI Arguments
```python
parser.add_argument(
    "--eos-token-id",
    type=int,
    default=None,
    help="EOS token ID for stopping generation (defaults to tokenizer.eos_token_id)",
)

# ✅ Mutually exclusive group for stop-on-eos control
eos_group = parser.add_mutually_exclusive_group()
eos_group.add_argument(
    "--stop-on-eos",
    dest="stop_on_eos",
    action="store_true",
    default=True,
    help="Stop generation when EOS token is encountered (default)",
)
eos_group.add_argument(
    "--no-stop-on-eos",
    dest="stop_on_eos",
    action="store_false",
    help="Continue generation even after EOS token",
)
```

#### Updated Function Signature
```python
def generate_text(
    model: LLMTransformer,
    tokenizer: AutoTokenizer,
    prompt: str,
    # ... other params ...
    eos_token_id: int | None = None,  # NEW
    stop_on_eos: bool = True,          # NEW
    device: str = "cuda",
) -> str:
```

#### Automatic EOS Resolution
```python
# In generate_text()
if eos_token_id is None:
    eos_token_id = tokenizer.eos_token_id

# In main()
eos_token_id = args.eos_token_id if args.eos_token_id is not None else tokenizer.eos_token_id

# Pass to model
output_ids = model.generate(
    input_ids,
    # ...
    eos_token_id=eos_token_id,
    stop_on_eos=stop_on_eos,
)
```

### Impact
- ✅ Proper EOS token handling in generation
- ✅ Automatic detection from tokenizer
- ✅ Manual override via CLI if needed
- ✅ Informative logging of EOS configuration
- ✅ Generation stops naturally when model outputs EOS

### EOS Configuration Logging
```
EOS token configuration:
  Token ID: 50256
  Stop on EOS: True
  Token string: '<|endoftext|>'
```

## Fix 3: Top-K + Top-P Warning System

### Problem
Using both top_k and top_p simultaneously can be confusing and overly restrictive. Users may not realize they're applying both filters sequentially.

### Solution
**File**: `scripts/generate.py:314-324, 338-345, 498-504`

#### Validation Function Enhancement
```python
def validate_generation_params(args):
    errors = []
    warnings = []  # NEW

    # ... existing validations ...

    # NEW: Warning for top_k + top_p combination
    if (args.top_k is not None and args.top_k > 0 and
        args.top_p is not None and args.top_p < 1.0):
        warnings.append(
            f"⚠️  Using both top_k={args.top_k} and top_p={args.top_p} simultaneously.\n"
            f"    This applies BOTH filters sequentially (top_k THEN top_p).\n"
            f"    For most use cases, using only one is recommended:\n"
            f"    - Creative text: Use top_p=0.9 (nucleus sampling)\n"
            f"    - Focused generation: Use top_k=40\n"
            f"    - Greedy decoding: Set temperature=0.0 (disables both)"
        )

    # Display warnings (non-blocking)
    if warnings:
        print("=" * 80)
        print("⚠️  PARAMETER WARNINGS")
        print("=" * 80)
        for warn in warnings:
            print(f"  {warn}")
        print("=" * 80)
        print()
```

#### Runtime Logging
```python
# Log explicite si top_k + top_p sont utilisés ensemble
if (args.top_k is not None and args.top_k > 0 and
    args.top_p is not None and args.top_p < 1.0):
    print("⚠️  NOTE: Using BOTH top_k and top_p filtering")
    print(f"   Sampling will apply: top_k={args.top_k} THEN top_p={args.top_p}")
    print(f"   This may result in very restrictive sampling.")
    print()
```

### Impact
- ✅ Non-blocking warning (doesn't prevent execution)
- ✅ Clear explanation of behavior
- ✅ Best practice recommendations
- ✅ Runtime logging for transparency
- ✅ Helps users understand sampling behavior

### Example Warning Output
```
================================================================================
⚠️  PARAMETER WARNINGS
================================================================================
  ⚠️  Using both top_k=50 and top_p=0.95 simultaneously.
    This applies BOTH filters sequentially (top_k THEN top_p).
    For most use cases, using only one is recommended:
    - Creative text: Use top_p=0.9 (nucleus sampling)
    - Focused generation: Use top_k=40
    - Greedy decoding: Set temperature=0.0 (disables both)
================================================================================

⚠️  NOTE: Using BOTH top_k and top_p filtering
   Sampling will apply: top_k=50 THEN top_p=0.95
   This may result in very restrictive sampling.
```

## Testing

### Comprehensive Test Script
**File**: `tests/test_all_three_fixes.py`

Run validation:
```bash
python tests/test_all_three_fixes.py
```

### Test Coverage

#### Test 1: Gumbel Training Activation
- ✅ Verifies model.train() enables Gumbel
- ✅ Checks gradient flow through landmark scores
- ✅ Validates eval mode behavior
- ✅ Tests landmark selection shapes

#### Test 2: EOS Token Handling
- ✅ Tests explicit EOS token ID
- ✅ Validates generation with stop_on_eos
- ✅ Checks early stopping behavior
- ✅ Verifies output shape consistency

#### Test 3: Top-K + Top-P Warning
- ✅ Validates warning trigger conditions
- ✅ Tests warning suppression when appropriate
- ✅ Checks multiple edge cases (top_k=0, top_p=1.0)
- ✅ Verifies recommendation messages

## Usage Examples

### Example 1: Training with Proper Gumbel
```python
model = LLMTransformer(cfg)
model.train()  # Automatically enables use_gumbel=True in landmark_selector

# Forward pass with gradient flow
logits, aux = model(input_ids, return_aux=True)
loss = criterion(logits, targets)
loss.backward()  # Gradients flow through landmarks!
```

### Example 2: Generation with EOS Stopping
```bash
# Default behavior (stop on EOS)
python scripts/generate.py \
    --checkpoint out_slga/ckpt_11000 \
    --config config.yaml \
    --prompt "The future of AI" \
    --max-tokens 100

# Explicit stop on EOS
python scripts/generate.py \
    --checkpoint out_slga/ckpt_11000 \
    --config config.yaml \
    --prompt "The future of AI" \
    --stop-on-eos

# Continue after EOS (NEW!)
python scripts/generate.py \
    --checkpoint out_slga/ckpt_11000 \
    --config config.yaml \
    --prompt "The future of AI" \
    --no-stop-on-eos

# Manual EOS token ID
python scripts/generate.py \
    --checkpoint out_slga/ckpt_11000 \
    --config config.yaml \
    --prompt "The future of AI" \
    --eos-token-id 50256
```

### Example 3: Understanding Sampling Warnings
```bash
# This will trigger warning (both filters active)
python scripts/generate.py \
    --checkpoint out_slga/ckpt_11000 \
    --config config.yaml \
    --top-k 50 \
    --top-p 0.95

# Output:
# ⚠️  PARAMETER WARNINGS
# Using both top_k=50 and top_p=0.95 simultaneously.
# Recommendation: Use only one for most use cases

# No warning (only top_p)
python scripts/generate.py \
    --checkpoint out_slga/ckpt_11000 \
    --config config.yaml \
    --top-k 0 \
    --top-p 0.95
```

## Technical Details

### Fix Interaction
These three fixes are complementary:

1. **Training (Gumbel)**: Improves model quality during training
2. **Generation (EOS)**: Improves generation quality at inference
3. **Validation (Warning)**: Improves user experience and understanding

### Backward Compatibility
- ✅ All fixes are backward compatible
- ✅ Default behaviors preserved
- ✅ Optional parameters have sensible defaults
- ✅ Existing code continues to work

### Performance Impact
- **Gumbel**: Negligible overhead during training (only when landmarks enabled)
- **EOS**: No overhead (early stopping can save compute)
- **Warning**: Zero runtime overhead (validation only)

## Verification Checklist

- [x] Gumbel activation works in training mode
- [x] Landmark selection has proper gradients
- [x] EOS token ID detected from tokenizer
- [x] EOS stopping works correctly
- [x] Top-k + top-p warning triggers appropriately
- [x] Warning messages are clear and helpful
- [x] CLI arguments added and documented
- [x] Test script passes all checks
- [x] No breaking changes introduced

## Related Files

### Modified Files
- `src/model.py` (1 line: Gumbel activation)
- `scripts/generate.py` (7 locations: EOS + warning system)

### New Files
- `tests/test_all_three_fixes.py` (comprehensive validation)
- `docs/HIVE_MIND_FIXES_2025-10-30.md` (this document)

## Conclusion

All three fixes have been successfully applied and validated:

1. ✅ **Gumbel Training**: Proper gradient flow during landmark selection
2. ✅ **EOS Handling**: Natural generation stopping with proper token detection
3. ✅ **Sampling Warning**: Clear user guidance for top_k + top_p combination

The codebase now has:
- Better training dynamics (Fix 1)
- Better generation quality (Fix 2)
- Better user experience (Fix 3)

Run `python tests/test_all_three_fixes.py` to verify all fixes.
