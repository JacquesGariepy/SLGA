#!/usr/bin/env python3
"""
Test for BUG #11: Verify landmarks update dynamically during generation

This test validates that cache_global_ids is recomputed at each generation step
as the context grows, rather than being frozen after first computation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.model import LLMTransformer, Config
from src.data import get_tokenizer


def test_bug11_landmarks_update_during_generation():
    """
    BUG #11 Test: Verify landmarks are recomputed as context grows.

    Expected behavior:
        - Landmarks span current context at each step
        - max(landmark_positions) ≈ current_length - 1
        - Landmarks change as sequence grows

    Broken behavior (before fix):
        - Landmarks computed once (step 0)
        - Frozen for all subsequent steps
        - New tokens never become landmarks
    """
    print("=" * 80)
    print("BUG #11 TEST: Dynamic Landmark Updates During Generation")
    print("=" * 80)

    # Setup model
    cfg = Config(
        vocab_size=50257,
        max_seq_len=512,
        embed_dim=256,
        num_heads=4,
        n_layers=4,
        local_window=64,
        global_k=16,
        learned_landmarks=False,  # Use heuristic landmarks (linspace)
    )

    model = LLMTransformer(cfg)
    model.eval()

    tokenizer = get_tokenizer("gpt2")

    # Create prompt
    prompt_text = "Once upon a time"
    prompt = tokenizer.encode(prompt_text, return_tensors="pt")
    prompt_length = prompt.size(1)

    print(f"\nPrompt: '{prompt_text}'")
    print(f"Prompt length: {prompt_length} tokens")

    # Generate with tracking
    max_new_tokens = 50
    print(f"\nGenerating {max_new_tokens} new tokens...")

    # Hook to capture landmarks during generation
    landmark_history = []
    context_lengths = []

    original_forward = model.forward

    def forward_with_tracking(input_ids, cache_global_ids=None, **kwargs):
        # Record context length and landmarks
        context_lengths.append(input_ids.size(1))
        if cache_global_ids is not None:
            landmark_history.append(cache_global_ids.clone())
        else:
            landmark_history.append(None)

        # Call original forward
        return original_forward(input_ids, cache_global_ids=cache_global_ids, **kwargs)

    model.forward = forward_with_tracking

    # Generate
    output = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,  # Greedy for determinism
        stop_on_eos=False,  # Generate full length to test all steps
    )

    # Restore original forward
    model.forward = original_forward

    final_length = output.size(1)
    tokens_generated = final_length - prompt_length

    print(f"\n✓ Generation completed")
    print(f"  Final length: {final_length} tokens")
    print(f"  Tokens generated: {tokens_generated}")
    print(f"  Forward passes: {len(landmark_history)}")

    # ========================================================================
    # VALIDATION CHECKS
    # ========================================================================

    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)

    # Check 1: Landmarks were provided at each step
    print("\n✓ Check 1: Landmarks provided at each generation step")

    none_count = sum(1 for lm in landmark_history if lm is None)
    if none_count > 0:
        print(f"  ⚠️  WARNING: {none_count}/{len(landmark_history)} steps had None landmarks")
        print(f"     This may indicate learned_landmarks=True or missing computation")
    else:
        print(f"  PASS: All {len(landmark_history)} steps had landmarks")

    # Filter out None landmarks for further analysis
    valid_landmarks = [(i, lm, ctx_len) for i, (lm, ctx_len) in enumerate(zip(landmark_history, context_lengths)) if lm is not None]

    if len(valid_landmarks) == 0:
        print("\n❌ ABORT: No valid landmarks captured (all None)")
        print("   Cannot test dynamic updates without landmark data")
        return False

    # Check 2: Landmark max position grows with context
    print("\n✓ Check 2: Landmark positions grow with context length")

    first_step, first_landmarks, first_ctx = valid_landmarks[0]
    last_step, last_landmarks, last_ctx = valid_landmarks[-1]

    first_max_pos = first_landmarks.max().item()
    last_max_pos = last_landmarks.max().item()

    print(f"  Step 0 (ctx={first_ctx}):")
    print(f"    Landmarks: {first_landmarks[0].tolist()}")
    print(f"    Max position: {first_max_pos}")

    print(f"  Step {last_step} (ctx={last_ctx}):")
    print(f"    Landmarks: {last_landmarks[0].tolist()}")
    print(f"    Max position: {last_max_pos}")

    if last_max_pos > first_max_pos:
        print(f"  ✓ PASS: Max landmark position grew from {first_max_pos} to {last_max_pos}")
        bug_present = False
    else:
        print(f"  ❌ FAIL: Max position did NOT grow ({first_max_pos} → {last_max_pos})")
        print(f"     This indicates BUG #11 is PRESENT (frozen landmarks)")
        bug_present = True

    # Check 3: Landmarks span current context at each step
    print("\n✓ Check 3: Landmarks span current context (not stuck at initial length)")

    span_errors = 0
    for step, landmarks, ctx_len in valid_landmarks[::10]:  # Sample every 10 steps
        max_pos = landmarks.max().item()
        expected_max = ctx_len - 1
        span_ratio = max_pos / expected_max if expected_max > 0 else 0

        if span_ratio < 0.8:  # Should span at least 80% of context
            print(f"  ⚠️  Step {step} (ctx={ctx_len}): max_pos={max_pos}, expected~{expected_max} (span={span_ratio:.1%})")
            span_errors += 1

    if span_errors == 0:
        print(f"  ✓ PASS: Landmarks properly span context at all sampled steps")
    else:
        print(f"  ❌ FAIL: {span_errors} steps had insufficient span (landmarks stuck at old positions)")
        bug_present = True

    # Check 4: Landmarks actually changed between steps
    print("\n✓ Check 4: Landmarks changed during generation")

    if torch.equal(first_landmarks, last_landmarks):
        print(f"  ❌ FAIL: First and last landmarks are IDENTICAL")
        print(f"     This definitively proves BUG #11 is PRESENT")
        bug_present = True
    else:
        print(f"  ✓ PASS: Landmarks changed over time")

        # Show how many changed
        diff = (first_landmarks != last_landmarks).sum().item()
        total = first_landmarks.numel()
        print(f"     Changed positions: {diff}/{total} ({diff/total:.1%})")

    # Check 5: Recent tokens became landmarks (coverage of generated text)
    print("\n✓ Check 5: Recent tokens included in landmarks")

    # In last step, check if any landmark points to recently generated tokens
    recent_threshold = final_length - 20  # Last 20 tokens
    recent_landmarks = (last_landmarks >= recent_threshold).sum().item()

    print(f"  Final context length: {final_length}")
    print(f"  Recent tokens (>{recent_threshold}): {final_length - recent_threshold}")
    print(f"  Landmarks in recent range: {recent_landmarks}/{cfg.global_k}")

    if recent_landmarks > 0:
        print(f"  ✓ PASS: {recent_landmarks} landmarks point to recent tokens")
    else:
        print(f"  ❌ FAIL: NO landmarks point to recent tokens (all stuck in prompt)")
        print(f"     This means model cannot attend to its own recent generation!")
        bug_present = True

    # ========================================================================
    # FINAL VERDICT
    # ========================================================================

    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    if not bug_present:
        print("✅ BUG #11 IS FIXED (or not present)")
        print("   Landmarks update dynamically during generation")
        print("   Max position grows with context length")
        print("   Recent tokens become landmarks")
        return True
    else:
        print("❌ BUG #11 IS PRESENT")
        print("   Landmarks frozen after first computation")
        print("   New tokens never become landmarks")
        print("   Model loses track of recent context during generation")
        print("\n📋 FIX: Remove 'and cache_global_ids is None' check in model.py:351")
        print("   Always recompute landmarks during generation")
        return False


def test_bug11_learned_vs_heuristic():
    """
    Test that learned landmarks don't suffer from same bug.
    """
    print("\n" + "=" * 80)
    print("BUG #11 TEST: Learned Landmarks (should be immune)")
    print("=" * 80)

    cfg = Config(
        vocab_size=50257,
        max_seq_len=256,
        embed_dim=128,
        num_heads=4,
        n_layers=2,
        local_window=32,
        global_k=8,
        learned_landmarks=True,  # Learned landmarks (via LearnableLandmarkSelector)
    )

    model = LLMTransformer(cfg)
    model.eval()

    tokenizer = get_tokenizer("gpt2")
    prompt = tokenizer.encode("Test", return_tensors="pt")

    print(f"Prompt length: {prompt.size(1)}")
    print("Generating 20 tokens with learned landmarks...")

    try:
        output = model.generate(prompt, max_new_tokens=20, temperature=0.0)
        print(f"✓ Generation successful with learned landmarks")
        print(f"  Output length: {output.size(1)}")
        return True
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TESTING BUG #11: Dynamic Landmark Updates During Generation")
    print("=" * 80)
    print("\nThis test validates the fix for BUG #11, where landmarks were")
    print("frozen after first computation, preventing new tokens from being selected.")
    print("=" * 80)

    # Run tests
    test1_passed = test_bug11_landmarks_update_during_generation()
    test2_passed = test_bug11_learned_vs_heuristic()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Test 1 (Heuristic landmarks): {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Test 2 (Learned landmarks): {'✅ PASS' if test2_passed else '❌ FAIL'}")

    if test1_passed and test2_passed:
        print("\n🎉 All tests PASSED - BUG #11 is FIXED")
        sys.exit(0)
    else:
        print("\n❌ Some tests FAILED - BUG #11 is still PRESENT")
        print("\n📋 To fix: Modify src/model.py line 351")
        print("   Remove 'and cache_global_ids is None' from condition")
        print("   Always recompute landmarks during generation")
        sys.exit(1)
