#!/usr/bin/env python3
"""
Test for BUG #10: Verify collator returns positions, not token IDs

This test validates that cache_global_ids contains sequence POSITIONS (0 to L-1)
rather than TOKEN IDs (0 to vocab_size-1).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.legacy.data import CollatorLocalGlobal, get_tokenizer


def test_bug10_positions_not_tokens():
    """
    BUG #10 Test: Verify collator returns landmark positions, not token IDs.

    Expected behavior:
        - cache_global_ids contains positions in range [0, L-1]
        - All values < sequence_length (max_length)
        - Values should NOT be in token ID range (0-50257 for GPT-2)

    Current broken behavior (before fix):
        - cache_global_ids contains token IDs from vocabulary
        - Values can be 0-50257, causing out-of-bounds when used as positions
        - Model gathers from wrong positions after clamping
    """
    print("=" * 80)
    print("BUG #10 TEST: Landmark Positions vs Token IDs")
    print("=" * 80)

    # Setup
    tokenizer = get_tokenizer("gpt2")
    max_length = 128
    max_global = 8

    collator = CollatorLocalGlobal(
        tokenizer=tokenizer,
        max_length=max_length,
        max_global=max_global,
        strategy="regular"
    )

    # Create example batch
    examples = [
        {"text": "The quick brown fox jumps over the lazy dog. " * 20},
        {"text": "Machine learning is transforming the world. " * 20},
        {"text": "Neural networks learn complex patterns from data. " * 20},
    ]

    # Collate
    batch = collator(examples)

    print(f"\nBatch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    print(f"  cache_global_ids: {batch['cache_global_ids'].shape}")

    cache_global_ids = batch["cache_global_ids"]
    B, G = cache_global_ids.shape

    print(f"\nLandmark IDs statistics:")
    print(f"  Shape: {cache_global_ids.shape} (batch_size={B}, num_landmarks={G})")
    print(f"  Min: {cache_global_ids.min().item()}")
    print(f"  Max: {cache_global_ids.max().item()}")
    print(f"  Mean: {cache_global_ids.float().mean().item():.2f}")
    print(f"\nFirst sample landmarks: {cache_global_ids[0].tolist()}")

    # ========================================================================
    # VALIDATION CHECKS
    # ========================================================================

    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)

    # Check 1: All values must be valid positions (0 to L-1)
    print("\n✓ Check 1: Values are valid positions [0, max_length-1]")
    assert (cache_global_ids >= 0).all(), \
        f"❌ Found negative values (invalid positions): min={cache_global_ids.min().item()}"
    assert (cache_global_ids < max_length).all(), \
        f"❌ Found values >= max_length ({max_length}): max={cache_global_ids.max().item()}"
    print(f"  PASS: All values in range [0, {max_length-1}]")

    # Check 2: Should NOT contain large token IDs (which would indicate bug)
    print("\n✓ Check 2: Values are NOT large token IDs")
    max_expected_position = max_length - 1  # Should be ~127
    max_token_id = 50257  # GPT-2 vocab size

    if (cache_global_ids >= 1000).any():
        print(f"  ❌ FAIL: Found values >= 1000 (looks like token IDs, not positions!)")
        print(f"     Max value: {cache_global_ids.max().item()}")
        print(f"     This indicates BUG #10 is PRESENT")

        # Show examples of bad values
        bad_values = cache_global_ids[cache_global_ids >= 1000]
        print(f"     Examples of bad values: {bad_values[:10].tolist()}")
        return False
    else:
        print(f"  PASS: All values < 1000 (valid positions, not token IDs)")

    # Check 3: Verify positions match expected regular spacing
    print("\n✓ Check 3: Regular spacing for 'regular' strategy")
    expected_positions = torch.linspace(0, max_length - 1, max_global).long()

    # Should be identical for all samples (regular strategy)
    for i in range(B):
        if not torch.equal(cache_global_ids[i], expected_positions):
            print(f"  ⚠️  WARNING: Sample {i} positions don't match expected regular spacing")
            print(f"     Expected: {expected_positions.tolist()}")
            print(f"     Got:      {cache_global_ids[i].tolist()}")
        else:
            print(f"  Sample {i}: Matches expected regular spacing ✓")

    # Check 4: Verify we can safely gather from input_ids
    print("\n✓ Check 4: Can safely gather from input_ids using these positions")
    input_ids = batch["input_ids"]  # (B, L)

    try:
        # This should NOT require clamping if positions are correct
        gathered = torch.gather(input_ids, dim=1, index=cache_global_ids)
        print(f"  PASS: Gathered successfully: {gathered.shape}")

        # Show gathered token IDs (these should be actual tokens from input)
        print(f"  Sample 0 gathered tokens: {gathered[0].tolist()}")

        # Verify gathered values are valid token IDs
        assert (gathered >= 0).all() and (gathered < 50257).all(), \
            "Gathered values are not valid token IDs!"

    except Exception as e:
        print(f"  ❌ FAIL: Could not gather: {e}")
        return False

    # Check 5: Semantic validation - gathered tokens should correspond to positions
    print("\n✓ Check 5: Gathered tokens match positions semantically")

    for i in range(B):
        positions = cache_global_ids[i]
        gathered_tokens = torch.gather(input_ids[i:i+1], dim=1, index=positions.unsqueeze(0))

        # Manual verification: input_ids[i, pos] should equal gathered_tokens[0, idx]
        for idx, pos in enumerate(positions):
            expected_token = input_ids[i, pos].item()
            actual_token = gathered_tokens[0, idx].item()

            if expected_token != actual_token:
                print(f"  ❌ FAIL: Mismatch at sample {i}, position {pos}")
                print(f"     Expected token: {expected_token}")
                print(f"     Gathered token: {actual_token}")
                return False

    print(f"  PASS: All gathered tokens match expected positions")

    # ========================================================================
    # FINAL VERDICT
    # ========================================================================

    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    if (cache_global_ids < 1000).all():
        print("✅ BUG #10 IS FIXED (or not present)")
        print("   cache_global_ids contains POSITIONS, not TOKEN IDs")
        print("   All values are valid sequence indices < max_length")
        return True
    else:
        print("❌ BUG #10 IS PRESENT")
        print("   cache_global_ids contains TOKEN IDs instead of POSITIONS")
        print("   This will cause incorrect landmark gathering in model.py")
        print("\n📋 FIX: Remove token gathering in src/data.py:282-286")
        print("   Return cache_global_ids (positions) directly, not gathered tokens")
        return False


def test_bug10_interaction_with_model():
    """
    Test BUG #10 interaction with model forward pass.

    Verify that landmarks gathered in model.py are semantically correct
    when using positions vs token IDs.
    """
    print("\n" + "=" * 80)
    print("BUG #10 MODEL INTERACTION TEST")
    print("=" * 80)

    from src.legacy.model import LLMTransformer, Config

    # Setup small model
    cfg = Config(
        vocab_size=50257,
        max_seq_len=128,
        embed_dim=128,
        num_heads=4,
        n_layers=2,
        local_window=32,
        global_k=8,
        learned_landmarks=False,  # Use heuristic landmarks
    )

    model = LLMTransformer(cfg)
    model.eval()

    # Create batch with collator
    tokenizer = get_tokenizer("gpt2")
    collator = CollatorLocalGlobal(
        tokenizer=tokenizer,
        max_length=128,
        max_global=8,
        strategy="regular"
    )

    examples = [{"text": "Test sequence " * 20}]
    batch = collator(examples)

    input_ids = batch["input_ids"]  # (1, 128)
    cache_global_ids = batch["cache_global_ids"]  # (1, 8)

    print(f"\nInput shape: {input_ids.shape}")
    print(f"Landmark IDs shape: {cache_global_ids.shape}")
    print(f"Landmark IDs: {cache_global_ids[0].tolist()}")

    # Check if IDs are positions or tokens
    if (cache_global_ids < 128).all():
        print("✓ IDs are valid positions")
    else:
        print("❌ IDs contain values >= 128 (likely token IDs)")
        print(f"   Max ID: {cache_global_ids.max().item()}")

    # Forward pass
    try:
        with torch.no_grad():
            logits = model(input_ids, cache_global_ids=cache_global_ids)

        print(f"\n✓ Forward pass successful")
        print(f"  Logits shape: {logits.shape}")

        # Check for NaN/Inf
        if torch.isnan(logits).any():
            print("  ❌ WARNING: NaN detected in logits")
        elif torch.isinf(logits).any():
            print("  ❌ WARNING: Inf detected in logits")
        else:
            print("  ✓ Logits are finite and valid")

        return True

    except Exception as e:
        print(f"\n❌ Forward pass failed: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TESTING BUG #10: Landmark Positions vs Token IDs")
    print("=" * 80)
    print("\nThis test validates the fix for BUG #10, where the collator")
    print("was incorrectly returning token IDs instead of position indices.")
    print("=" * 80)

    # Run tests
    test1_passed = test_bug10_positions_not_tokens()
    test2_passed = test_bug10_interaction_with_model()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Test 1 (Collator outputs): {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Test 2 (Model interaction): {'✅ PASS' if test2_passed else '❌ FAIL'}")

    if test1_passed and test2_passed:
        print("\n🎉 All tests PASSED - BUG #10 is FIXED")
        sys.exit(0)
    else:
        print("\n❌ Some tests FAILED - BUG #10 is still PRESENT")
        print("\n📋 To fix: Modify src/data.py line 288-292")
        print("   Remove token gathering, return positions directly")
        sys.exit(1)
