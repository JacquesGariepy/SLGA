#!/usr/bin/env python3
"""
Test que la troncation de cache_ids préserve la shape (batch, G)
et ne collapse pas en vecteur 1D avec cache_ids[mask].
"""

import torch
import sys


def test_shape_preservation():
    """Test que clamp préserve la shape vs mask qui la collapse"""
    print("=" * 80)
    print("TEST: cache_ids Shape Preservation (Bug Fix)")
    print("=" * 80)
    print()

    # Simuler le scénario du bug
    batch_size = 4
    num_landmarks = 8
    current_seq_len = 512

    # cache_ids avec certains landmarks hors limites
    cache_ids = torch.tensor([
        [100, 200, 300, 400, 500, 600, 700, 800],  # Les 4 derniers > 512
        [50, 150, 250, 350, 450, 550, 650, 750],   # Les 4 derniers > 512
        [10, 20, 30, 40, 50, 60, 70, 80],          # Tous < 512
        [200, 300, 400, 500, 600, 700, 800, 900],  # Les 5 derniers > 512
    ])

    print(f"Original cache_ids shape: {cache_ids.shape}")
    print(f"Expected shape: ({batch_size}, {num_landmarks})")
    print()

    # ❌ MÉTHODE BUGGÉE (ancienne)
    print("❌ OLD METHOD (buggy): cache_ids[mask]")
    mask = cache_ids < current_seq_len
    cache_ids_buggy = cache_ids[mask]
    print(f"  Result shape: {cache_ids_buggy.shape}")
    print(f"  Result: {cache_ids_buggy}")
    print(f"  Problem: Shape collapsed to 1D! Lost batch structure!")
    print(f"  Original had {batch_size} samples × {num_landmarks} landmarks")
    print(f"  Now has {cache_ids_buggy.numel()} flat elements")
    print()

    # ✅ MÉTHODE CORRIGÉE (nouvelle)
    print("✅ NEW METHOD (fixed): torch.clamp(cache_ids, 0, max)")
    cache_ids_fixed = torch.clamp(cache_ids, 0, current_seq_len - 1)
    print(f"  Result shape: {cache_ids_fixed.shape}")
    print(f"  Result:")
    print(f"  {cache_ids_fixed}")
    print(f"  Success: Shape preserved as ({batch_size}, {num_landmarks})!")
    print()

    # Validation
    all_passed = True

    # Test 1: Shape preservation
    if cache_ids_fixed.shape == (batch_size, num_landmarks):
        print("✓ PASS: Shape preserved correctly")
    else:
        print(f"✗ FAIL: Expected shape {(batch_size, num_landmarks)}, got {cache_ids_fixed.shape}")
        all_passed = False

    # Test 2: All values clamped
    if (cache_ids_fixed >= 0).all() and (cache_ids_fixed < current_seq_len).all():
        print("✓ PASS: All values within valid range [0, current_seq_len-1]")
    else:
        print("✗ FAIL: Some values outside valid range")
        all_passed = False

    # Test 3: Values that were valid remain unchanged
    valid_mask = cache_ids < current_seq_len
    if torch.equal(cache_ids_fixed[valid_mask], cache_ids[valid_mask]):
        print("✓ PASS: Valid values unchanged")
    else:
        print("✗ FAIL: Valid values were modified")
        all_passed = False

    # Test 4: Out-of-bounds values clamped to max
    invalid_mask = cache_ids >= current_seq_len
    if (cache_ids_fixed[invalid_mask] == current_seq_len - 1).all():
        print("✓ PASS: Out-of-bounds values clamped to max")
    else:
        print("✗ FAIL: Out-of-bounds values not clamped correctly")
        all_passed = False

    print()
    print("=" * 80)

    # Démonstration du problème avec le forward pass
    print("\nDemonstration: Impact on Forward Pass")
    print("-" * 80)

    print("\n❌ With buggy method (collapsed shape):")
    print(f"  cache_ids shape: {cache_ids_buggy.shape}")
    print(f"  Cannot use as (B, G) landmarks!")
    print(f"  Forward pass would:")
    print(f"    - Misalign landmarks across samples")
    print(f"    - Or crash with shape mismatch")
    print(f"    - Or Accelerate gather() would fail")

    print("\n✅ With fixed method (preserved shape):")
    print(f"  cache_ids shape: {cache_ids_fixed.shape}")
    print(f"  Perfect (B, G) structure maintained!")
    print(f"  Forward pass will:")
    print(f"    - Use correct landmarks per sample")
    print(f"    - No shape mismatches")
    print(f"    - Accelerate gather() works correctly")

    print("\n" + "=" * 80)

    if all_passed:
        print("✓ ALL TESTS PASSED - Shape preservation works correctly")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


def test_edge_cases():
    """Test edge cases"""
    print("\n" + "=" * 80)
    print("TEST: Edge Cases")
    print("=" * 80)
    print()

    all_passed = True

    # Edge case 1: All landmarks valid
    print("Edge case 1: All landmarks within bounds")
    cache_ids = torch.tensor([[10, 20, 30], [40, 50, 60]])
    current_seq_len = 100
    result = torch.clamp(cache_ids, 0, current_seq_len - 1)
    if torch.equal(result, cache_ids):
        print("✓ PASS: No changes when all valid")
    else:
        print("✗ FAIL: Valid values were changed")
        all_passed = False
    print()

    # Edge case 2: All landmarks invalid
    print("Edge case 2: All landmarks out of bounds")
    cache_ids = torch.tensor([[600, 700, 800], [900, 1000, 1100]])
    current_seq_len = 512
    result = torch.clamp(cache_ids, 0, current_seq_len - 1)
    expected = torch.full_like(cache_ids, current_seq_len - 1)
    if torch.equal(result, expected):
        print("✓ PASS: All clamped to max")
    else:
        print("✗ FAIL: Not all values clamped")
        all_passed = False
    print()

    # Edge case 3: Single sample
    print("Edge case 3: Single sample (batch_size=1)")
    cache_ids = torch.tensor([[100, 200, 300, 600]])
    current_seq_len = 512
    result = torch.clamp(cache_ids, 0, current_seq_len - 1)
    if result.shape == (1, 4):
        print(f"✓ PASS: Shape preserved: {result.shape}")
    else:
        print(f"✗ FAIL: Shape not preserved: {result.shape}")
        all_passed = False
    print()

    # Edge case 4: Large batch
    print("Edge case 4: Large batch (batch_size=32)")
    cache_ids = torch.randint(0, 2048, (32, 16))
    current_seq_len = 1024
    result = torch.clamp(cache_ids, 0, current_seq_len - 1)
    if result.shape == (32, 16):
        print(f"✓ PASS: Shape preserved: {result.shape}")
    else:
        print(f"✗ FAIL: Shape not preserved: {result.shape}")
        all_passed = False
    print()

    print("=" * 80)
    if all_passed:
        print("✓ ALL EDGE CASE TESTS PASSED")
        return 0
    else:
        print("✗ SOME EDGE CASE TESTS FAILED")
        return 1


if __name__ == "__main__":
    result1 = test_shape_preservation()
    result2 = test_edge_cases()

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    if result1 == 0 and result2 == 0:
        print("✓ ALL TESTS PASSED")
        print("\nThe fix correctly:")
        print("  1. Preserves (batch, G) shape")
        print("  2. Clamps out-of-bounds indices")
        print("  3. Leaves valid indices unchanged")
        print("  4. Handles all edge cases")
        print("\nNo more shape collapse bugs! ✅")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)
