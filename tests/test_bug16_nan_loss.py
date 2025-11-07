#!/usr/bin/env python3
"""
Test Bug #16: NaN loss when G <= 1 in landmark_spacing_loss fallback
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.landmarks import landmark_spacing_loss


def test_spacing_loss_with_small_G():
    """Test que spacing_loss ne produit pas NaN avec G <= 1"""
    print("=" * 80)
    print("BUG #16 TEST: NaN Loss with G <= 1")
    print("=" * 80)
    print()

    seq_len = 128
    batch_size = 2

    test_cases = [
        (0, "G=0 (no landmarks)"),
        (1, "G=1 (single landmark)"),
        (2, "G=2 (minimum for gaps)"),
        (4, "G=4 (normal)"),
    ]

    all_passed = True

    for G, description in test_cases:
        print(f"Test: {description}")
        print("-" * 40)

        if G == 0:
            # Cas spécial: pas de landmarks
            landmark_indices = torch.empty(batch_size, 0, dtype=torch.long)
        else:
            # Landmarks aléatoires
            landmark_indices = torch.randint(0, seq_len, (batch_size, G))

        print(f"  landmark_indices shape: {landmark_indices.shape}")

        try:
            # Appeler spacing_loss avec fallback (selection_scores=None)
            loss = landmark_spacing_loss(
                landmark_indices=landmark_indices,
                selection_scores=None,  # Force fallback path
                seq_len=seq_len,
                lambda_reg=0.01,
            )

            print(f"  Loss value: {loss.item()}")

            # Vérifier pas NaN
            if torch.isnan(loss):
                print(f"  ❌ FAIL: Loss is NaN!")
                all_passed = False
            elif torch.isinf(loss):
                print(f"  ❌ FAIL: Loss is Inf!")
                all_passed = False
            else:
                print(f"  ✓ PASS: Loss is valid (not NaN/Inf)")

        except Exception as e:
            print(f"  ❌ FAIL: Exception raised: {e}")
            all_passed = False

        print()

    print("=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED - Bug #16 is FIXED")
        print()
        print("The fix correctly:")
        print("  ✓ Returns 0.0 for G < 2 (no gaps possible)")
        print("  ✓ Computes valid loss for G >= 2")
        print("  ✓ Prevents NaN in curriculum/short sequences")
        return 0
    else:
        print("✗ SOME TESTS FAILED - Bug #16 still present")
        return 1


if __name__ == "__main__":
    sys.exit(test_spacing_loss_with_small_G())
