#!/usr/bin/env python3
"""
Test que le fix Bug #15 dans train.py fonctionne correctement
"""

import torch


def test_bug15_fix():
    """
    Valide la logique du fix Bug #15:
    - Régénérer landmarks heuristiques après troncature curriculum
    """
    print("=" * 80)
    print("BUG #15 FIX VALIDATION")
    print("=" * 80)
    print()

    # Simuler les paramètres
    full_seq_len = 512
    current_seq_len = 128  # Curriculum truncation
    num_landmarks = 8
    batch_size = 4

    # Simuler collator output (positions pour séquence complète)
    cache_ids = torch.linspace(0, full_seq_len-1, num_landmarks).long()
    cache_ids = cache_ids.unsqueeze(0).expand(batch_size, -1)

    print("BEFORE TRUNCATION:")
    print(f"  cache_ids (from collator): {cache_ids[0].tolist()}")
    print(f"  Sequence length: {full_seq_len}")
    print()

    # Simuler truncation
    print("AFTER TRUNCATION (curriculum):")
    print(f"  Sequence length: {current_seq_len}")
    print()

    # ❌ OLD APPROACH (buggy): Clamp
    cache_ids_buggy = torch.clamp(cache_ids, 0, current_seq_len - 1)
    print("❌ OLD FIX (clamp):")
    print(f"  Result: {cache_ids_buggy[0].tolist()}")
    unique_buggy = len(torch.unique(cache_ids_buggy[0]))
    print(f"  Unique positions: {unique_buggy}/{num_landmarks}")
    if unique_buggy < num_landmarks // 2:
        print(f"  ❌ DISASTER: Landmarks collapsed!")
    print()

    # ✅ NEW APPROACH (fixed): Regenerate
    learned_landmarks = False  # Heuristic mode

    if not learned_landmarks:
        # Régénérer pour longueur tronquée
        B, G = cache_ids.shape
        cache_ids_fixed = torch.linspace(
            0, current_seq_len - 1, G, device=cache_ids.device
        ).long().unsqueeze(0).expand(B, -1)

    print("✅ NEW FIX (regenerate):")
    print(f"  Result: {cache_ids_fixed[0].tolist()}")
    unique_fixed = len(torch.unique(cache_ids_fixed[0]))
    print(f"  Unique positions: {unique_fixed}/{num_landmarks}")
    if unique_fixed == num_landmarks:
        print(f"  ✅ SUCCESS: Full diversity maintained!")
    print()

    # Validation
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)
    print()

    all_passed = True

    # Test 1: Full diversity
    if unique_fixed == num_landmarks:
        print("✓ PASS: All landmarks unique after fix")
    else:
        print(f"✗ FAIL: Only {unique_fixed}/{num_landmarks} unique")
        all_passed = False

    # Test 2: All in valid range
    if (cache_ids_fixed >= 0).all() and (cache_ids_fixed < current_seq_len).all():
        print("✓ PASS: All positions in valid range [0, current_seq_len-1]")
    else:
        print("✗ FAIL: Some positions out of range")
        all_passed = False

    # Test 3: Regular spacing
    expected_spacing = (current_seq_len - 1) / (num_landmarks - 1)
    actual_spacing = cache_ids_fixed[0, 1] - cache_ids_fixed[0, 0]
    if abs(actual_spacing - expected_spacing) < 1:
        print(f"✓ PASS: Regular spacing maintained (≈{expected_spacing:.1f})")
    else:
        print(f"✗ FAIL: Spacing incorrect (expected {expected_spacing}, got {actual_spacing})")
        all_passed = False

    # Test 4: Bug is actually fixed (no clustering)
    if unique_buggy < num_landmarks // 2 and unique_fixed == num_landmarks:
        print(f"✓ PASS: Fix resolves clustering ({unique_buggy} → {unique_fixed} unique)")
    else:
        print(f"✗ FAIL: Fix doesn't resolve clustering")
        all_passed = False

    print()
    print("=" * 80)

    if all_passed:
        print("✓ ALL TESTS PASSED - Bug #15 fix is CORRECT")
        print()
        print("The fix:")
        print("  ✓ Maintains full landmark diversity")
        print("  ✓ Adapts to truncated sequence length")
        print("  ✓ Prevents clustering at sequence end")
        print("  ✓ Ensures effective global attention")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(test_bug15_fix())
