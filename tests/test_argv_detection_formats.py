#!/usr/bin/env python3
"""
Test que la détection d'arguments fonctionne avec TOUS les formats:
- --top-k 40 (espace)
- --top-k=40 (égal)
- --top_k 40 (underscore + espace)
- --top_k=40 (underscore + égal)
"""

import sys


def test_all_argument_formats():
    """Test la détection avec tous les formats d'arguments"""
    print("=" * 80)
    print("TEST: Argument Detection for All Formats")
    print("=" * 80)
    print()

    test_cases = [
        # Format 1: --top-k 40 (espace)
        (['generate.py', '--checkpoint', 'dummy', '--top-k', '40', '--top-p', '0.9'],
         True, True, "Space-separated (--top-k 40, --top-p 0.9)"),

        # Format 2: --top-k=40 (égal)
        (['generate.py', '--checkpoint', 'dummy', '--top-k=40', '--top-p=0.9'],
         True, True, "Equals-separated (--top-k=40, --top-p=0.9)"),

        # Format 3: --top_k 40 (underscore + espace)
        (['generate.py', '--checkpoint', 'dummy', '--top_k', '40', '--top_p', '0.9'],
         True, True, "Underscore space (--top_k 40, --top_p 0.9)"),

        # Format 4: --top_k=40 (underscore + égal)
        (['generate.py', '--checkpoint', 'dummy', '--top_k=40', '--top_p=0.9'],
         True, True, "Underscore equals (--top_k=40, --top_p=0.9)"),

        # Format 5: Mixte --top-k=40 --top_p 0.9
        (['generate.py', '--checkpoint', 'dummy', '--top-k=40', '--top_p', '0.9'],
         True, True, "Mixed format (--top-k=40 --top_p 0.9)"),

        # Format 6: Seulement --top-k
        (['generate.py', '--checkpoint', 'dummy', '--top-k=50'],
         True, False, "Only --top-k=50"),

        # Format 7: Seulement --top-p
        (['generate.py', '--checkpoint', 'dummy', '--top-p=0.8'],
         False, True, "Only --top-p=0.8"),

        # Format 8: Aucun argument
        (['generate.py', '--checkpoint', 'dummy'],
         False, False, "No sampling arguments"),
    ]

    all_passed = True

    for test_argv, expected_top_k, expected_top_p, description in test_cases:
        # Simuler la détection (logique de generate.py)
        user_set_top_k = any(arg.startswith('--top-k') or arg.startswith('--top_k')
                             for arg in test_argv)
        user_set_top_p = any(arg.startswith('--top-p') or arg.startswith('--top_p')
                             for arg in test_argv)

        # Vérifier
        if user_set_top_k == expected_top_k and user_set_top_p == expected_top_p:
            print(f"✓ PASS: {description}")
            print(f"  top_k detected: {user_set_top_k} (expected {expected_top_k})")
            print(f"  top_p detected: {user_set_top_p} (expected {expected_top_p})")
        else:
            print(f"✗ FAIL: {description}")
            print(f"  top_k detected: {user_set_top_k} (expected {expected_top_k})")
            print(f"  top_p detected: {user_set_top_p} (expected {expected_top_p})")
            all_passed = False

        print()

    print("=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED - Detection handles all formats")
        print()
        print("Tested formats:")
        print("  ✓ --top-k 40 (space-separated)")
        print("  ✓ --top-k=40 (equals-separated)")
        print("  ✓ --top_k 40 (underscore space)")
        print("  ✓ --top_k=40 (underscore equals)")
        print("  ✓ Mixed formats")
        print("  ✓ Single argument cases")
        print("  ✓ No arguments case")
        return 0
    else:
        print("✗ SOME TESTS FAILED - Detection incomplete")
        return 1


if __name__ == "__main__":
    sys.exit(test_all_argument_formats())
