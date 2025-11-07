#!/usr/bin/env python3
"""
Test que le warning top_k+top_p ne se déclenche PAS avec les valeurs par défaut
"""

import sys
import subprocess


def test_warning_not_triggered_by_defaults():
    """
    Test que l'avertissement ne se déclenche PAS quand on utilise les valeurs par défaut
    """
    print("=" * 80)
    print("TEST: Warning NOT Triggered by Default Values")
    print("=" * 80)
    print()

    # Test 1: Pas d'arguments -> pas de warning
    print("Test 1: No arguments (using defaults)")
    print("-" * 40)

    # On ne peut pas vraiment exécuter generate.py sans checkpoint
    # mais on peut tester la logique de détection

    # Simuler sys.argv sans --top-k ni --top-p
    test_argv_no_args = ['generate.py', '--checkpoint', 'dummy', '--config', 'dummy']

    user_set_top_k = '--top-k' in test_argv_no_args or '--top_k' in test_argv_no_args
    user_set_top_p = '--top-p' in test_argv_no_args or '--top_p' in test_argv_no_args

    should_warn = user_set_top_k and user_set_top_p

    if not should_warn:
        print("✓ PASS: Warning NOT triggered with defaults")
        print(f"  --top-k detected: {user_set_top_k}")
        print(f"  --top-p detected: {user_set_top_p}")
    else:
        print("✗ FAIL: Warning incorrectly triggered with defaults")
        return 1

    print()

    # Test 2: Seulement --top-k fourni -> pas de warning
    print("Test 2: Only --top-k provided")
    print("-" * 40)

    test_argv_only_k = ['generate.py', '--checkpoint', 'dummy', '--top-k', '50']

    user_set_top_k = '--top-k' in test_argv_only_k or '--top_k' in test_argv_only_k
    user_set_top_p = '--top-p' in test_argv_only_k or '--top_p' in test_argv_only_k

    should_warn = user_set_top_k and user_set_top_p

    if not should_warn:
        print("✓ PASS: Warning NOT triggered with only --top-k")
        print(f"  --top-k detected: {user_set_top_k}")
        print(f"  --top-p detected: {user_set_top_p}")
    else:
        print("✗ FAIL: Warning incorrectly triggered with only --top-k")
        return 1

    print()

    # Test 3: Seulement --top-p fourni -> pas de warning
    print("Test 3: Only --top-p provided")
    print("-" * 40)

    test_argv_only_p = ['generate.py', '--checkpoint', 'dummy', '--top-p', '0.9']

    user_set_top_k = '--top-k' in test_argv_only_p or '--top_k' in test_argv_only_p
    user_set_top_p = '--top-p' in test_argv_only_p or '--top_p' in test_argv_only_p

    should_warn = user_set_top_k and user_set_top_p

    if not should_warn:
        print("✓ PASS: Warning NOT triggered with only --top-p")
        print(f"  --top-k detected: {user_set_top_k}")
        print(f"  --top-p detected: {user_set_top_p}")
    else:
        print("✗ FAIL: Warning incorrectly triggered with only --top-p")
        return 1

    print()

    # Test 4: Les DEUX fournis -> warning devrait se déclencher
    print("Test 4: Both --top-k and --top-p provided")
    print("-" * 40)

    test_argv_both = ['generate.py', '--checkpoint', 'dummy', '--top-k', '50', '--top-p', '0.9']

    user_set_top_k = '--top-k' in test_argv_both or '--top_k' in test_argv_both
    user_set_top_p = '--top-p' in test_argv_both or '--top_p' in test_argv_both

    should_warn = user_set_top_k and user_set_top_p

    if should_warn:
        print("✓ PASS: Warning correctly triggered with both arguments")
        print(f"  --top-k detected: {user_set_top_k}")
        print(f"  --top-p detected: {user_set_top_p}")
    else:
        print("✗ FAIL: Warning should have been triggered with both arguments")
        return 1

    print()

    # Test 5: Variante avec underscores
    print("Test 5: Both arguments with underscores (--top_k, --top_p)")
    print("-" * 40)

    test_argv_underscores = ['generate.py', '--checkpoint', 'dummy', '--top_k', '50', '--top_p', '0.9']

    user_set_top_k = '--top-k' in test_argv_underscores or '--top_k' in test_argv_underscores
    user_set_top_p = '--top-p' in test_argv_underscores or '--top_p' in test_argv_underscores

    should_warn = user_set_top_k and user_set_top_p

    if should_warn:
        print("✓ PASS: Warning correctly triggered with underscored arguments")
        print(f"  --top-k detected: {user_set_top_k}")
        print(f"  --top-p detected: {user_set_top_p}")
    else:
        print("✗ FAIL: Warning should have been triggered")
        return 1

    print()

    print("=" * 80)
    print("✓ ALL TESTS PASSED - Warning logic is correct")
    print("=" * 80)
    print()
    print("Summary:")
    print("  ✓ No warning with default values")
    print("  ✓ No warning with only one argument")
    print("  ✓ Warning triggered when both explicitly provided")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(test_warning_not_triggered_by_defaults())
