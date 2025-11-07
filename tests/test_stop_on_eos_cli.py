#!/usr/bin/env python3
"""
Test script pour valider les options --stop-on-eos / --no-stop-on-eos
"""

import sys
import os
import argparse

# Simuler la définition d'arguments de generate.py
parser = argparse.ArgumentParser(description="Test EOS CLI args")

# Reproduire la définition exacte de generate.py
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


def test_cli_behavior():
    """Test les différentes combinaisons d'arguments"""
    print("=" * 80)
    print("TEST: --stop-on-eos CLI Behavior")
    print("=" * 80)
    print()

    test_cases = [
        ([], True, "Default (no args)"),
        (["--stop-on-eos"], True, "Explicit --stop-on-eos"),
        (["--no-stop-on-eos"], False, "Explicit --no-stop-on-eos"),
    ]

    all_passed = True

    for args_list, expected_value, description in test_cases:
        try:
            args = parser.parse_args(args_list)
            actual_value = args.stop_on_eos

            if actual_value == expected_value:
                print(f"✓ PASS: {description}")
                print(f"  Args: {args_list if args_list else '(none)'}")
                print(f"  stop_on_eos = {actual_value} (expected {expected_value})")
            else:
                print(f"✗ FAIL: {description}")
                print(f"  Args: {args_list if args_list else '(none)'}")
                print(f"  stop_on_eos = {actual_value} (expected {expected_value})")
                all_passed = False

            print()
        except Exception as e:
            print(f"✗ FAIL: {description}")
            print(f"  Error: {e}")
            print()
            all_passed = False

    # Test que --stop-on-eos et --no-stop-on-eos sont mutuellement exclusifs
    print("Testing mutually exclusive behavior...")
    try:
        args = parser.parse_args(["--stop-on-eos", "--no-stop-on-eos"])
        print("✗ FAIL: Should have raised error for conflicting args")
        all_passed = False
    except SystemExit:
        print("✓ PASS: Correctly rejects conflicting --stop-on-eos --no-stop-on-eos")
    print()

    print("=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(test_cli_behavior())
