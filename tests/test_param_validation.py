#!/usr/bin/env python3
"""
Test de validation des paramètres de génération.
Vérifie que generate.py rejette correctement les paramètres invalides.
"""

import subprocess
import sys
import os

# Ajouter le chemin du projet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_validation(description, args, should_fail=True):
    """Test un ensemble de paramètres."""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"{'='*80}")
    print(f"Args: {' '.join(args)}")

    cmd = ["python3", "scripts/generate.py"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)

    if should_fail:
        if result.returncode != 0:
            print(f"✓ PASSED - Script rejected invalid parameters")
            if "PARAMETER VALIDATION ERRORS" in result.stdout:
                print(f"✓ Validation error message displayed correctly")
                print(f"\nOutput:\n{result.stdout[:500]}")
            return True
        else:
            print(f"❌ FAILED - Script should have rejected parameters but didn't")
            return False
    else:
        if result.returncode == 0 or "PARAMETER VALIDATION ERRORS" not in result.stdout:
            print(f"✓ PASSED - Script accepted valid parameters")
            return True
        else:
            print(f"❌ FAILED - Script rejected valid parameters")
            print(f"\nOutput:\n{result.stdout[:500]}")
            return False

def main():
    print("=" * 80)
    print("PARAMETER VALIDATION TESTS")
    print("=" * 80)

    results = []

    # Test 1: Temperature négative
    results.append(test_validation(
        "Temperature négative (-0.5)",
        [
            "--checkpoint", "dummy.pt",
            "--config", "config/config.yaml",
            "--temperature", "-0.5",
            "--max-tokens", "10"
        ],
        should_fail=True
    ))

    # Test 2: top_k = 0
    results.append(test_validation(
        "top_k = 0 (doit être > 0)",
        [
            "--checkpoint", "dummy.pt",
            "--config", "config/config.yaml",
            "--top-k", "0",
            "--max-tokens", "10"
        ],
        should_fail=True
    ))

    # Test 3: top_k négatif
    results.append(test_validation(
        "top_k négatif (-5)",
        [
            "--checkpoint", "dummy.pt",
            "--config", "config/config.yaml",
            "--top-k", "-5",
            "--max-tokens", "10"
        ],
        should_fail=True
    ))

    # Test 4: top_p > 1
    results.append(test_validation(
        "top_p > 1 (1.5)",
        [
            "--checkpoint", "dummy.pt",
            "--config", "config/config.yaml",
            "--top-p", "1.5",
            "--max-tokens", "10"
        ],
        should_fail=True
    ))

    # Test 5: top_p < 0
    results.append(test_validation(
        "top_p < 0 (-0.1)",
        [
            "--checkpoint", "dummy.pt",
            "--config", "config/config.yaml",
            "--top-p", "-0.1",
            "--max-tokens", "10"
        ],
        should_fail=True
    ))

    # Test 6: max_tokens = 0
    results.append(test_validation(
        "max_tokens = 0",
        [
            "--checkpoint", "dummy.pt",
            "--config", "config/config.yaml",
            "--max-tokens", "0"
        ],
        should_fail=True
    ))

    # Test 7: max_tokens négatif
    results.append(test_validation(
        "max_tokens négatif (-10)",
        [
            "--checkpoint", "dummy.pt",
            "--config", "config/config.yaml",
            "--max-tokens", "-10"
        ],
        should_fail=True
    ))

    # Test 8: Multiples erreurs simultanées
    results.append(test_validation(
        "Multiples erreurs (temp=-1, top_k=0, top_p=2, max_tokens=-5)",
        [
            "--checkpoint", "dummy.pt",
            "--config", "config/config.yaml",
            "--temperature", "-1",
            "--top-k", "0",
            "--top-p", "2",
            "--max-tokens", "-5"
        ],
        should_fail=True
    ))

    # Résumé
    print("\n" + "=" * 80)
    print("RÉSUMÉ DES TESTS")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Tests réussis: {passed}/{total}")

    if passed == total:
        print("\n✓ TOUS LES TESTS SONT PASSÉS!")
        print("La validation des paramètres fonctionne correctement.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) ont échoué")
        return 1

if __name__ == "__main__":
    sys.exit(main())
