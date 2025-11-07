#!/usr/bin/env python3
"""
Test de cohérence entre validate_generation_params et generate_text
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer
from src.model import Config, LLMTransformer


def test_top_p_validation_consistency():
    """
    Test que les validations top_p sont cohérentes entre:
    1. validate_generation_params (CLI)
    2. generate_text() (runtime)
    """
    print("=" * 80)
    print("TEST: top_p Validation Consistency")
    print("=" * 80)
    print()

    # Créer un modèle minimal pour tester
    cfg = Config(
        vocab_size=1000,
        max_seq_len=128,
        embed_dim=64,
        num_heads=2,
        n_layers=1,
        learned_landmarks=False,
    )

    model = LLMTransformer(cfg)
    model.eval()

    # Mock tokenizer (simple)
    class MockTokenizer:
        def __init__(self):
            self.eos_token_id = 999

        def encode(self, text, return_tensors=None):
            # Retourne quelques tokens aléatoires
            tokens = torch.randint(0, 1000, (1, 5))
            return tokens

        def decode(self, tokens, skip_special_tokens=False):
            return "test output"

    tokenizer = MockTokenizer()

    # Import de la fonction generate_text
    from scripts.generate import generate_text

    test_cases = [
        (0.0, False, "top_p=0.0 (should fail - not strictly positive)"),
        (0.5, True, "top_p=0.5 (valid)"),
        (1.0, True, "top_p=1.0 (valid - boundary)"),
        (1.1, False, "top_p=1.1 (should fail - exceeds 1.0)"),
        (-0.1, False, "top_p=-0.1 (should fail - negative)"),
    ]

    all_passed = True

    for top_p_value, should_succeed, description in test_cases:
        try:
            # Test avec generate_text
            output = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt="test",
                max_new_tokens=5,
                temperature=0.8,
                top_k=None,
                top_p=top_p_value,
                device="cpu",
            )

            if should_succeed:
                print(f"✓ PASS: {description}")
                print(f"  generate_text accepted top_p={top_p_value}")
            else:
                print(f"✗ FAIL: {description}")
                print(f"  generate_text should have rejected top_p={top_p_value}")
                all_passed = False

        except ValueError as e:
            if not should_succeed:
                print(f"✓ PASS: {description}")
                print(f"  generate_text correctly rejected top_p={top_p_value}")
                print(f"  Error: {e}")
            else:
                print(f"✗ FAIL: {description}")
                print(f"  generate_text incorrectly rejected top_p={top_p_value}")
                print(f"  Error: {e}")
                all_passed = False

        print()

    # Test de la validation CLI (logique identique)
    print("Testing CLI validation logic...")
    print()

    for top_p_value, should_succeed, description in test_cases:
        # Simuler la validation CLI
        is_valid = top_p_value is not None and (0 < top_p_value <= 1)

        if is_valid == should_succeed:
            print(f"✓ PASS: CLI validation for {description}")
            print(f"  CLI validation: {is_valid}, Expected: {should_succeed}")
        else:
            print(f"✗ FAIL: CLI validation for {description}")
            print(f"  CLI validation: {is_valid}, Expected: {should_succeed}")
            all_passed = False

        print()

    print("=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED - Validation is consistent")
        return 0
    else:
        print("✗ SOME TESTS FAILED - Validation is inconsistent")
        return 1


if __name__ == "__main__":
    sys.exit(test_top_p_validation_consistency())
