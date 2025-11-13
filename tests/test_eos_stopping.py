#!/usr/bin/env python3
"""
Test EOS Token Stopping Feature

Vérifie que l'arrêt sur EOS fonctionne correctement:
1. Arrêt précoce quand tous les samples génèrent EOS
2. Désactivation possible avec stop_on_eos=False
3. Gestion correcte du batch (arrêt quand TOUS finissent)
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.legacy.model import Config, LLMTransformer


def test_eos_stopping_basic():
    """Test 1: Arrêt précoce sur EOS basique"""
    print("\n" + "="*60)
    print("TEST 1: Arrêt précoce sur EOS (single sample)")
    print("="*60)

    # Configuration minimale
    cfg = Config(
        vocab_size=50257,
        embed_dim=128,
        num_heads=4,
        n_layers=2,
        max_seq_len=512,
        global_k=8,
    )

    model = LLMTransformer(cfg)
    model.eval()

    # Input simple
    input_ids = torch.randint(0, 100, (1, 10))  # (1, 10)

    # Générer avec stop_on_eos=True
    print(f"Input shape: {input_ids.shape}")
    print(f"Max new tokens: 100")

    # Override forward pour forcer EOS après quelques tokens
    original_forward = model.forward
    counter = [0]

    def mock_forward_with_eos(input_ids, cache_global_ids=None, return_aux=False):
        counter[0] += 1
        logits = original_forward(input_ids, cache_global_ids, return_aux=False)

        # Après 5 étapes, forcer EOS token (50256)
        if counter[0] >= 5:
            logits[:, -1, :] = float('-inf')
            logits[:, -1, 50256] = 100.0  # Force EOS

        return logits

    model.forward = mock_forward_with_eos

    # Générer
    output = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.0,  # Greedy
        stop_on_eos=True,
        eos_token_id=50256,
    )

    tokens_generated = output.size(1) - input_ids.size(1)
    print(f"✅ Tokens générés: {tokens_generated} (devrait être ~5)")
    print(f"✅ Arrêt précoce: {tokens_generated < 100}")

    # Vérifier que le dernier token est EOS
    assert output[0, -1].item() == 50256, "Le dernier token devrait être EOS"
    assert tokens_generated < 100, "Devrait s'arrêter avant max_new_tokens"

    print("✅ TEST 1 PASSED")

    return output


def test_eos_stopping_disabled():
    """Test 2: Désactivation de l'arrêt sur EOS"""
    print("\n" + "="*60)
    print("TEST 2: Désactivation stop_on_eos")
    print("="*60)

    cfg = Config(
        vocab_size=50257,
        embed_dim=128,
        num_heads=4,
        n_layers=2,
        max_seq_len=512,
        global_k=8,
    )

    model = LLMTransformer(cfg)
    model.eval()

    input_ids = torch.randint(0, 100, (1, 10))

    # Générer avec stop_on_eos=False
    output = model.generate(
        input_ids,
        max_new_tokens=20,
        temperature=0.0,
        stop_on_eos=False,  # Désactivé
    )

    tokens_generated = output.size(1) - input_ids.size(1)
    print(f"Tokens générés: {tokens_generated}")

    # Devrait générer TOUS les tokens même si EOS apparaît
    assert tokens_generated == 20, f"Devrait générer exactement 20 tokens, got {tokens_generated}"

    print("✅ TEST 2 PASSED")

    return output


def test_eos_stopping_batch():
    """Test 3: Arrêt batch-aware (arrêt quand TOUS ont EOS)"""
    print("\n" + "="*60)
    print("TEST 3: Arrêt batch-aware (B=3)")
    print("="*60)

    cfg = Config(
        vocab_size=50257,
        embed_dim=128,
        num_heads=4,
        n_layers=2,
        max_seq_len=512,
        global_k=8,
    )

    model = LLMTransformer(cfg)
    model.eval()

    # Batch de 3 samples
    input_ids = torch.randint(0, 100, (3, 10))

    # Mock pour simuler EOS à différents moments
    original_forward = model.forward
    counter = [0]

    def mock_forward_batch_eos(input_ids, cache_global_ids=None, return_aux=False):
        counter[0] += 1
        logits = original_forward(input_ids, cache_global_ids, return_aux=False)

        # Sample 0: EOS après 3 tokens
        if counter[0] >= 3:
            logits[0, -1, :] = float('-inf')
            logits[0, -1, 50256] = 100.0

        # Sample 1: EOS après 5 tokens
        if counter[0] >= 5:
            logits[1, -1, :] = float('-inf')
            logits[1, -1, 50256] = 100.0

        # Sample 2: EOS après 7 tokens
        if counter[0] >= 7:
            logits[2, -1, :] = float('-inf')
            logits[2, -1, 50256] = 100.0

        return logits

    model.forward = mock_forward_batch_eos

    # Générer
    output = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.0,
        stop_on_eos=True,
        eos_token_id=50256,
    )

    tokens_generated = output.size(1) - input_ids.size(1)
    print(f"Tokens générés: {tokens_generated}")
    print(f"Dernier token batch[0]: {output[0, -1].item()}")
    print(f"Dernier token batch[1]: {output[1, -1].item()}")
    print(f"Dernier token batch[2]: {output[2, -1].item()}")

    # Devrait s'arrêter après ~7 tokens (quand le DERNIER sample finit)
    assert tokens_generated <= 10, f"Devrait s'arrêter après ~7 tokens, got {tokens_generated}"
    assert tokens_generated < 100, "Arrêt précoce devrait fonctionner"

    # Tous les samples devraient avoir EOS
    assert output[0, -1].item() == 50256, "Sample 0 devrait finir par EOS"
    assert output[1, -1].item() == 50256, "Sample 1 devrait finir par EOS"
    assert output[2, -1].item() == 50256, "Sample 2 devrait finir par EOS"

    print("✅ TEST 3 PASSED")

    return output


def test_custom_eos_token():
    """Test 4: Token EOS personnalisé"""
    print("\n" + "="*60)
    print("TEST 4: Token EOS personnalisé (token_id=42)")
    print("="*60)

    cfg = Config(
        vocab_size=50257,
        embed_dim=128,
        num_heads=4,
        n_layers=2,
        max_seq_len=512,
        global_k=8,
    )

    model = LLMTransformer(cfg)
    model.eval()

    input_ids = torch.randint(0, 100, (1, 10))

    # Mock pour forcer token 42
    original_forward = model.forward
    counter = [0]

    def mock_forward_custom_eos(input_ids, cache_global_ids=None, return_aux=False):
        counter[0] += 1
        logits = original_forward(input_ids, cache_global_ids, return_aux=False)

        # Force token 42 après 4 étapes
        if counter[0] >= 4:
            logits[:, -1, :] = float('-inf')
            logits[:, -1, 42] = 100.0

        return logits

    model.forward = mock_forward_custom_eos

    # Générer avec EOS personnalisé
    output = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.0,
        stop_on_eos=True,
        eos_token_id=42,  # Token personnalisé
    )

    tokens_generated = output.size(1) - input_ids.size(1)
    print(f"Tokens générés: {tokens_generated}")
    print(f"Dernier token: {output[0, -1].item()}")

    assert output[0, -1].item() == 42, "Devrait s'arrêter sur token 42"
    assert tokens_generated < 100, "Arrêt précoce devrait fonctionner"

    print("✅ TEST 4 PASSED")

    return output


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 TEST SUITE: EOS Token Stopping Feature")
    print("="*70)

    try:
        test_eos_stopping_basic()
        test_eos_stopping_disabled()
        test_eos_stopping_batch()
        test_custom_eos_token()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\n📝 Résumé:")
        print("  • Arrêt précoce sur EOS: ✅")
        print("  • Désactivation possible: ✅")
        print("  • Batch-aware (arrêt quand TOUS finissent): ✅")
        print("  • Token EOS personnalisé: ✅")
        print()

    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
