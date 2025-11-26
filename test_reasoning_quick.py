#!/usr/bin/env python3
"""
🧪 Test Rapide du Reasoning Model

Vérifie que tout fonctionne correctement.

Usage:
    python test_reasoning_quick.py
"""

import torch
import sys


def test_imports():
    """Test des imports."""
    print("1. Test des imports...")
    try:
        from train_reasoning_simple import (
            ReasoningModel,
            TrainConfig,
            SimpleTokenizer,
            SPECIAL_TOKENS,
        )
        print("   ✓ Imports OK")
        return True
    except ImportError as e:
        print(f"   ✗ Erreur d'import: {e}")
        return False


def test_model_creation():
    """Test de création du modèle."""
    print("2. Test création du modèle...")
    try:
        from train_reasoning_simple import ReasoningModel, TrainConfig

        config = TrainConfig.small()  # Version légère
        model = ReasoningModel(config)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ Modèle créé: {num_params / 1e6:.1f}M params")
        return model, config
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
        return None, None


def test_forward_pass(model, config):
    """Test du forward pass."""
    print("3. Test forward pass...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # Input factice
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)

        # Forward
        outputs = model(input_ids, targets)

        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)

        print(f"   ✓ Forward OK - Loss: {outputs['loss'].item():.4f}")
        return True
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
        return False


def test_generation(model, config):
    """Test de génération."""
    print("4. Test génération...")
    try:
        from train_reasoning_simple import SimpleTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        tokenizer = SimpleTokenizer()

        prompt = "Question: What is 2 + 2?\n<think>"
        input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.7,
            )

        generated = tokenizer.decode(output_ids[0].tolist())
        print(f"   ✓ Génération OK")
        print(f"   → Output: {generated[:100]}...")
        return True
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
        return False


def test_dataset():
    """Test du chargement de données."""
    print("5. Test données synthétiques...")
    try:
        from train_reasoning_simple import load_synthetic_data, CoTDataset, SimpleTokenizer

        data = load_synthetic_data(n=10)
        assert len(data) == 10

        tokenizer = SimpleTokenizer()
        dataset = CoTDataset(data, tokenizer, max_length=256)

        sample = dataset[0]
        assert "input_ids" in sample
        assert "targets" in sample

        print(f"   ✓ Dataset OK - {len(data)} exemples")
        return True
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
        return False


def test_training_step(model, config):
    """Test d'une étape d'entraînement."""
    print("6. Test étape d'entraînement...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Forward
        input_ids = torch.randint(0, config.vocab_size, (2, 32)).to(device)
        targets = torch.randint(0, config.vocab_size, (2, 32)).to(device)

        outputs = model(input_ids, targets)
        loss = outputs["loss"]

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"   ✓ Training step OK - Loss: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
        return False


def main():
    print("=" * 60)
    print("🧪 SLGA Reasoning Model - Tests Rapides")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    print("=" * 60 + "\n")

    results = []

    # Tests
    results.append(("Imports", test_imports()))

    model, config = test_model_creation()
    results.append(("Model Creation", model is not None))

    if model:
        results.append(("Forward Pass", test_forward_pass(model, config)))
        results.append(("Generation", test_generation(model, config)))
        results.append(("Training Step", test_training_step(model, config)))

    results.append(("Dataset", test_dataset()))

    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")

    print(f"\nTotal: {passed}/{total} tests passés")

    if passed == total:
        print("\n🎉 Tous les tests sont passés!")
        return 0
    else:
        print("\n⚠️ Certains tests ont échoué.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
