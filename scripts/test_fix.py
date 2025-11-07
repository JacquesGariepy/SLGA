#!/usr/bin/env python3
"""
Script de test pour vérifier que le fix du double-shifting fonctionne.

Vérifie:
1. Alignement correct des labels
2. Loss raisonnable (pas NaN/Inf)
3. Génération fonctionne

Usage:
    python scripts/test_fix.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import yaml
from transformers import GPT2Tokenizer
from src.data import get_tokenizer, load_text_dataset, CollatorLocal
from src.model import LLMTransformer
from torch.utils.data import DataLoader

def test_label_alignment():
    """Test que les labels sont correctement alignés"""
    print("="*80)
    print("TEST 1: Vérification de l'alignement des labels")
    print("="*80)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    collator = CollatorLocal(tokenizer, max_length=10)

    # Exemple simple
    text = "The cat sat on the mat"

    # Tokenize manuellement pour vérifier
    tokens_full = tokenizer.encode(text)
    print(f"Tokens complets: {tokens_full}")
    print(f"Décodé: {[tokenizer.decode([t]) for t in tokens_full]}")

    # Créer un batch via le collator
    batch = collator([{"text": text}])
    input_ids = batch["input_ids"][0]  # Premier et seul exemple
    labels = batch["labels"][0]

    print(f"\nAprès collator:")
    print(f"input_ids: {input_ids.tolist()}")
    print(f"labels:    {labels.tolist()}")

    # Vérifier l'alignement
    print(f"\nVérification alignement:")
    for i in range(min(7, len(input_ids))):
        input_tok = tokenizer.decode([input_ids[i].item()])
        label_tok = tokenizer.decode([labels[i].item()]) if labels[i] != tokenizer.pad_token_id else "<PAD>"
        expected_tok = tokenizer.decode([input_ids[i+1].item()]) if i+1 < len(input_ids) else "<END>"

        match = "✅" if (i+1 < len(input_ids) and labels[i] == input_ids[i+1]) or (labels[i] == tokenizer.pad_token_id) else "❌"
        print(f"  Position {i}: input='{input_tok}' -> label='{label_tok}' (expected '{expected_tok}') {match}")

    # Vérifier que labels[i] == input_ids[i+1] (sauf fin)
    correct_positions = 0
    total_positions = 0
    for i in range(len(input_ids) - 1):
        if input_ids[i] != tokenizer.pad_token_id:
            total_positions += 1
            if labels[i] == input_ids[i+1]:
                correct_positions += 1

    print(f"\n{'✅' if correct_positions == total_positions else '❌'} Alignement: {correct_positions}/{total_positions} positions correctes")

    if correct_positions == total_positions:
        print("✅ TEST 1 RÉUSSI: Les labels sont correctement alignés!")
    else:
        print("❌ TEST 1 ÉCHOUÉ: Problème d'alignement des labels!")
        return False

    return True


def test_training_step():
    """Test qu'un step de training fonctionne"""
    print("\n" + "="*80)
    print("TEST 2: Vérification d'un step de training")
    print("="*80)

    # Load config
    with open("config_3090.yaml") as f:
        cfg = yaml.safe_load(f)

    # Setup simple
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer(cfg["tokenizer"])

    print(f"Device: {device}")

    # Small model for testing
    cfg_test = cfg["model"].copy()
    cfg_test["n_layers"] = 2  # Seulement 2 couches pour test rapide
    cfg_test["embed_dim"] = 128
    cfg_test["grad_checkpointing"] = False  # Désactiver pour test

    from types import SimpleNamespace
    model_cfg = SimpleNamespace(**cfg_test)

    model = LLMTransformer(model_cfg).to(device)
    print(f"✅ Modèle créé: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Create simple batch
    collator = CollatorLocal(tokenizer, max_length=128)
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence"
    ]
    batch = collator([{"text": t} for t in texts])

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    print(f"Batch: {input_ids.shape}")

    # Forward pass
    model.train()
    logits, _ = model(input_ids, return_aux=True, global_weight=0.0)
    print(f"✅ Forward pass: logits shape = {logits.shape}")

    # Check no NaN/Inf
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("❌ TEST 2 ÉCHOUÉ: NaN/Inf dans logits!")
        return False
    print(f"✅ Pas de NaN/Inf dans logits")

    # Compute loss with NEW logic (no double-shift)
    logits_shifted = logits[:, :-1, :].contiguous()
    labels_shifted = labels[:, :-1].contiguous()  # NEW: pas de shift additionnel

    loss = torch.nn.functional.cross_entropy(
        logits_shifted.reshape(-1, logits_shifted.size(-1)),
        labels_shifted.reshape(-1),
        ignore_index=tokenizer.pad_token_id
    )

    print(f"Loss: {loss.item():.4f}")

    if torch.isnan(loss) or torch.isinf(loss):
        print("❌ TEST 2 ÉCHOUÉ: Loss est NaN/Inf!")
        return False

    if loss.item() > 20 or loss.item() < 0:
        print(f"⚠️  WARNING: Loss semble étrange ({loss.item():.4f})")
    else:
        print(f"✅ Loss raisonnable: {loss.item():.4f}")

    # Backward pass
    loss.backward()
    print(f"✅ Backward pass réussi")

    # Check gradients
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5

    print(f"Gradient norm: {grad_norm:.4f}")

    if grad_norm == 0 or torch.isnan(torch.tensor(grad_norm)):
        print("❌ TEST 2 ÉCHOUÉ: Gradients problématiques!")
        return False

    print("✅ TEST 2 RÉUSSI: Training step fonctionne!")
    return True


def test_generation():
    """Test que la génération fonctionne"""
    print("\n" + "="*80)
    print("TEST 3: Vérification de la génération")
    print("="*80)

    # Load config
    with open("config_3090.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer(cfg["tokenizer"])

    # Small model
    cfg_test = cfg["model"].copy()
    cfg_test["n_layers"] = 2
    cfg_test["embed_dim"] = 128
    cfg_test["grad_checkpointing"] = False  # Désactiver pour test

    from types import SimpleNamespace
    model_cfg = SimpleNamespace(**cfg_test)

    model = LLMTransformer(model_cfg).to(device)
    model.eval()

    # Generate
    prompt = "The cat"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    print(f"Prompt: '{prompt}'")
    print(f"Generating 10 tokens...")

    try:
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=1.0,
                top_k=50
            )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"✅ Génération réussie: '{generated_text}'")

        # Check no repetition (common bug)
        tokens = tokenizer.encode(generated_text)
        if len(set(tokens[-5:])) == 1:  # Last 5 tokens all same
            print("⚠️  WARNING: Génération répétitive (possible problème)")
        else:
            print(f"✅ Génération variée (pas de répétition)")

        print("✅ TEST 3 RÉUSSI: Génération fonctionne!")
        return True

    except Exception as e:
        print(f"❌ TEST 3 ÉCHOUÉ: {e}")
        return False


def main():
    print("VÉRIFICATION DU FIX DU DOUBLE-SHIFTING")
    print("="*80)
    print()

    results = []

    # Test 1: Label alignment
    results.append(("Alignement des labels", test_label_alignment()))

    # Test 2: Training step
    results.append(("Step de training", test_training_step()))

    # Test 3: Generation
    results.append(("Génération", test_generation()))

    # Summary
    print("\n" + "="*80)
    print("RÉSUMÉ DES TESTS")
    print("="*80)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
        if not passed:
            all_passed = False

    print("="*80)

    if all_passed:
        print("✅ TOUS LES TESTS RÉUSSIS!")
        print("\nVous pouvez maintenant:")
        print("  1. Arrêter le training actuel (Ctrl+C)")
        print("  2. Nettoyer les anciens checkpoints:")
        print("     bash scripts/clean_restart.sh")
        print("  3. Relancer le training:")
        print("     python scripts/train.py")
        print()
        return 0
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ!")
        print("\nVérifiez les erreurs ci-dessus avant de continuer.")
        print()
        return 1


if __name__ == "__main__":
    exit(main())
