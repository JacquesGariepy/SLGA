#!/usr/bin/env python3
"""
Test complet du pipeline: Training + Perplexity + Generation

Vérifie que TOUT fonctionne avant de lancer le training complet.

Usage:
    python scripts/test_complete.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import yaml
import math
from transformers import GPT2Tokenizer
from src.data import get_tokenizer, load_text_dataset, CollatorLocal
from src.model import LLMTransformer
from torch.utils.data import DataLoader
from types import SimpleNamespace


def load_mini_config():
    """Charge config réduite pour test rapide"""
    with open("config_3090.yaml") as f:
        cfg = yaml.safe_load(f)

    # Réduire pour test rapide
    cfg["model"]["n_layers"] = 4  # Au lieu de 12
    cfg["model"]["embed_dim"] = 256  # Au lieu de 512
    cfg["train"]["batch_size"] = 2  # Au lieu de 8
    cfg["train"]["seq_len_start"] = 128  # Au lieu de 512
    cfg["train"]["grad_checkpointing"] = False  # Désactiver pour test rapide

    return cfg


def test_training(cfg, num_steps=20):
    """Test 20 steps de training réel"""
    print("="*80)
    print("TEST 1: TRAINING (20 steps réels)")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    model_cfg_dict = cfg["model"].copy()
    model_cfg_dict["grad_checkpointing"] = cfg["train"].get("grad_checkpointing", False)
    model_cfg = SimpleNamespace(**model_cfg_dict)
    model = LLMTransformer(model_cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ Modèle créé: {num_params:.1f}M params")

    # Data
    tokenizer = get_tokenizer(cfg["tokenizer"])

    print("Chargement dataset...")
    try:
        ds = load_text_dataset(
            cfg["data"]["dataset"],
            cfg["data"].get("subset"),
            "train[:1000]",  # Seulement 1000 exemples
        )
        print(f"✅ Dataset chargé: {len(ds)} exemples")
    except Exception as e:
        print(f"❌ Erreur chargement dataset: {e}")
        return False

    collator = CollatorLocal(tokenizer, cfg["train"]["seq_len_start"])
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        betas=tuple(cfg["train"]["betas"]),
        weight_decay=cfg["train"]["weight_decay"],
    )

    # Training loop
    model.train()
    losses = []
    grad_norms = []

    print(f"\nTraining {num_steps} steps...")
    print("-" * 80)

    step = 0
    for batch in loader:
        if step >= num_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward
        logits, aux = model(input_ids, return_aux=True, global_weight=0.0)

        # Loss (avec le FIX!)
        logits_shifted = logits[:, :-1, :].contiguous()
        labels_shifted = labels[:, :-1].contiguous()  # PAS labels[:, 1:]!

        loss = torch.nn.functional.cross_entropy(
            logits_shifted.view(-1, logits_shifted.size(-1)),
            labels_shifted.view(-1),
            ignore_index=tokenizer.pad_token_id
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Grad norm
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        # Clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Step
        optimizer.step()

        losses.append(loss.item())
        grad_norms.append(grad_norm)

        if (step + 1) % 5 == 0:
            avg_loss = sum(losses[-5:]) / 5
            avg_grad = sum(grad_norms[-5:]) / 5
            ppl = math.exp(min(avg_loss, 10))
            print(f"Step {step+1:3d} | Loss: {avg_loss:.4f} | PPL: {ppl:7.2f} | GradNorm: {avg_grad:.3f}")

        step += 1

    print("-" * 80)

    # Analyse
    print(f"\nAnalyse des {num_steps} steps:")

    initial_loss = sum(losses[:5]) / 5
    final_loss = sum(losses[-5:]) / 5
    loss_decrease = initial_loss - final_loss
    loss_decrease_pct = (loss_decrease / initial_loss) * 100

    print(f"  Loss initiale (steps 1-5):   {initial_loss:.4f}")
    print(f"  Loss finale (steps 16-20):   {final_loss:.4f}")
    print(f"  Diminution: {loss_decrease:.4f} ({loss_decrease_pct:.1f}%)")

    avg_grad = sum(grad_norms) / len(grad_norms)
    print(f"  Gradient norm moyen: {avg_grad:.3f}")

    # Vérifications
    issues = []

    if loss_decrease < 0:
        issues.append("❌ Loss augmente au lieu de diminuer!")
    elif loss_decrease < 0.1:
        issues.append("⚠️  Loss descend très peu")
    else:
        print(f"  ✅ Loss descend correctement")

    if avg_grad < 0.001:
        issues.append("❌ Gradients trop petits (vanishing)")
    elif avg_grad > 100:
        issues.append("❌ Gradients trop grands (exploding)")
    else:
        print(f"  ✅ Gradients dans une plage correcte")

    if any(torch.isnan(torch.tensor(l)) for l in losses):
        issues.append("❌ NaN dans les losses!")
    else:
        print(f"  ✅ Pas de NaN dans les losses")

    if issues:
        print("\n" + "\n".join(issues))
        return False

    print("\n✅ TEST 1 RÉUSSI: Training fonctionne!")
    return model, tokenizer


def test_perplexity(model, tokenizer, cfg):
    """Test évaluation de perplexité"""
    print("\n" + "="*80)
    print("TEST 2: ÉVALUATION PERPLEXITÉ")
    print("="*80)

    device = next(model.parameters()).device

    # Load val data
    print("Chargement validation set...")
    try:
        ds_val = load_text_dataset(
            cfg["data"]["dataset"],
            cfg["data"].get("subset"),
            "train[1000:1100]",  # 100 exemples
        )
        print(f"✅ Dataset validé: {len(ds_val)} exemples")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

    collator = CollatorLocal(tokenizer, cfg["train"]["seq_len_start"])
    loader = DataLoader(
        ds_val,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # Evaluate
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    print("Évaluation...")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 20:  # Limiter pour rapidité
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            logits = model(input_ids)

            # Loss (avec le FIX!)
            logits_shifted = logits[:, :-1, :].contiguous()
            labels_shifted = labels[:, :-1].contiguous()

            loss = torch.nn.functional.cross_entropy(
                logits_shifted.view(-1, logits_shifted.size(-1)),
                labels_shifted.view(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction='sum'
            )

            num_tokens = (labels != tokenizer.pad_token_id).sum().item()
            total_loss += loss.item()
            total_tokens += num_tokens
            num_batches += 1

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 10))

    print(f"\nRésultats sur {num_batches} batches:")
    print(f"  Loss moyenne: {avg_loss:.4f}")
    print(f"  Perplexité:   {perplexity:.2f}")
    print(f"  Tokens:       {total_tokens}")

    # Vérifications
    if torch.isnan(torch.tensor(avg_loss)):
        print("❌ Loss est NaN!")
        return False

    if perplexity > 100000:
        print(f"⚠️  Perplexité très élevée ({perplexity:.0f})")
        print("    Normal pour modèle non entraîné, mais devrait diminuer avec training")
    else:
        print(f"  ✅ Perplexité calculée correctement")

    print("\n✅ TEST 2 RÉUSSI: Évaluation fonctionne!")
    return True


def test_generation(model, tokenizer):
    """Test génération de texte"""
    print("\n" + "="*80)
    print("TEST 3: GÉNÉRATION DE TEXTE")
    print("="*80)

    device = next(model.parameters()).device
    model.eval()

    prompts = [
        "The cat",
        "In the year 2024,",
        "Machine learning is"
    ]

    print("Test de génération avec plusieurs prompts...\n")

    all_ok = True

    for prompt in prompts:
        print(f"Prompt: '{prompt}'")

        try:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids,
                    max_new_tokens=20,
                    temperature=0.8,
                    top_k=50
                )

            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"→ Généré: '{generated_text}'")

            # Vérifications
            tokens = tokenizer.encode(generated_text)

            # Check répétition
            if len(tokens) > 5:
                last_5 = tokens[-5:]
                if len(set(last_5)) == 1:
                    print("  ⚠️  Répétition détectée (dernier 5 tokens identiques)")
                else:
                    print("  ✅ Génération variée")

            # Check longueur
            if len(tokens) < len(input_ids[0]) + 5:
                print("  ⚠️  Génération très courte")
            else:
                print(f"  ✅ {len(tokens) - len(input_ids[0])} tokens générés")

            print()

        except Exception as e:
            print(f"  ❌ ERREUR: {e}")
            all_ok = False
            print()

    if not all_ok:
        return False

    print("✅ TEST 3 RÉUSSI: Génération fonctionne!")
    return True


def main():
    print("="*80)
    print("TEST COMPLET: TRAINING + PERPLEXITY + GENERATION")
    print("="*80)
    print("\nCe test vérifie que TOUT le pipeline fonctionne correctement.")
    print("Durée estimée: 2-3 minutes")
    print("="*80)
    print()

    # Load config
    print("Chargement configuration...")
    cfg = load_mini_config()
    print(f"✅ Config chargée (réduite pour test rapide)")
    print()

    # Test 1: Training
    result = test_training(cfg, num_steps=20)
    if not result:
        print("\n❌ TEST TRAINING ÉCHOUÉ!")
        return 1

    model, tokenizer = result

    # Test 2: Perplexity
    if not test_perplexity(model, tokenizer, cfg):
        print("\n❌ TEST PERPLEXITY ÉCHOUÉ!")
        return 1

    # Test 3: Generation
    if not test_generation(model, tokenizer):
        print("\n❌ TEST GENERATION ÉCHOUÉ!")
        return 1

    # Summary
    print("\n" + "="*80)
    print("RÉSUMÉ")
    print("="*80)
    print("✅ TEST 1: Training (20 steps) - RÉUSSI")
    print("✅ TEST 2: Perplexity evaluation - RÉUSSI")
    print("✅ TEST 3: Génération - RÉUSSI")
    print("="*80)
    print()
    print("🎉 TOUS LES TESTS RÉUSSIS!")
    print()
    print("Le pipeline complet fonctionne correctement.")
    print("Vous pouvez maintenant lancer le training complet en toute confiance:")
    print()
    print("  1. Arrêter le training actuel (Ctrl+C)")
    print("  2. Nettoyer: bash scripts/clean_restart.sh")
    print("  3. Lancer:   python scripts/train.py")
    print()
    print("Résultats attendus avec le fix:")
    print("  - Step 2000: PPL ~800-2000 (au lieu de 4424)")
    print("  - Génération: Mots cohérents (au lieu de gibberish)")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
