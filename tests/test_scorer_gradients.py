#!/usr/bin/env python3
"""
Test #1: Vérifier que les gradients flow vers le scorer du landmark selector
"""
import torch
import yaml
from transformers import GPT2Tokenizer
import sys
import os

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.model import LLMTransformer, Config
from src.landmarks import landmark_spacing_loss

def test_scorer_gradients(checkpoint_path='out_slga/ckpt_18000', config_path='config/config.wikipedia.yaml'):
    """
    Test si les gradients flow vers le scorer du landmark selector
    """
    print("\n" + "="*70)
    print("🔍 TEST #1: Vérification des Gradients du Scorer")
    print("="*70)

    # Load config
    print(f"\n[1/5] Chargement de la config depuis {config_path}...")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    cfg = Config(**config_dict['model'])
    print(f"  ✓ Config chargée")

    # Load model
    print(f"\n[2/5] Chargement du modèle...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LLMTransformer(cfg).to(device)

    checkpoint_file = os.path.join(checkpoint_path, 'model.pt')
    state_dict = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(state_dict)
    model.train()  # ← TRAIN mode pour activer gradients!
    print(f"  ✓ Modèle chargé en mode TRAIN")

    # Test forward + backward
    print(f"\n[3/5] Préparation du batch de test...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Répéter pour avoir assez de tokens pour landmarks
    prompt = "The capital of France is Paris. It is known for its art and culture. " * 5
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    seq_len = input_ids.size(1)
    print(f"  ✓ Séquence de {seq_len} tokens créée")

    # Forward pass
    print(f"\n[4/5] Forward pass avec return_aux=True...")
    logits, aux = model(input_ids, return_aux=True)

    if aux is None or 'landmark_indices' not in aux:
        print("  ❌ ERREUR: return_aux ne retourne pas landmark_indices!")
        return False

    print(f"  ✓ Forward pass réussi")
    print(f"  Landmarks sélectionnés: {aux['landmark_indices'].shape}")

    # Calculer spacing loss avec selection_scores (pour gradients!)
    # Utiliser lambda_reg élevé pour voir des gradients significatifs dans le test
    loss_spacing = landmark_spacing_loss(
        landmark_indices=aux['landmark_indices'],
        seq_len=seq_len,
        lambda_reg=1.0,  # ← Plus élevé pour le test (vs 0.01 en training)
        selection_scores=aux['landmark_scores']  # ← FIX: Ajouter scores!
    )

    print(f"  Spacing loss: {loss_spacing.item():.6f}")

    # Backward pass
    print(f"\n[5/5] Backward pass pour calculer les gradients...")
    loss_spacing.backward()
    print(f"  ✓ Backward pass réussi")

    # Vérifier gradients du scorer
    print(f"\n" + "="*70)
    print("📊 ANALYSE DES GRADIENTS DU SCORER")
    print("="*70)

    if model.landmark_selector is None:
        print("  ❌ ERREUR: Le modèle n'a pas de landmark_selector!")
        return False

    scorer = model.landmark_selector.scorer
    has_grad = False
    total_params = 0
    params_with_grad = 0

    print("\nParamètres du scorer:")
    for name, param in scorer.named_parameters():
        total_params += 1
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  ✓ {name}: grad_norm = {grad_norm:.6f}")
            if grad_norm > 1e-8:
                has_grad = True
                params_with_grad += 1
        else:
            print(f"  ❌ {name}: NO GRADIENT!")

    print(f"\n" + "="*70)
    print("RÉSULTAT DU TEST")
    print("="*70)
    print(f"  Paramètres totaux: {total_params}")
    print(f"  Paramètres avec gradients: {params_with_grad}")

    if has_grad:
        print("\n  ✅ SUCCÈS: Les gradients FLOW vers le scorer!")
        print("  → Le problème n'est PAS un blocage de gradients")
        return True
    else:
        print("\n  ❌ ÉCHEC: Les gradients NE FLOW PAS vers le scorer!")
        print("  → Le problème vient du flux de gradients")
        print("\n  Causes possibles:")
        print("    1. Le straight-through trick bloque les gradients")
        print("    2. Le scorer est en eval mode")
        print("    3. Les opérations sur scores utilisent .detach()")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="out_slga/ckpt_18000", help="Checkpoint directory")
    parser.add_argument("--config", default="config/config.wikipedia.yaml", help="Config file")
    args = parser.parse_args()

    success = test_scorer_gradients(args.checkpoint, args.config)
    sys.exit(0 if success else 1)
