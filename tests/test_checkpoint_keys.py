#!/usr/bin/env python3
"""
Test #2: Vérifier que le checkpoint contient les poids du landmark_selector
"""
import torch
import sys
import os

def test_checkpoint_keys(checkpoint_path='out_slga/ckpt_18000'):
    """
    Vérifie que le checkpoint contient bien les clés du landmark_selector
    """
    print("\n" + "="*70)
    print("🔍 TEST #2: Vérification des Clés du Checkpoint")
    print("="*70)

    checkpoint_file = os.path.join(checkpoint_path, 'model.pt')

    if not os.path.exists(checkpoint_file):
        print(f"\n  ❌ ERREUR: Checkpoint introuvable: {checkpoint_file}")
        return False

    print(f"\n[1/3] Chargement du checkpoint depuis {checkpoint_file}...")
    ckpt = torch.load(checkpoint_file, map_location='cpu')
    print(f"  ✓ Checkpoint chargé ({len(ckpt)} clés au total)")

    # Chercher les clés du landmark_selector
    print(f"\n[2/3] Recherche des clés 'landmark_selector'...")
    selector_keys = [k for k in ckpt.keys() if 'landmark_selector' in k]

    print(f"\n" + "="*70)
    print("📊 CLÉS DU LANDMARK_SELECTOR TROUVÉES")
    print("="*70)

    if not selector_keys:
        print("\n  ❌ AUCUNE CLÉ 'landmark_selector' trouvée dans le checkpoint!")
        print("\n  Cela signifie que:")
        print("    1. Le landmark_selector n'est pas sauvegardé")
        print("    2. Ou le nom du module ne correspond pas")
        print("\n  Causes possibles:")
        print("    - Le modèle n'a pas learned_landmarks=True au moment du save")
        print("    - Le save_checkpoint ne sauvegarde pas tous les paramètres")
        return False

    print(f"\n  ✅ {len(selector_keys)} clés trouvées!\n")

    # Grouper par sous-module
    scorer_keys = [k for k in selector_keys if 'scorer' in k]
    other_keys = [k for k in selector_keys if 'scorer' not in k]

    if scorer_keys:
        print("  📦 Clés du SCORER:")
        for k in sorted(scorer_keys):
            tensor = ckpt[k]
            print(f"    ✓ {k}")
            print(f"       Shape: {tensor.shape}, dtype: {tensor.dtype}")
    else:
        print("  ⚠️  Aucune clé 'scorer' trouvée!")

    if other_keys:
        print("\n  📦 Autres clés du landmark_selector:")
        for k in sorted(other_keys):
            tensor = ckpt[k]
            print(f"    ✓ {k}")
            print(f"       Shape: {tensor.shape}, dtype: {tensor.dtype}")

    # Vérifier les valeurs du scorer
    print(f"\n[3/3] Analyse des valeurs du scorer...")

    if not scorer_keys:
        print("  ⚠️  Impossible d'analyser (pas de clés scorer)")
        return True  # On a trouvé des clés selector, c'est déjà bien

    # Analyser le premier poids
    first_key = sorted(scorer_keys)[0]
    weights = ckpt[first_key]

    print(f"\n  Analyse de '{first_key}':")
    print(f"    Min: {weights.min().item():.6f}")
    print(f"    Max: {weights.max().item():.6f}")
    print(f"    Mean: {weights.mean().item():.6f}")
    print(f"    Std: {weights.std().item():.6f}")

    # Si std très faible, les poids sont peut-être initialisés mais pas entraînés
    if weights.std().item() < 0.01:
        print("\n  ⚠️  AVERTISSEMENT: Std très faible!")
        print("       Les poids semblent peu variés (possiblement non entraînés)")

    print(f"\n" + "="*70)
    print("RÉSULTAT DU TEST")
    print("="*70)
    print(f"  ✅ Le checkpoint CONTIENT les poids du landmark_selector")
    print(f"  ✅ {len(scorer_keys)} paramètres du scorer trouvés")
    print(f"\n  → Le problème n'est PAS un checkpoint incomplet")

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="out_slga/ckpt_18000", help="Checkpoint directory")
    args = parser.parse_args()

    success = test_checkpoint_keys(args.checkpoint)
    sys.exit(0 if success else 1)
