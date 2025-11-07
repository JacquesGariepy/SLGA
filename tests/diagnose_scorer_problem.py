#!/usr/bin/env python3
"""
Diagnostic Complet: Pourquoi le scorer n'apprend pas?

Lance les Tests #1 et #2 pour identifier le problème.
"""
import sys
import os
import argparse

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import des tests
from test_checkpoint_keys import test_checkpoint_keys
from test_scorer_gradients import test_scorer_gradients


def main():
    parser = argparse.ArgumentParser(description="Diagnostic complet du scorer")
    parser.add_argument("--checkpoint", default="out_slga/ckpt_18000", help="Checkpoint directory")
    parser.add_argument("--config", default="config/config.wikipedia.yaml", help="Config file")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("🔍 DIAGNOSTIC COMPLET: Pourquoi le Scorer n'apprend pas?")
    print("="*80)

    results = {}

    # Test #2 d'abord (plus rapide, pas de GPU)
    print("\n\n" + "🔬 " + "="*75)
    print("TEST #2: Vérification du Checkpoint")
    print("="*80)

    results['checkpoint'] = test_checkpoint_keys(args.checkpoint)

    # Test #1 (nécessite GPU)
    print("\n\n" + "🔬 " + "="*75)
    print("TEST #1: Vérification des Gradients")
    print("="*80)

    results['gradients'] = test_scorer_gradients(args.checkpoint, args.config)

    # Résumé final
    print("\n\n" + "="*80)
    print("📊 RÉSUMÉ DU DIAGNOSTIC")
    print("="*80)

    checkpoint_ok = results['checkpoint']
    gradients_ok = results['gradients']

    print("\n  Tests effectués:")
    print(f"    {'✅' if checkpoint_ok else '❌'} Test #2: Clés du checkpoint")
    print(f"    {'✅' if gradients_ok else '❌'} Test #1: Gradients du scorer")

    print("\n" + "="*80)
    print("🎯 DIAGNOSTIC FINAL")
    print("="*80)

    if not checkpoint_ok:
        print("\n  ❌ PROBLÈME IDENTIFIÉ: Checkpoint incomplet")
        print("\n  Le checkpoint ne contient pas les poids du landmark_selector!")
        print("\n  ✅ SOLUTION:")
        print("    1. Vérifier que learned_landmarks=True dans la config")
        print("    2. Vérifier que save_checkpoint sauvegarde tous les paramètres")
        print("    3. Re-entraîner depuis le début avec la config corrigée")

    elif not gradients_ok:
        print("\n  ❌ PROBLÈME IDENTIFIÉ: Gradients bloqués")
        print("\n  Les gradients ne flow pas vers le scorer du landmark_selector!")
        print("\n  ✅ SOLUTIONS possibles:")
        print("    1. Vérifier le straight-through trick dans landmarks.py:122")
        print("    2. S'assurer que le scorer est bien en mode train")
        print("    3. Vérifier qu'il n'y a pas de .detach() inapproprié")
        print("\n  📝 Fix suggéré dans landmarks.py:")
        print("    - Ligne 122: Vérifier que `scores - scores.detach()` ne bloque pas")
        print("    - Ligne 158: S'assurer que straight_through est bien implémenté")

    else:
        print("\n  ⚠️ PROBLÈME NON IDENTIFIÉ PAR LES TESTS")
        print("\n  Le checkpoint contient les poids ET les gradients flow...")
        print("  Mais le scorer n'apprend toujours pas!")
        print("\n  ✅ SOLUTIONS à tester:")
        print("\n  1. AUGMENTER LES LAMBDAS (cause la plus probable):")
        print("     Dans config/config.wikipedia.yaml:")
        print("       lambda_spacing: 0.1    # était 0.01 (×10)")
        print("       lambda_sparsity: 0.01  # était 0.001 (×10)")
        print("\n  2. Vérifier le learning rate du scorer:")
        print("     - Le scorer partage-t-il le même LR que le modèle?")
        print("     - Essayer un LR plus élevé pour le scorer uniquement")
        print("\n  3. Analyser les valeurs de loss:")
        print("     - Lancer training avec --log-every 10")
        print("     - Vérifier que spacing_loss et sparsity_loss sont non-nulles")

    print("\n" + "="*80)

    # Code de sortie
    if checkpoint_ok and gradients_ok:
        print("\n  ℹ️  Tests passés mais problème non résolu")
        print("     → Essayer d'augmenter les lambdas!")
        return 2  # Warning code
    elif not checkpoint_ok or not gradients_ok:
        print("\n  ❌ Problème identifié - action requise")
        return 1  # Error code
    else:
        return 0  # Success


if __name__ == "__main__":
    sys.exit(main())
