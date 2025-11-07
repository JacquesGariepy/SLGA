#!/usr/bin/env python3
"""
Vérification: Les loss auxiliaires de landmarks sont-elles calculées et appliquées?

Ce script vérifie si les loss de spacing, diversity et sparsity sont bien
ajoutées à la loss totale pendant l'entraînement.
"""

import sys
import os

print("="*70)
print("🔍 VÉRIFICATION: Loss Auxiliaires des Landmarks")
print("="*70)

# Chercher dans le code d'entraînement
trainer_files = [
    'scripts/train.py',
    'train.py',
    'scripts/train_slga.py',
    'trainer.py',
]

found = False
for filepath in trainer_files:
    if os.path.exists(filepath):
        print(f"\n✓ Fichier trouvé: {filepath}")
        found = True
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Chercher les imports de loss
        print("\n--- Imports de loss ---")
        if 'landmark_spacing_loss' in content:
            print("  ✓ landmark_spacing_loss importé")
        else:
            print("  ❌ landmark_spacing_loss NON importé")
        
        if 'landmark_diversity_loss' in content:
            print("  ✓ landmark_diversity_loss importé")
        else:
            print("  ⚠️  landmark_diversity_loss NON importé")
        
        if 'landmark_sparsity_loss' in content:
            print("  ✓ landmark_sparsity_loss importé")
        else:
            print("  ❌ landmark_sparsity_loss NON importé")
        
        # Chercher les calculs de loss
        print("\n--- Calculs de loss dans le code ---")
        
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'landmark_spacing_loss' in line and not line.strip().startswith('#'):
                print(f"  Ligne {i}: {line.strip()}")
            if 'landmark_diversity_loss' in line and not line.strip().startswith('#'):
                print(f"  Ligne {i}: {line.strip()}")
            if 'landmark_sparsity_loss' in line and not line.strip().startswith('#'):
                print(f"  Ligne {i}: {line.strip()}")
        
        # Chercher return_aux=True
        print("\n--- Utilisation de return_aux ---")
        if 'return_aux=True' in content:
            print("  ✓ return_aux=True trouvé")
            for i, line in enumerate(lines, 1):
                if 'return_aux' in line and not line.strip().startswith('#'):
                    print(f"  Ligne {i}: {line.strip()}")
        else:
            print("  ❌ return_aux=True NON trouvé")
            print("     → Le modèle ne retourne PAS les aux (indices, scores)")
            print("     → Les loss auxiliaires ne peuvent PAS être calculées!")
        
        break

if not found:
    print("\n❌ Aucun fichier d'entraînement trouvé!")
    print("\nFichiers recherchés:")
    for f in trainer_files:
        print(f"  - {f}")
    print("\nVérifie manuellement ton code d'entraînement!")

print("\n" + "="*70)
print("DIAGNOSTIC:")
print("="*70)

if not found:
    print("⚠️  Impossible de vérifier - fichier non trouvé")
else:
    print("""
Si tu vois:
  ❌ landmark_spacing_loss NON importé
  ❌ return_aux=True NON trouvé

ALORS le problème est là!

Le LearnableLandmarkSelector génère des scores, mais ces scores
ne sont JAMAIS utilisés pour entraîner le scorer.

Sans loss auxiliaires:
  → Le scorer reste à son initialisation aléatoire
  → Tous les tokens ont le même score (~1/L)
  → Les landmarks sont sélectionnés au hasard
  → L'attention globale est inefficace
""")

print("\n" + "="*70)