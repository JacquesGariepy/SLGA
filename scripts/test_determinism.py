#!/usr/bin/env python3
"""
Script de Test Automatisé: Validation du Fix Non-Déterminisme
==============================================================

Ce script teste si le fix résout le problème de non-déterminisme.

Usage:
    python test_determinism.py --checkpoint out_slga/ckpt_18000

Tests effectués:
1. ✓ Déterminisme avec temperature=0.0 (greedy)
2. ✓ Reproductibilité avec seed (stochastique)
3. ✓ Qualité de génération (pas de boucles)
"""

import sys
import argparse
import torch
from collections import Counter


def test_greedy_determinism(model, tokenizer, device, num_runs=5):
    """
    Test #1: Vérifier que temperature=0.0 est DÉTERMINISTE.
    
    Tous les runs doivent produire la MÊME séquence.
    """
    print("\n" + "="*70)
    print("TEST #1: Déterminisme Greedy (temperature=0.0)")
    print("="*70)
    
    prompt = "The capital of France is"
    outputs = []
    
    for run in range(num_runs):
        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate (greedy)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.0,  # GREEDY
            top_k=40,
        )
        
        # Decode
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        outputs.append(generated_text)
        
        print(f"  Run {run+1}: {generated_text}")
    
    # Vérifier que tous sont identiques
    unique_outputs = set(outputs)
    
    if len(unique_outputs) == 1:
        print("\n  ✅ SUCCÈS: Tous les runs sont identiques (déterministe)!")
        return True
    else:
        print(f"\n  ❌ ÉCHEC: {len(unique_outputs)} outputs différents détectés!")
        print("  Outputs uniques:")
        for i, out in enumerate(unique_outputs, 1):
            print(f"    {i}. {out}")
        return False


def test_seed_reproducibility(model, tokenizer, device, num_runs=3):
    """
    Test #2: Vérifier que le seed rend le sampling REPRODUCTIBLE.
    
    Avec le même seed, tous les runs doivent être identiques.
    """
    print("\n" + "="*70)
    print("TEST #2: Reproductibilité avec Seed (temperature=0.8)")
    print("="*70)
    
    prompt = "The future of AI is"
    seed = 42
    outputs = []
    
    for run in range(num_runs):
        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate avec seed fixe
        output_ids = model.generate(
            input_ids,
            max_new_tokens=15,
            temperature=0.8,  # Stochastique
            top_k=40,
            seed=seed,  # ← Seed fixe
        )
        
        # Decode
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        outputs.append(generated_text)
        
        print(f"  Run {run+1}: {generated_text}")
    
    # Vérifier que tous sont identiques avec le même seed
    unique_outputs = set(outputs)
    
    if len(unique_outputs) == 1:
        print("\n  ✅ SUCCÈS: Seed rend le sampling reproductible!")
        return True
    else:
        print(f"\n  ❌ ÉCHEC: {len(unique_outputs)} outputs différents avec le même seed!")
        return False


def test_no_loops(model, tokenizer, device):
    """
    Test #3: Vérifier qu'il n'y a PAS de boucles de répétition.
    
    Le modèle ne devrait pas répéter "of France of France of France..."
    """
    print("\n" + "="*70)
    print("TEST #3: Détection de Boucles de Répétition")
    print("="*70)
    
    prompt = "The capital of France is"
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate
    output_ids = model.generate(
        input_ids,
        max_new_tokens=20,
        temperature=0.0,  # Greedy
        top_k=40,
    )
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"  Output: {generated_text}")
    
    # Extraire les tokens générés (après le prompt)
    prompt_length = len(tokenizer.encode(prompt))
    generated_tokens = output_ids[0, prompt_length:].tolist()
    
    # Détecter répétitions
    # Si un token apparaît plus de 3 fois, c'est suspect
    token_counts = Counter(generated_tokens)
    max_repetition = max(token_counts.values()) if token_counts else 0
    
    print(f"\n  Max répétition d'un token: {max_repetition}")
    
    if max_repetition <= 3:
        print("  ✅ SUCCÈS: Pas de boucles détectées!")
        return True
    else:
        most_common = token_counts.most_common(3)
        print(f"  ⚠️  AVERTISSEMENT: Répétitions détectées!")
        print(f"  Tokens les plus répétés:")
        for token_id, count in most_common:
            token_str = tokenizer.decode([token_id])
            print(f"    '{token_str}' répété {count} fois")
        
        # Tolérer quelques répétitions, mais pas trop
        return max_repetition <= 5


def main():
    parser = argparse.ArgumentParser(description="Test de déterminisme SLGA")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Chemin vers le checkpoint")
    parser.add_argument("--config", type=str, default="config/config.wikipedia.yaml",
                        help="Chemin vers le config")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("VALIDATION DU FIX NON-DÉTERMINISME")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Import nécessaire (adapté à ta structure)
    # NOTE: Tu devras adapter ces imports selon ton setup
    try:
        from transformers import GPT2Tokenizer
        import yaml
        import sys
        import os
        
        # Ajouter le répertoire racine au path pour imports absolus
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from src.model import LLMTransformer, Config
        
        print("✓ Imports réussis")
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("\nAssure-toi que:")
        print("  1. Tu es dans le répertoire racine du projet")
        print("  2. transformers est installé: pip install transformers")
        sys.exit(1)
    
    # Load tokenizer
    print("\nChargement du tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer chargé")
    
    # Load config
    print(f"\nChargement du config depuis {args.config}...")
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    cfg = Config(**config_dict['model'])
    print("✓ Config chargé")
    
    # Create model
    print("\nCréation du modèle...")
    model = LLMTransformer(cfg).to(device)
    print(f"✓ Modèle créé: {model.get_num_params() / 1e6:.2f}M paramètres")
    
    # Load checkpoint
    print(f"\nChargement du checkpoint depuis {args.checkpoint}...")
    checkpoint_path = f"{args.checkpoint}/model.pt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("✓ Checkpoint chargé")
    
    # Run tests
    print("\n" + "="*70)
    print("EXÉCUTION DES TESTS")
    print("="*70)
    
    results = {}
    
    # Test 1: Déterminisme greedy
    results['greedy'] = test_greedy_determinism(model, tokenizer, device)
    
    # Test 2: Reproductibilité avec seed
    # NOTE: Ce test nécessite que le paramètre 'seed' soit ajouté à generate()
    try:
        results['seed'] = test_seed_reproducibility(model, tokenizer, device)
    except TypeError as e:
        print(f"\n  ⚠️  AVERTISSEMENT: Paramètre 'seed' non trouvé dans generate()")
        print(f"     Error: {e}")
        print("     → Ajoute le paramètre 'seed' à la signature de generate()")
        results['seed'] = False
    
    # Test 3: Pas de boucles
    results['no_loops'] = test_no_loops(model, tokenizer, device)
    
    # Summary
    print("\n" + "="*70)
    print("RÉSUMÉ DES RÉSULTATS")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n  Score: {passed}/{total} tests réussis")
    
    if passed == total:
        print("\n  🎉 TOUS LES TESTS SONT PASSÉS!")
        print("  Le fix fonctionne correctement!")
        return 0
    else:
        print("\n  ⚠️  CERTAINS TESTS ONT ÉCHOUÉ")
        print("  Vérifie que le fix a été appliqué correctement.")
        return 1


if __name__ == "__main__":
    sys.exit(main())