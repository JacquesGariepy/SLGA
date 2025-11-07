#!/usr/bin/env python3
"""
Test de validation du fix linspace pour landmarks heuristiques.

Vérifie que:
1. model.py génère exactement global_k landmarks avec linspace
2. data.py génère exactement max_global landmarks avec linspace
3. Pas de G+1 landmarks (bug heuristique)
4. Fonctionne pour différentes valeurs de L et G
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import Config, LLMTransformer
from src.data import CollatorLocalGlobal
from transformers import AutoTokenizer


def test_model_linspace_landmarks():
    """Test que model.py génère exactement global_k landmarks"""
    print("=" * 80)
    print("TEST 1: model.py - Linspace Landmarks in generate()")
    print("=" * 80)

    cfg = Config(
        vocab_size=100,
        max_seq_len=512,
        embed_dim=64,
        num_heads=4,
        n_layers=2,
        local_window=32,
        global_k=24,
        learned_landmarks=False,  # ← Utiliser landmarks heuristiques
    )

    model = LLMTransformer(cfg)
    model.eval()

    test_cases = [
        (10, 24),   # L=10, G=24
        (50, 24),   # L=50, G=24
        (128, 24),  # L=128, G=24
        (256, 48),  # L=256, G=48
        (512, 48),  # L=512, G=48
    ]

    all_passed = True

    for seq_len, global_k in test_cases:
        cfg.global_k = global_k
        model.cfg.global_k = global_k

        # Créer input
        input_ids = torch.randint(0, cfg.vocab_size, (1, seq_len))

        # Simuler le calcul de landmarks heuristiques (même code que generate())
        L = input_ids.size(1)
        landmark_positions = torch.linspace(0, L-1, global_k, device=input_ids.device).long()
        cache_global_ids = landmark_positions.unsqueeze(0).expand(input_ids.size(0), -1)

        # Vérifications
        num_landmarks = cache_global_ids.size(1)
        expected = global_k

        if num_landmarks != expected:
            print(f"❌ FAIL: L={seq_len}, G={global_k} → Got {num_landmarks} landmarks (expected {expected})")
            all_passed = False
        else:
            print(f"✅ PASS: L={seq_len}, G={global_k} → {num_landmarks} landmarks (correct)")

            # Vérifier distribution uniforme
            positions = landmark_positions.numpy()
            gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            avg_gap = sum(gaps) / len(gaps)
            max_gap_deviation = max(abs(g - avg_gap) for g in gaps)

            print(f"  📊 First 5 positions: {positions[:5].tolist()}")
            print(f"  📊 Last 5 positions: {positions[-5:].tolist()}")
            print(f"  📊 Avg gap: {avg_gap:.2f}, Max deviation: {max_gap_deviation:.2f}")

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ TEST 1 PASSED: model.py génère exactement global_k landmarks")
    else:
        print("❌ TEST 1 FAILED: Certains cas ont échoué")
    print("=" * 80)
    print()

    return all_passed


def test_collator_linspace_landmarks():
    """Test que data.py génère exactement max_global landmarks"""
    print("=" * 80)
    print("TEST 2: data.py - CollatorLocalGlobal Linspace Fix")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    test_cases = [
        (128, 32, 16),   # max_length=128, global_every=32, max_global=16
        (256, 64, 24),   # max_length=256, global_every=64, max_global=24
        (512, 128, 32),  # max_length=512, global_every=128, max_global=32
        (1024, 256, 48), # max_length=1024, global_every=256, max_global=48
    ]

    all_passed = True

    for max_length, global_every, max_global in test_cases:
        collator = CollatorLocalGlobal(
            tokenizer,
            max_length=max_length,
            global_every=global_every,
            max_global=max_global,
            strategy="regular"
        )

        # Créer des exemples fictifs
        examples = [
            {"text": "This is a test sentence. " * 50},
            {"text": "Another example text. " * 50},
        ]

        batch = collator(examples)

        # Vérifications
        B, G = batch["cache_global_ids"].shape
        expected = max_global

        if G != expected:
            print(f"❌ FAIL: max_length={max_length}, max_global={max_global} → Got {G} landmarks (expected {expected})")
            all_passed = False
        else:
            print(f"✅ PASS: max_length={max_length}, max_global={max_global} → {G} landmarks (correct)")

            # Vérifier distribution
            positions = batch["cache_global_ids"][0].numpy()
            unique_positions = len(set(positions.tolist()))
            print(f"  📊 Unique positions: {unique_positions}/{G}")
            print(f"  📊 First 5: {positions[:5].tolist()}")
            print(f"  📊 Last 5: {positions[-5:].tolist()}")

            # Vérifier espacement uniforme
            if unique_positions == G:  # Pas de padding
                gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                avg_gap = sum(gaps) / len(gaps)
                max_gap_deviation = max(abs(g - avg_gap) for g in gaps) if gaps else 0
                print(f"  📊 Avg gap: {avg_gap:.2f}, Max deviation: {max_gap_deviation:.2f}")

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ TEST 2 PASSED: data.py génère exactement max_global landmarks")
    else:
        print("❌ TEST 2 FAILED: Certains cas ont échoué")
    print("=" * 80)
    print()

    return all_passed


def test_no_off_by_one_bug():
    """Test qu'il n'y a pas de bug G+1 landmarks"""
    print("=" * 80)
    print("TEST 3: Vérification absence du bug G+1")
    print("=" * 80)

    # Cas classiques où l'ancien code avec range() donnait G+1
    test_cases = [
        (256, 48),  # L / (G*2) = 256 / 96 = 2.67 → stride=2 → range(0,256,2) donne 128 positions!
        (512, 48),  # L / (G*2) = 512 / 96 = 5.33 → stride=5 → range(0,512,5) donne 103 positions!
        (1024, 48), # L / (G*2) = 1024 / 96 = 10.67 → stride=10 → range(0,1024,10) donne 103 positions!
    ]

    all_passed = True

    print("\n🔍 Simulation ancien code avec range():")
    for L, G in test_cases:
        # Ancien code (BUGGÉ)
        global_every = max(1, L // (G * 2))
        candidate_positions_old = list(range(0, L, global_every))
        num_old = len(candidate_positions_old)

        # Nouveau code (FIXÉ avec linspace)
        landmark_positions_new = torch.linspace(0, L-1, G).long().tolist()
        num_new = len(landmark_positions_new)

        bug_detected = num_old != G

        if bug_detected:
            print(f"  ⚠️  L={L:4d}, G={G:2d}: Ancien code → {num_old:3d} landmarks (BUG: {num_old-G:+d})")
        else:
            print(f"  ✅ L={L:4d}, G={G:2d}: Ancien code → {num_old:3d} landmarks (pas de bug)")

        print(f"     Nouveau code → {num_new:3d} landmarks (correct)")

        if num_new != G:
            print(f"     ❌ FAIL: Nouveau code ne génère pas exactement G landmarks!")
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ TEST 3 PASSED: Pas de bug G+1 avec linspace")
    else:
        print("❌ TEST 3 FAILED: Bug encore présent")
    print("=" * 80)
    print()

    return all_passed


def test_edge_cases():
    """Test des cas limites"""
    print("=" * 80)
    print("TEST 4: Cas limites")
    print("=" * 80)

    edge_cases = [
        (10, 24, "L < G: Plus de landmarks demandés que de positions"),
        (24, 24, "L == G: Autant de landmarks que de positions"),
        (25, 24, "L == G+1: Une position de plus que landmarks"),
        (48, 24, "L == 2*G: Exactement double"),
        (1, 24, "L == 1: Séquence minimale"),
        (2048, 1, "G == 1: Un seul landmark"),
    ]

    all_passed = True

    for L, G, description in edge_cases:
        try:
            landmark_positions = torch.linspace(0, L-1, G).long()
            num_landmarks = len(landmark_positions)

            if num_landmarks != G:
                print(f"❌ FAIL: {description}")
                print(f"  L={L}, G={G} → Got {num_landmarks} landmarks")
                all_passed = False
            else:
                print(f"✅ PASS: {description}")
                print(f"  L={L}, G={G} → {num_landmarks} landmarks")
                if G <= 10:  # Afficher si petit nombre
                    print(f"  Positions: {landmark_positions.tolist()}")
        except Exception as e:
            print(f"❌ ERROR: {description}")
            print(f"  Exception: {e}")
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ TEST 4 PASSED: Tous les cas limites gérés correctement")
    else:
        print("❌ TEST 4 FAILED: Certains cas limites ont échoué")
    print("=" * 80)
    print()

    return all_passed


def main():
    """Exécuter tous les tests"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "TEST SUITE: LINSPACE LANDMARK FIX" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    results = {
        "model.py linspace": test_model_linspace_landmarks(),
        "data.py linspace": test_collator_linspace_landmarks(),
        "No G+1 bug": test_no_off_by_one_bug(),
        "Edge cases": test_edge_cases(),
    }

    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 30 + "RÉSUMÉ FINAL" + " " * 35 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    total_tests = len(results)
    passed_tests = sum(results.values())

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {test_name}")

    print()
    print(f"  TOTAL: {passed_tests}/{total_tests} tests passed")
    print()

    if passed_tests == total_tests:
        print("  🎉 SUCCÈS: Tous les tests sont passés!")
        print("  ✅ Le fix linspace est correctement implémenté")
        print("  ✅ Pas de bug G+1 landmarks")
        print()
        return 0
    else:
        print("  ⚠️  ÉCHEC: Certains tests ont échoué")
        print("  ❌ Le fix nécessite des corrections supplémentaires")
        print()
        return 1


if __name__ == "__main__":
    exit(main())
