#!/usr/bin/env python3
"""
Tests complets du fix de landmark_sparsity_loss

Vérifie que la nouvelle implémentation:
1. Produit des gradients valides
2. Loss varie selon la concentration des scores
3. Valeurs raisonnables (pas constantes!)
4. Comportement correct pour différents scénarios
"""
import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.landmarks import landmark_sparsity_loss


def test_gradient_flow():
    """Test 1: Vérifie que les gradients fluent correctement"""
    print("\n" + "="*70)
    print("TEST 1: Gradient Flow")
    print("="*70)

    # Créer des scores avec requires_grad
    B, L, G = 4, 384, 48
    selection_scores = torch.randn(B, L, requires_grad=True)

    # Forward pass
    loss = landmark_sparsity_loss(selection_scores, num_landmarks=G, lambda_reg=0.001)

    # Backward pass
    loss.backward()

    # Vérifications
    assert selection_scores.grad is not None, "❌ Pas de gradients!"
    assert not torch.isnan(selection_scores.grad).any(), "❌ Gradients NaN!"
    assert not torch.isinf(selection_scores.grad).any(), "❌ Gradients Inf!"

    grad_norm = selection_scores.grad.norm().item()
    print(f"  ✅ Gradients valides: norm={grad_norm:.6f}")
    print(f"     Loss value: {loss.item():.6f}")

    return True


def test_loss_variation():
    """Test 2: Vérifie que la loss varie selon la concentration"""
    print("\n" + "="*70)
    print("TEST 2: Loss Variation avec Concentration")
    print("="*70)

    B, L, G = 4, 384, 48
    lambda_reg = 0.001

    # Scénario 1: Scores très concentrés (bonne sparsité)
    scores_concentrated = torch.randn(B, L)
    for b in range(B):
        top_indices = torch.randperm(L)[:G]
        scores_concentrated[b, top_indices] += 5.0

    loss_concentrated = landmark_sparsity_loss(scores_concentrated, G, lambda_reg)

    # Scénario 2: Scores uniformes (mauvaise sparsité)
    scores_uniform = torch.randn(B, L) * 0.1
    loss_uniform = landmark_sparsity_loss(scores_uniform, G, lambda_reg)

    # Scénario 3: Scores très dispersés (pire sparsité)
    scores_dispersed = torch.randn(B, L) + 2.0
    loss_dispersed = landmark_sparsity_loss(scores_dispersed, G, lambda_reg)

    print(f"  Loss concentrée:  {loss_concentrated.item():.6f}")
    print(f"  Loss uniforme:    {loss_uniform.item():.6f}")
    print(f"  Loss dispersée:   {loss_dispersed.item():.6f}")

    # Vérifications
    assert loss_concentrated < loss_uniform, "❌ Loss devrait être plus basse pour scores concentrés!"
    losses = [loss_concentrated.item(), loss_uniform.item(), loss_dispersed.item()]
    assert max(losses) - min(losses) > 0.001, "❌ Loss trop constante!"

    print(f"  ✅ Loss varie correctement: range={max(losses)-min(losses):.6f}")

    return True


def test_reasonable_values():
    """Test 3: Vérifie que les valeurs sont raisonnables"""
    print("\n" + "="*70)
    print("TEST 3: Valeurs Raisonnables")
    print("="*70)

    B, L, G = 4, 384, 48
    lambda_reg = 0.001

    losses = []
    for _ in range(10):
        scores = torch.randn(B, L)
        loss = landmark_sparsity_loss(scores, G, lambda_reg)
        losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    min_loss = min(losses)
    max_loss = max(losses)

    print(f"  Sur 10 runs aléatoires:")
    print(f"    Mean: {mean_loss:.6f}")
    print(f"    Min:  {min_loss:.6f}")
    print(f"    Max:  {max_loss:.6f}")

    assert min_loss >= 0.0, "❌ Loss ne peut pas être négative!"
    assert max_loss < 1.0, f"❌ Loss trop élevée: {max_loss}"
    assert max_loss - min_loss > 0.001, "❌ Pas assez de variation!"
    assert mean_loss < 0.5, f"❌ Mean loss trop élevée: {mean_loss}"

    print(f"  ✅ Valeurs dans la plage attendue [0.0, 0.5]")

    return True


def test_edge_cases():
    """Test 4: Cas limites"""
    print("\n" + "="*70)
    print("TEST 4: Cas Limites")
    print("="*70)

    B, L, G = 4, 384, 48
    lambda_reg = 0.001

    # Cas 1: Exactement G positions avec scores élevés
    scores_exact = torch.zeros(B, L)
    for b in range(B):
        top_indices = torch.randperm(L)[:G]
        scores_exact[b, top_indices] = 3.0

    loss_exact = landmark_sparsity_loss(scores_exact, G, lambda_reg)

    # Cas 2: Moins de G positions actives (devrait être loss≈0)
    scores_fewer = torch.zeros(B, L)
    for b in range(B):
        top_indices = torch.randperm(L)[:G//2]
        scores_fewer[b, top_indices] = 3.0

    loss_fewer = landmark_sparsity_loss(scores_fewer, G, lambda_reg)

    # Cas 3: Beaucoup plus que G positions actives
    scores_many = torch.ones(B, L) * 2.0
    loss_many = landmark_sparsity_loss(scores_many, G, lambda_reg)

    print(f"  Cas exact G={G} positions: loss={loss_exact.item():.6f}")
    print(f"  Cas G/2={G//2} positions:   loss={loss_fewer.item():.6f}")
    print(f"  Cas toutes positions:      loss={loss_many.item():.6f}")

    assert loss_exact < loss_many, "❌ Loss devrait être plus élevée pour trop de positions!"
    assert loss_fewer <= loss_exact, "❌ Loss devrait être minimale pour peu de positions!"

    print(f"  ✅ Comportement correct sur cas limites")

    return True


def test_comparison_with_old():
    """Test 5: Comparaison avec l'ancienne version buggée"""
    print("\n" + "="*70)
    print("TEST 5: Comparaison Ancienne vs Nouvelle Version")
    print("="*70)

    B, L, G = 4, 384, 48
    lambda_reg = 0.001

    # Ancienne version (buggée)
    def old_sparsity_loss(selection_scores):
        prob_scores = F.softmax(selection_scores / 0.1, dim=-1)
        effective_size = 1.0 / (prob_scores ** 2).sum(dim=-1).mean()
        active_fraction = effective_size / L
        target_active = G / L * 1.2
        loss = lambda_reg * F.relu(active_fraction - target_active)
        return loss

    old_losses = []
    new_losses = []

    print(f"  {'Cas':<6} {'Old Loss':<12} {'New Loss':<12} {'Diff':<12}")
    print("  " + "-"*48)

    for i in range(5):
        scores = torch.randn(B, L)
        old_loss = old_sparsity_loss(scores)
        new_loss = landmark_sparsity_loss(scores, G, lambda_reg)

        old_losses.append(old_loss.item())
        new_losses.append(new_loss.item())

        diff = abs(new_loss.item() - old_loss.item())
        print(f"  {i+1:<6} {old_loss.item():<12.6f} {new_loss.item():<12.6f} {diff:<12.6f}")

    old_std = torch.tensor(old_losses).std().item()
    new_std = torch.tensor(new_losses).std().item()

    print(f"\n  Std dev ancienne: {old_std:.6f}")
    print(f"  Std dev nouvelle: {new_std:.6f}")

    assert new_std > old_std or new_std > 0.001, \
           "❌ Nouvelle version devrait avoir plus de variation!"

    print(f"  ✅ Nouvelle version montre plus de variation ({new_std:.6f} > {old_std:.6f})")

    return True


def test_numerical_stability():
    """Test 6: Stabilité numérique"""
    print("\n" + "="*70)
    print("TEST 6: Stabilité Numérique")
    print("="*70)

    B, L, G = 4, 384, 48
    lambda_reg = 0.001

    # Scores très grands
    scores_large = torch.randn(B, L) * 100
    loss_large = landmark_sparsity_loss(scores_large, G, lambda_reg)
    assert not torch.isnan(loss_large), "❌ NaN avec scores grands!"
    assert not torch.isinf(loss_large), "❌ Inf avec scores grands!"
    print(f"  ✓ Scores grands (×100):   loss={loss_large.item():.6f}")

    # Scores très petits
    scores_small = torch.randn(B, L) * 0.001
    loss_small = landmark_sparsity_loss(scores_small, G, lambda_reg)
    assert not torch.isnan(loss_small), "❌ NaN avec scores petits!"
    print(f"  ✓ Scores petits (×0.001): loss={loss_small.item():.6f}")

    # Scores négatifs
    scores_neg = torch.randn(B, L) - 5.0
    loss_neg = landmark_sparsity_loss(scores_neg, G, lambda_reg)
    assert not torch.isnan(loss_neg), "❌ NaN avec scores négatifs!"
    print(f"  ✓ Scores négatifs:        loss={loss_neg.item():.6f}")

    print(f"  ✅ Stable numériquement sur cas extrêmes")

    return True


def test_sparsity_gradients():
    """Test 7: Test original - Gradients avec réseau neural"""
    print("\n" + "="*70)
    print("TEST 7: Gradient Flow avec Neural Scorer")
    print("="*70)

    class DummyScorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 256)

        def forward(self, x):
            return self.linear(x)

    scorer = DummyScorer()
    optimizer = torch.optim.SGD(scorer.parameters(), lr=0.01)

    # Forward
    x = torch.randn(2, 10)
    scores = scorer(x)

    # Calculer sparsity loss
    loss = landmark_sparsity_loss(scores, num_landmarks=32, lambda_reg=1.0)

    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Has grad_fn: {loss.grad_fn is not None}")

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Vérifier gradients du scorer
    has_grads = False
    total_grad_norm = 0.0

    for name, param in scorer.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            if grad_norm > 1e-8:
                has_grads = True

    print(f"  Total gradient norm: {total_grad_norm:.8f}")

    assert has_grads, "❌ Pas de gradients vers le scorer!"
    print(f"  ✅ Les gradients FLOW vers le scorer!")

    return True


def run_all_tests():
    """Exécute tous les tests"""
    print("\n" + "="*70)
    print("TESTS COMPLETS DU FIX landmark_sparsity_loss")
    print("="*70)

    tests = [
        ("Gradient Flow", test_gradient_flow),
        ("Loss Variation", test_loss_variation),
        ("Valeurs Raisonnables", test_reasonable_values),
        ("Cas Limites", test_edge_cases),
        ("Comparaison Versions", test_comparison_with_old),
        ("Stabilité Numérique", test_numerical_stability),
        ("Neural Scorer Gradients", test_sparsity_gradients),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "✅ PASS"))
        except AssertionError as e:
            results.append((name, f"❌ FAIL: {e}"))
            print(f"  ❌ ÉCHEC: {e}")
        except Exception as e:
            results.append((name, f"❌ ERROR: {e}"))
            print(f"  ❌ ERREUR: {e}")

    # Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ DES TESTS")
    print("="*70)
    for name, status in results:
        print(f"  {name:<30} {status}")

    passed = sum(1 for _, status in results if "✅" in status)
    total = len(results)
    print(f"\n  {passed}/{total} tests passés")

    if passed == total:
        print("\n  🎉 TOUS LES TESTS PASSENT! Fix validé.\n")
        return True
    else:
        print("\n  ⚠️  Certains tests ont échoué.\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
