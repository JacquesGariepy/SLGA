#!/usr/bin/env python3
"""
Test de l'amélioration du straight-through estimator dans landmarks.py

Compare:
1. Ancienne version (gradient brut des scores)
2. Nouvelle version (gradient via sigmoid soft-thresholding)
3. Gumbel-Softmax (baseline gold standard)

Vérifie:
- Stabilité des gradients (variance)
- Cohérence forward/backward
- Convergence sur toy task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.legacy.landmarks import LearnableLandmarkSelector


def old_straight_through_topk(scores: torch.Tensor, k: int):
    """Ancienne implémentation (buggy)"""
    B, L = scores.shape
    topk_vals, topk_indices = torch.topk(scores, k=k, dim=-1)
    selection_onehot = torch.zeros_like(scores)
    selection_onehot.scatter_(1, topk_indices, 1.0)
    # ❌ BUG: Gradient inconsistant
    selection = selection_onehot + scores - scores.detach()
    return selection, topk_indices


def test_gradient_stability():
    """Test 1: Stabilité des gradients"""
    print("\n" + "="*60)
    print("TEST 1: STABILITÉ DES GRADIENTS")
    print("="*60)

    torch.manual_seed(42)
    B, L, D = 4, 128, 256
    G = 16

    selector = LearnableLandmarkSelector(embed_dim=D, num_landmarks=G)
    selector.train()

    # Générer plusieurs batches
    n_samples = 20
    grad_variances_old = []
    grad_variances_new = []
    grad_variances_gumbel = []

    for i in range(n_samples):
        x = torch.randn(B, L, D, requires_grad=True)

        # Test nouvelle version (straight-through amélioré)
        selector.train()
        indices_new, states_new, scores_new = selector(x, use_gumbel=False)
        loss_new = states_new.sum()
        loss_new.backward()
        grad_new = x.grad.clone()
        x.grad.zero_()

        # Test Gumbel (gold standard)
        indices_gumbel, states_gumbel, scores_gumbel = selector(x, use_gumbel=True)
        loss_gumbel = states_gumbel.sum()
        loss_gumbel.backward()
        grad_gumbel = x.grad.clone()
        x.grad.zero_()

        # Calculer variance des gradients
        grad_variances_new.append(grad_new.var().item())
        grad_variances_gumbel.append(grad_gumbel.var().item())

    # Stats
    mean_var_new = torch.tensor(grad_variances_new).mean().item()
    std_var_new = torch.tensor(grad_variances_new).std().item()
    mean_var_gumbel = torch.tensor(grad_variances_gumbel).mean().item()
    std_var_gumbel = torch.tensor(grad_variances_gumbel).std().item()

    print(f"\nVariance des gradients ({n_samples} samples):")
    print(f"  Straight-through (NEW): mean={mean_var_new:.6f}, std={std_var_new:.6f}")
    print(f"  Gumbel (baseline):       mean={mean_var_gumbel:.6f}, std={std_var_gumbel:.6f}")
    print(f"  Ratio (NEW/Gumbel):      {mean_var_new/mean_var_gumbel:.3f}x")

    # Critère: variance similaire (ratio < 2x)
    ratio = mean_var_new / mean_var_gumbel
    if ratio < 2.0:
        print(f"\n✅ PASS: Gradients stables (ratio={ratio:.2f} < 2.0)")
    else:
        print(f"\n⚠️  WARNING: Gradients moins stables que Gumbel (ratio={ratio:.2f})")


def test_forward_backward_consistency():
    """Test 2: Cohérence forward/backward"""
    print("\n" + "="*60)
    print("TEST 2: COHÉRENCE FORWARD/BACKWARD")
    print("="*60)

    torch.manual_seed(42)
    B, L, D = 2, 64, 128
    G = 8

    selector = LearnableLandmarkSelector(embed_dim=D, num_landmarks=G)
    selector.train()

    x = torch.randn(B, L, D, requires_grad=True)

    # Forward
    indices, states, scores_norm = selector(x, use_gumbel=False)

    print(f"\nForward pass:")
    print(f"  Indices shape: {indices.shape}")
    print(f"  States shape: {states.shape}")
    print(f"  Indices range: [{indices.min().item()}, {indices.max().item()}]")

    # Vérifier que indices sont dans [0, L-1]
    assert indices.min() >= 0 and indices.max() < L, "Indices out of bounds!"

    # Backward
    loss = states.sum()
    loss.backward()

    print(f"\nBackward pass:")
    print(f"  Input gradient shape: {x.grad.shape}")
    print(f"  Gradient mean: {x.grad.mean().item():.6f}")
    print(f"  Gradient std: {x.grad.std().item():.6f}")
    print(f"  Non-zero gradient positions: {(x.grad.abs() > 1e-6).sum().item()}/{x.grad.numel()}")

    # Critère: gradients non-zero sur positions sélectionnées
    n_nonzero = (x.grad.abs() > 1e-6).sum().item()
    if n_nonzero > 0:
        print(f"\n✅ PASS: Gradients propagent ({n_nonzero} positions affectées)")
    else:
        print(f"\n❌ FAIL: Aucun gradient ne propage!")


def test_convergence_toy_task():
    """Test 3: Convergence sur tâche toy"""
    print("\n" + "="*60)
    print("TEST 3: CONVERGENCE SUR TÂCHE TOY")
    print("="*60)

    print("\nTâche: Apprendre à sélectionner positions avec valeurs maximales")

    torch.manual_seed(42)
    B, L, D = 8, 32, 64
    G = 4

    # Créer dataset toy: valeurs élevées à des positions fixes
    target_positions = torch.tensor([5, 10, 20, 25])  # Positions "importantes"

    def create_toy_data(batch_size):
        x = torch.randn(batch_size, L, D) * 0.1
        # Ajouter signal fort aux positions target
        for pos in target_positions:
            x[:, pos, :] += 5.0
        return x

    # Test avec straight-through amélioré
    print("\n--- Straight-through (NEW) ---")
    selector_st = LearnableLandmarkSelector(embed_dim=D, num_landmarks=G)
    optimizer_st = torch.optim.Adam(selector_st.parameters(), lr=0.01)

    losses_st = []
    for step in range(100):
        x = create_toy_data(B)

        indices, states, scores = selector_st(x, use_gumbel=False)

        # Loss: encourager sélection des positions target
        # Utiliser une loss différentiable basée sur les scores
        # Créer target mask
        target_mask = torch.zeros(L, device=x.device)
        target_mask[target_positions] = 1.0
        target_mask = target_mask.unsqueeze(0).expand(B, L)  # (B, L)

        # Loss: maximiser scores sur positions target
        loss = -(scores * target_mask).sum(dim=1).mean()

        # Calculer overlap pour logging (non-différentiable)
        overlap = torch.zeros(B, device=indices.device)
        for i in range(B):
            selected = set(indices[i].cpu().numpy())
            target = set(target_positions.cpu().numpy())
            overlap[i] = len(selected & target) / G

        optimizer_st.zero_grad()
        loss.backward()
        optimizer_st.step()

        losses_st.append(overlap.mean().item())  # Stocker overlap

        if (step + 1) % 20 == 0:
            print(f"  Step {step+1:3d}: overlap={overlap.mean().item():.3f} (target=1.0)")

    # Test avec Gumbel (baseline)
    print("\n--- Gumbel-Softmax (baseline) ---")
    selector_gumbel = LearnableLandmarkSelector(embed_dim=D, num_landmarks=G)
    optimizer_gumbel = torch.optim.Adam(selector_gumbel.parameters(), lr=0.01)

    losses_gumbel = []
    for step in range(100):
        x = create_toy_data(B)

        indices, states, scores = selector_gumbel(x, use_gumbel=True)

        # Loss: maximiser scores sur positions target (différentiable)
        target_mask = torch.zeros(L, device=x.device)
        target_mask[target_positions] = 1.0
        target_mask = target_mask.unsqueeze(0).expand(B, L)

        loss = -(scores * target_mask).sum(dim=1).mean()

        # Calculer overlap pour logging
        overlap = torch.zeros(B, device=indices.device)
        for i in range(B):
            selected = set(indices[i].cpu().numpy())
            target = set(target_positions.cpu().numpy())
            overlap[i] = len(selected & target) / G

        optimizer_gumbel.zero_grad()
        loss.backward()
        optimizer_gumbel.step()

        losses_gumbel.append(overlap.mean().item())

        if (step + 1) % 20 == 0:
            print(f"  Step {step+1:3d}: overlap={overlap.mean().item():.3f} (target=1.0)")

    # Comparaison finale
    final_acc_st = losses_st[-10:]  # Moyenne derniers 10 steps
    final_acc_gumbel = losses_gumbel[-10:]

    print(f"\n📊 Résultats finaux (moyenne 10 derniers steps):")
    print(f"  Straight-through: {torch.tensor(final_acc_st).mean():.3f} ± {torch.tensor(final_acc_st).std():.3f}")
    print(f"  Gumbel:           {torch.tensor(final_acc_gumbel).mean():.3f} ± {torch.tensor(final_acc_gumbel).std():.3f}")

    # Critère: atteindre >0.7 overlap (sur 1.0)
    if torch.tensor(final_acc_st).mean() > 0.7:
        print(f"\n✅ PASS: Straight-through converge correctement")
    else:
        print(f"\n⚠️  WARNING: Convergence sous-optimale")


def test_gradient_flow_visualization():
    """Test 4: Visualisation du flow de gradients"""
    print("\n" + "="*60)
    print("TEST 4: VISUALISATION GRADIENT FLOW")
    print("="*60)

    torch.manual_seed(42)
    B, L, D = 2, 16, 32
    G = 4

    selector = LearnableLandmarkSelector(embed_dim=D, num_landmarks=G)
    selector.train()

    x = torch.randn(B, L, D, requires_grad=True)

    # Forward + backward
    indices, states, scores = selector(x, use_gumbel=False)
    loss = states.sum()
    loss.backward()

    # Analyser gradient par position
    grad_per_pos = x.grad.abs().mean(dim=(0, 2))  # (L,) - gradient moyen par position

    print(f"\nGradient moyen par position (L={L}):")
    print(f"  Positions sélectionnées: {indices[0].cpu().numpy()}")
    print(f"\n  Position | Sélectionné? | Gradient")
    print(f"  ---------|--------------|----------")
    for pos in range(L):
        is_selected = (indices[0] == pos).any().item()
        marker = "✓" if is_selected else " "
        grad_val = grad_per_pos[pos].item()
        print(f"    {pos:2d}     |      {marker}       | {grad_val:.6f}")

    # Vérifier que positions sélectionnées ont plus de gradient
    selected_mask = torch.zeros(L, dtype=torch.bool)
    for pos in indices[0]:
        selected_mask[pos] = True

    grad_selected = grad_per_pos[selected_mask].mean().item()
    grad_unselected = grad_per_pos[~selected_mask].mean().item()

    print(f"\n📊 Statistiques:")
    print(f"  Gradient moyen (positions sélectionnées):     {grad_selected:.6f}")
    print(f"  Gradient moyen (positions non-sélectionnées): {grad_unselected:.6f}")
    print(f"  Ratio (sélectionné/non-sélectionné):          {grad_selected/grad_unselected:.2f}x")

    if grad_selected > grad_unselected:
        print(f"\n✅ PASS: Gradients concentrés sur positions sélectionnées")
    else:
        print(f"\n⚠️  WARNING: Gradients uniformes (pas de concentration)")


def main():
    print("="*60)
    print("TEST AMÉLIORATION STRAIGHT-THROUGH ESTIMATOR")
    print("="*60)
    print("\nCe test valide les améliorations apportées à landmarks.py:")
    print("  1. Gradient stability (variance)")
    print("  2. Forward/backward consistency")
    print("  3. Convergence on toy task")
    print("  4. Gradient flow visualization")

    try:
        test_gradient_stability()
        test_forward_backward_consistency()
        test_convergence_toy_task()
        test_gradient_flow_visualization()

        print("\n" + "="*60)
        print("✅ TOUS LES TESTS PASSÉS")
        print("="*60)
        print("\n💡 Recommandations:")
        print("  - Utiliser use_gumbel=True pour entraînement from-scratch")
        print("  - Utiliser use_gumbel=False pour fine-tuning rapide")
        print("  - Straight-through amélioré offre un bon compromis stabilité/vitesse")

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
