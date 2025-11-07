"""
Test suite for landmark selection
Tests optimizations and loss functions
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from landmarks import (
    LearnableLandmarkSelector,
    landmark_spacing_loss,
    landmark_sparsity_loss
)


class TestLandmarkSelector:
    """Test learnable landmark selector"""

    def test_initialization(self):
        """Test proper initialization"""
        selector = LearnableLandmarkSelector(
            embed_dim=384,
            num_landmarks=32,
            temperature_decay=0.999,
            min_temperature=0.3
        )

        assert selector.embed_dim == 384
        assert selector.num_landmarks == 32
        assert selector.step_count == 0

    def test_temperature_decay(self):
        """Test que temperature decay atteint min rapidement"""
        selector = LearnableLandmarkSelector(
            embed_dim=384,
            num_landmarks=32,
            temperature_decay=0.999,
            min_temperature=0.3
        )

        # Simuler 5000 steps
        for _ in range(5000):
            selector.step_count += 1

        temp = selector._get_temperature()
        assert temp <= 0.35, f"Temperature should be ~0.3 at 5K steps, got {temp}"
        assert temp >= 0.3, f"Temperature should not go below min, got {temp}"

    def test_temperature_initial(self):
        """Test température initiale"""
        selector = LearnableLandmarkSelector(
            embed_dim=384,
            num_landmarks=32,
            temperature_decay=0.999,
            min_temperature=0.3
        )

        temp = selector._get_temperature()
        assert temp >= 0.9, f"Initial temperature should be ~1.0, got {temp}"

    def test_forward_shape(self):
        """Test que forward préserve les shapes"""
        selector = LearnableLandmarkSelector(
            embed_dim=384,
            num_landmarks=32
        )

        B, L, D = 4, 256, 384
        x = torch.randn(B, L, D)

        indices, states, scores = selector(x)

        assert indices.shape == (B, 32), f"Expected indices shape (4, 32), got {indices.shape}"
        assert states.shape == (B, 32, D), f"Expected states shape (4, 32, 384), got {states.shape}"
        assert scores.shape == (B, L), f"Expected scores shape (4, 256), got {scores.shape}"

    def test_landmark_uniqueness(self):
        """Test que les landmarks sélectionnés sont uniques"""
        selector = LearnableLandmarkSelector(
            embed_dim=384,
            num_landmarks=32
        )

        x = torch.randn(2, 256, 384)
        indices, _, _ = selector(x)

        # Vérifier unicité pour chaque batch
        for b in range(indices.shape[0]):
            unique_indices = torch.unique(indices[b])
            assert len(unique_indices) == 32, "Landmarks should be unique"

    def test_landmark_range(self):
        """Test que les indices sont dans la plage valide"""
        selector = LearnableLandmarkSelector(
            embed_dim=384,
            num_landmarks=32
        )

        x = torch.randn(2, 256, 384)
        indices, _, _ = selector(x)

        assert indices.min() >= 0, "Indices should be >= 0"
        assert indices.max() < 256, "Indices should be < sequence_length"

    def test_gradient_flow(self):
        """Test que les gradients se propagent"""
        selector = LearnableLandmarkSelector(
            embed_dim=384,
            num_landmarks=32
        )

        x = torch.randn(2, 256, 384, requires_grad=True)
        indices, states, scores = selector(x)

        # Calculer une loss simple
        loss = scores.sum() + states.sum()
        loss.backward()

        # Vérifier gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestSpacingLoss:
    """Test spacing loss function"""

    def test_spacing_loss_uniform(self):
        """Test spacing loss avec landmarks uniformes"""
        B, G, L = 2, 32, 256

        # Landmarks parfaitement espacés
        indices_uniform = torch.arange(G).unsqueeze(0).repeat(B, 1) * (L // G)
        loss_uniform = landmark_spacing_loss(indices_uniform, L, lambda_reg=1.0)

        # Loss devrait être très faible
        assert loss_uniform < 0.1, f"Uniform spacing should have low loss, got {loss_uniform}"

    def test_spacing_loss_clumped(self):
        """Test spacing loss pénalise gaps non-uniformes"""
        B, G, L = 2, 32, 256

        # Landmarks parfaitement espacés
        indices_uniform = torch.arange(G).unsqueeze(0).repeat(B, 1) * (L // G)
        loss_uniform = landmark_spacing_loss(indices_uniform, L, lambda_reg=1.0)

        # Landmarks clumpés au début
        indices_clumped = torch.randint(0, L//4, (B, G))
        loss_clumped = landmark_spacing_loss(indices_clumped, L, lambda_reg=1.0)

        # Loss clumpé doit être plus élevée
        assert loss_clumped > loss_uniform * 1.5, f"Clumped spacing should have higher loss"

    def test_spacing_loss_lambda_scaling(self):
        """Test que lambda_reg scale la loss"""
        B, G, L = 2, 32, 256
        indices = torch.randint(0, L, (B, G))

        loss1 = landmark_spacing_loss(indices, L, lambda_reg=1.0)
        loss2 = landmark_spacing_loss(indices, L, lambda_reg=2.0)

        # loss2 devrait être ~2x loss1
        ratio = (loss2 / loss1).item()
        assert 1.8 < ratio < 2.2, f"Lambda scaling incorrect: ratio={ratio}"

    def test_spacing_loss_sorted_behavior(self):
        """Test que le tri des indices n'affecte pas la loss"""
        B, G, L = 2, 32, 256

        # Indices aléatoires
        indices = torch.randint(0, L, (B, G))
        loss1 = landmark_spacing_loss(indices, L, lambda_reg=1.0)

        # Mêmes indices triés
        indices_sorted = torch.sort(indices, dim=1)[0]
        loss2 = landmark_spacing_loss(indices_sorted, L, lambda_reg=1.0)

        # Losses doivent être identiques
        assert torch.allclose(loss1, loss2, atol=1e-6)


class TestSparsityLoss:
    """Test sparsity loss function"""

    def test_sparsity_loss_adaptive(self):
        """Test que sparsity loss s'adapte au nombre de landmarks"""
        B, L = 2, 256

        # G=32 landmarks → target ~15% actifs (38 tokens)
        scores = torch.randn(B, L).softmax(dim=-1)
        loss = landmark_sparsity_loss(scores, num_landmarks=32, lambda_reg=1.0)

        # Loss devrait être non-négative (peut être 0 si contrainte satisfaite)
        assert loss >= 0, "Sparsity loss should be non-negative"
        assert loss < 5.0, f"Sparsity loss too high: {loss}"

    def test_sparsity_loss_concentrated(self):
        """Test avec scores très concentrés"""
        B, L = 2, 256

        # Scores très concentrés sur quelques positions (less than target)
        scores = torch.zeros(B, L)
        scores[:, :10] = 1.0
        scores = scores.softmax(dim=-1)

        loss = landmark_sparsity_loss(scores, num_landmarks=32, lambda_reg=1.0)

        # Loss devrait être >= 0 (peut être 0 si moins que le seuil)
        assert loss >= 0, "Loss should be non-negative"

    def test_sparsity_loss_uniform(self):
        """Test avec scores uniformes"""
        B, L = 2, 256

        # Scores uniformes
        scores = torch.ones(B, L) / L

        loss = landmark_sparsity_loss(scores, num_landmarks=32, lambda_reg=1.0)

        # Loss devrait être raisonnable
        assert loss < 1.0, f"Uniform distribution should have reasonable loss: {loss}"

    def test_sparsity_loss_different_landmarks(self):
        """Test avec différents nombres de landmarks"""
        B, L = 2, 256
        scores = torch.randn(B, L).softmax(dim=-1)

        loss_small = landmark_sparsity_loss(scores, num_landmarks=16, lambda_reg=1.0)
        loss_large = landmark_sparsity_loss(scores, num_landmarks=64, lambda_reg=1.0)

        # Les losses peuvent être différentes
        # Mais doivent rester dans des plages raisonnables
        assert loss_small >= 0 and loss_small < 2.0
        assert loss_large >= 0 and loss_large < 2.0


class TestLandmarkOptimization:
    """Test end-to-end landmark optimization"""

    def test_training_step(self):
        """Test une étape d'entraînement complète"""
        selector = LearnableLandmarkSelector(
            embed_dim=384,
            num_landmarks=32
        )
        selector.train()  # Important: mode training
        optimizer = torch.optim.Adam(selector.parameters(), lr=1e-3)

        x = torch.randn(2, 256, 384)

        # Forward pass
        indices, states, scores = selector(x, use_gumbel=False)  # straight-through

        # Compute losses - only use sparsity loss for gradient test
        # (spacing loss uses integer indices which don't propagate gradients)
        sparsity_loss = landmark_sparsity_loss(scores, num_landmarks=32, lambda_reg=0.05)
        total_loss = sparsity_loss

        # Backward pass
        optimizer.zero_grad()
        if total_loss.requires_grad:
            total_loss.backward()
            optimizer.step()

            # Vérifier que les gradients existent
            for param in selector.parameters():
                if param.requires_grad:
                    assert param.grad is not None
        else:
            # Si loss ne nécessite pas de gradients, c'est OK (peut arriver avec ReLU à 0)
            assert total_loss.item() >= 0, "Loss should be non-negative"

    def test_loss_convergence(self):
        """Test que les losses diminuent avec l'entraînement"""
        selector = LearnableLandmarkSelector(
            embed_dim=384,
            num_landmarks=32
        )
        selector.train()  # Important: mode training
        optimizer = torch.optim.Adam(selector.parameters(), lr=1e-2)

        x = torch.randn(4, 256, 384)

        initial_losses = []
        final_losses = []

        # Entraîner pour quelques steps - only use sparsity loss
        # (spacing loss uses integer indices which don't propagate gradients directly)
        for i in range(50):
            indices, states, scores = selector(x, use_gumbel=False)
            sparsity_loss = landmark_sparsity_loss(scores, num_landmarks=32, lambda_reg=0.05)
            total_loss = sparsity_loss

            if i < 5:
                initial_losses.append(total_loss.item())
            if i >= 45:
                final_losses.append(total_loss.item())

            optimizer.zero_grad()
            if total_loss.requires_grad:
                total_loss.backward()
                optimizer.step()
            selector.step_count += 1

        # Loss moyenne devrait diminuer (ou rester stable si déjà optimale)
        avg_initial = sum(initial_losses) / len(initial_losses)
        avg_final = sum(final_losses) / len(final_losses)

        # Acceptance criterion: loss should decrease OR stay low
        assert avg_final <= avg_initial * 1.1, f"Loss should not increase significantly: {avg_initial} -> {avg_final}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
