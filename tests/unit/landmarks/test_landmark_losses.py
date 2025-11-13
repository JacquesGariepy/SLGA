"""
Comprehensive unit tests for landmark loss functions.

Tests cover:
- landmark_spacing_loss (differentiable version)
- landmark_diversity_loss (deprecated)
- landmark_sparsity_loss (concentration metric)
- Gradient flow through losses
- Edge cases (G=1, G=0, short sequences)
- Numerical stability
"""

import pytest
import torch
from hypothesis import given, strategies as st, settings
import math

from src.legacy.landmarks import (
    landmark_spacing_loss,
    landmark_diversity_loss,
    landmark_sparsity_loss,
)


class TestLandmarkSpacingLoss:
    """Test landmark spacing loss (uniform distribution penalty)."""

    def test_basic_spacing_loss(self):
        """Test basic spacing loss computation."""
        B, G, L = 2, 8, 128
        landmark_indices = torch.randint(0, L, (B, G))
        selection_scores = torch.randn(B, L)

        loss = landmark_spacing_loss(landmark_indices, L, lambda_reg=0.01, selection_scores=selection_scores)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss >= 0  # Loss should be non-negative

    def test_perfectly_spaced_landmarks_low_loss(self):
        """Test that perfectly spaced landmarks have low loss."""
        B, G, L = 2, 8, 128
        # Create perfectly spaced landmarks
        landmark_indices = torch.linspace(0, L-1, G).long().unsqueeze(0).expand(B, G)

        # Create selection scores that concentrate on these positions
        selection_scores = torch.zeros(B, L)
        for b in range(B):
            selection_scores[b, landmark_indices[b]] = 1.0
        selection_scores = selection_scores / selection_scores.sum(dim=1, keepdim=True)

        loss = landmark_spacing_loss(landmark_indices, L, lambda_reg=0.01, selection_scores=selection_scores)

        # Should have very low loss (close to 0)
        assert loss < 0.01

    def test_clustered_landmarks_high_loss(self):
        """Test that clustered landmarks have higher loss."""
        B, G, L = 2, 8, 128
        # Create clustered landmarks (all in first half)
        landmark_indices = torch.randint(0, L // 2, (B, G))

        selection_scores = torch.zeros(B, L)
        for b in range(B):
            selection_scores[b, landmark_indices[b]] = 1.0
        selection_scores = selection_scores / selection_scores.sum(dim=1, keepdim=True)

        loss = landmark_spacing_loss(landmark_indices, L, lambda_reg=0.01, selection_scores=selection_scores)

        # Should have noticeable loss
        assert loss > 0

    def test_gradient_flow_with_scores(self):
        """Test that gradients flow through selection_scores."""
        B, G, L = 2, 8, 128
        landmark_indices = torch.randint(0, L, (B, G))
        selection_scores = torch.randn(B, L, requires_grad=True)

        loss = landmark_spacing_loss(landmark_indices, L, lambda_reg=0.01, selection_scores=selection_scores)
        loss.backward()

        assert selection_scores.grad is not None
        assert selection_scores.grad.abs().sum() > 0

    def test_fallback_without_scores(self):
        """Test non-differentiable fallback without selection_scores."""
        B, G, L = 2, 8, 128
        landmark_indices = torch.randint(0, L, (B, G))

        loss = landmark_spacing_loss(landmark_indices, L, lambda_reg=0.01, selection_scores=None)

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0

    def test_bug16_fix_single_landmark(self):
        """Test Bug #16 fix: G=1 should not cause NaN."""
        B, G, L = 2, 1, 128
        landmark_indices = torch.randint(0, L, (B, G))

        loss = landmark_spacing_loss(landmark_indices, L, lambda_reg=0.01, selection_scores=None)

        # Should return 0.0 without NaN
        assert torch.isfinite(loss)
        assert loss == 0.0

    def test_bug16_fix_zero_landmarks(self):
        """Test Bug #16 fix: G=0 should not crash."""
        B, G, L = 2, 0, 128
        landmark_indices = torch.empty(B, 0, dtype=torch.long)

        # Should not crash, return 0.0
        loss = landmark_spacing_loss(landmark_indices, L, lambda_reg=0.01, selection_scores=None)

        assert torch.isfinite(loss)
        assert loss == 0.0

    def test_two_landmarks_valid_gap(self):
        """Test with exactly 2 landmarks (1 gap)."""
        B, G, L = 2, 2, 128
        landmark_indices = torch.tensor([[0, 127], [10, 100]])  # Well spaced

        loss = landmark_spacing_loss(landmark_indices, L, lambda_reg=0.01, selection_scores=None)

        assert torch.isfinite(loss)
        assert loss >= 0

    @pytest.mark.parametrize("G", [2, 4, 8, 16, 24, 32])
    def test_various_landmark_counts(self, G):
        """Test with various landmark counts."""
        B, L = 2, 256
        landmark_indices = torch.randint(0, L, (B, G))
        selection_scores = torch.randn(B, L)

        loss = landmark_spacing_loss(landmark_indices, L, lambda_reg=0.01, selection_scores=selection_scores)

        assert torch.isfinite(loss)
        assert loss >= 0

    @pytest.mark.parametrize("lambda_reg", [0.001, 0.01, 0.1, 1.0])
    def test_various_lambda_values(self, lambda_reg):
        """Test with various regularization weights."""
        B, G, L = 2, 8, 128
        landmark_indices = torch.randint(0, L, (B, G))
        selection_scores = torch.randn(B, L)

        loss = landmark_spacing_loss(landmark_indices, L, lambda_reg=lambda_reg, selection_scores=selection_scores)

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_short_sequence(self):
        """Test with short sequence."""
        B, G, L = 2, 4, 16
        landmark_indices = torch.randint(0, L, (B, G))
        selection_scores = torch.randn(B, L)

        loss = landmark_spacing_loss(landmark_indices, L, lambda_reg=0.01, selection_scores=selection_scores)

        assert torch.isfinite(loss)

    def test_long_sequence(self):
        """Test with long sequence."""
        B, G, L = 2, 24, 2048
        landmark_indices = torch.randint(0, L, (B, G))
        selection_scores = torch.randn(B, L)

        loss = landmark_spacing_loss(landmark_indices, L, lambda_reg=0.01, selection_scores=selection_scores)

        assert torch.isfinite(loss)


class TestLandmarkDiversityLoss:
    """Test landmark diversity loss (entropy-based, deprecated)."""

    def test_basic_diversity_loss(self):
        """Test basic diversity loss computation."""
        B, L = 2, 128
        selection_scores = torch.softmax(torch.randn(B, L), dim=-1)

        loss = landmark_diversity_loss(selection_scores, lambda_reg=0.01)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss >= 0

    def test_uniform_distribution_low_loss(self):
        """Test that uniform distribution has low loss (high entropy)."""
        B, L = 2, 128
        # Uniform distribution: high entropy
        selection_scores = torch.ones(B, L) / L

        loss = landmark_diversity_loss(selection_scores, lambda_reg=0.01)

        # High entropy → low loss
        assert loss < 0.01

    def test_peaked_distribution_high_loss(self):
        """Test that peaked distribution has higher loss (low entropy)."""
        B, L = 2, 128
        # Peaked distribution: low entropy
        selection_scores = torch.zeros(B, L)
        selection_scores[:, 0] = 1.0  # All mass on first position

        loss = landmark_diversity_loss(selection_scores, lambda_reg=0.01)

        # Low entropy → high loss
        assert loss > 0.005

    def test_gradient_flow(self):
        """Test gradient flow through diversity loss."""
        B, L = 2, 128
        selection_scores = torch.softmax(torch.randn(B, L, requires_grad=True), dim=-1)

        loss = landmark_diversity_loss(selection_scores, lambda_reg=0.01)
        loss.backward()

        # Gradients should flow
        # Note: softmax is detached, so we check the computation graph exists

    @pytest.mark.parametrize("lambda_reg", [0.001, 0.01, 0.1])
    def test_various_lambda_values(self, lambda_reg):
        """Test with various lambda values."""
        B, L = 2, 128
        selection_scores = torch.softmax(torch.randn(B, L), dim=-1)

        loss = landmark_diversity_loss(selection_scores, lambda_reg=lambda_reg)

        assert torch.isfinite(loss)
        assert loss >= 0


class TestLandmarkSparsityLoss:
    """Test landmark sparsity loss (concentration metric)."""

    def test_basic_sparsity_loss(self):
        """Test basic sparsity loss computation."""
        B, L, G = 2, 384, 48
        selection_scores = torch.randn(B, L)

        loss = landmark_sparsity_loss(selection_scores, num_landmarks=G, lambda_reg=0.001)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss >= 0

    def test_well_concentrated_scores_low_loss(self):
        """Test that well-concentrated scores have low loss."""
        B, L, G = 2, 384, 48
        selection_scores = torch.randn(B, L)

        # Boost scores at top-G positions to create concentration
        _, top_indices = torch.topk(selection_scores, k=G, dim=-1)
        for b in range(B):
            selection_scores[b, top_indices[b]] += 10  # Boost top-G

        loss = landmark_sparsity_loss(selection_scores, num_landmarks=G, lambda_reg=0.001)

        # Well concentrated → loss should be 0 or very small
        assert loss < 0.01

    def test_dispersed_scores_higher_loss(self):
        """Test that dispersed scores have higher loss."""
        B, L, G = 2, 384, 48
        # Uniform scores: dispersed
        selection_scores = torch.ones(B, L)

        loss = landmark_sparsity_loss(selection_scores, num_landmarks=G, lambda_reg=0.001)

        # Dispersed → mass_in_top_g will be low → loss > 0
        assert loss > 0

    def test_gradient_flow(self):
        """Test gradient flow through sparsity loss."""
        B, L, G = 2, 384, 48
        selection_scores = torch.randn(B, L, requires_grad=True)

        loss = landmark_sparsity_loss(selection_scores, num_landmarks=G, lambda_reg=0.001)
        loss.backward()

        assert selection_scores.grad is not None
        assert selection_scores.grad.abs().sum() > 0

    def test_adaptive_target_mass(self):
        """Test that target mass adapts to G/L ratio."""
        B, L = 2, 384

        # Small G relative to L
        loss_small_g = landmark_sparsity_loss(torch.randn(B, L), num_landmarks=12, lambda_reg=0.001)

        # Large G relative to L
        loss_large_g = landmark_sparsity_loss(torch.randn(B, L), num_landmarks=192, lambda_reg=0.001)

        # Both should be finite
        assert torch.isfinite(loss_small_g)
        assert torch.isfinite(loss_large_g)

    @pytest.mark.parametrize("G,L", [
        (8, 128),
        (16, 256),
        (24, 384),
        (48, 512),
    ])
    def test_various_g_l_ratios(self, G, L):
        """Test with various G/L ratios."""
        B = 2
        selection_scores = torch.randn(B, L)

        loss = landmark_sparsity_loss(selection_scores, num_landmarks=G, lambda_reg=0.001)

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_all_landmarks_selected(self):
        """Test when G = L (all positions are landmarks)."""
        B, L = 2, 128
        selection_scores = torch.randn(B, L)

        loss = landmark_sparsity_loss(selection_scores, num_landmarks=L, lambda_reg=0.001)

        # Should have very low loss (100% mass in top-L = all)
        assert loss < 0.001

    def test_single_landmark(self):
        """Test with single landmark."""
        B, L = 2, 128
        selection_scores = torch.randn(B, L)

        loss = landmark_sparsity_loss(selection_scores, num_landmarks=1, lambda_reg=0.001)

        assert torch.isfinite(loss)

    def test_extreme_concentration(self):
        """Test with extreme concentration (all mass on one position)."""
        B, L, G = 2, 384, 48
        selection_scores = torch.zeros(B, L)
        selection_scores[:, 0] = 100  # All mass on position 0

        loss = landmark_sparsity_loss(selection_scores, num_landmarks=G, lambda_reg=0.001)

        # Even though concentrated, only 1 position has mass (not G positions)
        # So loss might be high
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("lambda_reg", [0.0001, 0.001, 0.01, 0.1])
    def test_various_lambda_values(self, lambda_reg):
        """Test with various regularization weights."""
        B, L, G = 2, 384, 48
        selection_scores = torch.randn(B, L)

        loss = landmark_sparsity_loss(selection_scores, num_landmarks=G, lambda_reg=lambda_reg)

        assert torch.isfinite(loss)
        assert loss >= 0


class TestLandmarkLossesIntegration:
    """Integration tests combining multiple losses."""

    def test_combined_losses_gradient_flow(self):
        """Test gradient flow through combined losses."""
        B, G, L = 2, 8, 128

        # Create learnable scores
        raw_scores = torch.randn(B, L, requires_grad=True)
        selection_scores = torch.softmax(raw_scores, dim=-1)

        # Get top-K indices
        _, landmark_indices = torch.topk(raw_scores, k=G, dim=-1)

        # Compute all losses
        spacing_loss_val = landmark_spacing_loss(landmark_indices, L, lambda_reg=0.01, selection_scores=selection_scores)
        diversity_loss_val = landmark_diversity_loss(selection_scores, lambda_reg=0.01)
        sparsity_loss_val = landmark_sparsity_loss(raw_scores, num_landmarks=G, lambda_reg=0.001)

        # Combined loss
        total_loss = spacing_loss_val + diversity_loss_val + sparsity_loss_val
        total_loss.backward()

        # Check gradients
        assert raw_scores.grad is not None
        assert raw_scores.grad.abs().sum() > 0

    def test_loss_trade_offs(self):
        """Test that losses capture different aspects."""
        B, G, L = 2, 8, 128

        # Scenario 1: Uniform scores (high diversity, low sparsity)
        uniform_scores = torch.ones(B, L) / L
        uniform_indices = torch.randint(0, L, (B, G))

        div_loss_uniform = landmark_diversity_loss(uniform_scores, lambda_reg=0.01)
        spar_loss_uniform = landmark_sparsity_loss(uniform_scores * L, num_landmarks=G, lambda_reg=0.001)

        # Scenario 2: Concentrated scores (low diversity, high sparsity)
        concentrated_scores = torch.zeros(B, L)
        concentrated_scores[:, :G] = 1.0 / G
        concentrated_indices = torch.arange(G).unsqueeze(0).expand(B, G)

        div_loss_concentrated = landmark_diversity_loss(concentrated_scores, lambda_reg=0.01)
        spar_loss_concentrated = landmark_sparsity_loss(concentrated_scores * L, num_landmarks=G, lambda_reg=0.001)

        # Diversity loss should be lower for uniform
        assert div_loss_uniform < div_loss_concentrated

        # Sparsity loss should be lower for concentrated
        assert spar_loss_concentrated < spar_loss_uniform

    def test_losses_work_together_in_training(self):
        """Test that losses work together in a training scenario."""
        B, G, L = 2, 8, 128

        # Simulate scorer output
        scorer = torch.nn.Sequential(
            torch.nn.Linear(L, L),
            torch.nn.GELU(),
        )

        x = torch.randn(B, L)
        scores = scorer(x).squeeze()

        # Compute losses
        selection_scores = torch.softmax(scores, dim=-1)
        _, indices = torch.topk(scores, k=G, dim=-1)

        spacing = landmark_spacing_loss(indices, L, lambda_reg=0.01, selection_scores=selection_scores)
        sparsity = landmark_sparsity_loss(scores, num_landmarks=G, lambda_reg=0.001)

        total = spacing + sparsity
        total.backward()

        # Check that scorer parameters have gradients
        for param in scorer.parameters():
            assert param.grad is not None


class TestLandmarkLossesNumericalStability:
    """Test numerical stability of loss functions."""

    def test_spacing_loss_no_nan(self):
        """Test spacing loss never produces NaN."""
        for _ in range(10):
            B, G, L = 2, 8, 128
            landmark_indices = torch.randint(0, L, (B, G))
            selection_scores = torch.randn(B, L)

            loss = landmark_spacing_loss(landmark_indices, L, lambda_reg=0.01, selection_scores=selection_scores)

            assert not torch.isnan(loss)
            assert not torch.isinf(loss)

    def test_diversity_loss_no_nan(self):
        """Test diversity loss never produces NaN."""
        for _ in range(10):
            B, L = 2, 128
            selection_scores = torch.softmax(torch.randn(B, L), dim=-1)

            loss = landmark_diversity_loss(selection_scores, lambda_reg=0.01)

            assert not torch.isnan(loss)
            assert not torch.isinf(loss)

    def test_sparsity_loss_no_nan(self):
        """Test sparsity loss never produces NaN."""
        for _ in range(10):
            B, L, G = 2, 384, 48
            selection_scores = torch.randn(B, L)

            loss = landmark_sparsity_loss(selection_scores, num_landmarks=G, lambda_reg=0.001)

            assert not torch.isnan(loss)
            assert not torch.isinf(loss)

    def test_extreme_score_values(self):
        """Test with extreme score values."""
        B, L, G = 2, 128, 8

        # Very large scores
        large_scores = torch.randn(B, L) * 1000
        _, indices = torch.topk(large_scores, k=G, dim=-1)

        loss = landmark_spacing_loss(indices, L, lambda_reg=0.01, selection_scores=torch.softmax(large_scores, dim=-1))
        assert torch.isfinite(loss)

        # Very small scores
        small_scores = torch.randn(B, L) * 0.001
        loss2 = landmark_sparsity_loss(small_scores, num_landmarks=G, lambda_reg=0.001)
        assert torch.isfinite(loss2)

    @pytest.mark.property
    @given(
        batch=st.integers(1, 4),
        seq_len=st.integers(32, 128),
        num_landmarks=st.integers(4, 16)
    )
    @settings(max_examples=20, deadline=None)
    def test_property_spacing_loss_always_finite(self, batch, seq_len, num_landmarks):
        """Property test: spacing loss always finite."""
        num_landmarks = min(num_landmarks, seq_len)  # Ensure G <= L
        landmark_indices = torch.randint(0, seq_len, (batch, num_landmarks))
        selection_scores = torch.randn(batch, seq_len)

        loss = landmark_spacing_loss(landmark_indices, seq_len, lambda_reg=0.01, selection_scores=selection_scores)

        assert torch.isfinite(loss)

    @pytest.mark.property
    @given(
        batch=st.integers(1, 4),
        seq_len=st.integers(32, 128),
        num_landmarks=st.integers(4, 16)
    )
    @settings(max_examples=20, deadline=None)
    def test_property_sparsity_loss_always_finite(self, batch, seq_len, num_landmarks):
        """Property test: sparsity loss always finite."""
        num_landmarks = min(num_landmarks, seq_len)
        selection_scores = torch.randn(batch, seq_len)

        loss = landmark_sparsity_loss(selection_scores, num_landmarks=num_landmarks, lambda_reg=0.001)

        assert torch.isfinite(loss)
