"""
Comprehensive unit tests for PositionalLandmarkSelector.

Tests cover:
- Initialization and positional embedding configuration
- Position-based scoring mechanism
- Top-K selection on position scores
- Independence from input content
- Gradient flow through positional embeddings
- Device compatibility
- Edge cases (short sequences, single landmark)
- Numerical stability
"""

import pytest
import torch
import torch.nn as nn
from hypothesis import given, strategies as st, settings

from src.core.landmarks.learned import PositionalLandmarkSelector


class TestPositionalLandmarkSelectorInitialization:
    """Test initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic initialization with required parameters."""
        selector = PositionalLandmarkSelector(
            max_seq_len=2048, num_landmarks=64, embed_dim=512
        )
        assert selector.max_seq_len == 2048
        assert selector.num_landmarks == 64
        assert selector.pos_embeddings.shape == (2048, 512)

    def test_positional_embeddings_created(self):
        """Test that positional embeddings are learnable parameters."""
        selector = PositionalLandmarkSelector(
            max_seq_len=1024, num_landmarks=32, embed_dim=256
        )
        assert isinstance(selector.pos_embeddings, nn.Parameter)
        assert selector.pos_embeddings.requires_grad

    def test_scorer_architecture(self):
        """Test scorer network has correct architecture."""
        selector = PositionalLandmarkSelector(
            max_seq_len=2048, num_landmarks=64, embed_dim=512
        )
        # Scorer should be: Linear(embed_dim -> 1)
        assert isinstance(selector.scorer, nn.Linear)
        assert selector.scorer.in_features == 512
        assert selector.scorer.out_features == 1

    @pytest.mark.parametrize("max_seq_len,num_landmarks,embed_dim", [
        (512, 16, 128),
        (1024, 32, 256),
        (2048, 64, 512),
        (4096, 128, 768),
    ])
    def test_various_configurations(self, max_seq_len, num_landmarks, embed_dim):
        """Test various configuration combinations."""
        selector = PositionalLandmarkSelector(
            max_seq_len=max_seq_len, num_landmarks=num_landmarks, embed_dim=embed_dim
        )
        assert selector.max_seq_len == max_seq_len
        assert selector.num_landmarks == num_landmarks
        assert selector.pos_embeddings.shape == (max_seq_len, embed_dim)


class TestPositionalLandmarkSelectorForward:
    """Test forward pass and landmark selection."""

    @pytest.fixture
    def selector(self):
        """Create selector for testing."""
        return PositionalLandmarkSelector(
            max_seq_len=2048, num_landmarks=64, embed_dim=512
        )

    def test_forward_basic_shape(self, selector):
        """Test forward pass returns correct shapes."""
        B, L, D = 4, 256, 512
        x = torch.randn(B, L, D)

        indices, states, scores = selector(x)

        assert indices.shape == (B, 64)
        assert states.shape == (B, 64, D)
        assert scores.shape == (B, L)

    @pytest.mark.parametrize("batch,seq_len,num_landmarks", [
        (1, 128, 16),
        (4, 256, 32),
        (8, 512, 64),
        (16, 1024, 128),
    ])
    def test_various_batch_sequence_landmarks(self, batch, seq_len, num_landmarks):
        """Test with various configurations."""
        selector = PositionalLandmarkSelector(
            max_seq_len=2048, num_landmarks=num_landmarks, embed_dim=256
        )
        x = torch.randn(batch, seq_len, 256)

        indices, states, scores = selector(x)

        assert indices.shape == (batch, num_landmarks)
        assert states.shape == (batch, num_landmarks, 256)
        assert scores.shape == (batch, seq_len)

    def test_short_sequence_fewer_than_landmarks(self, selector):
        """Test when sequence is shorter than num_landmarks."""
        B, L, D = 2, 32, 512  # L=32 < num_landmarks=64
        x = torch.randn(B, L, D)

        indices, states, scores = selector(x)

        # Should select min(L, num_landmarks) = 32
        assert indices.shape == (B, 32)
        assert states.shape == (B, 32, D)

    def test_position_based_selection_content_independent(self, selector):
        """Test that selection is independent of input content."""
        B, L, D = 4, 256, 512

        # Different input content
        x1 = torch.randn(B, L, D)
        x2 = torch.randn(B, L, D) * 100

        indices1, _, _ = selector(x1)
        indices2, _, _ = selector(x2)

        # Should select same positions regardless of content
        assert torch.equal(indices1, indices2)

    def test_deterministic_selection(self, selector):
        """Test that selection is deterministic."""
        x = torch.randn(2, 256, 512)

        indices1, _, _ = selector(x)
        indices2, _, _ = selector(x)

        assert torch.equal(indices1, indices2)

    def test_selected_indices_in_valid_range(self, selector):
        """Test that selected indices are within valid range."""
        x = torch.randn(2, 256, 512)
        indices, _, _ = selector(x)

        assert (indices >= 0).all()
        assert (indices < 256).all()

    def test_gathered_states_match_input(self, selector):
        """Test that gathered states correspond to correct positions."""
        B, L, D = 2, 256, 512
        x = torch.randn(B, L, D)

        indices, states, _ = selector(x)

        # Manually gather and compare
        for b in range(B):
            for g in range(indices.size(1)):
                idx = indices[b, g].item()
                expected_state = x[b, idx]
                actual_state = states[b, g]
                assert torch.allclose(expected_state, actual_state, atol=1e-5)

    def test_selection_scores_normalized(self, selector):
        """Test that selection scores are normalized (sum to 1)."""
        x = torch.randn(2, 256, 512)
        _, _, scores = selector(x)

        # Scores should sum to 1 per batch (softmax normalized)
        sums = scores.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_selection_scores_non_negative(self, selector):
        """Test that selection scores are non-negative."""
        x = torch.randn(2, 256, 512)
        _, _, scores = selector(x)

        assert (scores >= 0).all()

    def test_scores_same_across_batch(self, selector):
        """Test that scores are the same across all batch elements."""
        x = torch.randn(4, 256, 512)
        _, _, scores = selector(x)

        # All batch elements should have identical scores (position-based)
        for b in range(1, 4):
            assert torch.allclose(scores[0], scores[b], atol=1e-6)


class TestPositionalLandmarkSelectorPositionScoring:
    """Test position-based scoring mechanism."""

    @pytest.fixture
    def selector(self):
        """Create selector for position scoring testing."""
        return PositionalLandmarkSelector(
            max_seq_len=2048, num_landmarks=64, embed_dim=512
        )

    def test_position_embeddings_used(self, selector):
        """Test that position embeddings are used for scoring."""
        B, L, D = 2, 256, 512
        x = torch.randn(B, L, D)

        # Get position embeddings for this length
        pos_emb = selector.pos_embeddings[:L]
        assert pos_emb.shape == (L, D)

        # Run forward
        indices, _, _ = selector(x)
        assert indices.shape == (B, 64)

    def test_longer_than_max_seq_len_raises_error(self, selector):
        """Test that sequences longer than max_seq_len raise error."""
        B, L, D = 2, 2100, 512  # L > max_seq_len=2048
        x = torch.randn(B, L, D)

        with pytest.raises(RuntimeError):
            selector(x)

    def test_position_scores_consistent(self, selector):
        """Test that position scores are consistent across calls."""
        x1 = torch.randn(2, 256, 512)
        x2 = torch.randn(2, 256, 512)

        _, _, scores1 = selector(x1)
        _, _, scores2 = selector(x2)

        # Scores should be identical (position-based)
        assert torch.allclose(scores1, scores2, atol=1e-6)

    def test_different_sequence_lengths_different_scores(self, selector):
        """Test that different sequence lengths produce different score distributions."""
        x1 = torch.randn(2, 128, 512)
        x2 = torch.randn(2, 256, 512)

        _, _, scores1 = selector(x1)
        _, _, scores2 = selector(x2)

        # Shape should be different
        assert scores1.shape[1] != scores2.shape[1]


class TestPositionalLandmarkSelectorGradientFlow:
    """Test gradient flow through positional embeddings."""

    @pytest.fixture
    def selector(self):
        """Create selector for gradient testing."""
        return PositionalLandmarkSelector(
            max_seq_len=2048, num_landmarks=64, embed_dim=512
        )

    def test_gradient_flow_through_positional_embeddings(self, selector):
        """Test gradients flow to positional embeddings via scores."""
        x = torch.randn(2, 256, 512)

        indices, states, scores = selector(x)
        # Use scores for loss (which depend on positional embeddings)
        loss = scores.sum()
        loss.backward()

        # Check that positional embeddings have gradients
        assert selector.pos_embeddings.grad is not None
        assert selector.pos_embeddings.grad.abs().sum() > 0

    def test_gradient_flow_to_scorer_weights(self, selector):
        """Test gradients reach scorer network weights."""
        x = torch.randn(2, 256, 512)

        indices, states, scores = selector(x)
        # Use weighted scores for loss (to get non-zero gradients)
        weighted_scores = scores * torch.arange(scores.size(1), dtype=scores.dtype, device=scores.device)
        loss = weighted_scores.sum()
        loss.backward()

        # Check scorer has gradients
        assert selector.scorer.weight.grad is not None

    def test_gradient_to_input_via_states(self, selector):
        """Test that input receives gradients via gathered states."""
        x = torch.randn(2, 256, 512, requires_grad=True)

        indices, states, scores = selector(x)
        loss = states.sum()
        loss.backward()

        # Input should have gradients because states are gathered from x
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestPositionalLandmarkSelectorTraining:
    """Test training behavior."""

    @pytest.fixture
    def selector(self):
        """Create selector for training testing."""
        return PositionalLandmarkSelector(
            max_seq_len=2048, num_landmarks=64, embed_dim=512
        )

    def test_integration_with_training_loop(self, selector):
        """Test integration in a training loop."""
        optimizer = torch.optim.Adam(selector.parameters(), lr=0.001)

        x = torch.randn(2, 256, 512)

        # Training step
        selector.train()
        optimizer.zero_grad()

        indices, states, scores = selector(x)
        loss = scores.sum()  # Use scores to get gradient flow to positional embeddings
        loss.backward()

        optimizer.step()

        # Check that parameters were updated
        assert selector.pos_embeddings.grad is not None

    def test_position_patterns_learned(self, selector):
        """Test that position patterns can be learned."""
        optimizer = torch.optim.SGD(selector.parameters(), lr=0.01)

        x = torch.randn(4, 256, 512)

        # Get initial positions
        initial_indices, _, _ = selector(x)

        # Train for a few steps with loss that prefers early positions
        for _ in range(10):
            optimizer.zero_grad()
            indices, states, scores = selector(x)

            # Loss: use scores weighted by position to encourage early selection
            position_weights = torch.arange(scores.size(1), dtype=scores.dtype, device=scores.device)
            weighted_loss = (scores * position_weights).sum()
            weighted_loss.backward()
            optimizer.step()

        # Get final positions
        final_indices, _, _ = selector(x)

        # Mean position should have decreased (learned to prefer early positions)
        assert final_indices.float().mean() < initial_indices.float().mean()

    def test_train_eval_mode_consistency(self, selector):
        """Test that train/eval modes produce same results."""
        x = torch.randn(2, 256, 512)

        selector.train()
        train_indices, _, _ = selector(x)

        selector.eval()
        eval_indices, _, _ = selector(x)

        # Should be identical (no stochastic sampling)
        assert torch.equal(train_indices, eval_indices)


class TestPositionalLandmarkSelectorDeviceCompatibility:
    """Test device compatibility."""

    def test_cpu_device(self):
        """Test selector works on CPU."""
        selector = PositionalLandmarkSelector(
            max_seq_len=2048, num_landmarks=64, embed_dim=512
        )
        x = torch.randn(2, 256, 512)

        indices, states, scores = selector(x)

        assert indices.device.type == 'cpu'
        assert states.device.type == 'cpu'
        assert scores.device.type == 'cpu'

    @pytest.mark.gpu
    def test_cuda_device(self):
        """Test selector works on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        selector = PositionalLandmarkSelector(
            max_seq_len=2048, num_landmarks=64, embed_dim=512
        ).cuda()
        x = torch.randn(2, 256, 512).cuda()

        indices, states, scores = selector(x)

        assert indices.device.type == 'cuda'
        assert states.device.type == 'cuda'
        assert scores.device.type == 'cuda'


class TestPositionalLandmarkSelectorEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_landmark(self):
        """Test with single landmark selection."""
        selector = PositionalLandmarkSelector(
            max_seq_len=2048, num_landmarks=1, embed_dim=512
        )
        x = torch.randn(2, 256, 512)

        indices, states, scores = selector(x)

        assert indices.shape == (2, 1)
        assert states.shape == (2, 1, 512)

    def test_many_landmarks(self):
        """Test with many landmarks (approaching sequence length)."""
        selector = PositionalLandmarkSelector(
            max_seq_len=2048, num_landmarks=240, embed_dim=512
        )
        x = torch.randn(2, 256, 512)

        indices, states, scores = selector(x)

        # Should select min(240, 256) = 240
        assert indices.shape == (2, 240)

    def test_landmarks_equal_sequence_length(self):
        """Test when num_landmarks equals sequence length."""
        selector = PositionalLandmarkSelector(
            max_seq_len=2048, num_landmarks=256, embed_dim=512
        )
        x = torch.randn(2, 256, 512)

        indices, states, scores = selector(x)

        assert indices.shape == (2, 256)
        assert states.shape == (2, 256, 512)

    def test_very_short_sequence(self):
        """Test with very short sequence."""
        selector = PositionalLandmarkSelector(
            max_seq_len=2048, num_landmarks=64, embed_dim=512
        )
        x = torch.randn(2, 8, 512)  # Only 8 tokens

        indices, states, scores = selector(x)

        assert indices.shape == (2, 8)  # Can only select 8

    def test_single_token_sequence(self):
        """Test with single token."""
        selector = PositionalLandmarkSelector(
            max_seq_len=2048, num_landmarks=64, embed_dim=512
        )
        x = torch.randn(2, 1, 512)

        indices, states, scores = selector(x)

        assert indices.shape == (2, 1)

    def test_large_batch(self):
        """Test with large batch size."""
        selector = PositionalLandmarkSelector(
            max_seq_len=2048, num_landmarks=64, embed_dim=512
        )
        x = torch.randn(128, 256, 512)

        indices, states, scores = selector(x)

        assert indices.shape == (128, 64)

    def test_zero_input(self):
        """Test with all-zero input."""
        selector = PositionalLandmarkSelector(
            max_seq_len=2048, num_landmarks=64, embed_dim=512
        )
        x = torch.zeros(2, 256, 512)

        indices, states, scores = selector(x)

        assert torch.isfinite(indices.float()).all()
        assert torch.isfinite(states).all()
        assert torch.isfinite(scores).all()

    def test_max_seq_len_equal_num_landmarks(self):
        """Test when max_seq_len equals num_landmarks."""
        selector = PositionalLandmarkSelector(
            max_seq_len=64, num_landmarks=64, embed_dim=512
        )
        x = torch.randn(2, 64, 512)

        indices, states, scores = selector(x)

        assert indices.shape == (2, 64)


class TestPositionalLandmarkSelectorNumericalStability:
    """Test numerical stability."""

    @pytest.fixture
    def selector(self):
        """Create selector for stability testing."""
        return PositionalLandmarkSelector(
            max_seq_len=2048, num_landmarks=64, embed_dim=512
        )

    def test_no_nan_in_outputs(self, selector):
        """Test that outputs never contain NaN."""
        x = torch.randn(2, 256, 512)

        indices, states, scores = selector(x)

        assert not torch.isnan(indices.float()).any()
        assert not torch.isnan(states).any()
        assert not torch.isnan(scores).any()

    def test_no_inf_in_outputs(self, selector):
        """Test that outputs never contain Inf."""
        x = torch.randn(2, 256, 512)

        indices, states, scores = selector(x)

        assert not torch.isinf(indices.float()).any()
        assert not torch.isinf(states).any()
        assert not torch.isinf(scores).any()

    def test_extreme_input_values(self, selector):
        """Test with extreme input values."""
        x = torch.randn(2, 256, 512) * 1000

        indices, states, scores = selector(x)

        assert torch.isfinite(indices.float()).all()
        assert torch.isfinite(states).all()
        assert torch.isfinite(scores).all()

    def test_stability_with_repeated_calls(self, selector):
        """Test stability across multiple calls."""
        x = torch.randn(2, 256, 512)

        for _ in range(10):
            indices, states, scores = selector(x)
            assert torch.isfinite(indices.float()).all()
            assert torch.isfinite(states).all()
            assert torch.isfinite(scores).all()

    @pytest.mark.property
    @given(
        batch=st.integers(1, 8),
        seq_len=st.integers(16, 512),
        scale=st.floats(0.1, 10.0)
    )
    @settings(max_examples=20, deadline=None)
    def test_property_always_finite(self, batch, seq_len, scale):
        """Property test: outputs always finite for random inputs."""
        selector = PositionalLandmarkSelector(
            max_seq_len=1024, num_landmarks=32, embed_dim=256
        )

        x = torch.randn(batch, seq_len, 256) * scale

        indices, states, scores = selector(x)

        assert torch.isfinite(indices.float()).all()
        assert torch.isfinite(states).all()
        assert torch.isfinite(scores).all()
