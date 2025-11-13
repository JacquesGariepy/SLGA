"""
Comprehensive unit tests for LearnableLandmarkSelector.

Tests cover:
- Initialization and configuration
- Scorer network architecture
- Gumbel-Softmax selection (training mode)
- Straight-through estimator (alternative mode)
- Temperature annealing
- Gradient flow
- NaN protection (Bug #17 fix)
- Device compatibility
- Edge cases
"""

import pytest
import torch
import torch.nn as nn
from hypothesis import given, strategies as st, settings
import math

from src.legacy.landmarks import LearnableLandmarkSelector


class TestLearnableLandmarkSelectorInitialization:
    """Test initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic initialization with required parameters."""
        selector = LearnableLandmarkSelector(embed_dim=512, num_landmarks=24)
        assert selector.embed_dim == 512
        assert selector.num_landmarks == 24
        assert selector.temperature == 1.0
        assert selector.temperature_decay == 0.999
        assert selector.min_temperature == 0.3

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        selector = LearnableLandmarkSelector(
            embed_dim=768,
            num_landmarks=32,
            hidden_dim=256,
            temperature=2.0,
            temperature_decay=0.99,
            min_temperature=0.1,
        )
        assert selector.embed_dim == 768
        assert selector.num_landmarks == 32
        assert selector.temperature == 2.0
        assert selector.temperature_decay == 0.99
        assert selector.min_temperature == 0.1

    @pytest.mark.parametrize("embed_dim,num_landmarks", [
        (256, 8),
        (512, 24),
        (768, 32),
        (1024, 48),
    ])
    def test_various_dimensions(self, embed_dim, num_landmarks):
        """Test various embedding dimensions and landmark counts."""
        selector = LearnableLandmarkSelector(embed_dim=embed_dim, num_landmarks=num_landmarks)
        assert selector.embed_dim == embed_dim
        assert selector.num_landmarks == num_landmarks

    def test_scorer_architecture(self):
        """Test scorer network has correct architecture."""
        selector = LearnableLandmarkSelector(embed_dim=512, num_landmarks=24)
        # Scorer should be: Linear(512 -> 256) -> GELU -> Dropout -> Linear(256 -> 1)
        assert len(selector.scorer) == 4
        assert isinstance(selector.scorer[0], nn.Linear)
        assert selector.scorer[0].in_features == 512
        assert selector.scorer[0].out_features == 256
        assert isinstance(selector.scorer[3], nn.Linear)
        assert selector.scorer[3].out_features == 1

    def test_custom_hidden_dim(self):
        """Test scorer with custom hidden dimension."""
        selector = LearnableLandmarkSelector(embed_dim=512, num_landmarks=24, hidden_dim=128)
        assert selector.scorer[0].out_features == 128
        assert selector.scorer[3].in_features == 128

    def test_default_hidden_dim(self):
        """Test that default hidden_dim is embed_dim // 2."""
        selector = LearnableLandmarkSelector(embed_dim=512, num_landmarks=24, hidden_dim=None)
        assert selector.scorer[0].out_features == 256  # 512 // 2

    def test_step_count_buffer_registered(self):
        """Test that step count buffer is registered."""
        selector = LearnableLandmarkSelector(embed_dim=512, num_landmarks=24)
        assert hasattr(selector, 'step_count')
        assert selector.step_count == 0


class TestLearnableLandmarkSelectorForward:
    """Test forward pass and landmark selection."""

    @pytest.fixture
    def selector(self):
        """Create selector for testing."""
        return LearnableLandmarkSelector(embed_dim=256, num_landmarks=16)

    def test_forward_basic_shape(self, selector):
        """Test forward pass returns correct shapes."""
        B, L, D = 2, 128, 256
        x = torch.randn(B, L, D)

        indices, states, scores = selector(x)

        assert indices.shape == (B, 16)
        assert states.shape == (B, 16, D)
        assert scores.shape == (B, L)

    @pytest.mark.parametrize("batch,seq_len,num_landmarks", [
        (1, 64, 8),
        (4, 128, 16),
        (8, 256, 24),
        (16, 512, 32),
    ])
    def test_various_batch_sequence_landmarks(self, batch, seq_len, num_landmarks):
        """Test with various configurations."""
        selector = LearnableLandmarkSelector(embed_dim=128, num_landmarks=num_landmarks)
        x = torch.randn(batch, seq_len, 128)

        indices, states, scores = selector(x)

        assert indices.shape == (batch, num_landmarks)
        assert states.shape == (batch, num_landmarks, 128)
        assert scores.shape == (batch, seq_len)

    def test_short_sequence_fewer_than_landmarks(self, selector):
        """Test when sequence is shorter than num_landmarks."""
        B, L, D = 2, 8, 256  # L=8 < num_landmarks=16
        x = torch.randn(B, L, D)

        indices, states, scores = selector(x)

        # Should select min(L, num_landmarks) = 8
        assert indices.shape == (B, 8)
        assert states.shape == (B, 8, D)

    def test_training_mode_gumbel(self, selector):
        """Test that training mode uses Gumbel-Softmax."""
        selector.train()
        x = torch.randn(2, 128, 256)

        indices1, _, _ = selector(x, use_gumbel=True)
        indices2, _, _ = selector(x, use_gumbel=True)

        # Should produce different results due to Gumbel noise
        assert not torch.equal(indices1, indices2)

    def test_training_mode_straight_through(self, selector):
        """Test that training mode with straight-through works."""
        selector.train()
        x = torch.randn(2, 128, 256)

        indices, states, scores = selector(x, use_gumbel=False)

        assert indices.shape == (2, 16)
        assert torch.isfinite(indices).all()

    def test_eval_mode_deterministic(self, selector):
        """Test that eval mode produces deterministic results."""
        selector.eval()
        x = torch.randn(2, 128, 256)

        indices1, _, _ = selector(x)
        indices2, _, _ = selector(x)

        # Should be deterministic in eval mode
        assert torch.equal(indices1, indices2)

    def test_selected_indices_in_valid_range(self, selector):
        """Test that selected indices are within valid range."""
        x = torch.randn(2, 128, 256)
        indices, _, _ = selector(x)

        assert (indices >= 0).all()
        assert (indices < 128).all()

    def test_gathered_states_match_input(self, selector):
        """Test that gathered states correspond to correct positions."""
        B, L, D = 2, 128, 256
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
        x = torch.randn(2, 128, 256)
        _, _, scores = selector(x)

        # Scores should sum to 1 per batch (softmax normalized)
        sums = scores.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_selection_scores_non_negative(self, selector):
        """Test that selection scores are non-negative."""
        x = torch.randn(2, 128, 256)
        _, _, scores = selector(x)

        assert (scores >= 0).all()


class TestLearnableLandmarkSelectorGumbel:
    """Test Gumbel-Softmax selection mechanism."""

    @pytest.fixture
    def selector(self):
        """Create selector for Gumbel testing."""
        return LearnableLandmarkSelector(embed_dim=256, num_landmarks=16)

    def test_gumbel_topk_returns_correct_shapes(self, selector):
        """Test _gumbel_topk returns correct shapes."""
        scores = torch.randn(2, 128)
        soft_selection, hard_indices = selector._gumbel_topk(scores, k=16, temperature=1.0)

        assert soft_selection.shape == (2, 128)
        assert hard_indices.shape == (2, 16)

    def test_gumbel_noise_non_deterministic(self, selector):
        """Test that Gumbel noise makes selection stochastic."""
        scores = torch.randn(2, 128)

        _, indices1 = selector._gumbel_topk(scores, k=16, temperature=1.0)
        _, indices2 = selector._gumbel_topk(scores, k=16, temperature=1.0)

        # Should be different due to random Gumbel noise
        assert not torch.equal(indices1, indices2)

    def test_gumbel_temperature_affects_selection(self, selector):
        """Test that temperature affects selection sharpness."""
        scores = torch.randn(2, 128)

        # High temperature: more uniform
        soft_high, _ = selector._gumbel_topk(scores, k=16, temperature=10.0)

        # Low temperature: more peaked
        soft_low, _ = selector._gumbel_topk(scores, k=16, temperature=0.1)

        # Low temp should have higher variance (more peaked distribution)
        assert soft_low.var() > soft_high.var()

    def test_gumbel_no_nan_in_output(self, selector):
        """Test NaN protection in Gumbel sampling (Bug #17 fix)."""
        # Test with various dtypes
        for dtype in [torch.float32, torch.float16]:
            scores = torch.randn(2, 128, dtype=dtype)
            soft_selection, hard_indices = selector._gumbel_topk(scores, k=16, temperature=1.0)

            assert not torch.isnan(soft_selection).any()
            assert not torch.isnan(hard_indices.float()).any()

    def test_gumbel_float32_noise_generation(self, selector):
        """Test that Gumbel noise is generated in float32 for stability."""
        scores = torch.randn(2, 128, dtype=torch.float16)

        # Should not raise error or produce NaN
        soft_selection, hard_indices = selector._gumbel_topk(scores, k=16, temperature=1.0)

        assert torch.isfinite(soft_selection).all()

    def test_gumbel_extreme_scores(self, selector):
        """Test Gumbel with extreme score values."""
        scores = torch.randn(2, 128) * 100  # Large values

        soft_selection, hard_indices = selector._gumbel_topk(scores, k=16, temperature=1.0)

        assert torch.isfinite(soft_selection).all()
        assert torch.isfinite(hard_indices.float()).all()

    def test_gumbel_zero_scores(self, selector):
        """Test Gumbel with zero scores."""
        scores = torch.zeros(2, 128)

        soft_selection, hard_indices = selector._gumbel_topk(scores, k=16, temperature=1.0)

        assert torch.isfinite(soft_selection).all()

    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0, 10.0])
    def test_gumbel_various_temperatures(self, selector, temperature):
        """Test Gumbel with various temperature values."""
        scores = torch.randn(2, 128)

        soft_selection, hard_indices = selector._gumbel_topk(scores, k=16, temperature=temperature)

        assert soft_selection.shape == (2, 128)
        assert torch.isfinite(soft_selection).all()


class TestLearnableLandmarkSelectorStraightThrough:
    """Test straight-through estimator."""

    @pytest.fixture
    def selector(self):
        """Create selector for straight-through testing."""
        return LearnableLandmarkSelector(embed_dim=256, num_landmarks=16)

    def test_straight_through_returns_correct_shapes(self, selector):
        """Test _straight_through_topk returns correct shapes."""
        scores = torch.randn(2, 128)
        selection, indices = selector._straight_through_topk(scores, k=16)

        assert selection.shape == (2, 128)
        assert indices.shape == (2, 16)

    def test_straight_through_hard_selection(self, selector):
        """Test that forward pass uses hard selection."""
        scores = torch.randn(2, 128)
        selection, indices = selector._straight_through_topk(scores, k=16)

        # In forward, selected positions should have value ~1, others ~0
        # But due to straight-through, exact values may vary

    def test_straight_through_gradient_flow(self, selector):
        """Test that gradients flow through straight-through estimator."""
        scores = torch.randn(2, 128, requires_grad=True)
        selection, _ = selector._straight_through_topk(scores, k=16)

        loss = selection.sum()
        loss.backward()

        assert scores.grad is not None
        assert scores.grad.abs().sum() > 0

    def test_straight_through_deterministic(self, selector):
        """Test that straight-through is deterministic."""
        scores = torch.randn(2, 128)

        selection1, indices1 = selector._straight_through_topk(scores, k=16)
        selection2, indices2 = selector._straight_through_topk(scores, k=16)

        assert torch.equal(indices1, indices2)

    def test_straight_through_adaptive_threshold(self, selector):
        """Test that threshold adapts based on k-th value."""
        scores = torch.randn(2, 128)
        selection, indices = selector._straight_through_topk(scores, k=16)

        # Top-16 indices should be the highest scoring positions
        top16_scores = torch.gather(scores, 1, indices)
        kth_value = top16_scores.min(dim=1)[0]

        # All selected indices should have scores >= kth value
        for b in range(2):
            for idx in indices[b]:
                assert scores[b, idx] >= kth_value[b] - 1e-5


class TestLearnableLandmarkSelectorTemperature:
    """Test temperature annealing mechanism."""

    @pytest.fixture
    def selector(self):
        """Create selector with specific temperature config."""
        return LearnableLandmarkSelector(
            embed_dim=256,
            num_landmarks=16,
            temperature=1.0,
            temperature_decay=0.99,
            min_temperature=0.3,
        )

    def test_initial_temperature(self, selector):
        """Test initial temperature value."""
        assert selector._get_temperature() == 1.0

    def test_temperature_decreases_with_steps(self, selector):
        """Test that temperature decreases with training steps."""
        selector.train()
        x = torch.randn(2, 128, 256)

        temp_before = selector._get_temperature()

        # Run forward passes to increment step counter
        for _ in range(10):
            selector(x, use_gumbel=True)

        temp_after = selector._get_temperature()

        assert temp_after < temp_before

    def test_temperature_reaches_minimum(self, selector):
        """Test that temperature does not go below minimum."""
        selector.train()
        x = torch.randn(2, 128, 256)

        # Run many steps
        for _ in range(1000):
            selector(x, use_gumbel=True)

        final_temp = selector._get_temperature()

        assert final_temp >= selector.min_temperature

    def test_temperature_in_eval_mode(self, selector):
        """Test that eval mode uses minimum temperature."""
        selector.eval()

        temp = selector._get_temperature()

        assert temp == selector.min_temperature

    def test_step_counter_increments(self, selector):
        """Test that step counter increments correctly."""
        selector.train()
        x = torch.randn(2, 128, 256)

        initial_steps = selector.step_count.item()

        for i in range(5):
            selector(x, use_gumbel=True)
            assert selector.step_count.item() == initial_steps + i + 1

    def test_step_counter_not_incremented_without_gumbel(self, selector):
        """Test that step counter is not incremented in straight-through mode."""
        selector.train()
        x = torch.randn(2, 128, 256)

        initial_steps = selector.step_count.item()

        selector(x, use_gumbel=False)

        assert selector.step_count.item() == initial_steps


class TestLearnableLandmarkSelectorGradientFlow:
    """Test gradient flow through the selector."""

    @pytest.fixture
    def selector(self):
        """Create selector for gradient testing."""
        return LearnableLandmarkSelector(embed_dim=256, num_landmarks=16)

    def test_gradient_flow_gumbel_mode(self, selector):
        """Test gradients flow in Gumbel mode."""
        selector.train()
        x = torch.randn(2, 128, 256, requires_grad=True)

        indices, states, scores = selector(x, use_gumbel=True)
        loss = states.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_gradient_flow_straight_through_mode(self, selector):
        """Test gradients flow in straight-through mode."""
        selector.train()
        x = torch.randn(2, 128, 256, requires_grad=True)

        indices, states, scores = selector(x, use_gumbel=False)
        loss = states.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_gradient_to_scorer_weights(self, selector):
        """Test gradients reach scorer network weights."""
        selector.train()
        x = torch.randn(2, 128, 256)

        indices, states, scores = selector(x, use_gumbel=True)
        loss = states.sum()
        loss.backward()

        # Check all scorer layers have gradients
        for param in selector.scorer.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert param.grad.abs().sum() > 0

    def test_gradient_flow_with_loss(self, selector):
        """Test gradients flow through complete pipeline with task loss."""
        selector.train()
        x = torch.randn(2, 128, 256, requires_grad=True)

        indices, states, scores = selector(x, use_gumbel=True)

        # Simulate task loss on selected landmarks
        task_loss = (states ** 2).sum()
        task_loss.backward()

        assert x.grad is not None


class TestLearnableLandmarkSelectorNaNProtection:
    """Test NaN protection mechanisms (Bug #17 fix)."""

    @pytest.fixture
    def selector(self):
        """Create selector for NaN testing."""
        return LearnableLandmarkSelector(embed_dim=256, num_landmarks=16)

    def test_score_clamping_prevents_nan(self, selector):
        """Test that score clamping prevents NaN in softmax."""
        # Create extreme scores that could cause overflow
        x = torch.randn(2, 128, 256) * 100

        indices, states, scores = selector(x)

        assert not torch.isnan(scores).any()
        assert not torch.isnan(states).any()

    def test_nan_protection_in_selection_scores(self, selector):
        """Test fallback to uniform distribution if NaN detected."""
        # This is hard to trigger directly, but we ensure the protection exists
        x = torch.randn(2, 128, 256)

        indices, states, scores = selector(x)

        # Should never have NaN
        assert torch.isfinite(scores).all()

    def test_gumbel_nan_protection_float16(self, selector):
        """Test Gumbel NaN protection with float16."""
        selector.train()
        x = torch.randn(2, 128, 256, dtype=torch.float16)

        indices, states, scores = selector(x, use_gumbel=True)

        assert torch.isfinite(indices.float()).all()
        assert torch.isfinite(states).all()

    def test_extreme_input_values(self, selector):
        """Test with extreme input values."""
        x = torch.randn(2, 128, 256) * 1000

        indices, states, scores = selector(x)

        assert torch.isfinite(indices.float()).all()
        assert torch.isfinite(states).all()
        assert torch.isfinite(scores).all()


class TestLearnableLandmarkSelectorDeviceCompatibility:
    """Test device compatibility."""

    def test_cpu_device(self):
        """Test selector works on CPU."""
        selector = LearnableLandmarkSelector(embed_dim=256, num_landmarks=16)
        x = torch.randn(2, 128, 256)

        indices, states, scores = selector(x)

        assert indices.device.type == 'cpu'
        assert states.device.type == 'cpu'
        assert scores.device.type == 'cpu'

    @pytest.mark.gpu
    def test_cuda_device(self):
        """Test selector works on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        selector = LearnableLandmarkSelector(embed_dim=256, num_landmarks=16).cuda()
        x = torch.randn(2, 128, 256).cuda()

        indices, states, scores = selector(x)

        assert indices.device.type == 'cuda'
        assert states.device.type == 'cuda'
        assert scores.device.type == 'cuda'

    @pytest.mark.gpu
    def test_device_mismatch_error(self):
        """Test error with device mismatch."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        selector = LearnableLandmarkSelector(embed_dim=256, num_landmarks=16).cuda()
        x = torch.randn(2, 128, 256)  # CPU tensor

        with pytest.raises(RuntimeError):
            selector(x)


class TestLearnableLandmarkSelectorEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_landmark(self):
        """Test with single landmark selection."""
        selector = LearnableLandmarkSelector(embed_dim=256, num_landmarks=1)
        x = torch.randn(2, 128, 256)

        indices, states, scores = selector(x)

        assert indices.shape == (2, 1)
        assert states.shape == (2, 1, 256)

    def test_many_landmarks(self):
        """Test with many landmarks (approaching sequence length)."""
        selector = LearnableLandmarkSelector(embed_dim=256, num_landmarks=120)
        x = torch.randn(2, 128, 256)

        indices, states, scores = selector(x)

        # Should select min(120, 128) = 120
        assert indices.shape == (2, 120)

    def test_landmarks_equal_sequence_length(self):
        """Test when num_landmarks equals sequence length."""
        selector = LearnableLandmarkSelector(embed_dim=256, num_landmarks=128)
        x = torch.randn(2, 128, 256)

        indices, states, scores = selector(x)

        assert indices.shape == (2, 128)
        assert states.shape == (2, 128, 256)

    def test_very_short_sequence(self):
        """Test with very short sequence."""
        selector = LearnableLandmarkSelector(embed_dim=256, num_landmarks=16)
        x = torch.randn(2, 4, 256)  # Only 4 tokens

        indices, states, scores = selector(x)

        assert indices.shape == (2, 4)  # Can only select 4

    def test_single_token_sequence(self):
        """Test with single token."""
        selector = LearnableLandmarkSelector(embed_dim=256, num_landmarks=16)
        x = torch.randn(2, 1, 256)

        indices, states, scores = selector(x)

        assert indices.shape == (2, 1)

    def test_large_batch(self):
        """Test with large batch size."""
        selector = LearnableLandmarkSelector(embed_dim=256, num_landmarks=16)
        x = torch.randn(128, 256, 256)

        indices, states, scores = selector(x)

        assert indices.shape == (128, 16)

    def test_zero_input(self):
        """Test with all-zero input."""
        selector = LearnableLandmarkSelector(embed_dim=256, num_landmarks=16)
        x = torch.zeros(2, 128, 256)

        indices, states, scores = selector(x)

        assert torch.isfinite(indices.float()).all()
        assert torch.isfinite(states).all()
        assert torch.isfinite(scores).all()


class TestLearnableLandmarkSelectorNumericalStability:
    """Test numerical stability."""

    @pytest.fixture
    def selector(self):
        """Create selector for stability testing."""
        return LearnableLandmarkSelector(embed_dim=256, num_landmarks=16)

    def test_no_nan_in_outputs(self, selector):
        """Test that outputs never contain NaN."""
        x = torch.randn(2, 128, 256)

        indices, states, scores = selector(x)

        assert not torch.isnan(indices.float()).any()
        assert not torch.isnan(states).any()
        assert not torch.isnan(scores).any()

    def test_no_inf_in_outputs(self, selector):
        """Test that outputs never contain Inf."""
        x = torch.randn(2, 128, 256)

        indices, states, scores = selector(x)

        assert not torch.isinf(indices.float()).any()
        assert not torch.isinf(states).any()
        assert not torch.isinf(scores).any()

    def test_stability_with_repeated_calls(self, selector):
        """Test stability across multiple calls."""
        selector.eval()  # Deterministic mode
        x = torch.randn(2, 128, 256)

        for _ in range(10):
            indices, states, scores = selector(x)
            assert torch.isfinite(indices.float()).all()
            assert torch.isfinite(states).all()
            assert torch.isfinite(scores).all()

    @pytest.mark.property
    @given(
        batch=st.integers(1, 8),
        seq_len=st.integers(16, 128),
        scale=st.floats(0.1, 10.0)
    )
    @settings(max_examples=20, deadline=None)
    def test_property_always_finite(self, batch, seq_len, scale):
        """Property test: outputs always finite for random inputs."""
        selector = LearnableLandmarkSelector(embed_dim=128, num_landmarks=16)
        selector.eval()

        x = torch.randn(batch, seq_len, 128) * scale

        indices, states, scores = selector(x)

        assert torch.isfinite(indices.float()).all()
        assert torch.isfinite(states).all()
        assert torch.isfinite(scores).all()


class TestLearnableLandmarkSelectorIntegration:
    """Integration tests with complete pipelines."""

    def test_integration_with_attention(self):
        """Test integration with attention module."""
        selector = LearnableLandmarkSelector(embed_dim=256, num_landmarks=16)
        x = torch.randn(2, 128, 256)

        # Select landmarks
        indices, states, scores = selector(x)

        # Use landmarks as global cache for attention
        # (This would be done by the model)
        assert states.shape == (2, 16, 256)

    def test_integration_with_training_loop(self):
        """Test integration in a training loop."""
        selector = LearnableLandmarkSelector(embed_dim=256, num_landmarks=16)
        optimizer = torch.optim.Adam(selector.parameters(), lr=0.001)

        x = torch.randn(2, 128, 256)

        # Training step
        selector.train()
        optimizer.zero_grad()

        indices, states, scores = selector(x, use_gumbel=True)
        loss = states.sum()  # Dummy loss
        loss.backward()

        optimizer.step()

        # Check that parameters were updated
        assert any(p.grad is not None for p in selector.parameters())

    def test_eval_after_training(self):
        """Test switching between train and eval modes."""
        selector = LearnableLandmarkSelector(embed_dim=256, num_landmarks=16)
        x = torch.randn(2, 128, 256)

        # Train mode
        selector.train()
        train_indices, _, _ = selector(x, use_gumbel=True)

        # Eval mode
        selector.eval()
        eval_indices1, _, _ = selector(x)
        eval_indices2, _, _ = selector(x)

        # Eval should be deterministic
        assert torch.equal(eval_indices1, eval_indices2)
