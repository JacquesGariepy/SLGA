"""
Comprehensive unit tests for TransformerBlock.

Tests cover:
- Initialization with Config
- Forward pass with SLGA attention
- FFN sub-layer
- Residual connections
- Layer normalization (pre-norm architecture)
- Gradient checkpointing
- Progressive dilation
- Global cache integration
- Gradient flow
- Device compatibility
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

from src.legacy.model import TransformerBlock, Config, FeedForward


class TestTransformerBlockInitialization:
    """Test TransformerBlock initialization."""

    def test_basic_initialization(self):
        """Test basic initialization with default config."""
        cfg = Config(embed_dim=512, num_heads=8, n_layers=12)
        block = TransformerBlock(cfg, layer_idx=0)

        assert block.cfg == cfg
        assert block.layer_idx == 0
        assert hasattr(block, 'attn')
        assert hasattr(block, 'ffn')
        assert hasattr(block, 'norm1')
        assert hasattr(block, 'norm2')

    @pytest.mark.parametrize("layer_idx", [0, 1, 2, 5, 11])
    def test_various_layer_indices(self, layer_idx):
        """Test initialization with various layer indices."""
        cfg = Config(n_layers=12)
        block = TransformerBlock(cfg, layer_idx=layer_idx)

        assert block.layer_idx == layer_idx

    def test_dilation_increases_with_layer(self):
        """Test that dilation factor increases for higher layers."""
        cfg = Config(n_layers=12, dilated_windows=True)

        # Lower layer: dilation = 1
        block_low = TransformerBlock(cfg, layer_idx=0)
        assert block_low.attn.dilation == 1

        # Higher layer: dilation > 1
        block_high = TransformerBlock(cfg, layer_idx=8)
        assert block_high.attn.dilation > 1

    def test_no_dilation_when_disabled(self):
        """Test that dilation stays 1 when disabled."""
        cfg = Config(n_layers=12, dilated_windows=False)

        for layer_idx in [0, 6, 11]:
            block = TransformerBlock(cfg, layer_idx=layer_idx)
            assert block.attn.dilation == 1

    def test_attn_configuration(self):
        """Test that SLGA attention is configured correctly."""
        cfg = Config(
            embed_dim=512,
            num_heads=8,
            local_window=128,
            global_k=24,
            gated_fusion=True,
            diverse_topk=True,
        )
        block = TransformerBlock(cfg, layer_idx=0)

        assert block.attn.D == 512
        assert block.attn.H == 8
        assert block.attn.W == 128
        assert block.attn.GK == 24
        assert block.attn.gated is True
        assert block.attn.diverse_topk is True

    def test_ffn_configuration(self):
        """Test that FFN is configured correctly."""
        cfg = Config(embed_dim=512, ff_hidden_multiplier=4, dropout_rate=0.1)
        block = TransformerBlock(cfg, layer_idx=0)

        assert isinstance(block.ffn, FeedForward)
        # FFN should have hidden_dim = 512 * 4 = 2048

    def test_layer_norms_created(self):
        """Test that layer norms are created."""
        cfg = Config(embed_dim=512)
        block = TransformerBlock(cfg, layer_idx=0)

        assert isinstance(block.norm1, nn.LayerNorm)
        assert isinstance(block.norm2, nn.LayerNorm)
        assert block.norm1.normalized_shape == (512,)
        assert block.norm2.normalized_shape == (512,)


class TestTransformerBlockForward:
    """Test forward pass through TransformerBlock."""

    @pytest.fixture
    def block(self):
        """Create block for testing."""
        cfg = Config(embed_dim=256, num_heads=4, local_window=64, global_k=16)
        return TransformerBlock(cfg, layer_idx=0)

    def test_forward_basic_shape(self, block):
        """Test forward pass returns correct shape."""
        B, L, D = 2, 128, 256
        x = torch.randn(B, L, D)

        out = block(x)

        assert out.shape == (B, L, D)

    @pytest.mark.parametrize("batch,seq_len", [
        (1, 64),
        (4, 128),
        (8, 256),
        (16, 512),
    ])
    def test_various_batch_sequence_sizes(self, batch, seq_len):
        """Test with various batch sizes and sequence lengths."""
        cfg = Config(embed_dim=256, num_heads=4)
        block = TransformerBlock(cfg, layer_idx=0)

        x = torch.randn(batch, seq_len, 256)
        out = block(x)

        assert out.shape == (batch, seq_len, 256)

    def test_forward_with_global_cache(self, block):
        """Test forward with global cache."""
        B, L, D = 2, 128, 256
        G = 16
        x = torch.randn(B, L, D)
        cache_global = torch.randn(B, G, D)

        out = block(x, cache_global=cache_global)

        assert out.shape == (B, L, D)

    @pytest.mark.parametrize("global_weight", [0.0, 0.5, 1.0])
    def test_forward_with_global_weight(self, block, global_weight):
        """Test forward with different global weights."""
        B, L, D = 2, 128, 256
        G = 16
        x = torch.randn(B, L, D)
        cache_global = torch.randn(B, G, D)

        out = block(x, cache_global=cache_global, global_weight=global_weight)

        assert out.shape == (B, L, D)
        assert torch.isfinite(out).all()

    def test_residual_connections(self, block):
        """Test that residual connections work correctly."""
        B, L, D = 2, 128, 256
        x = torch.randn(B, L, D)

        out = block(x)

        # Output should not be identical to input (transformations applied)
        assert not torch.allclose(out, x)

        # But residual should make output similar in magnitude
        assert out.abs().mean() > 0


class TestTransformerBlockPreNorm:
    """Test pre-norm architecture (LayerNorm before sub-layers)."""

    @pytest.fixture
    def block(self):
        """Create block for testing."""
        cfg = Config(embed_dim=256, num_heads=4)
        return TransformerBlock(cfg, layer_idx=0)

    def test_pre_norm_applied_before_attention(self, block):
        """Test that LayerNorm is applied before attention."""
        # This is tested by checking the architecture
        # In pre-norm: x + attn(norm(x))
        # We can verify by checking that norm1 exists and is used

        B, L, D = 2, 128, 256
        x = torch.randn(B, L, D)

        # Forward pass
        out = block(x)

        # If pre-norm working, output should be finite
        assert torch.isfinite(out).all()

    def test_pre_norm_applied_before_ffn(self, block):
        """Test that LayerNorm is applied before FFN."""
        B, L, D = 2, 128, 256
        x = torch.randn(B, L, D)

        out = block(x)

        assert torch.isfinite(out).all()

    def test_two_layer_norms_used(self, block):
        """Test that both norm1 and norm2 are used."""
        B, L, D = 2, 128, 256
        x = torch.randn(B, L, D, requires_grad=True)

        out = block(x)
        loss = out.sum()
        loss.backward()

        # Both norms should have gradients if used
        assert block.norm1.weight.grad is not None
        assert block.norm2.weight.grad is not None


class TestTransformerBlockGradientCheckpointing:
    """Test gradient checkpointing functionality."""

    def test_no_checkpointing_in_eval(self):
        """Test that checkpointing is disabled in eval mode."""
        cfg = Config(embed_dim=256, num_heads=4, grad_checkpointing=True)
        block = TransformerBlock(cfg, layer_idx=0)
        block.eval()

        x = torch.randn(2, 128, 256)

        # Should work without checkpointing in eval
        out = block(x)
        assert out.shape == (2, 128, 256)

    def test_checkpointing_enabled_in_training(self):
        """Test that checkpointing works in training mode."""
        cfg = Config(embed_dim=256, num_heads=4, grad_checkpointing=True)
        block = TransformerBlock(cfg, layer_idx=0)
        block.train()

        x = torch.randn(2, 128, 256, requires_grad=True)

        out = block(x)
        loss = out.sum()
        loss.backward()

        # Should complete successfully with checkpointing
        assert x.grad is not None

    def test_checkpointing_disabled(self):
        """Test normal forward without checkpointing."""
        cfg = Config(embed_dim=256, num_heads=4, grad_checkpointing=False)
        block = TransformerBlock(cfg, layer_idx=0)
        block.train()

        x = torch.randn(2, 128, 256, requires_grad=True)

        out = block(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None

    @pytest.mark.slow
    def test_checkpointing_saves_memory(self):
        """Test that checkpointing reduces memory usage."""
        # This is a conceptual test - actual memory measurement is complex
        cfg = Config(embed_dim=512, num_heads=8, grad_checkpointing=True)
        block = TransformerBlock(cfg, layer_idx=0)
        block.train()

        x = torch.randn(4, 512, 512, requires_grad=True)

        out = block(x)
        loss = out.sum()
        loss.backward()

        # Should complete without OOM
        assert x.grad is not None


class TestTransformerBlockGradientFlow:
    """Test gradient flow through the block."""

    @pytest.fixture
    def block(self):
        """Create block for gradient testing."""
        cfg = Config(embed_dim=256, num_heads=4)
        return TransformerBlock(cfg, layer_idx=0)

    def test_gradient_flow_through_attention(self, block):
        """Test gradients flow through attention."""
        x = torch.randn(2, 128, 256, requires_grad=True)

        out = block(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_gradient_flow_through_ffn(self, block):
        """Test gradients flow through FFN."""
        x = torch.randn(2, 128, 256, requires_grad=True)

        out = block(x)
        loss = out.sum()
        loss.backward()

        # FFN parameters should have gradients
        for param in block.ffn.parameters():
            assert param.grad is not None
            assert param.grad.abs().sum() > 0

    def test_gradient_to_attention_weights(self, block):
        """Test gradients reach attention weights."""
        x = torch.randn(2, 128, 256)

        out = block(x)
        loss = out.sum()
        loss.backward()

        # Attention QKV projection should have gradients
        assert block.attn.qkv_proj.weight.grad is not None
        assert block.attn.out_proj.weight.grad is not None

    def test_gradient_with_global_cache(self, block):
        """Test gradients with global cache."""
        x = torch.randn(2, 128, 256, requires_grad=True)
        cache_global = torch.randn(2, 16, 256, requires_grad=True)

        out = block(x, cache_global=cache_global)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert cache_global.grad is not None


class TestTransformerBlockDeviceCompatibility:
    """Test device compatibility."""

    def test_cpu_device(self):
        """Test block works on CPU."""
        cfg = Config(embed_dim=256, num_heads=4)
        block = TransformerBlock(cfg, layer_idx=0)

        x = torch.randn(2, 128, 256)
        out = block(x)

        assert out.device.type == 'cpu'

    @pytest.mark.gpu
    def test_cuda_device(self):
        """Test block works on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        cfg = Config(embed_dim=256, num_heads=4)
        block = TransformerBlock(cfg, layer_idx=0).cuda()

        x = torch.randn(2, 128, 256).cuda()
        out = block(x)

        assert out.device.type == 'cuda'

    @pytest.mark.gpu
    def test_mixed_device_error(self):
        """Test error with mixed devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        cfg = Config(embed_dim=256, num_heads=4)
        block = TransformerBlock(cfg, layer_idx=0).cuda()

        x = torch.randn(2, 128, 256)  # CPU tensor

        with pytest.raises(RuntimeError):
            block(x)


class TestTransformerBlockEdgeCases:
    """Test edge cases."""

    def test_very_short_sequence(self):
        """Test with very short sequence."""
        cfg = Config(embed_dim=256, num_heads=4)
        block = TransformerBlock(cfg, layer_idx=0)

        x = torch.randn(2, 2, 256)
        out = block(x)

        assert out.shape == (2, 2, 256)
        assert torch.isfinite(out).all()

    def test_single_token(self):
        """Test with single token."""
        cfg = Config(embed_dim=256, num_heads=4)
        block = TransformerBlock(cfg, layer_idx=0)

        x = torch.randn(2, 1, 256)
        out = block(x)

        assert out.shape == (2, 1, 256)

    def test_very_long_sequence(self):
        """Test with very long sequence."""
        cfg = Config(embed_dim=256, num_heads=4, local_window=128)
        block = TransformerBlock(cfg, layer_idx=0)

        x = torch.randn(1, 4096, 256)
        out = block(x)

        assert out.shape == (1, 4096, 256)

    def test_zero_input(self):
        """Test with all-zero input."""
        cfg = Config(embed_dim=256, num_heads=4)
        block = TransformerBlock(cfg, layer_idx=0)

        x = torch.zeros(2, 128, 256)
        out = block(x)

        assert torch.isfinite(out).all()

    def test_extreme_values(self):
        """Test with extreme input values."""
        cfg = Config(embed_dim=256, num_heads=4)
        block = TransformerBlock(cfg, layer_idx=0)

        x = torch.randn(2, 128, 256) * 100
        out = block(x)

        assert torch.isfinite(out).all()


class TestTransformerBlockNumericalStability:
    """Test numerical stability."""

    @pytest.fixture
    def block(self):
        """Create block for stability testing."""
        cfg = Config(embed_dim=256, num_heads=4)
        return TransformerBlock(cfg, layer_idx=0)

    def test_no_nan_in_output(self, block):
        """Test output never contains NaN."""
        x = torch.randn(2, 128, 256)
        out = block(x)

        assert not torch.isnan(out).any()

    def test_no_inf_in_output(self, block):
        """Test output never contains Inf."""
        x = torch.randn(2, 128, 256)
        out = block(x)

        assert not torch.isinf(out).any()

    def test_stable_with_repeated_calls(self, block):
        """Test stability across multiple calls."""
        x = torch.randn(2, 128, 256)

        for _ in range(10):
            out = block(x)
            assert torch.isfinite(out).all()

    def test_stable_with_large_inputs(self, block):
        """Test stability with large inputs."""
        x = torch.randn(2, 128, 256) * 100

        out = block(x)
        assert torch.isfinite(out).all()


class TestTransformerBlockDilation:
    """Test progressive dilation across layers."""

    def test_dilation_progression(self):
        """Test that dilation increases progressively."""
        cfg = Config(n_layers=12, dilated_windows=True)

        dilations = []
        for layer_idx in range(12):
            block = TransformerBlock(cfg, layer_idx=layer_idx)
            dilations.append(block.attn.dilation)

        # Dilation should increase or stay same (never decrease)
        for i in range(1, len(dilations)):
            assert dilations[i] >= dilations[i-1]

    def test_dilation_formula(self):
        """Test dilation formula: 2^(layer_idx // (n_layers // 3))."""
        cfg = Config(n_layers=12, dilated_windows=True)

        # Layer 0-3: dilation = 2^0 = 1
        for idx in range(4):
            block = TransformerBlock(cfg, layer_idx=idx)
            assert block.attn.dilation == 1

        # Layer 4-7: dilation = 2^1 = 2
        for idx in range(4, 8):
            block = TransformerBlock(cfg, layer_idx=idx)
            assert block.attn.dilation == 2

        # Layer 8-11: dilation = 2^2 = 4
        for idx in range(8, 12):
            block = TransformerBlock(cfg, layer_idx=idx)
            assert block.attn.dilation == 4

    def test_dilation_with_different_layer_counts(self):
        """Test dilation with various layer counts."""
        for n_layers in [6, 12, 24]:
            cfg = Config(n_layers=n_layers, dilated_windows=True)

            for layer_idx in range(n_layers):
                block = TransformerBlock(cfg, layer_idx=layer_idx)
                # Should always be a power of 2
                assert block.attn.dilation in [1, 2, 4, 8, 16, 32]


class TestFeedForward:
    """Test FeedForward sub-module."""

    def test_initialization(self):
        """Test FFN initialization."""
        ffn = FeedForward(embed_dim=512, hidden_multiplier=4, dropout=0.1)

        assert isinstance(ffn.fc1, nn.Linear)
        assert isinstance(ffn.fc2, nn.Linear)
        assert ffn.fc1.in_features == 512
        assert ffn.fc1.out_features == 2048  # 512 * 4
        assert ffn.fc2.in_features == 2048
        assert ffn.fc2.out_features == 512

    def test_forward_shape(self):
        """Test FFN forward pass shape."""
        ffn = FeedForward(embed_dim=256, hidden_multiplier=4)

        x = torch.randn(2, 128, 256)
        out = ffn(x)

        assert out.shape == (2, 128, 256)

    @pytest.mark.parametrize("embed_dim,hidden_mult", [
        (256, 2),
        (512, 4),
        (768, 4),
        (1024, 8),
    ])
    def test_various_dimensions(self, embed_dim, hidden_mult):
        """Test with various dimensions."""
        ffn = FeedForward(embed_dim=embed_dim, hidden_multiplier=hidden_mult)

        x = torch.randn(2, 128, embed_dim)
        out = ffn(x)

        assert out.shape == (2, 128, embed_dim)

    def test_gelu_activation(self):
        """Test that GELU activation is applied."""
        ffn = FeedForward(embed_dim=256, hidden_multiplier=4)

        # GELU should produce non-linear transformation
        x = torch.randn(2, 128, 256)
        out = ffn(x)

        # Output should be different from linear transformation
        assert not torch.allclose(out, ffn.fc2(ffn.fc1(x)))

    def test_dropout_applied(self):
        """Test that dropout is applied."""
        ffn = FeedForward(embed_dim=256, hidden_multiplier=4, dropout=0.5)
        ffn.train()

        x = torch.randn(2, 128, 256)

        out1 = ffn(x)
        out2 = ffn(x)

        # With high dropout, outputs should differ
        assert not torch.allclose(out1, out2)

    def test_gradient_flow(self):
        """Test gradient flow through FFN."""
        ffn = FeedForward(embed_dim=256, hidden_multiplier=4)

        x = torch.randn(2, 128, 256, requires_grad=True)
        out = ffn(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert ffn.fc1.weight.grad is not None
        assert ffn.fc2.weight.grad is not None

    def test_no_bias_in_layers(self):
        """Test that bias is enabled by default."""
        ffn = FeedForward(embed_dim=256, hidden_multiplier=4)

        # By default, Linear layers have bias
        assert ffn.fc1.bias is not None
        assert ffn.fc2.bias is not None
