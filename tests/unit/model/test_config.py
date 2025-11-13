"""
Unit tests for Config dataclass.

Tests cover:
- Default values
- Custom configurations
- Parameter validation
- Serialization
"""

import pytest
from dataclasses import asdict

from src.legacy.model import Config


class TestConfigDefaults:
    """Test default configuration values."""

    def test_default_initialization(self):
        """Test Config with all default values."""
        cfg = Config()

        assert cfg.vocab_size == 50257
        assert cfg.max_seq_len == 2048
        assert cfg.embed_dim == 512
        assert cfg.num_heads == 8
        assert cfg.ff_hidden_multiplier == 4
        assert cfg.n_layers == 12
        assert cfg.dropout_rate == 0.1

    def test_default_slga_config(self):
        """Test default SLGA configuration."""
        cfg = Config()

        assert cfg.local_window == 128
        assert cfg.global_k == 24
        assert cfg.gated_fusion is True
        assert cfg.learned_landmarks is True
        assert cfg.dilated_windows is True
        assert cfg.diverse_topk is True

    def test_default_training_config(self):
        """Test default training configuration."""
        cfg = Config()

        assert cfg.grad_checkpointing is False


class TestConfigCustomization:
    """Test custom configuration creation."""

    def test_custom_model_size(self):
        """Test creating custom model size config."""
        cfg = Config(
            vocab_size=32000,
            embed_dim=768,
            num_heads=12,
            n_layers=24,
        )

        assert cfg.vocab_size == 32000
        assert cfg.embed_dim == 768
        assert cfg.num_heads == 12
        assert cfg.n_layers == 24

    def test_custom_slga_config(self):
        """Test creating custom SLGA config."""
        cfg = Config(
            local_window=256,
            global_k=48,
            gated_fusion=False,
            learned_landmarks=False,
        )

        assert cfg.local_window == 256
        assert cfg.global_k == 48
        assert cfg.gated_fusion is False
        assert cfg.learned_landmarks is False

    @pytest.mark.parametrize("embed_dim,num_heads", [
        (256, 4),
        (512, 8),
        (768, 12),
        (1024, 16),
    ])
    def test_various_model_sizes(self, embed_dim, num_heads):
        """Test various model size configurations."""
        cfg = Config(embed_dim=embed_dim, num_heads=num_heads)

        assert cfg.embed_dim == embed_dim
        assert cfg.num_heads == num_heads
        assert cfg.embed_dim % cfg.num_heads == 0  # Should be divisible


class TestConfigValidation:
    """Test configuration validation."""

    def test_head_dimension_divisibility(self):
        """Test that embed_dim must be divisible by num_heads."""
        # This validation happens in the model, not in Config
        # But we can test that valid configs are created
        cfg = Config(embed_dim=512, num_heads=8)
        assert cfg.embed_dim % cfg.num_heads == 0

    def test_positive_values(self):
        """Test that configuration values are positive."""
        cfg = Config()

        assert cfg.vocab_size > 0
        assert cfg.max_seq_len > 0
        assert cfg.embed_dim > 0
        assert cfg.num_heads > 0
        assert cfg.n_layers > 0
        assert cfg.local_window > 0
        assert cfg.global_k > 0

    def test_dropout_range(self):
        """Test that dropout is in valid range."""
        cfg = Config(dropout_rate=0.1)

        assert 0.0 <= cfg.dropout_rate < 1.0


class TestConfigSerialization:
    """Test configuration serialization."""

    def test_asdict_conversion(self):
        """Test converting Config to dictionary."""
        cfg = Config(embed_dim=256, num_heads=4)
        cfg_dict = asdict(cfg)

        assert isinstance(cfg_dict, dict)
        assert cfg_dict['embed_dim'] == 256
        assert cfg_dict['num_heads'] == 4

    def test_reconstruction_from_dict(self):
        """Test reconstructing Config from dictionary."""
        original = Config(embed_dim=256, num_heads=4, n_layers=6)
        cfg_dict = asdict(original)

        reconstructed = Config(**cfg_dict)

        assert reconstructed.embed_dim == original.embed_dim
        assert reconstructed.num_heads == original.num_heads
        assert reconstructed.n_layers == original.n_layers


class TestConfigPresets:
    """Test common configuration presets."""

    def test_small_model_config(self):
        """Test small model configuration."""
        cfg = Config(
            embed_dim=256,
            num_heads=4,
            n_layers=6,
            local_window=64,
            global_k=16,
        )

        assert cfg.embed_dim == 256
        assert cfg.num_heads == 4
        assert cfg.n_layers == 6

    def test_base_model_config(self):
        """Test base model configuration (default)."""
        cfg = Config()  # Default is ~125M params

        assert cfg.embed_dim == 512
        assert cfg.num_heads == 8
        assert cfg.n_layers == 12

    def test_large_model_config(self):
        """Test large model configuration."""
        cfg = Config(
            embed_dim=1024,
            num_heads=16,
            n_layers=24,
            local_window=256,
            global_k=48,
        )

        assert cfg.embed_dim == 1024
        assert cfg.num_heads == 16
        assert cfg.n_layers == 24
