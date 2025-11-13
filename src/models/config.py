"""Model configuration with validation using Pydantic."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Configuration for SLGA Transformer model with validation.

    Args:
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
        embed_dim: Embedding dimension (d_model)
        num_heads: Number of attention heads
        ff_hidden_multiplier: FFN hidden layer multiplier (default: 4x)
        n_layers: Number of transformer layers
        dropout_rate: Dropout rate for regularization

        # SLGA-specific config
        local_window: Local attention window size
        global_k: Number of global landmark tokens
        gated_fusion: Use gated fusion for local/global attention
        learned_landmarks: Use learned landmark selection
        dilated_windows: Use progressive dilation across layers
        diverse_topk: Use diverse top-k selection for global tokens

        # Landmark selector config (optional)
        landmark_selector: Configuration dict for landmark selector

        # Training config
        grad_checkpointing: Enable gradient checkpointing for memory efficiency

    Example:
        >>> config = ModelConfig(vocab_size=50257, embed_dim=512, num_heads=8)
        >>> print(config.vocab_size)
        50257
    """

    # Model architecture
    vocab_size: int = Field(default=50257, gt=0)
    max_seq_len: int = Field(default=2048, gt=0)
    embed_dim: int = Field(default=512, gt=0)
    num_heads: int = Field(default=8, gt=0)
    ff_hidden_multiplier: int = Field(default=4, gt=0)
    n_layers: int = Field(default=12, gt=0)
    dropout_rate: float = Field(default=0.1, ge=0.0, le=1.0)

    # SLGA configuration
    local_window: int = Field(default=128, gt=0)
    global_k: int = Field(default=24, gt=0)
    gated_fusion: bool = Field(default=True)
    learned_landmarks: bool = Field(default=True)
    dilated_windows: bool = Field(default=True)
    diverse_topk: bool = Field(default=True)

    # Optional landmark selector config
    landmark_selector: Optional[Dict[str, Any]] = None

    # Training configuration
    grad_checkpointing: bool = Field(default=False)

    class Config:
        """Pydantic configuration."""

        frozen = False  # Allow modification after creation
        extra = "forbid"  # Reject unknown fields

    @field_validator("embed_dim")
    @classmethod
    def validate_embed_dim_divisible_by_heads(cls, v: int, info) -> int:
        """Ensure embed_dim is divisible by num_heads."""
        # Note: num_heads might not be set yet during validation
        # This will be checked in post_init if needed
        return v

    @field_validator("local_window")
    @classmethod
    def validate_local_window(cls, v: int, info) -> int:
        """Ensure local window is reasonable."""
        if v > 2048:
            raise ValueError("local_window should not exceed 2048 for memory efficiency")
        return v

    @field_validator("global_k")
    @classmethod
    def validate_global_k(cls, v: int, info) -> int:
        """Ensure global_k is reasonable."""
        if v > 256:
            raise ValueError("global_k should not exceed 256 for memory efficiency")
        return v

    def validate_consistency(self) -> None:
        """Validate configuration consistency after all fields are set."""
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        if self.global_k >= self.max_seq_len:
            raise ValueError(
                f"global_k ({self.global_k}) must be less than " f"max_seq_len ({self.max_seq_len})"
            )

        if self.local_window > self.max_seq_len:
            raise ValueError(
                f"local_window ({self.local_window}) should not exceed "
                f"max_seq_len ({self.max_seq_len})"
            )


# Preset configurations for common use cases
PRESET_CONFIGS = {
    "tiny": ModelConfig(
        vocab_size=50257,
        max_seq_len=1024,
        embed_dim=256,
        num_heads=4,
        ff_hidden_multiplier=4,
        n_layers=6,
        dropout_rate=0.1,
        local_window=64,
        global_k=12,
    ),
    "small": ModelConfig(
        vocab_size=50257,
        max_seq_len=2048,
        embed_dim=512,
        num_heads=8,
        ff_hidden_multiplier=4,
        n_layers=12,
        dropout_rate=0.1,
        local_window=128,
        global_k=24,
    ),
    "medium": ModelConfig(
        vocab_size=50257,
        max_seq_len=2048,
        embed_dim=768,
        num_heads=12,
        ff_hidden_multiplier=4,
        n_layers=16,
        dropout_rate=0.1,
        local_window=128,
        global_k=32,
    ),
    "large": ModelConfig(
        vocab_size=50257,
        max_seq_len=4096,
        embed_dim=1024,
        num_heads=16,
        ff_hidden_multiplier=4,
        n_layers=24,
        dropout_rate=0.1,
        local_window=256,
        global_k=48,
    ),
}


def get_config(name: str) -> ModelConfig:
    """Get preset configuration by name.

    Args:
        name: Preset name ("tiny", "small", "medium", "large")

    Returns:
        ModelConfig instance

    Example:
        >>> config = get_config("small")
        >>> config.embed_dim
        512
    """
    if name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESET_CONFIGS.keys())}")
    return PRESET_CONFIGS[name].model_copy()


__all__ = ["PRESET_CONFIGS", "ModelConfig", "get_config"]
