"""Transformer block with SLGA attention."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.core.layers.feedforward import FeedForward
from src.core.attention.slga import SLGAAttention

if TYPE_CHECKING:
    from src.models.config import ModelConfig


class TransformerBlock(nn.Module):
    """Transformer Block with SLGA (Sparse Local-Global Attention).

    Architecture: Pre-norm (LayerNorm before attention/FFN).
    Each block consists of SLGA attention followed by Feed-Forward Network (FFN),
    with residual connections around each sub-layer.
    Supports progressive dilation across layers for hierarchical processing.

    Args:
        config: Model configuration
        layer_idx: Layer index for progressive dilation

    Example:
        >>> from src.models.config import ModelConfig
        >>> config = ModelConfig(embed_dim=512, num_heads=8)
        >>> block = TransformerBlock(config, layer_idx=0)
        >>> x = torch.randn(2, 10, 512)
        >>> output = block(x)
        >>> output.shape
        torch.Size([2, 10, 512])
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        # Progressive dilation per layer if enabled
        # Lower layers: dense, higher layers: dilated for hierarchical processing
        if config.dilated_windows:
            dilation_factor = 2 ** (layer_idx // max(1, config.n_layers // 3))
        else:
            dilation_factor = 1

        # SLGA Attention module
        # Handles sparse attention with local windows and global landmarks
        self.attn = SLGAAttention(
            d_model=config.embed_dim,  # Fixed: SLGAAttention uses d_model, not embed_dim
            num_heads=config.num_heads,
            local_window=config.local_window,
            global_k=config.global_k,
            dropout=config.dropout_rate,  # Fixed: SLGAAttention uses dropout, not attn_drop/proj_drop
            causal=True,  # Causal attention for autoregressive generation
            gated_fusion=config.gated_fusion,  # Gated fusion of local and global attention
            dilation=dilation_factor,  # Dilation factor for this layer
            diverse_topk=config.diverse_topk,  # Diverse top-k selection for global tokens
        )

        # Feed-Forward Network (FFN)
        self.ffn = FeedForward(
            embed_dim=config.embed_dim,
            hidden_multiplier=config.ff_hidden_multiplier,
            dropout=config.dropout_rate,
        )

        # Layer norms (pre-norm architecture)
        # Applied before attention and FFN for better gradient flow
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)

    def _attn_forward(
        self,
        x: torch.Tensor,
        cache_global: Optional[torch.Tensor],
        global_weight: float = 1.0,
    ) -> torch.Tensor:
        """Wrapper for attention forward pass.

        Used with gradient checkpointing.

        Args:
            x: Input tensor
            cache_global: Global landmark states
            global_weight: Weight for global attention

        Returns:
            Attention output
        """
        return self.attn(x, cache_global=cache_global, global_weight=global_weight)

    def _ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Wrapper for FFN forward pass.

        Used with gradient checkpointing.

        Args:
            x: Input tensor

        Returns:
            FFN output
        """
        return self.ffn(x)

    def forward(
        self,
        x: torch.Tensor,
        cache_global: Optional[torch.Tensor] = None,
        global_weight: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass through the Transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            cache_global: Global landmark states of shape (batch, global_k, embed_dim)
            global_weight: Weight for global attention component (0.0 to 1.0)

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        # Attention with residual connection (pre-norm)
        # Use gradient checkpointing during training to save memory
        if self.config.grad_checkpointing and self.training:
            attn_out = checkpoint(
                self._attn_forward,
                self.norm1(x),
                cache_global,
                global_weight,
                use_reentrant=False,
            )
        else:
            attn_out = self.attn(
                self.norm1(x),
                cache_global=cache_global,
                global_weight=global_weight,
            )

        # Add residual connection
        x = x + attn_out

        # FFN with residual connection (pre-norm)
        if self.config.grad_checkpointing and self.training:
            ffn_out = checkpoint(
                self._ffn_forward,
                self.norm2(x),
                use_reentrant=False,
            )
        else:
            ffn_out = self.ffn(self.norm2(x))

        # Add residual connection
        x = x + ffn_out

        return x


__all__ = ["TransformerBlock"]
