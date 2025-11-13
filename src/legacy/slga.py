# slga.py
"""
Backward compatibility layer for SLGA Module.

DEPRECATED: This module is deprecated. Use src.core.attention.slga instead.

This module maintains the original API while delegating to the new modular architecture.
All new code should import from src.core.attention instead.

The SLGA implementation has been refactored into:
- src/core/attention/slga.py - Main SLGA attention mechanism
- src/core/attention/local.py - Local windowed attention
- src/core/attention/global.py - Global landmark attention
- src/core/attention/config.py - Attention configuration

Migration guide:
    OLD:
        from src.slga import SLGAModule

    NEW:
        from src.core.attention.slga import SLGAAttention

    Or use the config-based approach:
        from src.core.attention import SLGAAttention
        from src.core.attention.config import AttentionConfig

        config = AttentionConfig(
            d_model=512,
            num_heads=8,
            local_window_size=256,
            max_global_landmarks=64
        )
        attention = SLGAAttention(config)
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import torch
from torch import nn

# Import from new modular location
from src.core.attention.slga import SLGAAttention


class SLGAModule(nn.Module):
    """
    Backward compatibility wrapper for SLGA Module.

    This class maintains the original API while delegating to the new
    modular SLGAAttention implementation.

    DEPRECATED: Use src.core.attention.slga.SLGAAttention directly.

    Parameters:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        local_window: Size of local window
        global_k: Top-K for global selection per head
        attn_drop: Dropout on attention weights
        proj_drop: Dropout on output projection
        causal: If True, applies causal masking
        gated_fusion: If True, uses learned fusion (else additive)
        dilation: Dilation factor for window (1=dense, 2=skip every 2)
        diverse_topk: If True, encourages inter-head diversity in top-K
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        local_window: int = 256,
        global_k: int = 32,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        causal: bool = True,
        gated_fusion: bool = True,
        dilation: int = 1,
        diverse_topk: bool = True,
    ):
        super().__init__()

        # Issue deprecation warning
        warnings.warn(
            "SLGAModule is deprecated. Use src.core.attention.slga.SLGAAttention instead. "
            "This wrapper will be removed in version 3.0.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Store config for backward compatibility
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.local_window = local_window
        self.global_k = global_k
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.gated_fusion = gated_fusion
        self.dilation = dilation
        self.diverse_topk = diverse_topk

        # Create underlying attention module directly (no config class needed)
        self._attention = SLGAAttention(
            d_model=embed_dim,
            num_heads=num_heads,
            local_window=local_window,
            global_k=global_k,
            dropout=attn_drop,
            causal=causal,
            gated_fusion=gated_fusion,
            dilation=dilation,
            diverse_topk=diverse_topk,
        )

        # Expose internal components for backward compatibility
        # Note: SLGAAttention uses qkv_proj (unified) instead of separate q/k/v projections
        self.qkv_proj = self._attention.qkv_proj
        self.out_proj = self._attention.out_proj

        if gated_fusion and hasattr(self._attention, "fusion_gate"):
            self.fusion_gate = self._attention.fusion_gate

        # Expose attention modules
        self.local_attn = self._attention.local_attn
        self.global_attn = self._attention.global_attn

        # Dropouts
        self.proj_dropout = nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cache_global_indices: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through SLGA attention.

        Delegates to SLGAAttention.forward().

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            cache_global_indices: Optional landmark indices (batch, global_k)
            mask: Optional attention mask (batch, seq_len)
            return_aux: If True, return auxiliary metrics

        Returns:
            output: Attention output of shape (batch, seq_len, embed_dim)
            aux: Optional auxiliary data dictionary
        """
        # SLGAAttention expects (x, cache_global, global_weight)
        # For backward compatibility, use cache_global=None (internal selection)
        result = self._attention(
            x=x,
            cache_global=None,  # Let SLGAAttention handle landmark selection
            global_weight=1.0,  # Use full global attention
        )

        # Apply output projection dropout for backward compatibility
        result = self.proj_dropout(result)

        if return_aux:
            # Return with empty aux dict for backward compatibility
            return result, {}

        return result


# Maintain backward compatibility for direct imports
__all__ = [
    "SLGAModule",
]
