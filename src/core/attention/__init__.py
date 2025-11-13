"""Attention mechanisms for SLGA-Plus.

This package provides modular attention implementations:
- BaseAttention: Abstract base class with common utilities
- LocalAttention: Efficient sliding window attention
- GlobalAttention: Top-K landmark selection with diversity
- SLGAAttention: Fused local-global attention with learned gating

All attention modules follow the AttentionMechanism protocol defined in
src.domain.services.attention_protocol.

Example:
    >>> from src.core.attention import SLGAAttention
    >>>
    >>> # Create SLGA attention module
    >>> slga = SLGAAttention(
    ...     d_model=512,
    ...     num_heads=8,
    ...     local_window=128,
    ...     global_k=32
    ... )
    >>>
    >>> # Forward pass
    >>> x = torch.randn(2, 1000, 512)
    >>> cache = torch.randn(2, 100, 512)
    >>> output = slga(x, cache_global=cache)
"""

from src.core.attention.base import BaseAttention
from src.core.attention.global_ import GlobalAttention
from src.core.attention.local import LocalAttention
from src.core.attention.slga import SLGAAttention

__all__ = [
    "BaseAttention",
    "GlobalAttention",
    "LocalAttention",
    "SLGAAttention",
]
