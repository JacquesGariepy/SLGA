"""Grouped Query Attention (GQA) module.

GQA uses fewer KV heads than Q heads:
- MHA (Multi-Head Attention): num_kv_heads = num_heads
- MQA (Multi-Query Attention): num_kv_heads = 1
- GQA (Grouped Query Attention): num_kv_heads = num_heads // group_size

This reduces memory footprint and KV cache size during inference while
maintaining model quality better than MQA.

Memory savings:
- MHA: 3 * D * D (Q, K, V projections)
- GQA: D * D + 2 * (D * D * num_kv_heads / num_heads)

For example with num_heads=8, num_kv_heads=2:
- MHA: 3 * D * D parameters
- GQA: D * D + 2 * (D * D * 2/8) = 1.5 * D * D parameters (50% reduction)

Reference:
- GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
  https://arxiv.org/abs/2305.13245
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with configurable KV head reduction.

    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of query heads (must be divisible by num_kv_heads)
        num_kv_heads: Number of key/value heads (< num_heads for GQA)
        dropout: Dropout probability for attention weights
        bias: Whether to use bias in projections
        max_seq_len: Maximum sequence length for positional encoding
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        max_seq_len: int = 131072,
    ):
        super().__init__()

        # Validation
        assert embed_dim % num_heads == 0, \
            f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"

        # Default: use standard MHA
        if num_kv_heads is None:
            num_kv_heads = num_heads

        assert num_kv_heads > 0 and num_kv_heads <= num_heads, \
            f"num_kv_heads={num_kv_heads} must be in range (0, num_heads={num_heads}]"

        assert num_heads % num_kv_heads == 0, \
            f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.kv_head_dim = embed_dim // num_heads  # Keep same head_dim for compatibility

        # Number of Q heads per KV head
        self.num_queries_per_kv = num_heads // num_kv_heads

        self.scale = self.head_dim ** -0.5
        self.dropout_p = dropout

        # Projections
        # Q: Full heads (num_heads * head_dim = embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # K, V: Reduced heads (num_kv_heads * head_dim)
        kv_dim = num_kv_heads * self.kv_head_dim
        self.k_proj = nn.Linear(embed_dim, kv_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, kv_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads to match Q heads.

        Args:
            x: (B, num_kv_heads, L, head_dim)
            n_rep: Number of repetitions (num_queries_per_kv)

        Returns:
            (B, num_heads, L, head_dim)
        """
        if n_rep == 1:
            return x

        B, num_kv_heads, L, head_dim = x.shape

        # Repeat each KV head n_rep times
        # (B, num_kv_heads, L, head_dim) -> (B, num_kv_heads, n_rep, L, head_dim)
        x = x.unsqueeze(2).expand(B, num_kv_heads, n_rep, L, head_dim)

        # (B, num_kv_heads, n_rep, L, head_dim) -> (B, num_heads, L, head_dim)
        x = x.reshape(B, num_kv_heads * n_rep, L, head_dim)

        return x

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of GQA.

        Args:
            x: (B, L, embed_dim) input tensor
            attention_mask: (B, L, L) or (B, 1, L, L) mask (True = masked)
            return_attn_weights: Whether to return attention weights

        Returns:
            output: (B, L, embed_dim)
            attn_weights: (B, num_heads, L, L) if return_attn_weights else None
        """
        B, L, D = x.shape

        # Project Q, K, V
        q = self.q_proj(x)  # (B, L, embed_dim)
        k = self.k_proj(x)  # (B, L, num_kv_heads * head_dim)
        v = self.v_proj(x)  # (B, L, num_kv_heads * head_dim)

        # Reshape to multi-head format
        # Q: (B, L, embed_dim) -> (B, num_heads, L, head_dim)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # K, V: (B, L, num_kv_heads * head_dim) -> (B, num_kv_heads, L, head_dim)
        k = k.view(B, L, self.num_kv_heads, self.kv_head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_kv_heads, self.kv_head_dim).transpose(1, 2)

        # Repeat KV heads to match Q heads
        k = self._repeat_kv(k, self.num_queries_per_kv)  # (B, num_heads, L, head_dim)
        v = self._repeat_kv(v, self.num_queries_per_kv)  # (B, num_heads, L, head_dim)

        # Scaled dot-product attention
        # scores: (B, num_heads, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            # Ensure mask is broadcastable to (B, num_heads, L, L)
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)  # (B, 1, L, L)
            scores = scores.masked_fill(attention_mask, float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        # (B, num_heads, L, L) @ (B, num_heads, L, head_dim) -> (B, num_heads, L, head_dim)
        output = torch.matmul(attn_weights, v)

        # Merge heads: (B, num_heads, L, head_dim) -> (B, L, embed_dim)
        output = output.transpose(1, 2).contiguous().view(B, L, self.embed_dim)

        # Output projection
        output = self.out_proj(output)
        output = self.resid_dropout(output)

        if return_attn_weights:
            return output, attn_weights
        return output, None

    def get_memory_savings(self) -> dict:
        """Calculate memory savings compared to MHA.

        Returns:
            dict with parameter counts and reduction percentage
        """
        # MHA parameters
        mha_params = 3 * self.embed_dim * self.embed_dim  # Q, K, V projections

        # GQA parameters
        q_params = self.embed_dim * self.embed_dim
        kv_params = 2 * self.embed_dim * (self.num_kv_heads * self.kv_head_dim)
        gqa_params = q_params + kv_params

        reduction = (mha_params - gqa_params) / mha_params * 100

        return {
            "mha_params": mha_params,
            "gqa_params": gqa_params,
            "params_saved": mha_params - gqa_params,
            "reduction_percent": reduction,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "queries_per_kv": self.num_queries_per_kv,
        }


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal mask for autoregressive attention.

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        (seq_len, seq_len) boolean mask where True = masked
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    return mask


__all__ = ["GroupedQueryAttention", "create_causal_mask"]
