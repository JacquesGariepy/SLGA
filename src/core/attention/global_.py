"""Global attention with top-K landmark selection.

This module implements efficient global attention by selecting top-K most relevant
positions from a global cache, reducing complexity while maintaining long-range context.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from src.core.attention.base import BaseAttention

# Global attention configuration constants
DEFAULT_TOP_K = 24
DEFAULT_DIVERSITY_PENALTY = 0.1
MIN_TOP_K = 1


class GlobalAttention(BaseAttention):
    """Global attention with top-K landmark selection and diversity.

    Implements efficient global attention by:
    1. Selecting top-K most relevant positions from global cache
    2. Encouraging diversity across attention heads
    3. Supporting causal masking with position tracking
    4. Caching global states for efficiency

    Key features:
    - Top-K selection reduces O(L*G) to O(L*K) where K << G
    - Diverse top-K prevents head degeneration (all heads selecting same positions)
    - NaN protection for fully masked rows
    - Gradient-friendly clamping

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        top_k: Number of top positions to select per query (default: 24)
        dropout: Dropout probability for attention weights (default: 0.1)
        causal: Whether to use causal masking (default: True)
        diverse_topk: Enable diversity penalty across heads (default: True)
        diversity_penalty: Penalty weight for repeated selections (default: 0.1)

    Raises:
        ValueError: If top_k <= 0
        ValueError: If diversity_penalty < 0

    Example:
        >>> attn = GlobalAttention(d_model=512, num_heads=8, top_k=32)
        >>> q = torch.randn(2, 100, 512)  # Queries
        >>> cache = torch.randn(2, 1000, 512)  # Global cache
        >>> k = v = cache
        >>> output = attn(q, k, v)
        >>> assert output.shape == (2, 100, 512)

    Complexity:
        Time: O(L * G) for scoring + O(L * K * log(G)) for top-K
        Space: O(B * H * L * K) for selected keys/values
        where K << G, so effectively O(L * K) complexity
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = DEFAULT_TOP_K,
        dropout: float = 0.1,
        causal: bool = True,
        diverse_topk: bool = True,
        diversity_penalty: float = DEFAULT_DIVERSITY_PENALTY,
    ) -> None:
        super().__init__(d_model, num_heads, dropout)

        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if diversity_penalty < 0:
            raise ValueError(f"diversity_penalty must be >= 0, got {diversity_penalty}")

        self.top_k = top_k
        self.causal = causal
        self.diverse_topk = diverse_topk
        self.diversity_penalty = diversity_penalty

        # Cache for storing last forward pass metrics
        self.last_topk_indices: Optional[Tensor] = None
        self.last_topk_scores: Optional[Tensor] = None

    def _diverse_topk_selection(
        self,
        scores: Tensor,
        k: int,
    ) -> Tuple[Tensor, Tensor]:
        """Select top-K with diversity encouragement across heads.

        Standard top-K can lead to all attention heads selecting the same
        positions, causing head degeneration. This method applies a diversity
        penalty to encourage different heads to focus on different positions.

        The penalty works by tracking how many times each global position has
        been selected by previous heads, and reducing the score for positions
        that have been selected frequently.

        Args:
            scores: Attention scores of shape [batch, num_heads, seq_len, global_size]
            k: Number of elements to select per query

        Returns:
            topk_values: Selected scores of shape [batch, num_heads, seq_len, k]
            topk_indices: Indices of selected positions [batch, num_heads, seq_len, k]

        Example:
            >>> scores = torch.randn(2, 8, 100, 1000)
            >>> values, indices = self._diverse_topk_selection(scores, k=32)
            >>> # Each head should select somewhat different positions
            >>> # Check diversity: compute overlap between head 0 and head 1
            >>> overlap = (indices[:, 0] == indices[:, 1].unsqueeze(-1)).any(-1).float().mean()
            >>> # With diversity, overlap should be < 1.0
        """
        if not self.diverse_topk:
            # Standard top-K (no diversity penalty)
            return torch.topk(scores, k=k, dim=-1)

        batch_size, num_heads, seq_len, global_size = scores.shape
        k_actual = min(k, global_size)

        # Track how many times each position has been selected
        selection_counts = torch.zeros(
            batch_size, seq_len, global_size, device=scores.device, dtype=scores.dtype
        )

        topk_values_list = []
        topk_indices_list = []

        # Process each head sequentially to apply diversity penalty
        for h in range(num_heads):
            scores_h = scores[:, h]  # [B, L, G]

            # Apply penalty for positions selected by previous heads
            if h > 0:
                penalty = self.diversity_penalty * selection_counts
                scores_h = scores_h - penalty

            # Select top-K for this head
            topk_values_h, topk_indices_h = torch.topk(scores_h, k=k_actual, dim=-1)

            # Update selection counter
            selection_counts.scatter_add_(
                2,  # dim=2 (global_size dimension)
                topk_indices_h,
                torch.ones_like(topk_indices_h, dtype=scores.dtype),
            )

            topk_values_list.append(topk_values_h)
            topk_indices_list.append(topk_indices_h)

        # Stack results back into [B, H, L, K] format
        topk_values = torch.stack(topk_values_list, dim=1)
        topk_indices = torch.stack(topk_indices_list, dim=1)

        return topk_values, topk_indices

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        cache_positions: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute global attention with top-K selection.

        Args:
            query: Query tensor of shape [batch, seq_len, d_model]
            key: Key tensor (global cache) of shape [batch, cache_size, d_model]
            value: Value tensor (global cache) of shape [batch, cache_size, d_model]
            mask: Optional external mask (unused, causal mask applied via positions)
            cache_positions: Optional absolute positions of cache entries [batch, cache_size]
                           Used for causal masking in autoregressive generation

        Returns:
            Attended output of shape [batch, seq_len, d_model]

        Raises:
            ValueError: If input tensors have invalid shapes
            ValueError: If cache_positions shape doesn't match cache size

        Implementation notes:
            1. Compute attention scores between queries and global cache
            2. Apply causal mask if cache_positions provided
            3. Select top-K positions (with diversity if enabled)
            4. Protect against fully masked rows (NaN prevention)
            5. Gather values for selected positions
            6. Compute weighted sum
        """
        # Note: We don't call validate_inputs because key/value have different seq_len
        # Instead, do custom validation
        if query.dim() != 3:
            raise ValueError(f"Query must be 3D [batch, seq_len, d_model], got shape {query.shape}")
        if key.dim() != 3 or value.dim() != 3:
            raise ValueError(
                f"Key and value must be 3D [batch, cache_size, d_model], "
                f"got shapes {key.shape}, {value.shape}"
            )

        batch_size, seq_len, _ = query.shape
        cache_batch, cache_size, _ = key.shape

        if batch_size != cache_batch:
            raise ValueError(f"Batch size mismatch: query={batch_size}, cache={cache_batch}")
        if key.shape != value.shape:
            raise ValueError(f"Key and value shapes must match: {key.shape} vs {value.shape}")
        if cache_positions is not None:
            if cache_positions.shape != (cache_batch, cache_size):
                raise ValueError(
                    f"cache_positions shape must be [batch, cache_size], "
                    f"got {cache_positions.shape}"
                )

        device = query.device

        # Split into attention heads
        q = self.split_heads(query)  # [B, H, L, Dh]
        k = self.split_heads(key)  # [B, H, G, Dh]
        v = self.split_heads(value)  # [B, H, G, Dh]

        # Compute attention scores: Q @ K^T
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, G]

        # Apply causal mask if positions provided
        if self.causal and cache_positions is not None:
            # Query positions: assume sequential [0, 1, 2, ..., L-1]
            query_positions = torch.arange(seq_len, device=device).view(1, 1, seq_len, 1)
            cache_pos_expanded = cache_positions.view(batch_size, 1, 1, cache_size)

            # Mask: True where cache position is in the future (invalid)
            future_mask = cache_pos_expanded > query_positions
            scores = scores.masked_fill(future_mask, float("-inf"))

        # Select top-K positions (with or without diversity)
        k_select = min(self.top_k, cache_size)
        topk_scores, topk_indices = self._diverse_topk_selection(scores, k=k_select)
        # topk_scores: [B, H, L, K]
        # topk_indices: [B, H, L, K]

        # Store for monitoring/debugging
        self.last_topk_indices = topk_indices.detach()
        self.last_topk_scores = topk_scores.detach()

        # Detect fully masked rows (all top-K scores are -inf after causal masking)
        all_masked = (topk_scores == float("-inf")).all(dim=-1)  # [B, H, L]

        # Compute attention weights with safe softmax
        attn_weights = F.softmax(topk_scores, dim=-1)  # [B, H, L, K]

        # Replace NaN with zeros for fully masked rows
        if all_masked.any():
            attn_weights = torch.where(
                all_masked.unsqueeze(-1),  # [B, H, L, 1]
                torch.zeros_like(attn_weights),
                attn_weights,
            )

        attn_weights = self.dropout(attn_weights)

        # Gather values according to top-K indices
        # v: [B, H, G, Dh] -> need to select according to topk_indices: [B, H, L, K]

        # Clamp indices to valid range (protection against edge cases)
        topk_indices_safe = torch.clamp(topk_indices, 0, cache_size - 1)

        # Expand dimensions for gathering
        v_expanded = v.unsqueeze(2).expand(
            batch_size, self.num_heads, seq_len, cache_size, self.head_dim
        )  # [B, H, L, G, Dh]

        topk_indices_expanded = topk_indices_safe.unsqueeze(-1).expand(
            batch_size, self.num_heads, seq_len, k_select, self.head_dim
        )  # [B, H, L, K, Dh]

        # Gather selected values
        v_selected = torch.gather(
            v_expanded, dim=3, index=topk_indices_expanded
        )  # [B, H, L, K, Dh]

        # Compute weighted sum of values
        context = (attn_weights.unsqueeze(-1) * v_selected).sum(dim=3)  # [B, H, L, Dh]

        # Merge heads back to d_model
        output = self.merge_heads(context)  # [B, L, D]

        return output
