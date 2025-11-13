"""
Attention Mechanism Protocol Interfaces

This module defines the Protocol interfaces for all attention mechanisms in the SLGA architecture.
All attention implementations must conform to these interfaces for clean architecture compliance.
"""

from typing import Dict, Optional, Protocol, Tuple

from torch import Tensor


class AttentionMechanism(Protocol):
    """Protocol for all attention mechanisms.

    All attention implementations (local, global, fusion) must conform to this interface.
    This ensures interchangeability and testability of different attention strategies.
    """

    def forward(
        self,
        query: Tensor,  # [batch, seq_len, d_model]
        key: Tensor,  # [batch, seq_len, d_model]
        value: Tensor,  # [batch, seq_len, d_model]
        mask: Optional[
            Tensor
        ] = None,  # [batch, seq_len, seq_len] or [batch, num_heads, seq_len, seq_len]
    ) -> Tensor:  # [batch, seq_len, d_model]
        """Compute attention output.

        Args:
            query: Query tensor of shape [batch, seq_len, d_model]
            key: Key tensor of shape [batch, seq_len, d_model]
            value: Value tensor of shape [batch, seq_len, d_model]
            mask: Optional attention mask. True/1 indicates positions to MASK OUT (invalid).
                  Shape can be [batch, seq_len, seq_len] or [batch, num_heads, seq_len, seq_len]

        Returns:
            Attention output of shape [batch, seq_len, d_model]

        Example:
            >>> mechanism = LocalAttention(window_size=512)
            >>> output = mechanism(query, key, value, mask)
            >>> assert output.shape == query.shape
        """
        ...


class LocalAttention(Protocol):
    """Protocol for local attention with sliding window mechanism.

    Implements efficient local attention by restricting attention to a sliding window
    around each position. Supports both causal and non-causal variants.
    """

    window_size: int
    causal: bool
    dilation: int

    def forward(
        self,
        query: Tensor,  # [batch, num_heads, seq_len, head_dim]
        key: Tensor,  # [batch, num_heads, seq_len, head_dim]
        value: Tensor,  # [batch, num_heads, seq_len, head_dim]
        mask: Optional[Tensor] = None,  # [seq_len, window_size] boolean mask
    ) -> Tensor:  # [batch, num_heads, seq_len, head_dim]
        """Compute local attention within sliding window.

        Args:
            query: Query tensor split by heads [batch, num_heads, seq_len, head_dim]
            key: Key tensor split by heads [batch, num_heads, seq_len, head_dim]
            value: Value tensor split by heads [batch, num_heads, seq_len, head_dim]
            mask: Window mask [seq_len, window_size]. True = invalid position.
                  If None, computed based on causal setting and window_size.

        Returns:
            Local attention context [batch, num_heads, seq_len, head_dim]

        Example:
            >>> local_attn = SlidingWindowAttention(window_size=128, causal=True)
            >>> context = local_attn(q, k, v)
            >>> assert context.shape == q.shape

        Notes:
            - Window is centered around each position (or left-aligned if causal)
            - Dilation factor creates strided windows (1=dense, 2=every other position)
            - Invalid positions (padding) are masked with -inf before softmax
        """
        ...

    def _create_window_mask(
        self,
        seq_len: int,
        device: Tensor.device,  # type: ignore
    ) -> Tensor:  # [seq_len, window_size]
        """Create local window mask for efficient caching.

        Args:
            seq_len: Sequence length
            device: PyTorch device for tensor allocation

        Returns:
            Boolean mask [seq_len, window_size] where True = invalid position

        Notes:
            - Result is cached for performance (5-10x speedup)
            - Handles causal masking and boundary conditions
        """
        ...


class GlobalAttention(Protocol):
    """Protocol for global attention with sparse selection.

    Implements sparse global attention by attending to a subset of landmark tokens
    selected via top-K mechanism. Supports diverse top-K for multi-head specialization.
    """

    num_global_tokens: int
    diverse_topk: bool

    def forward(
        self,
        query: Tensor,  # [batch, num_heads, seq_len, head_dim]
        key_global: Tensor,  # [batch, num_heads, num_global, head_dim]
        value_global: Tensor,  # [batch, num_heads, num_global, head_dim]
        cache_positions: Optional[Tensor] = None,  # [batch, num_global] absolute positions
        causal: bool = True,
    ) -> Tensor:  # [batch, num_heads, seq_len, head_dim]
        """Compute global attention with sparse landmark selection.

        Args:
            query: Query tensor [batch, num_heads, seq_len, head_dim]
            key_global: Global landmark keys [batch, num_heads, num_global, head_dim]
            value_global: Global landmark values [batch, num_heads, num_global, head_dim]
            cache_positions: Absolute positions of landmarks [batch, num_global].
                           Used for causal masking to prevent attending to future landmarks.
            causal: If True, applies causal masking based on cache_positions

        Returns:
            Global attention context [batch, num_heads, seq_len, head_dim]

        Example:
            >>> global_attn = TopKGlobalAttention(num_global_tokens=48, diverse_topk=True)
            >>> context = global_attn(query, key_cache, value_cache, positions)
            >>> assert context.shape[2:] == query.shape[2:]  # seq_len, head_dim preserved

        Notes:
            - Uses top-K selection per query position to choose most relevant landmarks
            - diverse_topk encourages different heads to select different landmarks
            - Causal masking prevents attention to future landmark positions
            - NaN protection for fully masked rows (all landmarks in future)
        """
        ...

    def _diverse_topk(
        self,
        scores: Tensor,  # [batch, num_heads, seq_len, num_global]
        k: int,
        diversity_penalty: float = 0.1,
    ) -> Tuple[Tensor, Tensor]:  # ([batch, num_heads, seq_len, k], [batch, num_heads, seq_len, k])
        """Select top-K with inter-head diversity encouragement.

        Args:
            scores: Global attention scores [batch, num_heads, seq_len, num_global]
            k: Number of landmarks to select per position
            diversity_penalty: Weight for penalizing repeated selections across heads

        Returns:
            topk_values: Top-K scores [batch, num_heads, seq_len, k]
            topk_indices: Top-K landmark indices [batch, num_heads, seq_len, k]

        Notes:
            - Prevents head degeneration by penalizing common selections
            - Each head sequentially selects top-K with penalty on prior selections
            - Maintains gradient flow through soft selection mechanism
        """
        ...


class AttentionFusion(Protocol):
    """Protocol for fusing local and global attention contexts.

    Combines local and global attention outputs using learned gating mechanism
    or simple additive fusion. Enables dynamic balancing of context sources.
    """

    gated: bool
    head_dim: int

    def forward(
        self,
        local_context: Tensor,  # [batch, num_heads, seq_len, head_dim]
        global_context: Tensor,  # [batch, num_heads, seq_len, head_dim]
        global_weight: float = 1.0,
    ) -> Tensor:  # [batch, num_heads, seq_len, head_dim]
        """Fuse local and global attention contexts.

        Args:
            local_context: Local attention output [batch, num_heads, seq_len, head_dim]
            global_context: Global attention output [batch, num_heads, seq_len, head_dim]
            global_weight: Multiplicative weight for global context (0.0 to 1.0).
                          Used for progressive warmup during training.

        Returns:
            Fused context [batch, num_heads, seq_len, head_dim]

        Example:
            >>> fusion = GatedAttentionFusion(gated=True, head_dim=64)
            >>> fused = fusion(local_ctx, global_ctx, global_weight=0.8)
            >>> assert fused.shape == local_ctx.shape

        Notes:
            - If gated=True: Uses learned sigmoid gate to balance contexts
              fused = gate * local + (1 - gate) * global * global_weight
            - If gated=False: Simple additive fusion
              fused = local + global * global_weight
            - global_weight enables curriculum learning (start with local, add global gradually)
        """
        ...

    def get_gate_values(self) -> Optional[Tensor]:  # [batch, num_heads, seq_len]
        """Retrieve last computed gate values for monitoring.

        Returns:
            Gate scalar values [batch, num_heads, seq_len] if gated=True, else None.
            Values in [0, 1] where 1 = prefer local, 0 = prefer global.

        Notes:
            - Used for interpretability and debugging
            - Detached from computation graph
            - Only available if gated=True and after forward() call
        """
        ...


class SparseLocalGlobalAttention(Protocol):
    """Protocol for complete SLGA module combining local and global attention.

    This is the main attention interface that orchestrates local sliding window
    attention, global sparse attention, and their fusion. Corresponds to SLGAModule.
    """

    embed_dim: int
    num_heads: int
    head_dim: int
    local_window: int
    global_k: int
    causal: bool
    gated_fusion: bool
    dilation: int
    diverse_topk: bool

    def forward(
        self,
        x: Tensor,  # [batch, seq_len, embed_dim]
        cache_global: Optional[Tensor] = None,  # [batch, num_global, embed_dim]
        cache_positions: Optional[Tensor] = None,  # [batch, num_global]
        global_weight: float = 1.0,
    ) -> Tensor:  # [batch, seq_len, embed_dim]
        """Forward pass combining local and global attention.

        Args:
            x: Input embeddings [batch, seq_len, embed_dim]
            cache_global: Global landmark states [batch, num_global, embed_dim].
                         These are the pre-computed embeddings of landmark tokens.
            cache_positions: Absolute positions of landmarks [batch, num_global].
                            Used for causal masking in global attention.
            global_weight: Weight for global attention component (0.0 to 1.0).
                          Enables progressive warmup (start 0.0, ramp to 1.0).

        Returns:
            Combined attention output [batch, seq_len, embed_dim]

        Example:
            >>> slga = SLGAModule(
            ...     embed_dim=512, num_heads=8, local_window=128,
            ...     global_k=48, causal=True, gated_fusion=True
            ... )
            >>> output = slga(x, cache_global=landmarks, global_weight=0.5)
            >>> assert output.shape == x.shape

        Algorithm:
            1. Project x to Q, K, V and split heads
            2. Compute local attention: Q @ K_window^T (sliding window)
            3. Compute global attention: Q @ K_global^T (top-K selection)
            4. Fuse contexts: gate * local + (1-gate) * global
            5. Merge heads and output projection

        Notes:
            - Local attention uses causal sliding window with dilation
            - Global attention selects top-K landmarks per query via diverse top-K
            - Fusion is learned via gating if gated_fusion=True
            - All intermediate computations use head-split tensors for efficiency
        """
        ...

    def get_monitoring_data(self) -> Dict[str, Optional[Tensor]]:
        """Retrieve internal metrics for monitoring and debugging.

        Returns:
            Dictionary with keys:
            - 'gate_scalar': [batch, num_heads, seq_len] gate values if gated
            - 'global_topk_indices': [batch, num_heads, seq_len, k] selected landmarks
            - 'global_attention_weights': [batch, num_heads, seq_len, k] attention weights

        Notes:
            - All tensors are detached from computation graph
            - Used for visualization and analysis
            - Returns None for unavailable metrics
        """
        ...


__all__ = [
    "AttentionFusion",
    "AttentionMechanism",
    "GlobalAttention",
    "LocalAttention",
    "SparseLocalGlobalAttention",
]
