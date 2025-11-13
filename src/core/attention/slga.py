"""SLGA attention: fusion of local and global attention mechanisms.

This module implements the complete Sparse Local-Global Attention (SLGA) mechanism
by composing local sliding window attention with global top-K attention, using
learned gating for adaptive fusion.
"""

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from src.core.attention.global_ import GlobalAttention
from src.core.attention.local import LocalAttention

# Fusion configuration constants
DEFAULT_GLOBAL_WEIGHT = 1.0
MIN_GLOBAL_WEIGHT = 0.0
MAX_GLOBAL_WEIGHT = 1.0


class SLGAAttention(nn.Module):
    """Sparse Local-Global Attention with learned fusion.

    Combines two complementary attention mechanisms:
    1. Local attention: Sliding window for nearby context (efficient)
    2. Global attention: Top-K landmarks for long-range dependencies

    Fusion strategies:
    - Gated fusion: Learned per-position gate weights (adaptive)
    - Additive fusion: Simple addition (baseline)

    Architecture:
        Input: [batch, seq_len, d_model]
        ├─ QKV Projection → Q, K, V
        ├─ Local Branch: LocalAttention(Q, K, V) → context_local
        ├─ Global Branch: GlobalAttention(Q, K_cache, V_cache) → context_global
        ├─ Fusion: Gate(context_local, context_global) → context_fused
        └─ Output Projection → [batch, seq_len, d_model]

    Args:
        d_model: Model dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        local_window: Size of local attention window (default: 128)
        global_k: Number of global landmarks to select (default: 24)
        dropout: Dropout probability for attention and projection (default: 0.1)
        causal: Whether to use causal masking (default: True)
        gated_fusion: Use learned gating (True) or simple addition (False)
        dilation: Dilation factor for local window (default: 1)
        diverse_topk: Enable diversity across heads for global attention

    Raises:
        ValueError: If d_model not divisible by num_heads
        ValueError: If local_window or global_k <= 0

    Example:
        >>> slga = SLGAAttention(
        ...     d_model=512,
        ...     num_heads=8,
        ...     local_window=128,
        ...     global_k=32,
        ...     gated_fusion=True
        ... )
        >>>
        >>> # Forward pass with global cache
        >>> x = torch.randn(2, 1000, 512)
        >>> cache = torch.randn(2, 100, 512)  # 100 global landmarks
        >>> output = slga(x, cache_global=cache)
        >>>
        >>> # Access fusion metrics
        >>> gate_weights = slga.get_gate_weights()  # [batch, num_heads, seq_len]
        >>> topk_indices = slga.get_topk_indices()  # [batch, num_heads, seq_len, k]

    Complexity:
        Time: O(L*W + L*K) where W=local_window, K=global_k
        Space: O(B*H*(L*W + L*K))
        Much more efficient than O(L²) for long sequences
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        local_window: int = 128,
        global_k: int = 24,
        dropout: float = 0.1,
        causal: bool = True,
        gated_fusion: bool = True,
        dilation: int = 1,
        diverse_topk: bool = True,
    ) -> None:
        super().__init__()

        # Validate parameters
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        if local_window <= 0:
            raise ValueError(f"local_window must be > 0, got {local_window}")
        if global_k <= 0:
            raise ValueError(f"global_k must be > 0, got {global_k}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.local_window = local_window
        self.global_k = global_k
        self.causal = causal
        self.gated_fusion = gated_fusion

        # Unified QKV projection for both local and global attention
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)

        # Local attention module
        self.local_attn = LocalAttention(
            d_model=d_model,
            num_heads=num_heads,
            window_size=local_window,
            dropout=dropout,
            causal=causal,
            dilation=dilation,
        )

        # Global attention module
        self.global_attn = GlobalAttention(
            d_model=d_model,
            num_heads=num_heads,
            top_k=global_k,
            dropout=dropout,
            causal=causal,
            diverse_topk=diverse_topk,
        )

        # Learned gating mechanism (if enabled)
        if self.gated_fusion:
            # Input: concatenation of local and global contexts [2*head_dim]
            # Output: gate weights [head_dim]
            self.gate_proj = nn.Linear(2 * self.head_dim, self.head_dim)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.proj_dropout = nn.Dropout(dropout)

        # Storage for monitoring metrics
        self._last_gate_weights: Optional[Tensor] = None
        self._last_topk_indices: Optional[Tensor] = None
        self._last_topk_scores: Optional[Tensor] = None

    def _split_heads(self, tensor: Tensor) -> Tensor:
        """Reshape [B, L, D] -> [B, H, L, Dh]."""
        batch_size, seq_len, d_model = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def _merge_heads(self, tensor: Tensor) -> Tensor:
        """Reshape [B, H, L, Dh] -> [B, L, D]."""
        batch_size, num_heads, seq_len, head_dim = tensor.shape
        tensor = tensor.transpose(1, 2)
        return tensor.contiguous().view(batch_size, seq_len, num_heads * head_dim)

    def _fuse_contexts(
        self,
        context_local: Tensor,
        context_global: Tensor,
    ) -> Tensor:
        """Fuse local and global contexts using learned gating or addition.

        Args:
            context_local: Local attention context [B, H, L, Dh]
            context_global: Global attention context [B, H, L, Dh]

        Returns:
            Fused context [B, H, L, Dh]
        """
        if not self.gated_fusion:
            # Simple additive fusion
            return context_local + context_global

        # Learned gated fusion
        batch_size, num_heads, seq_len, head_dim = context_local.shape

        # Concatenate contexts along head_dim
        context_cat = torch.cat([context_local, context_global], dim=-1)  # [B, H, L, 2*Dh]

        # Reshape for linear projection
        context_cat_flat = context_cat.reshape(batch_size * num_heads * seq_len, 2 * self.head_dim)

        # Compute gate weights (sigmoid for [0, 1] range)
        gate_flat = torch.sigmoid(self.gate_proj(context_cat_flat))  # [B*H*L, Dh]
        gate = gate_flat.view(batch_size, num_heads, seq_len, self.head_dim)

        # Store for monitoring (detached from computation graph)
        self._last_gate_weights = gate.mean(dim=-1).detach()  # [B, H, L]

        # Weighted fusion: gate * local + (1 - gate) * global
        context_fused = gate * context_local + (1 - gate) * context_global

        return context_fused

    def forward(
        self,
        x: Tensor,
        cache_global: Optional[Tensor] = None,
        cache_positions: Optional[Tensor] = None,
        global_weight: float = DEFAULT_GLOBAL_WEIGHT,
    ) -> Tensor:
        """Forward pass of SLGA attention.

        Args:
            x: Input embeddings of shape [batch, seq_len, d_model]
            cache_global: Optional global cache [batch, cache_size, d_model]
                         If None, only local attention is computed
            cache_positions: Optional positions of cache entries [batch, cache_size]
                           Used for causal masking in autoregressive generation
            global_weight: Weight for global attention contribution (0.0 to 1.0)
                          Useful for progressive warmup during training

        Returns:
            Output tensor of shape [batch, seq_len, d_model]

        Raises:
            ValueError: If input tensor has invalid shape
            ValueError: If global_weight not in [0, 1]

        Implementation flow:
            1. Project input to Q, K, V
            2. Compute local attention (sliding window)
            3. Compute global attention (if cache provided)
            4. Fuse local and global contexts
            5. Project to output
        """
        if x.dim() != 3:
            raise ValueError(f"Input must be 3D [batch, seq_len, d_model], got shape {x.shape}")
        if not MIN_GLOBAL_WEIGHT <= global_weight <= MAX_GLOBAL_WEIGHT:
            raise ValueError(f"global_weight must be in [0, 1], got {global_weight}")

        batch_size, seq_len, d_model = x.shape

        if d_model != self.d_model:
            raise ValueError(f"Input d_model mismatch: expected {self.d_model}, got {d_model}")

        # ===== QKV Projection =====
        qkv = self.qkv_proj(x)  # [B, L, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)  # [B, L, D] each

        # ===== Local Attention =====
        context_local = self.local_attn(q, k, v)  # [B, L, D]

        # Convert to multi-head format for fusion
        context_local_heads = self._split_heads(context_local)  # [B, H, L, Dh]

        # ===== Global Attention (if cache provided) =====
        if cache_global is not None and global_weight > 0.0:
            # Project global cache to K, V using same projection
            cache_qkv = self.qkv_proj(cache_global)  # [B, G, 3*D]
            _, k_global, v_global = cache_qkv.chunk(3, dim=-1)  # [B, G, D]

            # Compute global attention
            context_global = self.global_attn(
                q, k_global, v_global, cache_positions=cache_positions
            )  # [B, L, D]

            # Convert to multi-head format
            context_global_heads = self._split_heads(context_global)  # [B, H, L, Dh]

            # Apply warmup weight
            context_global_heads = context_global_heads * global_weight

            # Store global attention metrics
            self._last_topk_indices = self.global_attn.last_topk_indices
            self._last_topk_scores = self.global_attn.last_topk_scores

            # Fuse local and global contexts
            context_fused = self._fuse_contexts(context_local_heads, context_global_heads)
        else:
            # No global attention (cache not provided or weight=0)
            context_fused = context_local_heads
            self._last_topk_indices = None
            self._last_topk_scores = None

        # ===== Output Projection =====
        context_merged = self._merge_heads(context_fused)  # [B, L, D]
        output = self.out_proj(context_merged)
        output = self.proj_dropout(output)

        return output

    def get_gate_weights(self) -> Optional[Tensor]:
        """Get gate weights from last forward pass.

        Returns:
            Gate weights of shape [batch, num_heads, seq_len] if gated fusion
            is enabled and a forward pass has been completed, None otherwise.

            Values are in [0, 1] where:
            - 1.0 = fully local context
            - 0.0 = fully global context
        """
        return self._last_gate_weights

    def get_topk_indices(self) -> Optional[Tensor]:
        """Get top-K indices selected by global attention.

        Returns:
            Indices of shape [batch, num_heads, seq_len, k] if global attention
            was used in last forward pass, None otherwise.
        """
        return self._last_topk_indices

    def get_topk_scores(self) -> Optional[Tensor]:
        """Get top-K attention scores from global attention.

        Returns:
            Scores of shape [batch, num_heads, seq_len, k] if global attention
            was used in last forward pass, None otherwise.
        """
        return self._last_topk_scores

    def get_metrics(self) -> Dict[str, Optional[Tensor]]:
        """Get all monitoring metrics from last forward pass.

        Returns:
            Dictionary containing:
            - 'gate_weights': Gate weights [B, H, L] (if gated fusion)
            - 'topk_indices': Selected global positions [B, H, L, K]
            - 'topk_scores': Global attention scores [B, H, L, K]
        """
        return {
            "gate_weights": self._last_gate_weights,
            "topk_indices": self._last_topk_indices,
            "topk_scores": self._last_topk_scores,
        }
