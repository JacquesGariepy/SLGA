"""Base attention module with shared utilities.

This module provides the abstract base class and common utilities for all
attention mechanisms in SLGA-Plus.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

# Constants for numerical stability
EPSILON = 1e-10
NEG_INF = float("-inf")


class BaseAttention(nn.Module, ABC):
    """Abstract base class for attention mechanisms.

    Provides common functionality for all attention types including:
    - Scaling factor computation
    - Safe masked softmax with NaN protection
    - Head splitting and merging utilities
    - Input validation

    Args:
        d_model: Model/embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability for attention weights

    Raises:
        ValueError: If d_model is not divisible by num_heads
        ValueError: If dropout is not in [0, 1)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.dropout = nn.Dropout(dropout)

    @abstractmethod
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute attention mechanism (must be implemented by subclasses).

        Args:
            query: Query tensor [batch, seq_len, d_model]
            key: Key tensor [batch, seq_len, d_model]
            value: Value tensor [batch, seq_len, d_model]
            mask: Optional boolean mask [batch, seq_len, seq_len]

        Returns:
            Attended output [batch, seq_len, d_model]
        """

    def split_heads(self, tensor: Tensor) -> Tensor:
        """Split tensor into multiple attention heads.

        Args:
            tensor: Input tensor of shape [batch, seq_len, d_model]

        Returns:
            Reshaped tensor of shape [batch, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, d_model = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)  # [B, H, L, Dh]

    def merge_heads(self, tensor: Tensor) -> Tensor:
        """Merge attention heads back to d_model.

        Args:
            tensor: Input tensor of shape [batch, num_heads, seq_len, head_dim]

        Returns:
            Merged tensor of shape [batch, seq_len, d_model]
        """
        batch_size, num_heads, seq_len, head_dim = tensor.shape
        tensor = tensor.transpose(1, 2)  # [B, L, H, Dh]
        return tensor.contiguous().view(batch_size, seq_len, num_heads * head_dim)

    def safe_masked_softmax(
        self,
        scores: Tensor,
        mask: Optional[Tensor] = None,
        dim: int = -1,
    ) -> Tensor:
        """Compute softmax with NaN protection for fully masked rows.

        This method handles edge cases where all positions in a row are masked,
        which would normally produce NaN values from softmax. Instead, it returns
        zeros for such rows.

        Args:
            scores: Attention scores of any shape
            mask: Boolean mask where True indicates invalid positions
            dim: Dimension along which to apply softmax

        Returns:
            Attention weights with same shape as scores

        Example:
            >>> scores = torch.randn(2, 4, 10)
            >>> mask = torch.zeros(2, 4, 10, dtype=torch.bool)
            >>> mask[0, 2, :] = True  # Fully mask one row
            >>> weights = self.safe_masked_softmax(scores, mask, dim=-1)
            >>> assert not torch.isnan(weights).any()
        """
        if mask is not None:
            # Apply mask by setting invalid positions to -inf
            scores_masked = scores.masked_fill(mask, NEG_INF)
        else:
            scores_masked = scores

        # Detect fully masked rows (all values are -inf)
        if mask is not None:
            all_masked = mask.all(dim=dim, keepdim=True)
        else:
            all_masked = None

        # Compute softmax (may produce nan for fully masked rows)
        attn_weights = F.softmax(scores_masked, dim=dim)

        # Replace NaN with zeros for fully masked rows
        if all_masked is not None and all_masked.any():
            attn_weights = torch.where(
                all_masked.expand_as(attn_weights), torch.zeros_like(attn_weights), attn_weights
            )

        return attn_weights

    def validate_inputs(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> None:
        """Validate input tensor shapes and dimensions.

        Args:
            query: Query tensor [batch, seq_len, d_model]
            key: Key tensor [batch, seq_len, d_model]
            value: Value tensor [batch, seq_len, d_model]
            mask: Optional mask tensor

        Raises:
            ValueError: If any tensor has invalid shape
        """
        # Check that all tensors have 3 dimensions
        if query.dim() != 3:
            raise ValueError(f"Query must be 3D [batch, seq_len, d_model], got shape {query.shape}")
        if key.dim() != 3:
            raise ValueError(f"Key must be 3D [batch, seq_len, d_model], got shape {key.shape}")
        if value.dim() != 3:
            raise ValueError(f"Value must be 3D [batch, seq_len, d_model], got shape {value.shape}")

        # Check batch size consistency
        batch_q, seq_q, dim_q = query.shape
        batch_k, seq_k, dim_k = key.shape
        batch_v, seq_v, dim_v = value.shape

        if batch_q != batch_k or batch_q != batch_v:
            raise ValueError(
                f"Batch size mismatch: query={batch_q}, key={batch_k}, value={batch_v}"
            )

        if seq_k != seq_v:
            raise ValueError(
                f"Key and value sequence lengths must match: key={seq_k}, value={seq_v}"
            )

        # Check model dimension
        if dim_q != self.d_model or dim_k != self.d_model or dim_v != self.d_model:
            raise ValueError(
                f"Model dimension mismatch: expected {self.d_model}, "
                f"got query={dim_q}, key={dim_k}, value={dim_v}"
            )

        # Validate mask shape if provided
        if mask is not None:
            if mask.dim() not in [3, 4]:
                raise ValueError(
                    f"Mask must be 3D [batch, seq_q, seq_k] or "
                    f"4D [batch, num_heads, seq_q, seq_k], got shape {mask.shape}"
                )

            expected_batch = batch_q
            expected_seq_q = seq_q
            expected_seq_k = seq_k

            if mask.dim() == 3:
                mask_batch, mask_seq_q, mask_seq_k = mask.shape
                if (
                    mask_batch != expected_batch
                    or mask_seq_q != expected_seq_q
                    or mask_seq_k != expected_seq_k
                ):
                    raise ValueError(
                        f"3D mask shape mismatch: expected [{expected_batch}, "
                        f"{expected_seq_q}, {expected_seq_k}], got {mask.shape}"
                    )
            else:  # 4D mask
                mask_batch, mask_heads, mask_seq_q, mask_seq_k = mask.shape
                if (
                    mask_batch != expected_batch
                    or mask_heads != self.num_heads
                    or mask_seq_q != expected_seq_q
                    or mask_seq_k != expected_seq_k
                ):
                    raise ValueError(
                        f"4D mask shape mismatch: expected [{expected_batch}, "
                        f"{self.num_heads}, {expected_seq_q}, {expected_seq_k}], "
                        f"got {mask.shape}"
                    )
