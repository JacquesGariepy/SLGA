"""
KV Cache for efficient autoregressive generation.
VERSION 1.0

Implements key-value caching for transformer inference to avoid recomputing
attention keys and values for previously generated tokens.

Features:
- Per-layer KV storage
- Dynamic sequence length tracking
- Efficient tensor pre-allocation
- Support for multi-head attention
- Compatible with RoPE (positions tracked separately)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch


@dataclass
class KVCache:
    """
    Cache for key-value pairs across generation steps.

    Stores K and V tensors for each transformer layer to enable efficient
    autoregressive generation by avoiding redundant computation.

    Attributes:
        key_cache: List of (B, H, seq_len, Dh) tensors per layer
        value_cache: List of (B, H, seq_len, Dh) tensors per layer
        seq_len: Current cached sequence length
        max_seq_len: Maximum sequence length capacity
        device: Device where cache is stored
        dtype: Data type of cached tensors
    """

    key_cache: List[torch.Tensor]
    value_cache: List[torch.Tensor]
    seq_len: int
    max_seq_len: int
    device: torch.device
    dtype: torch.dtype

    @classmethod
    def create(
        cls,
        batch_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "KVCache":
        """
        Pre-allocate KV cache for all layers.

        Args:
            batch_size: Batch size (B)
            num_layers: Number of transformer layers
            num_heads: Number of attention heads (H)
            head_dim: Dimension per head (Dh)
            max_seq_len: Maximum sequence length
            device: Device to allocate on
            dtype: Data type for cache

        Returns:
            KVCache instance with pre-allocated storage
        """
        key_cache = []
        value_cache = []

        for _ in range(num_layers):
            # Pre-allocate full cache: (B, H, max_seq_len, Dh)
            k = torch.zeros(
                batch_size, num_heads, max_seq_len, head_dim,
                device=device, dtype=dtype
            )
            v = torch.zeros(
                batch_size, num_heads, max_seq_len, head_dim,
                device=device, dtype=dtype
            )
            key_cache.append(k)
            value_cache.append(v)

        return cls(
            key_cache=key_cache,
            value_cache=value_cache,
            seq_len=0,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

    def update(
        self,
        layer_idx: int,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
    ) -> None:
        """
        Append new key-value pairs to the cache.

        Args:
            layer_idx: Layer index (0 to num_layers-1)
            new_keys: (B, H, new_L, Dh) new keys to append
            new_values: (B, H, new_L, Dh) new values to append
        """
        B, H, new_L, Dh = new_keys.shape

        # Validate shapes
        assert new_values.shape == (B, H, new_L, Dh), "K and V shapes must match"
        assert layer_idx < len(self.key_cache), f"Invalid layer_idx {layer_idx}"

        # Check capacity
        if self.seq_len + new_L > self.max_seq_len:
            raise ValueError(
                f"Cache overflow: current={self.seq_len}, new={new_L}, "
                f"max={self.max_seq_len}"
            )

        # Write to cache at current position
        start_idx = self.seq_len
        end_idx = self.seq_len + new_L

        self.key_cache[layer_idx][:, :, start_idx:end_idx, :] = new_keys
        self.value_cache[layer_idx][:, :, start_idx:end_idx, :] = new_values

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve cached K and V for a layer.

        Args:
            layer_idx: Layer index (0 to num_layers-1)

        Returns:
            keys: (B, H, seq_len, Dh) cached keys
            values: (B, H, seq_len, Dh) cached values
        """
        assert layer_idx < len(self.key_cache), f"Invalid layer_idx {layer_idx}"

        # Return only valid cached portion
        k = self.key_cache[layer_idx][:, :, :self.seq_len, :]
        v = self.value_cache[layer_idx][:, :, :self.seq_len, :]

        return k, v

    def increment_seq_len(self, increment: int) -> None:
        """
        Increment the current sequence length counter.

        Should be called after all layers have been updated with new tokens.

        Args:
            increment: Number of new tokens added
        """
        self.seq_len += increment
        if self.seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {self.seq_len} exceeds max {self.max_seq_len}"
            )

    def clear(self) -> None:
        """
        Reset cache to empty state without deallocating memory.
        """
        self.seq_len = 0
        # Zero out cached tensors
        for k_cache in self.key_cache:
            k_cache.zero_()
        for v_cache in self.value_cache:
            v_cache.zero_()

    def get_batch_size(self) -> int:
        """Get the batch size of the cache."""
        if len(self.key_cache) > 0:
            return self.key_cache[0].shape[0]
        return 0

    def get_num_layers(self) -> int:
        """Get the number of layers in the cache."""
        return len(self.key_cache)

    def get_num_heads(self) -> int:
        """Get the number of attention heads."""
        if len(self.key_cache) > 0:
            return self.key_cache[0].shape[1]
        return 0

    def get_head_dim(self) -> int:
        """Get the dimension per head."""
        if len(self.key_cache) > 0:
            return self.key_cache[0].shape[3]
        return 0

    def __repr__(self) -> str:
        return (
            f"KVCache(layers={len(self.key_cache)}, "
            f"seq_len={self.seq_len}/{self.max_seq_len}, "
            f"batch_size={self.get_batch_size()}, "
            f"num_heads={self.get_num_heads()}, "
            f"head_dim={self.get_head_dim()}, "
            f"device={self.device}, dtype={self.dtype})"
        )


__all__ = ["KVCache"]
