"""Embedding layers for token and position embeddings."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Token embedding layer.

    Maps token IDs to dense embedding vectors.

    Args:
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension

    Example:
        >>> token_emb = TokenEmbedding(vocab_size=50257, embed_dim=512)
        >>> input_ids = torch.randint(0, 50257, (2, 10))
        >>> embeddings = token_emb(input_ids)
        >>> embeddings.shape
        torch.Size([2, 10, 512])
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)

        Returns:
            Embeddings of shape (batch, seq_len, embed_dim)
        """
        return self.embedding(input_ids)


class PositionEmbedding(nn.Module):
    """Position embedding layer.

    Adds positional information to token embeddings using learned embeddings.

    Args:
        max_seq_len: Maximum sequence length
        embed_dim: Embedding dimension

    Example:
        >>> pos_emb = PositionEmbedding(max_seq_len=2048, embed_dim=512)
        >>> batch_size, seq_len = 2, 10
        >>> positions = pos_emb(batch_size, seq_len, device='cuda')
        >>> positions.shape
        torch.Size([2, 10, 512])
    """

    def __init__(self, max_seq_len: int, embed_dim: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(max_seq_len, embed_dim)

    def forward(
        self,
        batch_size: int,
        seq_len: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            device: Target device

        Returns:
            Position embeddings of shape (batch, seq_len, embed_dim)
        """
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")

        # Create position indices
        positions = torch.arange(seq_len, device=device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        return self.embedding(positions)


class EmbeddingLayer(nn.Module):
    """Combined token and position embedding layer with dropout.

    Combines token embeddings and position embeddings, applies dropout.

    Args:
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
        embed_dim: Embedding dimension
        dropout_rate: Dropout rate for regularization

    Example:
        >>> emb_layer = EmbeddingLayer(
        ...     vocab_size=50257,
        ...     max_seq_len=2048,
        ...     embed_dim=512,
        ...     dropout_rate=0.1
        ... )
        >>> input_ids = torch.randint(0, 50257, (2, 10))
        >>> embeddings = emb_layer(input_ids)
        >>> embeddings.shape
        torch.Size([2, 10, 512])
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Token and position embeddings
        self.token_emb = TokenEmbedding(vocab_size, embed_dim)
        self.pos_emb = PositionEmbedding(max_seq_len, embed_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)

        Returns:
            Combined embeddings of shape (batch, seq_len, embed_dim)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get token and position embeddings
        token_embeddings = self.token_emb(input_ids)
        position_embeddings = self.pos_emb(batch_size, seq_len, device)

        # Combine and apply dropout
        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


__all__ = ["EmbeddingLayer", "PositionEmbedding", "TokenEmbedding"]
