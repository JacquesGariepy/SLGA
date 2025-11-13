"""Feed-forward network layers."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Standard Feed-Forward Network (FFN) used in transformer blocks.

    Consists of two linear layers with activation and dropout.
    Supports GELU and SwiGLU activation functions.

    Args:
        embed_dim: Embedding dimension
        hidden_multiplier: Hidden layer multiplier (default: 4x)
        dropout: Dropout rate for regularization
        activation: Activation function ("gelu" or "swiglu")

    Example:
        >>> ffn = FeedForward(embed_dim=512, hidden_multiplier=4, dropout=0.1)
        >>> x = torch.randn(2, 10, 512)
        >>> output = ffn(x)
        >>> output.shape
        torch.Size([2, 10, 512])
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_multiplier: int = 4,
        dropout: float = 0.1,
        activation: Literal["gelu", "swiglu"] = "gelu",
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_multiplier = hidden_multiplier
        self.activation_type = activation

        # Hidden layer dimension is typically 4x the embedding dimension
        hidden_dim = embed_dim * hidden_multiplier

        if activation == "swiglu":
            # SwiGLU requires gating, so we need two projections
            self.fc1 = nn.Linear(embed_dim, hidden_dim)
            self.fc_gate = nn.Linear(embed_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, embed_dim)
        else:
            # Standard GELU activation
            self.fc1 = nn.Linear(embed_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the FFN.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        if self.activation_type == "swiglu":
            # SwiGLU: x * silu(gate)
            hidden = F.silu(self.fc_gate(x)) * self.fc1(x)
        else:
            # Standard GELU activation
            hidden = F.gelu(self.fc1(x))

        # Project back to embedding dimension
        output = self.fc2(hidden)

        # Apply dropout
        output = self.dropout(output)

        return output


class GLUFeedForward(nn.Module):
    """Gated Linear Unit (GLU) variant of Feed-Forward Network.

    Uses gating mechanism for better expressiveness.
    Equivalent to FeedForward with activation="swiglu".

    Args:
        embed_dim: Embedding dimension
        hidden_multiplier: Hidden layer multiplier (default: 4x)
        dropout: Dropout rate for regularization

    Example:
        >>> ffn = GLUFeedForward(embed_dim=512, hidden_multiplier=4, dropout=0.1)
        >>> x = torch.randn(2, 10, 512)
        >>> output = ffn(x)
        >>> output.shape
        torch.Size([2, 10, 512])
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_multiplier: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        hidden_dim = embed_dim * hidden_multiplier

        # Projection to hidden dimension
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        # Gate projection
        self.fc_gate = nn.Linear(embed_dim, hidden_dim)
        # Projection back to embedding dimension
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for GLU FFN.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        # Gated activation: x * sigmoid(gate)
        hidden = torch.sigmoid(self.fc_gate(x)) * self.fc1(x)

        # Project back and apply dropout
        output = self.fc2(hidden)
        output = self.dropout(output)

        return output


__all__ = ["FeedForward", "GLUFeedForward"]
