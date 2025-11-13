"""
Base classes and utilities for landmark selection.

Provides common functionality shared across all landmark selectors.
"""


import torch
from torch import nn


class BaseLandmarkSelector(nn.Module):
    """Abstract base class for all landmark selectors.

    Provides common utility methods for landmark selection operations.

    Args:
        num_landmarks: Number of landmarks to select (G)
    """

    def __init__(self, num_landmarks: int):
        super().__init__()
        self.num_landmarks = num_landmarks

    def _safe_gather(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Safely gather elements with index clamping.

        Prevents index out-of-bounds errors by clamping indices.

        Args:
            x: (B, L, D) input tensor
            indices: (B, G) indices to gather

        Returns:
            gathered: (B, G, D) gathered elements
        """
        B, L, D = x.shape
        G = indices.size(1)

        # Clamp indices to valid range
        indices_safe = torch.clamp(indices, 0, L - 1)

        # Expand indices for gather operation
        indices_exp = indices_safe.unsqueeze(-1).expand(B, G, D)

        # Gather
        return torch.gather(x, dim=1, index=indices_exp)

    def _validate_sequence_length(self, seq_len: int) -> int:
        """Validate and adjust number of landmarks for sequence length.

        Args:
            seq_len: Actual sequence length L

        Returns:
            k: Adjusted number of landmarks (min(G, L))
        """
        return min(self.num_landmarks, seq_len)
