"""
Heuristic landmark selectors without learnable parameters.

Implements fixed strategies: uniform, stride, random, and first-K.
Useful for baselines and ablation studies.
"""

from typing import Literal, Tuple

import torch

from .base import BaseLandmarkSelector

StrategyType = Literal["uniform", "stride", "random", "first"]


class HeuristicLandmarkSelector(BaseLandmarkSelector):
    """Fixed landmark selection strategies without learning.

    Supports multiple heuristic strategies:
    - uniform: Evenly spaced landmarks across sequence
    - stride: Fixed stride selection (alias for uniform)
    - random: Random sampling (changes each forward pass)
    - first: Select first G positions

    No learnable parameters. Useful for baselines and ablations.

    Args:
        num_landmarks: Number of landmarks to select (G)
        strategy: Selection strategy (default: "uniform")
        seed: Random seed for "random" strategy (default: None)

    Example:
        >>> # Uniform spacing
        >>> selector = HeuristicLandmarkSelector(num_landmarks=64, strategy="uniform")
        >>> x = torch.randn(4, 256, 512)
        >>> indices, states, scores = selector(x)
        >>>
        >>> # Random selection with fixed seed
        >>> selector = HeuristicLandmarkSelector(
        ...     num_landmarks=64, strategy="random", seed=42
        ... )
    """

    def __init__(
        self,
        num_landmarks: int,
        strategy: StrategyType = "uniform",
        seed: int = None,
    ):
        super().__init__(num_landmarks)

        self.strategy = strategy
        self.seed = seed

        # For random strategy with fixed seed
        if seed is not None:
            self.rng = torch.Generator()
            self.rng.manual_seed(seed)
        else:
            self.rng = None

    def _select_uniform(self, L: int, k: int) -> torch.Tensor:
        """Select evenly spaced landmarks.

        Args:
            L: Sequence length
            k: Number of landmarks

        Returns:
            indices: (k,) uniform indices
        """
        # Calculate stride for uniform spacing
        stride = L / k
        indices = torch.arange(k) * stride
        indices = indices.long()

        # Clamp to valid range
        return torch.clamp(indices, 0, L - 1)

    def _select_random(self, L: int, k: int, device: torch.device) -> torch.Tensor:
        """Select random landmarks.

        Args:
            L: Sequence length
            k: Number of landmarks
            device: Device for tensor

        Returns:
            indices: (k,) random indices
        """
        if self.rng is not None:
            # Use fixed seed generator for reproducibility
            perm = torch.randperm(L, generator=self.rng, device=device)
        else:
            # Random each time
            perm = torch.randperm(L, device=device)

        return perm[:k].sort()[0]  # Sort for consistent ordering

    def _select_first(self, L: int, k: int) -> torch.Tensor:
        """Select first k positions.

        Args:
            L: Sequence length
            k: Number of landmarks

        Returns:
            indices: (k,) first k indices
        """
        return torch.arange(k, dtype=torch.long)

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select landmarks using heuristic strategy.

        Args:
            x: (B, L, D) input sequence
            **kwargs: Unused (protocol compatibility)

        Returns:
            landmark_indices: (B, G) indices (same for all batches)
            landmark_states: (B, G, D) corresponding states
            selection_scores: (B, L) uniform scores (1/L)
        """
        B, L, D = x.shape
        k = self._validate_sequence_length(L)

        # Select based on strategy
        if self.strategy == "uniform" or self.strategy == "stride":
            indices = self._select_uniform(L, k)
        elif self.strategy == "random":
            indices = self._select_random(L, k, x.device)
        elif self.strategy == "first":
            indices = self._select_first(L, k)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Move to device and expand for batch
        indices = indices.to(x.device)
        landmark_indices = indices.unsqueeze(0).expand(B, k)  # (B, G)

        # Gather states
        landmark_states = self._safe_gather(x, landmark_indices)

        # Uniform selection scores (no preference)
        selection_scores = torch.ones(B, L, device=x.device) / L

        return landmark_indices, landmark_states, selection_scores
