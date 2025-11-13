"""
Gumbel-Softmax landmark selector.

Implements differentiable landmark selection using Gumbel-Softmax relaxation.
Provides smooth gradients through temperature annealing.
"""

from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from src.core.landmarks.base import BaseLandmarkSelector


class GumbelSoftmaxSelector(BaseLandmarkSelector):
    """Differentiable landmark selection via Gumbel-Softmax.

    Uses a neural scorer to assign importance scores, then selects
    top-K landmarks through Gumbel-Softmax relaxation for differentiability.

    Training: Gumbel-Softmax with temperature annealing
    Inference: Hard top-K deterministic selection

    Args:
        embed_dim: Input embedding dimension
        num_landmarks: Number of landmarks to select (G)
        hidden_dim: Hidden dimension of scorer network (default: embed_dim // 2)
        temperature: Initial Gumbel temperature (default: 1.0)
        temperature_decay: Temperature decay factor (default: 0.999)
        min_temperature: Minimum temperature (default: 0.3)

    Example:
        >>> selector = GumbelSoftmaxSelector(embed_dim=512, num_landmarks=64)
        >>> x = torch.randn(4, 256, 512)  # (B, L, D)
        >>> indices, states, scores = selector(x)
        >>> indices.shape  # (4, 64)
        >>> states.shape   # (4, 64, 512)
    """

    def __init__(
        self,
        embed_dim: int,
        num_landmarks: int,
        hidden_dim: Optional[int] = None,
        temperature: float = 1.0,
        temperature_decay: float = 0.999,
        min_temperature: float = 0.3,
    ):
        super().__init__(num_landmarks)

        self.embed_dim = embed_dim
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature

        # Scorer network: embed -> hidden -> 1 (importance score)
        hidden = hidden_dim or (embed_dim // 2)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

        # Step counter for temperature decay (non-persistent buffer)
        self.register_buffer("step_count", torch.tensor(0), persistent=False)

    def _get_temperature(self) -> float:
        """Compute current temperature with exponential decay."""
        if self.training:
            temp = self.temperature * (self.temperature_decay ** self.step_count.item())
            return max(temp, self.min_temperature)
        return self.min_temperature

    def _gumbel_topk(
        self, scores: torch.Tensor, k: int, temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable top-K via Gumbel-Softmax.

        Args:
            scores: (B, L) importance scores
            k: Number of elements to select
            temperature: Temperature for relaxation

        Returns:
            soft_selection: (B, L) continuous weights
            hard_indices: (B, k) hard indices
        """
        B, L = scores.shape
        eps = 1e-10

        # Generate Gumbel noise in float32 for numerical stability
        # Avoids NaN in AMP (automatic mixed precision)
        original_dtype = scores.dtype
        uniform_noise = torch.rand(scores.shape, dtype=torch.float32, device=scores.device)
        gumbel_noise = -torch.log(-torch.log(uniform_noise + eps) + eps)
        gumbel_noise = gumbel_noise.to(original_dtype)

        # Safety check for NaN/Inf
        if torch.isnan(gumbel_noise).any() or torch.isinf(gumbel_noise).any():
            gumbel_noise = torch.zeros_like(scores)

        # Perturbed scores
        perturbed_scores = (scores + gumbel_noise) / temperature

        # Hard top-K (forward pass)
        _, hard_indices = torch.topk(perturbed_scores, k=k, dim=-1)

        # Soft selection via softmax (backward pass)
        soft_scores = F.softmax(perturbed_scores, dim=-1)

        # Safety check after softmax
        if torch.isnan(soft_scores).any():
            soft_scores = torch.ones_like(soft_scores) / L

        return soft_scores, hard_indices

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select landmarks using Gumbel-Softmax.

        Args:
            x: (B, L, D) input sequence
            **kwargs: Unused (for protocol compatibility)

        Returns:
            landmark_indices: (B, G) indices of selected landmarks
            landmark_states: (B, G, D) corresponding states
            selection_scores: (B, L) normalized selection scores
        """
        B, L, D = x.shape

        # Score each position
        scores = self.scorer(x).squeeze(-1)  # (B, L)

        # Clamp scores to prevent overflow in softmax
        scores = torch.clamp(scores, min=-20, max=20)

        # Validate number of landmarks
        k = self._validate_sequence_length(L)

        if self.training:
            # Gumbel-Softmax with temperature annealing
            temp = self._get_temperature()
            _, landmark_indices = self._gumbel_topk(scores, k, temp)
            self.step_count += 1
        else:
            # Inference: hard top-K
            _, landmark_indices = torch.topk(scores, k=k, dim=-1)

        # Gather corresponding states (safe gather with clamping)
        landmark_states = self._safe_gather(x, landmark_indices)

        # Normalized selection scores for auxiliary losses
        selection_scores = F.softmax(scores, dim=-1)

        # Safety check for NaN
        if torch.isnan(selection_scores).any():
            selection_scores = torch.ones_like(selection_scores) / L

        return landmark_indices, landmark_states, selection_scores
