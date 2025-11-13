"""
Learned landmark selectors using neural networks.

Implements attention-based and positional landmark selection strategies.
"""

from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .base import BaseLandmarkSelector


class LearnedLandmarkSelector(BaseLandmarkSelector):
    """Neural network-based landmark selection with straight-through estimator.

    Similar to Gumbel-Softmax but uses sigmoid soft-thresholding for
    backward pass (straight-through estimator).

    Faster than Gumbel-Softmax but less stable gradients. Suitable for
    fine-tuning or when quick prototyping is needed.

    Args:
        embed_dim: Input embedding dimension
        num_landmarks: Number of landmarks to select (G)
        hidden_dim: Hidden dimension of scorer (default: embed_dim // 2)

    Example:
        >>> selector = LearnedLandmarkSelector(embed_dim=512, num_landmarks=64)
        >>> x = torch.randn(4, 256, 512)
        >>> indices, states, scores = selector(x)
    """

    def __init__(
        self,
        embed_dim: int,
        num_landmarks: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__(num_landmarks)

        self.embed_dim = embed_dim

        # Scorer network
        hidden = hidden_dim or (embed_dim // 2)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def _straight_through_topk(
        self, scores: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-K with straight-through estimator.

        Forward: Hard top-K (one-hot)
        Backward: Gradient via sigmoid soft-thresholding

        Args:
            scores: (B, L) raw importance scores
            k: Number of elements to select

        Returns:
            selection: (B, L) selection weights (forward=hard, backward=soft)
            topk_indices: (B, k) hard indices of top-k
        """
        B, L = scores.shape

        # Forward: Hard top-K
        topk_vals, topk_indices = torch.topk(scores, k=k, dim=-1)

        # One-hot encoding of selections
        selection_onehot = torch.zeros_like(scores)
        selection_onehot.scatter_(1, topk_indices, 1.0)

        # Soft selection for backward pass
        # Uses sigmoid soft-thresholding based on k-th value
        threshold = topk_vals[:, -1:].detach()  # k-th score
        temp = 0.1  # Temperature: lower = closer to hard

        selection_soft = torch.sigmoid((scores - threshold) / temp)

        # Straight-through: forward=hard, backward=soft
        selection = selection_onehot + selection_soft - selection_soft.detach()

        return selection, topk_indices

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select landmarks using straight-through estimator.

        Args:
            x: (B, L, D) input sequence
            **kwargs: Unused (protocol compatibility)

        Returns:
            landmark_indices: (B, G) indices
            landmark_states: (B, G, D) states
            selection_scores: (B, L) normalized scores
        """
        B, L, D = x.shape

        # Score each position
        scores = self.scorer(x).squeeze(-1)  # (B, L)
        scores = torch.clamp(scores, min=-20, max=20)

        k = self._validate_sequence_length(L)

        if self.training:
            _, landmark_indices = self._straight_through_topk(scores, k)
        else:
            _, landmark_indices = torch.topk(scores, k=k, dim=-1)

        landmark_states = self._safe_gather(x, landmark_indices)
        selection_scores = F.softmax(scores, dim=-1)

        if torch.isnan(selection_scores).any():
            selection_scores = torch.ones_like(selection_scores) / L

        return landmark_indices, landmark_states, selection_scores


class PositionalLandmarkSelector(BaseLandmarkSelector):
    """Landmark selection based on learned positional patterns.

    Instead of scoring individual tokens, learns patterns of important
    positions (e.g., paragraph starts, every N tokens).

    Args:
        max_seq_len: Maximum sequence length
        num_landmarks: Number of landmarks to select (G)
        embed_dim: Embedding dimension for positional embeddings

    Example:
        >>> selector = PositionalLandmarkSelector(
        ...     max_seq_len=2048, num_landmarks=64, embed_dim=512
        ... )
        >>> x = torch.randn(4, 256, 512)
        >>> indices, states, scores = selector(x)
    """

    def __init__(
        self,
        max_seq_len: int,
        num_landmarks: int,
        embed_dim: int,
    ):
        super().__init__(num_landmarks)

        self.max_seq_len = max_seq_len

        # Learnable positional embeddings
        self.pos_embeddings = nn.Parameter(torch.randn(max_seq_len, embed_dim))

        # Projector to scores
        self.scorer = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select landmarks based on positional patterns.

        Args:
            x: (B, L, D) input sequence
            **kwargs: Unused

        Returns:
            landmark_indices: (B, G) indices
            landmark_states: (B, G, D) states
            selection_scores: (B, L) scores
        """
        B, L, D = x.shape

        # Get positional embeddings for this length
        pos_emb = self.pos_embeddings[:L]  # (L, embed_dim)

        # Score positions
        scores = self.scorer(pos_emb).squeeze(-1)  # (L,)
        scores = scores.unsqueeze(0).expand(B, L)  # (B, L)

        # Top-K
        k = self._validate_sequence_length(L)
        _, landmark_indices = torch.topk(scores, k=k, dim=-1)

        # Gather states
        landmark_states = self._safe_gather(x, landmark_indices)
        selection_scores = F.softmax(scores, dim=-1)

        return landmark_indices, landmark_states, selection_scores


class HybridLandmarkSelector(BaseLandmarkSelector):
    """Hybrid selector combining content-based and positional strategies.

    Uses gating to dynamically balance between learned content-based
    selection and positional pattern-based selection.

    Args:
        embed_dim: Input embedding dimension
        max_seq_len: Maximum sequence length
        num_landmarks: Number of landmarks to select (G)

    Example:
        >>> selector = HybridLandmarkSelector(
        ...     embed_dim=512, max_seq_len=2048, num_landmarks=64
        ... )
    """

    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int,
        num_landmarks: int,
    ):
        super().__init__(num_landmarks)

        self.content_selector = LearnedLandmarkSelector(embed_dim, num_landmarks)
        self.position_selector = PositionalLandmarkSelector(max_seq_len, num_landmarks, embed_dim)

        # Gate to combine strategies
        self.gate = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select landmarks using hybrid strategy.

        Args:
            x: (B, L, D) input sequence
            **kwargs: Unused

        Returns:
            landmark_indices: (B, G) indices
            landmark_states: (B, G, D) states
            selection_scores: (B, L) combined scores
        """
        B, L, D = x.shape

        # Get selections from both strategies
        _, _, scores_content = self.content_selector(x)
        _, _, scores_position = self.position_selector(x)

        # Gate based on global sequence representation
        x_pooled = x.mean(dim=1)  # (B, D)
        gate_weight = torch.sigmoid(self.gate(x_pooled))  # (B, 1)

        # Combine scores
        scores_combined = gate_weight * scores_content + (1 - gate_weight) * scores_position

        # Re-select top-K from combined scores
        k = self._validate_sequence_length(L)
        _, landmark_indices = torch.topk(scores_combined, k=k, dim=-1)

        # Gather states
        landmark_states = self._safe_gather(x, landmark_indices)

        return landmark_indices, landmark_states, scores_combined
