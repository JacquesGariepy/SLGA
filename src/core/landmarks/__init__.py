"""
Landmark selection module for SLGA architecture.

Provides modular, extensible landmark selection strategies following
Factory and Strategy design patterns.

Usage:
    >>> from src.core.landmarks import LandmarkSelectorFactory
    >>>
    >>> # Create a Gumbel-Softmax selector
    >>> selector = LandmarkSelectorFactory.create(
    ...     "gumbel",
    ...     embed_dim=512,
    ...     num_landmarks=64
    ... )
    >>>
    >>> # Use in model
    >>> x = torch.randn(4, 256, 512)  # (B, L, D)
    >>> indices, states, scores = selector(x)

Available Selectors:
    - gumbel: Differentiable selection via Gumbel-Softmax (recommended)
    - learned: Straight-through estimator (faster, less stable)
    - positional: Learned positional patterns
    - hybrid: Combination of content and positional
    - uniform: Evenly spaced landmarks (no learning)
    - random: Random selection (no learning)
    - first: First-K positions (no learning)
"""

from src.core.landmarks.base import BaseLandmarkSelector
from src.core.landmarks.curriculum import CurriculumLandmarkSelector, CurriculumSchedule
from src.core.landmarks.factory import LandmarkSelectorFactory
from src.core.landmarks.gumbel import GumbelSoftmaxSelector
from src.core.landmarks.heuristic import HeuristicLandmarkSelector
from src.core.landmarks.learned import (
    HybridLandmarkSelector,
    LearnedLandmarkSelector,
    PositionalLandmarkSelector,
)
from src.legacy.landmarks import (
    landmark_diversity_loss,
    landmark_spacing_loss,
    landmark_sparsity_loss,
)

__all__ = [
    # Base class
    "BaseLandmarkSelector",
    # Selector implementations
    "GumbelSoftmaxSelector",
    "LearnedLandmarkSelector",
    "PositionalLandmarkSelector",
    "HybridLandmarkSelector",
    "HeuristicLandmarkSelector",
    # Curriculum learning
    "CurriculumSchedule",
    "CurriculumLandmarkSelector",
    # Factory
    "LandmarkSelectorFactory",
    # Loss functions
    "landmark_spacing_loss",
    "landmark_diversity_loss",
    "landmark_sparsity_loss",
    # Convenience function
    "create_landmark_selector",
]


def create_landmark_selector(selector_type: str, **kwargs) -> BaseLandmarkSelector:
    """Convenience function to create a landmark selector.

    Equivalent to LandmarkSelectorFactory.create().

    Args:
        selector_type: Type of selector ("gumbel", "learned", etc.)
        **kwargs: Arguments for selector constructor

    Returns:
        selector: Landmark selector instance

    Example:
        >>> selector = create_landmark_selector("gumbel", embed_dim=512, num_landmarks=64)
    """
    return LandmarkSelectorFactory.create(selector_type, **kwargs)
