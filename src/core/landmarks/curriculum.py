"""
Curriculum learning for landmark selection.

Implements progressive landmark selection during training with
temperature annealing and adaptive scheduling.
"""


import torch
from torch import nn

from .base import BaseLandmarkSelector


class CurriculumSchedule:
    """Curriculum schedule for progressive landmark selection.

    Gradually increases the number of landmarks during training,
    starting from fewer landmarks and building up to the target.

    Helps model learn to focus on most important positions first
    before handling more complex landmark patterns.

    Args:
        initial_landmarks: Starting number of landmarks
        target_landmarks: Final number of landmarks
        warmup_steps: Steps to reach target (default: 10000)
        schedule: Schedule type - "linear" or "cosine" (default: "linear")

    Example:
        >>> schedule = CurriculumSchedule(
        ...     initial_landmarks=16,
        ...     target_landmarks=64,
        ...     warmup_steps=10000
        ... )
        >>> for step in range(15000):
        ...     num_landmarks = schedule.get_num_landmarks(step)
        ...     # Train with num_landmarks
    """

    def __init__(
        self,
        initial_landmarks: int,
        target_landmarks: int,
        warmup_steps: int = 10000,
        schedule: str = "linear",
    ):
        self.initial_landmarks = initial_landmarks
        self.target_landmarks = target_landmarks
        self.warmup_steps = warmup_steps
        self.schedule = schedule

        if initial_landmarks > target_landmarks:
            raise ValueError("initial_landmarks must be <= target_landmarks")

    def get_num_landmarks(self, step: int) -> int:
        """Get number of landmarks for current training step.

        Args:
            step: Current training step

        Returns:
            num_landmarks: Number of landmarks to use
        """
        if step >= self.warmup_steps:
            return self.target_landmarks

        # Calculate progress ratio
        progress = step / self.warmup_steps

        if self.schedule == "linear":
            # Linear interpolation
            ratio = progress
        elif self.schedule == "cosine":
            # Cosine annealing (slower start, faster end)
            import math

            ratio = (1 - math.cos(progress * math.pi)) / 2
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        # Interpolate number of landmarks
        num = self.initial_landmarks + ratio * (self.target_landmarks - self.initial_landmarks)
        return max(self.initial_landmarks, int(num))

    def get_temperature(self, step: int, initial_temp: float = 1.0, min_temp: float = 0.3) -> float:
        """Get temperature for current step (optional temperature scheduling).

        Can be used to coordinate temperature annealing with curriculum.

        Args:
            step: Current training step
            initial_temp: Starting temperature
            min_temp: Minimum temperature

        Returns:
            temperature: Current temperature value
        """
        if step >= self.warmup_steps:
            return min_temp

        progress = step / self.warmup_steps
        temp = initial_temp * (1 - progress) + min_temp * progress
        return max(temp, min_temp)


class CurriculumLandmarkSelector(nn.Module):
    """Wrapper for landmark selectors with curriculum learning.

    Wraps any BaseLandmarkSelector and applies curriculum scheduling
    to progressively increase number of landmarks during training.

    Args:
        selector: Base landmark selector to wrap
        curriculum: CurriculumSchedule instance

    Example:
        >>> base_selector = GumbelSoftmaxSelector(embed_dim=512, num_landmarks=64)
        >>> schedule = CurriculumSchedule(
        ...     initial_landmarks=16,
        ...     target_landmarks=64,
        ...     warmup_steps=10000
        ... )
        >>> selector = CurriculumLandmarkSelector(base_selector, schedule)
        >>>
        >>> # Training loop
        >>> for step, batch in enumerate(dataloader):
        ...     selector.step_count = step  # Update step
        ...     indices, states, scores = selector(batch)
    """

    def __init__(
        self,
        selector: BaseLandmarkSelector,
        curriculum: CurriculumSchedule,
    ):
        super().__init__()

        self.selector = selector
        self.curriculum = curriculum
        self.step_count = 0  # Track training steps

    def forward(self, x: torch.Tensor, **kwargs):
        """Forward pass with curriculum-adjusted landmark count.

        Args:
            x: (B, L, D) input sequence
            **kwargs: Additional arguments for selector

        Returns:
            landmark_indices: (B, G_curr) indices (G_curr may vary by step)
            landmark_states: (B, G_curr, D) states
            selection_scores: (B, L) scores
        """
        # Get current number of landmarks from curriculum
        if self.training:
            curr_num_landmarks = self.curriculum.get_num_landmarks(self.step_count)

            # Temporarily override selector's num_landmarks
            original_num = self.selector.num_landmarks
            self.selector.num_landmarks = curr_num_landmarks

            # Forward pass
            result = self.selector(x, **kwargs)

            # Restore original
            self.selector.num_landmarks = original_num
        else:
            # Inference: use target landmarks
            result = self.selector(x, **kwargs)

        return result
