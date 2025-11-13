"""
Landmark Selection Protocol Interfaces

This module defines Protocol interfaces for landmark (global token) selection mechanisms.
Landmarks are important tokens selected for global attention in SLGA architecture.
"""

from typing import Protocol, Tuple

from torch import Tensor


class LandmarkSelector(Protocol):
    """Protocol for landmark selection mechanisms.

    Landmark selectors identify important positions in a sequence to serve as
    global attention tokens. Selection can be learned (content-based) or heuristic
    (positional patterns).
    """

    num_landmarks: int

    def forward(
        self,
        x: Tensor,  # [batch, seq_len, embed_dim]
        use_gumbel: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Select landmarks from input sequence.

        Args:
            x: Input sequence embeddings [batch, seq_len, embed_dim]
            use_gumbel: If True, uses Gumbel-Softmax for differentiable selection.
                       If False, uses straight-through estimator.
                       Recommended: use_gumbel=True for training from scratch.

        Returns:
            landmark_indices: Selected landmark positions [batch, num_landmarks]
            landmark_states: Corresponding embeddings [batch, num_landmarks, embed_dim]
            selection_scores: Normalized selection probabilities [batch, seq_len]

        Example:
            >>> selector = LearnableLandmarkSelector(embed_dim=512, num_landmarks=48)
            >>> indices, states, scores = selector(x, use_gumbel=True)
            >>> assert indices.shape == (batch, 48)
            >>> assert states.shape == (batch, 48, 512)
            >>> assert scores.shape == (batch, seq_len)
            >>> assert torch.allclose(scores.sum(dim=-1), torch.ones(batch))

        Notes:
            - landmark_indices: Integer positions in [0, seq_len-1]
            - landmark_states: Gathered embeddings at selected positions
            - selection_scores: Softmax probabilities, used for auxiliary losses
            - Gumbel mode: Smoother gradients, better for training
            - Straight-through: Faster, good for fine-tuning
        """
        ...

    def select_landmarks(
        self,
        x: Tensor,  # [batch, seq_len, embed_dim]
    ) -> Tuple[Tensor, Tensor]:
        """Simplified interface for inference (deterministic selection).

        Args:
            x: Input sequence embeddings [batch, seq_len, embed_dim]

        Returns:
            landmark_indices: Selected positions [batch, num_landmarks]
            landmark_states: Gathered embeddings [batch, num_landmarks, embed_dim]

        Example:
            >>> selector.eval()
            >>> indices, states = selector.select_landmarks(x)

        Notes:
            - Always uses hard top-K (no stochastic relaxation)
            - No gradient computation
            - Equivalent to forward(x, use_gumbel=False) but returns only indices and states
        """
        ...


class LearnedLandmarkSelector(Protocol):
    """Protocol for learned (content-based) landmark selection.

    Uses a neural network to score each position based on its embedding,
    then selects top-K via differentiable mechanisms (Gumbel-Softmax or
    straight-through estimator).
    """

    embed_dim: int
    num_landmarks: int
    temperature: float
    temperature_decay: float
    min_temperature: float

    def forward(
        self,
        x: Tensor,  # [batch, seq_len, embed_dim]
        use_gumbel: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Select landmarks based on learned importance scores.

        Args:
            x: Input embeddings [batch, seq_len, embed_dim]
            use_gumbel: Use Gumbel-Softmax (True) or straight-through (False)

        Returns:
            landmark_indices: [batch, num_landmarks]
            landmark_states: [batch, num_landmarks, embed_dim]
            selection_scores: [batch, seq_len] softmax probabilities

        Example:
            >>> selector = LearnableLandmarkSelector(
            ...     embed_dim=512, num_landmarks=48,
            ...     temperature=1.0, temperature_decay=0.999, min_temperature=0.3
            ... )
            >>> # Training mode: smooth gradients
            >>> selector.train()
            >>> indices, states, scores = selector(x, use_gumbel=True)
            >>> # Inference mode: deterministic
            >>> selector.eval()
            >>> indices, states, scores = selector(x, use_gumbel=False)

        Algorithm:
            1. Score each position: scores = scorer_network(x)  # [batch, seq_len]
            2. Select top-K via:
               - Gumbel mode: Perturb scores with Gumbel noise, soft top-K
               - Straight-through: Hard top-K with sigmoid-based gradients
            3. Gather embeddings at selected positions
            4. Return indices, states, and normalized scores

        Notes:
            - Temperature anneals during training for sharper selections
            - Gumbel-Softmax provides theoretically grounded gradients
            - Straight-through is faster but less stable early in training
            - NaN protection for edge cases (all scores equal, numerical issues)
        """
        ...

    def _get_temperature(self) -> float:
        """Get current Gumbel temperature with decay.

        Returns:
            Current temperature (decayed from initial value, clamped to minimum)

        Notes:
            - Temperature decays as: temp = initial * (decay ^ step_count)
            - Clamped to min_temperature to prevent over-sharpening
            - Only applies in training mode
        """
        ...


class PositionalLandmarkSelector(Protocol):
    """Protocol for positional (pattern-based) landmark selection.

    Learns to select landmarks based on positional patterns rather than content.
    Useful for capturing structural patterns (e.g., paragraph boundaries, every N tokens).
    """

    max_seq_len: int
    num_landmarks: int
    embed_dim: int

    def forward(
        self,
        x: Tensor,  # [batch, seq_len, embed_dim]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Select landmarks based on learned positional patterns.

        Args:
            x: Input embeddings [batch, seq_len, embed_dim]
               Note: Only shape is used, content is ignored.

        Returns:
            landmark_indices: [batch, num_landmarks]
            landmark_states: [batch, num_landmarks, embed_dim]
            selection_scores: [batch, seq_len]

        Example:
            >>> selector = PositionalLandmarkSelector(
            ...     max_seq_len=2048, num_landmarks=48, embed_dim=512
            ... )
            >>> indices, states, scores = selector(x)

        Notes:
            - Uses learned positional embeddings to determine importance
            - Position patterns are shared across all sequences (batch-independent)
            - Useful for structured documents (paragraphs, sections, etc.)
            - Can combine with content-based selection via HybridLandmarkSelector
        """
        ...


class HybridLandmarkSelector(Protocol):
    """Protocol for hybrid landmark selection combining content and position.

    Dynamically blends learned (content-based) and positional selection using
    a gating mechanism. Best of both worlds for complex patterns.
    """

    num_landmarks: int

    def forward(
        self,
        x: Tensor,  # [batch, seq_len, embed_dim]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Select landmarks via gated combination of content and position.

        Args:
            x: Input embeddings [batch, seq_len, embed_dim]

        Returns:
            landmark_indices: [batch, num_landmarks]
            landmark_states: [batch, num_landmarks, embed_dim]
            selection_scores: [batch, seq_len] combined scores

        Example:
            >>> selector = HybridLandmarkSelector(
            ...     embed_dim=512, max_seq_len=2048, num_landmarks=48
            ... )
            >>> indices, states, scores = selector(x)

        Algorithm:
            1. Get content-based selection: idx_c, states_c, scores_c = content_selector(x)
            2. Get positional selection: idx_p, states_p, scores_p = position_selector(x)
            3. Compute gate weight: gate = sigmoid(gate_network(x.mean(dim=1)))
            4. Combine scores: scores = gate * scores_c + (1 - gate) * scores_p
            5. Re-select top-K from combined scores

        Notes:
            - Gate weight is sequence-dependent (adaptive blending)
            - Inherits benefits of both content and position selection
            - More parameters but better expressivity
        """
        ...


class LearningSchedule(Protocol):
    """Protocol for curriculum learning schedules.

    Defines schedules for progressively increasing task difficulty during training.
    Used for landmark selection (start with few landmarks, gradually increase) and
    other progressive training strategies.
    """

    initial_value: float
    final_value: float
    total_steps: int

    def get_value(self, current_step: int) -> float:
        """Get curriculum value at current training step.

        Args:
            current_step: Current training iteration

        Returns:
            Value for current step (e.g., number of landmarks, sequence length)

        Example:
            >>> schedule = LinearSchedule(
            ...     initial_value=8, final_value=48, total_steps=10000
            ... )
            >>> num_landmarks = schedule.get_value(current_step=5000)
            >>> assert 8 <= num_landmarks <= 48

        Notes:
            - Used for progressive complexity increase
            - Common patterns: linear, exponential, cosine, step-wise
            - Helps with training stability and convergence
        """
        ...


class LinearSchedule(Protocol):
    """Linear curriculum schedule.

    Linearly interpolates from initial_value to final_value over total_steps.
    """

    initial_value: float
    final_value: float
    total_steps: int

    def get_value(self, current_step: int) -> float:
        """Get linearly interpolated value.

        Args:
            current_step: Current step

        Returns:
            value = initial + (final - initial) * min(current / total, 1.0)

        Example:
            >>> schedule = LinearSchedule(initial_value=8, final_value=48, total_steps=1000)
            >>> schedule.get_value(0)  # 8.0
            >>> schedule.get_value(500)  # 28.0
            >>> schedule.get_value(1000)  # 48.0
            >>> schedule.get_value(2000)  # 48.0 (clamped)
        """
        ...


class ExponentialSchedule(Protocol):
    """Exponential curriculum schedule.

    Exponentially interpolates from initial_value to final_value.
    Slow initial growth, rapid later growth.
    """

    initial_value: float
    final_value: float
    total_steps: int

    def get_value(self, current_step: int) -> float:
        """Get exponentially interpolated value.

        Args:
            current_step: Current step

        Returns:
            Exponential interpolation between initial and final values

        Example:
            >>> schedule = ExponentialSchedule(initial_value=8, final_value=48, total_steps=1000)
            >>> # Slow growth early, rapid growth later
            >>> schedule.get_value(500)  # ~15 (slower than linear)
            >>> schedule.get_value(900)  # ~40 (catching up)
        """
        ...


__all__ = [
    "ExponentialSchedule",
    "HybridLandmarkSelector",
    "LandmarkSelector",
    "LearnedLandmarkSelector",
    "LearningSchedule",
    "LinearSchedule",
    "PositionalLandmarkSelector",
]
