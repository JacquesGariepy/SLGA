"""
Generation configuration dataclass.
Defines all parameters for text generation with SLGA models.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 = deterministic, higher = more random)
        top_k: Keep only top K tokens with highest probability (None = disabled)
        top_p: Nucleus sampling - keep tokens with cumulative probability >= p (None = disabled)
        repetition_penalty: Penalty for repeating tokens (1.0 = disabled, >1.0 = penalize)
        no_repeat_ngram_size: Ban repeated n-grams of this size (None/0 = disabled)
        eos_token_id: End-of-sequence token ID (None = use tokenizer default)
        stop_on_eos: Whether to stop generation when EOS is encountered
    """

    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: Optional[int] = None
    eos_token_id: Optional[int] = None
    stop_on_eos: bool = True

    def validate(self) -> None:
        """Validate generation parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")

        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")

        if self.top_k is not None and self.top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {self.top_k}")

        if self.top_p is not None and not (0 < self.top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")

        if self.repetition_penalty < 1.0:
            raise ValueError(
                f"repetition_penalty must be >= 1.0 (1.0 disables it), got {self.repetition_penalty}"
            )

        if self.no_repeat_ngram_size is not None and self.no_repeat_ngram_size < 0:
            raise ValueError(
                f"no_repeat_ngram_size must be >= 0 (0 disables it), got {self.no_repeat_ngram_size}"
            )

    def is_greedy(self) -> bool:
        """Check if this config uses greedy decoding (temperature=0)."""
        return self.temperature == 0.0

    def has_top_k(self) -> bool:
        """Check if top-k filtering is enabled."""
        return self.top_k is not None and self.top_k > 0

    def has_top_p(self) -> bool:
        """Check if top-p (nucleus) sampling is enabled."""
        return self.top_p is not None and self.top_p < 1.0

    def has_repetition_penalty(self) -> bool:
        """Check if repetition penalty is enabled."""
        return self.repetition_penalty > 1.0

    def has_no_repeat_ngram(self) -> bool:
        """Check if n-gram blocking is enabled."""
        return self.no_repeat_ngram_size is not None and self.no_repeat_ngram_size > 0
