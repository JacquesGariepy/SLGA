"""
Stopping criteria for text generation.
Implements various conditions to halt generation.
"""

import torch
from typing import Optional, List
from abc import ABC, abstractmethod


class StoppingCriteria(ABC):
    """Abstract base class for stopping criteria."""

    @abstractmethod
    def should_stop(self, tokens: torch.Tensor, **kwargs) -> bool:
        """Check if generation should stop.

        Args:
            tokens: Generated token IDs [batch_size, seq_len]
            **kwargs: Additional context for stopping decision

        Returns:
            True if generation should stop, False otherwise
        """
        pass

    @abstractmethod
    def get_reason(self) -> str:
        """Get human-readable reason for stopping."""
        pass


class MaxLengthStoppingCriteria(StoppingCriteria):
    """Stop when maximum number of tokens is generated."""

    def __init__(self, max_length: int, initial_length: int = 0):
        """Initialize max length criterion.

        Args:
            max_length: Maximum number of NEW tokens to generate
            initial_length: Length of initial prompt (tokens already generated)
        """
        self.max_length = max_length
        self.initial_length = initial_length
        self._stopped_at = None

    def should_stop(self, tokens: torch.Tensor, **kwargs) -> bool:
        """Check if we've reached max length."""
        current_length = tokens.size(1)
        new_tokens = current_length - self.initial_length

        if new_tokens >= self.max_length:
            self._stopped_at = current_length
            return True

        return False

    def get_reason(self) -> str:
        """Get stopping reason."""
        return (
            f"max_length reached ({self._stopped_at - self.initial_length}/{self.max_length} tokens)"
        )


class EOSStoppingCriteria(StoppingCriteria):
    """Stop when EOS (End-of-Sequence) token is generated."""

    def __init__(self, eos_token_id: int, enabled: bool = True):
        """Initialize EOS criterion.

        Args:
            eos_token_id: Token ID that represents end-of-sequence
            enabled: Whether to actually stop on EOS (if False, just tracks it)
        """
        self.eos_token_id = eos_token_id
        self.enabled = enabled
        self._eos_position = None

    def should_stop(self, tokens: torch.Tensor, **kwargs) -> bool:
        """Check if EOS token was just generated."""
        if not self.enabled or self.eos_token_id is None:
            return False

        # Check if last token is EOS for any batch element
        last_tokens = tokens[:, -1]
        eos_mask = last_tokens == self.eos_token_id

        if eos_mask.any():
            self._eos_position = tokens.size(1)
            return True

        return False

    def get_reason(self) -> str:
        """Get stopping reason."""
        if not self.enabled:
            return "EOS stopping disabled"
        return f"EOS token generated at position {self._eos_position}"


class MultiStoppingCriteria(StoppingCriteria):
    """Combine multiple stopping criteria with OR logic."""

    def __init__(self, criteria: List[StoppingCriteria]):
        """Initialize multi-criteria.

        Args:
            criteria: List of stopping criteria to combine
        """
        self.criteria = criteria
        self._triggered_criterion = None

    def should_stop(self, tokens: torch.Tensor, **kwargs) -> bool:
        """Check if ANY criterion is met."""
        for criterion in self.criteria:
            if criterion.should_stop(tokens, **kwargs):
                self._triggered_criterion = criterion
                return True

        return False

    def get_reason(self) -> str:
        """Get reason from the triggered criterion."""
        if self._triggered_criterion is None:
            return "not stopped"
        return self._triggered_criterion.get_reason()


class CustomTokenStoppingCriteria(StoppingCriteria):
    """Stop when any of specified tokens is generated."""

    def __init__(self, stop_tokens: List[int]):
        """Initialize custom token criterion.

        Args:
            stop_tokens: List of token IDs that trigger stopping
        """
        self.stop_tokens = set(stop_tokens)
        self._stopped_token = None

    def should_stop(self, tokens: torch.Tensor, **kwargs) -> bool:
        """Check if any stop token was just generated."""
        if not self.stop_tokens:
            return False

        last_tokens = tokens[:, -1]

        for stop_token in self.stop_tokens:
            if (last_tokens == stop_token).any():
                self._stopped_token = stop_token
                return True

        return False

    def get_reason(self) -> str:
        """Get stopping reason."""
        return f"stop token {self._stopped_token} generated"


def create_default_stopping_criteria(
    max_new_tokens: int,
    initial_length: int,
    eos_token_id: Optional[int],
    stop_on_eos: bool,
) -> MultiStoppingCriteria:
    """Create standard stopping criteria for generation.

    Args:
        max_new_tokens: Maximum number of new tokens to generate
        initial_length: Length of initial prompt
        eos_token_id: EOS token ID (None = no EOS stopping)
        stop_on_eos: Whether to stop on EOS

    Returns:
        Combined stopping criteria
    """
    criteria = [
        MaxLengthStoppingCriteria(max_new_tokens, initial_length)
    ]

    if eos_token_id is not None:
        criteria.append(
            EOSStoppingCriteria(eos_token_id, enabled=stop_on_eos)
        )

    return MultiStoppingCriteria(criteria)
