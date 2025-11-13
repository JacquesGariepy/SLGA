"""
Penalty mechanisms for text generation.
Implements repetition penalty and n-gram blocking to improve generation quality.
"""

import torch
from typing import List, Set, Tuple


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty to discourage repeating tokens.

    Divides logits of previously generated tokens by the penalty factor.

    Args:
        logits: Current logits [batch_size, vocab_size]
        generated_tokens: Previously generated token IDs [batch_size, seq_len]
        penalty: Penalty factor (1.0 = no penalty, >1.0 = penalize repetition)

    Returns:
        Modified logits with repetition penalty applied
    """
    if penalty <= 1.0:
        return logits

    if penalty < 1.0:
        raise ValueError(f"repetition_penalty must be >= 1.0, got {penalty}")

    # Get unique tokens that have been generated
    batch_size = logits.size(0)

    for i in range(batch_size):
        # Get unique tokens for this batch element
        unique_tokens = generated_tokens[i].unique()

        # Apply penalty by dividing logits of repeated tokens
        logits[i, unique_tokens] /= penalty

    return logits


def get_ngrams(token_list: List[int], n: int) -> Set[Tuple[int, ...]]:
    """Extract all n-grams from a token sequence.

    Args:
        token_list: List of token IDs
        n: N-gram size

    Returns:
        Set of n-grams (tuples of n token IDs)
    """
    if n <= 0:
        return set()

    ngrams = set()
    for i in range(len(token_list) - n + 1):
        ngram = tuple(token_list[i:i + n])
        ngrams.add(ngram)

    return ngrams


def get_banned_tokens_for_ngram(
    generated_tokens: List[int],
    ngram_size: int,
) -> Set[int]:
    """Get tokens that would complete a repeated n-gram.

    Args:
        generated_tokens: Previously generated token IDs
        ngram_size: Size of n-grams to check

    Returns:
        Set of token IDs that should be banned
    """
    if ngram_size <= 0 or len(generated_tokens) < ngram_size - 1:
        return set()

    banned = set()

    # Get all existing n-grams
    existing_ngrams = get_ngrams(generated_tokens, ngram_size)

    # Check what tokens would complete a repeated n-gram
    # Look at the last (n-1) tokens
    context = tuple(generated_tokens[-(ngram_size - 1):])

    # Ban any token that would complete an n-gram we've already seen
    for ngram in existing_ngrams:
        if ngram[:-1] == context:
            banned.add(ngram[-1])

    return banned


def apply_no_repeat_ngram(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    ngram_size: int,
) -> torch.Tensor:
    """Apply n-gram blocking to prevent repeating n-grams.

    Sets logits of tokens that would complete a repeated n-gram to -inf.

    Args:
        logits: Current logits [batch_size, vocab_size]
        generated_tokens: Previously generated token IDs [batch_size, seq_len]
        ngram_size: Size of n-grams to block

    Returns:
        Modified logits with banned tokens set to -inf
    """
    if ngram_size <= 0:
        return logits

    if ngram_size < 0:
        raise ValueError(f"ngram_size must be >= 0, got {ngram_size}")

    batch_size = logits.size(0)

    for i in range(batch_size):
        # Convert to list for easier manipulation
        gen_tokens = generated_tokens[i].tolist()

        # Get tokens that would create repeated n-grams
        banned_tokens = get_banned_tokens_for_ngram(gen_tokens, ngram_size)

        # Set banned tokens to -inf
        if banned_tokens:
            banned_indices = torch.tensor(list(banned_tokens), device=logits.device)
            logits[i, banned_indices] = float('-inf')

    return logits


def get_penalty_info(
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    num_generated_tokens: int,
) -> dict:
    """Get human-readable penalty configuration info.

    Args:
        repetition_penalty: Repetition penalty value
        no_repeat_ngram_size: N-gram blocking size
        num_generated_tokens: Number of tokens generated so far

    Returns:
        Dictionary with penalty configuration details
    """
    info = {
        "repetition_penalty": {
            "value": repetition_penalty,
            "enabled": repetition_penalty > 1.0,
            "description": (
                "disabled" if repetition_penalty <= 1.0
                else f"{repetition_penalty:.2f}x penalty on repeated tokens"
            ),
        },
        "no_repeat_ngram": {
            "size": no_repeat_ngram_size,
            "enabled": no_repeat_ngram_size > 0,
            "description": (
                "disabled" if no_repeat_ngram_size <= 0
                else f"blocking repeated {no_repeat_ngram_size}-grams"
            ),
            "active": (
                no_repeat_ngram_size > 0 and num_generated_tokens >= no_repeat_ngram_size - 1
            ),
        },
    }

    return info
