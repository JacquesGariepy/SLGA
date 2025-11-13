"""
Sampling strategies for text generation.
Implements temperature scaling, top-k, top-p (nucleus sampling), and token sampling.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling to logits.

    Args:
        logits: Raw logits from model [batch_size, vocab_size]
        temperature: Temperature value (0.0 = greedy, >0 = sampling)

    Returns:
        Scaled logits
    """
    if temperature == 0.0:
        # Greedy decoding - return logits as-is (will use argmax)
        return logits

    if temperature <= 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")

    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Apply top-k filtering to logits.

    Keep only the k tokens with highest probability, set others to -inf.

    Args:
        logits: Input logits [batch_size, vocab_size]
        k: Number of top tokens to keep

    Returns:
        Filtered logits
    """
    if k <= 0:
        raise ValueError(f"top_k must be positive, got {k}")

    # Get top-k values and indices
    top_k_values, top_k_indices = torch.topk(logits, min(k, logits.size(-1)))

    # Create mask for top-k tokens
    mask = torch.full_like(logits, float('-inf'))
    mask.scatter_(-1, top_k_indices, top_k_values)

    return mask


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Apply top-p (nucleus) sampling to logits.

    Keep tokens with cumulative probability >= p, set others to -inf.

    Args:
        logits: Input logits [batch_size, vocab_size]
        p: Cumulative probability threshold (0 < p <= 1)

    Returns:
        Filtered logits
    """
    if not (0 < p <= 1):
        raise ValueError(f"top_p must be in (0, 1], got {p}")

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    # Keep at least one token
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 0] = False  # Always keep the best token

    # Scatter back to original indices
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )

    # Set removed tokens to -inf
    logits = logits.masked_fill(indices_to_remove, float('-inf'))

    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """Sample next token from logits with temperature, top-k, and top-p filtering.

    Applies filters in order: temperature -> top_k -> top_p -> sample

    Args:
        logits: Raw logits from model [batch_size, vocab_size]
        temperature: Temperature for sampling (0.0 = greedy)
        top_k: Top-k filtering (None = disabled)
        top_p: Nucleus sampling threshold (None = disabled)

    Returns:
        Sampled token indices [batch_size]
    """
    # Apply temperature scaling
    if temperature == 0.0:
        # Greedy decoding
        return torch.argmax(logits, dim=-1)

    logits = apply_temperature(logits, temperature)

    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        logits = apply_top_k(logits, top_k)

    # Apply top-p (nucleus) filtering
    if top_p is not None and top_p < 1.0:
        logits = apply_top_p(logits, top_p)

    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return next_token


def get_sampling_info(
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> dict:
    """Get human-readable sampling configuration info.

    Args:
        temperature: Temperature value
        top_k: Top-k value
        top_p: Top-p value

    Returns:
        Dictionary with sampling configuration details
    """
    info = {
        "strategy": "greedy" if temperature == 0.0 else "sampling",
        "temperature": temperature,
        "top_k": "disabled" if top_k is None or top_k <= 0 else top_k,
        "top_p": "disabled" if top_p is None or top_p >= 1.0 else top_p,
    }

    # Determine sampling mode
    if temperature == 0.0:
        info["mode"] = "greedy (deterministic)"
    elif top_k and top_p:
        info["mode"] = f"top-k ({top_k}) + top-p ({top_p})"
    elif top_k:
        info["mode"] = f"top-k sampling (k={top_k})"
    elif top_p:
        info["mode"] = f"nucleus sampling (p={top_p})"
    else:
        info["mode"] = "pure temperature sampling"

    return info
