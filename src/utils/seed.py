"""
Random seed utilities for reproducibility.

This module provides functions to set random seeds across multiple libraries
(random, numpy, torch) to ensure reproducible results.
"""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.

    This function sets the seed for:
    - Python's random module
    - NumPy's random number generator
    - PyTorch's CPU random number generator
    - PyTorch's CUDA random number generators (all GPUs)

    Args:
        seed: The seed value to use for all random number generators.
              Should be an integer value for consistent results.

    Example:
        >>> set_seed(42)
        >>> # All random operations will now be reproducible
        >>> torch.rand(3)
        tensor([0.8823, 0.9150, 0.3829])

    Note:
        For complete determinism, you may also need to set:
        - torch.backends.cudnn.deterministic = True
        - torch.backends.cudnn.benchmark = False
        However, this can significantly impact performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Optional: Uncomment for fully deterministic behavior (may slow down training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
