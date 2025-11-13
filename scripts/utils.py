"""
Backward compatibility wrapper for scripts.utils module.

This module is DEPRECATED and maintained only for backward compatibility.
All functionality has been moved to src.utils for better organization.

Please update your imports:
    OLD: from scripts.utils import set_seed, save_checkpoint
    NEW: from src.utils import set_seed, save_checkpoint

This wrapper will be removed in a future version.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "scripts.utils is deprecated and will be removed in a future version. "
    "Please use 'from src.utils import ...' instead of 'from scripts.utils import ...'",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from src.utils for backward compatibility
from src.utils import *  # noqa: F401, F403

# Preserve the original __all__ for explicit exports
__all__ = [
    "set_seed",
    "save_checkpoint",
    "load_checkpoint",
    "load_latest_checkpoint",
    "count_parameters",
    "get_memory_usage",
    "format_time",
    "estimate_training_time",
    "AverageMeter",
    "print_model_summary",
]
