"""
SLGA Utilities Package.

This package provides utility functions for training, checkpointing,
metrics tracking, and model analysis.

Modules:
    seed: Random seed utilities for reproducibility
    checkpoint: Model checkpoint save/load utilities
    metrics: Training metrics and memory monitoring
    time: Time formatting and estimation
    model_summary: Model analysis and summary functions
"""

# Random seed utilities
from .seed import set_seed

# Checkpoint utilities
from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_latest_checkpoint,
)

# Metrics utilities
from .metrics import (
    AverageMeter,
    count_parameters,
    get_memory_usage,
)

# Time utilities
from .time import (
    format_time,
    estimate_training_time,
)

# Model summary utilities
from .model_summary import print_model_summary


__all__ = [
    # Seed
    "set_seed",

    # Checkpoint
    "save_checkpoint",
    "load_checkpoint",
    "load_latest_checkpoint",

    # Metrics
    "AverageMeter",
    "count_parameters",
    "get_memory_usage",

    # Time
    "format_time",
    "estimate_training_time",

    # Model Summary
    "print_model_summary",
]
