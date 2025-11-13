"""
Domain Value Objects

This module contains immutable configuration dataclasses for the SLGA architecture.
Value objects ensure type safety and validation of configuration parameters.
"""

from .config import (
    AttentionConfig,
    DataConfig,
    LandmarkConfig,
    ModelConfig,
    TrainingConfig,
)

__all__ = [
    "AttentionConfig",
    "DataConfig",
    "LandmarkConfig",
    "ModelConfig",
    "TrainingConfig",
]
