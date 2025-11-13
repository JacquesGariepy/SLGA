"""
Domain Services

This module contains protocol interfaces for domain services in the SLGA architecture.
Services implement complex business logic that doesn't naturally fit within entities.
"""

from .attention_protocol import (
    AttentionFusion,
    AttentionMechanism,
    GlobalAttention,
    LocalAttention,
    SparseLocalGlobalAttention,
)
from .landmark_protocol import (
    ExponentialSchedule,
    HybridLandmarkSelector,
    LandmarkSelector,
    LearnedLandmarkSelector,
    LearningSchedule,
    LinearSchedule,
    PositionalLandmarkSelector,
)

__all__ = [
    # Attention protocols
    "AttentionMechanism",
    "LocalAttention",
    "GlobalAttention",
    "AttentionFusion",
    "SparseLocalGlobalAttention",
    # Landmark selection protocols
    "LandmarkSelector",
    "LearnedLandmarkSelector",
    "PositionalLandmarkSelector",
    "HybridLandmarkSelector",
    # Learning schedules
    "LearningSchedule",
    "LinearSchedule",
    "ExponentialSchedule",
]
