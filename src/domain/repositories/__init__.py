"""
Domain Repositories

This module contains protocol interfaces for data access in the SLGA architecture.
Repositories abstract data sources and persistence mechanisms.
"""

from .data_repository import (
    ICheckpointRepository,
    ICollatorRepository,
    IDatasetRepository,
    ITokenizerRepository,
)

__all__ = [
    "ICheckpointRepository",
    "ICollatorRepository",
    "IDatasetRepository",
    "ITokenizerRepository",
]
