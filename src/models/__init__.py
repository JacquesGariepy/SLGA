"""SLGA Transformer models and utilities."""

from src.models.config import (
    PRESET_CONFIGS,
    ModelConfig,
    get_config,
)
from src.models.factory import (
    ModelFactory,
)
from src.models.generation import (
    GenerationState,
    TextGenerator,
)
from src.models.slga_model import (
    SLGATransformer,
)

# Backward compatibility alias
Config = ModelConfig

__all__ = [
    "Config",  # Backward compatibility alias for ModelConfig
    "PRESET_CONFIGS",
    "GenerationState",
    "ModelConfig",
    "ModelFactory",
    "SLGATransformer",
    "TextGenerator",
    "get_config",
]
