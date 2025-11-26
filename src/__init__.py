"""
SLGA-Plus: Sparse Local-Global Attention Transformer

Main package for efficient long-context language modeling.
Now with Reasoning Model support!
"""

__version__ = "0.2.0"
__author__ = "SLGA Team"

# Core SLGA modules
from .slga import SLGAModule
from .landmarks import LearnableLandmarkSelector, landmark_diversity_loss, landmark_sparsity_loss
from .model import Config, LLMTransformer
from .data import get_tokenizer, load_text_dataset, CollatorLocal, CollatorLocalGlobal

# Reasoning modules
from .reasoning import (
    ReasoningConfig,
    ThoughtTokenEmbedding,
    ReasoningController,
    ProcessRewardModel,
    ChainOfThoughtLoss,
    ReasoningLandmarkSelector,
    SelfConsistencyDecoder,
    create_reasoning_tokens,
    format_cot_example,
    extract_reasoning_steps,
)
from .reasoning_model import (
    ReasoningModelConfig,
    SLGAReasoningModel,
    create_reasoning_model,
)

__all__ = [
    # Core
    "SLGAModule",
    "LearnableLandmarkSelector",
    "landmark_diversity_loss",
    "landmark_sparsity_loss",
    "Config",
    "LLMTransformer",
    "get_tokenizer",
    "load_text_dataset",
    "CollatorLocal",
    "CollatorLocalGlobal",
    # Reasoning
    "ReasoningConfig",
    "ThoughtTokenEmbedding",
    "ReasoningController",
    "ProcessRewardModel",
    "ChainOfThoughtLoss",
    "ReasoningLandmarkSelector",
    "SelfConsistencyDecoder",
    "create_reasoning_tokens",
    "format_cot_example",
    "extract_reasoning_steps",
    "ReasoningModelConfig",
    "SLGAReasoningModel",
    "create_reasoning_model",
]
