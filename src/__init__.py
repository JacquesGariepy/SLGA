"""
SLGA-Plus: Sparse Local-Global Attention Transformer

Main package for efficient long-context language modeling.
"""

__version__ = "0.1.0"
__author__ = "SLGA Team"

from .slga import SLGAModule
from .landmarks import LearnableLandmarkSelector, landmark_diversity_loss, landmark_sparsity_loss
from .model import Config, LLMTransformer
from .data import get_tokenizer, load_text_dataset, CollatorLocal, CollatorLocalGlobal

__all__ = [
    "SLGAModule",
    "LearnableLandmarkSelector",
    "landmark_diversity_loss",
    "landmark_sparsity_loss",
    "Config",
    "LLMTransformer",
    "get_tokenizer",
    "load_text_dataset",
    "CollatorLocal",
    "CollatorLocalGlobal"
]
