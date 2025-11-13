"""
SLGA-Plus: Sparse Local-Global Attention Transformer

Main package for efficient long-context language modeling.
"""

__version__ = "0.1.0"
__author__ = "SLGA Team"

# Import refactored data module components
from .legacy.data import (
    CollatorLocal,
    CollatorLocalGlobal,
    get_tokenizer,  # Backward compatible function
)

# Import new factory classes from modular data layer
try:
    from .data import (
        CollatorFactory,
        DataLoaderFactory,
        ProcessorFactory,
        TokenizerFactory,
    )
except ImportError:
    # Fallback if factories not available
    CollatorFactory = None
    DataLoaderFactory = None
    ProcessorFactory = None
    TokenizerFactory = None
from .legacy.landmarks import LearnableLandmarkSelector, landmark_diversity_loss, landmark_sparsity_loss
from .legacy.model import Config, LLMTransformer
from .legacy.slga import SLGAModule


# Backward compatibility: create load_text_dataset function
def load_text_dataset(dataset_name: str, subset: str = None, split: str = "train"):
    """Load text dataset (backward compatible wrapper).

    Args:
        dataset_name: Dataset name or local path
        subset: Subset/config name
        split: Split to load

    Returns:
        Dataset object

    Example:
        >>> dataset = load_text_dataset("openwebtext", split="train")
    """
    from .data.loaders.text_dataset import TextDatasetLoader

    loader = TextDatasetLoader(dataset_name)
    return loader.load_dataset(split=split, subset=subset)


__all__ = [
    "SLGAModule",
    "LearnableLandmarkSelector",
    "landmark_diversity_loss",
    "landmark_sparsity_loss",
    "Config",
    "LLMTransformer",
    # Data components (backward compatible)
    "get_tokenizer",
    "load_text_dataset",
    "CollatorLocal",
    "CollatorLocalGlobal",
    # New factory classes
    "DataLoaderFactory",
    "TokenizerFactory",
    "CollatorFactory",
    "ProcessorFactory",
]
