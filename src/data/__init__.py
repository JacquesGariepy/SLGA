"""
Data module for SLGA.

Provides loaders, processors, tokenizers, collators, and factories
for managing training data with clean architecture.
"""

# Collators
from src.data.collators.language_modeling_collator import (
    CollatorLocal,
    CollatorLocalGlobal,
    CollatorWithTFIDF,
)

# Factories
from src.data.factory import (
    CollatorFactory,
    DataLoaderFactory,
    ProcessorFactory,
    TokenizerFactory,
    get_tokenizer,
)

# Loaders
from src.data.loaders.text_dataset import TextDatasetLoader
from src.data.loaders.wiki_loader import WikipediaDatasetLoader

# Processors
from src.data.processors.cleaner import CleanedDataset, DataCleaner
from src.data.processors.text_processor import TextProcessor

# Tokenizers
from src.data.tokenizers.tokenizer_wrapper import TokenizerWrapper

__all__ = [
    # Loaders
    "TextDatasetLoader",
    "WikipediaDatasetLoader",
    # Processors
    "TextProcessor",
    "DataCleaner",
    "CleanedDataset",
    # Tokenizers
    "TokenizerWrapper",
    # Collators
    "CollatorLocal",
    "CollatorLocalGlobal",
    "CollatorWithTFIDF",
    # Factories
    "DataLoaderFactory",
    "ProcessorFactory",
    "CollatorFactory",
    "TokenizerFactory",
    "get_tokenizer",
]
