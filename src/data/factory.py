"""
Data Loader Factory with registry pattern.

Provides centralized creation of data loaders, processors, and collators.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from .collators.language_modeling_collator import (
    CollatorLocal,
    CollatorLocalGlobal,
    CollatorWithTFIDF,
)
from .loaders.text_dataset import TextDatasetLoader
from .loaders.wiki_loader import WikipediaDatasetLoader
from .processors.cleaner import DataCleaner
from .processors.text_processor import TextProcessor
from .tokenizers.tokenizer_wrapper import TokenizerWrapper


class DataLoaderFactory:
    """Factory for creating data loaders.

    Example:
        >>> factory = DataLoaderFactory()
        >>> loader = factory.create_loader("wikipedia", language="en")
        >>> dataset = loader.load_dataset(split="train")
    """

    _loader_registry: Dict[str, Type] = {
        "text": TextDatasetLoader,
        "wikipedia": WikipediaDatasetLoader,
    }

    @classmethod
    def create_loader(cls, loader_type: str, **kwargs) -> Any:
        """Create a data loader.

        Args:
            loader_type: Loader type ("text", "wikipedia")
            **kwargs: Arguments for loader constructor

        Returns:
            Loader instance

        Raises:
            ValueError: If loader type not found

        Example:
            >>> loader = DataLoaderFactory.create_loader("text", dataset_name="openwebtext")
        """
        if loader_type not in cls._loader_registry:
            raise ValueError(
                f"Unknown loader type: {loader_type}. "
                f"Available: {list(cls._loader_registry.keys())}"
            )
        return cls._loader_registry[loader_type](**kwargs)

    @classmethod
    def register_loader(cls, name: str, loader_class: Type) -> None:
        """Register a custom loader.

        Args:
            name: Loader name
            loader_class: Loader class

        Example:
            >>> DataLoaderFactory.register_loader("custom", CustomLoader)
        """
        cls._loader_registry[name] = loader_class
        print(f"Registered loader: {name}")


class ProcessorFactory:
    """Factory for creating text processors.

    Example:
        >>> factory = ProcessorFactory()
        >>> processor = factory.create_processor("text")
        >>> cleaned_dataset = processor.process_dataset(dataset)
    """

    _processor_registry: Dict[str, Type] = {
        "text": TextProcessor,
        "cleaner": DataCleaner,
    }

    @classmethod
    def create_processor(cls, processor_type: str, **kwargs) -> Any:
        """Create a text processor.

        Args:
            processor_type: Processor type ("text", "cleaner")
            **kwargs: Arguments for processor constructor

        Returns:
            Processor instance
        """
        if processor_type not in cls._processor_registry:
            raise ValueError(
                f"Unknown processor type: {processor_type}. "
                f"Available: {list(cls._processor_registry.keys())}"
            )
        return cls._processor_registry[processor_type](**kwargs)


class CollatorFactory:
    """Factory for creating collators.

    Example:
        >>> factory = CollatorFactory()
        >>> collator = factory.create_collator(
        ...     "local_global", tokenizer, max_length=512, strategy="regular"
        ... )
    """

    _collator_registry: Dict[str, Type] = {
        "local": CollatorLocal,
        "local_global": CollatorLocalGlobal,
        "tfidf": CollatorWithTFIDF,
    }

    @classmethod
    def create_collator(
        cls,
        collator_type: str,
        tokenizer: Any,
        max_length: int,
        **kwargs: Any,
    ) -> Any:
        """Create a collator.

        Args:
            collator_type: Collator type ("local", "local_global", "tfidf")
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            **kwargs: Additional collator-specific arguments

        Returns:
            Collator instance (callable)

        Example:
            >>> collator = CollatorFactory.create_collator(
            ...     "local_global", tokenizer, max_length=512,
            ...     max_global=48, strategy="regular"
            ... )
        """
        if collator_type not in cls._collator_registry:
            raise ValueError(
                f"Unknown collator type: {collator_type}. "
                f"Available: {list(cls._collator_registry.keys())}"
            )
        return cls._collator_registry[collator_type](
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )


class TokenizerFactory:
    """Factory for creating tokenizer wrappers.

    Example:
        >>> factory = TokenizerFactory()
        >>> wrapper = factory.create_tokenizer("gpt2")
        >>> tokenizer = wrapper.tokenizer
    """

    @classmethod
    def create_tokenizer(
        cls,
        tokenizer_name: str,
        cache_dir: Optional[str] = None,
    ) -> TokenizerWrapper:
        """Create a tokenizer wrapper.

        Args:
            tokenizer_name: Tokenizer identifier
            cache_dir: Cache directory

        Returns:
            TokenizerWrapper instance
        """
        return TokenizerWrapper(tokenizer_name, cache_dir)


# Convenience function for backward compatibility
def get_tokenizer(tokenizer_name: str, cache_dir: Optional[str] = None):
    """Get tokenizer (backward compatible with data.py).

    Args:
        tokenizer_name: Tokenizer identifier
        cache_dir: Cache directory

    Returns:
        Tokenizer instance
    """
    wrapper = TokenizerFactory.create_tokenizer(tokenizer_name, cache_dir)
    return wrapper.tokenizer
