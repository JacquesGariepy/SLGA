"""
Wikipedia Dataset Loader specialization.

Handles Wikipedia-specific preprocessing and configuration.
"""

from __future__ import annotations

from typing import Optional

from datasets import Dataset

from .text_dataset import TextDatasetLoader


class WikipediaDatasetLoader(TextDatasetLoader):
    """Wikipedia dataset loader with language filtering.

    Specialized loader for Wikipedia dumps from HuggingFace Hub.

    Args:
        language: Language code (e.g., "en", "fr", "de")
        date: Wikipedia dump date (e.g., "20220301")
        cache_dir: Directory for caching
        streaming: Whether to use streaming mode

    Example:
        >>> loader = WikipediaDatasetLoader(language="en", date="20220301")
        >>> dataset = loader.load_dataset(split="train")
        >>> print(f"Loaded {len(dataset)} Wikipedia articles")
    """

    def __init__(
        self,
        language: str = "en",
        date: str = "20220301",
        cache_dir: Optional[str] = None,
        streaming: bool = False,
    ) -> None:
        # Wikipedia dataset name on HuggingFace Hub
        dataset_name = "wikipedia"

        super().__init__(
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            streaming=streaming,
        )

        self.language = language
        self.date = date

    def load_dataset(
        self,
        split: str = "train",
        subset: Optional[str] = None,
    ) -> Dataset:
        """Load Wikipedia dataset with language configuration.

        Args:
            split: Split to load (Wikipedia only has "train")
            subset: Override subset (default: "{date}.{language}")

        Returns:
            Wikipedia dataset

        Example:
            >>> dataset = loader.load_dataset(split="train")
        """
        # Use language and date to construct subset
        if subset is None:
            subset = f"{self.date}.{self.language}"

        return super().load_dataset(split=split, subset=subset)

    def filter_by_length(
        self,
        dataset: Dataset,
        min_length: int = 100,
        max_length: Optional[int] = None,
        text_key: str = "text",
    ) -> Dataset:
        """Filter Wikipedia articles by text length.

        Args:
            dataset: Wikipedia dataset
            min_length: Minimum text length (characters)
            max_length: Maximum text length (optional)
            text_key: Key containing article text

        Returns:
            Filtered dataset

        Example:
            >>> dataset = loader.filter_by_length(dataset, min_length=500)
        """

        def length_filter(example):
            text_len = len(example[text_key])
            if text_len < min_length:
                return False
            if max_length is not None and text_len > max_length:
                return False
            return True

        return dataset.filter(length_filter)

    def extract_title_and_text(
        self,
        dataset: Dataset,
    ) -> Dataset:
        """Extract title and text from Wikipedia articles.

        Wikipedia datasets have 'title', 'text', and 'url' fields.
        This method ensures consistent field naming.

        Args:
            dataset: Wikipedia dataset

        Returns:
            Dataset with 'title' and 'text' fields
        """

        def extract_fields(example):
            return {
                "title": example.get("title", ""),
                "text": example.get("text", ""),
            }

        return dataset.map(extract_fields)
