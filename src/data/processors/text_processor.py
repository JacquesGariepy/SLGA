"""
Text Processor for cleaning and normalization.

Handles text preprocessing including cleaning, normalization,
and quality filtering for training data.
"""

from __future__ import annotations

from typing import Callable, Optional

from datasets import Dataset


class TextProcessor:
    """Process and normalize text data.

    Applies cleaning transformations to raw text for better training quality.

    Args:
        text_key: Key in dataset dict containing text
        cleaning_fn: Custom cleaning function (optional)

    Example:
        >>> processor = TextProcessor(text_key="text")
        >>> dataset = processor.process_dataset(dataset)
    """

    def __init__(
        self,
        text_key: str = "text",
        cleaning_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.text_key = text_key
        self.cleaning_fn = cleaning_fn

    def process_dataset(
        self,
        dataset: Dataset,
        batched: bool = True,
        num_proc: Optional[int] = None,
    ) -> Dataset:
        """Process entire dataset with cleaning.

        Args:
            dataset: Raw dataset
            batched: Whether to process in batches (faster)
            num_proc: Number of processes for parallel processing

        Returns:
            Processed dataset with cleaned text

        Example:
            >>> processed = processor.process_dataset(dataset, num_proc=4)
        """

        def process_function(examples):
            if batched:
                # Process batch of examples
                texts = examples[self.text_key]
                cleaned = [self._clean_text(text) for text in texts]
                return {self.text_key: cleaned}
            # Process single example
            text = examples[self.text_key]
            cleaned = self._clean_text(text)
            return {self.text_key: cleaned}

        return dataset.map(
            process_function,
            batched=batched,
            num_proc=num_proc,
        )

    def _clean_text(self, text: str) -> str:
        """Clean a single text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return text

        # Use custom cleaning function if provided
        if self.cleaning_fn is not None:
            return self.cleaning_fn(text)

        # Default: basic normalization
        text = self._normalize_whitespace(text)
        text = self._strip_control_characters(text)

        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text.

        - Removes leading/trailing whitespace
        - Collapses multiple spaces to single space
        - Normalizes newlines
        """
        # Strip lines
        lines = text.split("\n")
        lines = [line.strip() for line in lines]
        text = "\n".join(lines)

        # Normalize spaces (but preserve single spaces)
        import re

        text = re.sub(r" {2,}", " ", text)

        # Normalize newlines (max 2 consecutive)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @staticmethod
    def _strip_control_characters(text: str) -> str:
        """Remove control characters from text.

        Keeps: space (32), tab (9), newline (10), carriage return (13)
        Removes: all other characters < 32 and DEL (127)
        """
        return "".join(c for c in text if ord(c) >= 32 or c in "\n\t\r")

    def filter_by_quality(
        self,
        dataset: Dataset,
        min_length: int = 50,
        max_length: Optional[int] = None,
        min_alpha_ratio: float = 0.6,
    ) -> Dataset:
        """Filter dataset by quality metrics.

        Args:
            dataset: Dataset to filter
            min_length: Minimum text length (characters)
            max_length: Maximum text length (optional)
            min_alpha_ratio: Minimum ratio of alphabetic characters

        Returns:
            Filtered dataset

        Example:
            >>> filtered = processor.filter_by_quality(
            ...     dataset, min_length=100, min_alpha_ratio=0.7
            ... )
        """

        def quality_filter(example):
            text = example[self.text_key]

            # Length check
            text_len = len(text)
            if text_len < min_length:
                return False
            if max_length is not None and text_len > max_length:
                return False

            # Alpha ratio check
            alpha_count = sum(c.isalpha() for c in text)
            alpha_ratio = alpha_count / text_len if text_len > 0 else 0
            if alpha_ratio < min_alpha_ratio:
                return False

            return True

        return dataset.filter(quality_filter)
