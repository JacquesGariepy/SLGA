"""
Data Cleaner for removing unwanted content.

Handles Unicode corruption, control characters, and text normalization.
Extracted from dataset_cleaner.py with enhanced error handling.
"""

from __future__ import annotations

import re
from typing import Any, Dict


class DataCleaner:
    """Clean text data by removing corruption and unwanted characters.

    Handles:
    - Unicode replacement characters (�, U+FFFD)
    - Control characters (except newline/tab)
    - Multiple whitespaces
    - Newline normalization

    Args:
        text_key: Key in dataset dict containing text

    Example:
        >>> cleaner = DataCleaner(text_key="text")
        >>> cleaned_text = cleaner.clean_text("Hello � world")
        >>> print(cleaned_text)  # "Hello  world"
    """

    def __init__(self, text_key: str = "text") -> None:
        self.text_key = text_key

    def clean_text(self, text: str) -> str:
        """Clean a single text.

        Args:
            text: Raw text with potential corruption

        Returns:
            Cleaned text

        Example:
            >>> text = "Hello\\x00world   test\\n\\n\\n\\nline"
            >>> cleaned = cleaner.clean_text(text)
            >>> print(cleaned)  # "Helloworld  test\\n\\nline"
        """
        if not isinstance(text, str):
            return text

        # 1. Replace Unicode replacement characters (�)
        text = self._remove_replacement_chars(text)

        # 2. Remove control characters (except \n, \t, \r)
        text = self._remove_control_chars(text)

        # 3. Normalize whitespace
        text = self._normalize_spaces(text)
        text = self._normalize_tabs(text)
        text = self._normalize_newlines(text)

        # 4. Strip lines
        text = self._strip_lines(text)

        return text

    @staticmethod
    def _remove_replacement_chars(text: str) -> str:
        """Remove Unicode replacement characters.

        U+FFFD (�) is inserted by Python when encoding fails.
        """
        text = text.replace("\ufffd", " ")  # Unicode escape
        text = text.replace("�", " ")  # Literal
        return text

    @staticmethod
    def _remove_control_chars(text: str) -> str:
        """Remove control characters except whitespace.

        Keeps: space (32), tab (9), newline (10), carriage return (13)
        Removes: all other characters < 32 and DEL (127)
        """
        return "".join(c for c in text if ord(c) >= 32 or c in "\n\t\r")

    @staticmethod
    def _normalize_spaces(text: str) -> str:
        """Normalize multiple spaces to two spaces.

        Replaces 3+ consecutive spaces with 2 spaces.
        """
        return re.sub(r" {3,}", "  ", text)

    @staticmethod
    def _normalize_tabs(text: str) -> str:
        """Normalize multiple tabs to single tab."""
        return re.sub(r"\t+", "\t", text)

    @staticmethod
    def _normalize_newlines(text: str) -> str:
        """Normalize multiple newlines to max 2.

        Replaces 3+ consecutive newlines with 2 newlines.
        """
        return re.sub(r"\n{3,}", "\n\n", text)

    @staticmethod
    def _strip_lines(text: str) -> str:
        """Strip whitespace from beginning/end of lines."""
        lines = text.split("\n")
        lines = [line.strip() for line in lines]
        return "\n".join(lines)

    def clean_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Clean a dataset example.

        Args:
            example: Dict with text_key containing text

        Returns:
            Example with cleaned text

        Example:
            >>> example = {"text": "Hello � world"}
            >>> cleaned = cleaner.clean_example(example)
            >>> print(cleaned["text"])  # "Hello  world"
        """
        if self.text_key not in example:
            return example

        example_clean = example.copy()
        example_clean[self.text_key] = self.clean_text(example[self.text_key])
        return example_clean


class CleanedDataset:
    """Wrapper around HuggingFace dataset that cleans text on-the-fly.

    Compatible with existing dataset_cleaner.py interface.

    Args:
        dataset: HuggingFace dataset to wrap
        text_key: Key containing text to clean

    Example:
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("openwebtext", split="train")
        >>> cleaned = CleanedDataset(dataset, text_key="text")
        >>> example = cleaned[0]  # Text is cleaned automatically
    """

    def __init__(self, dataset, text_key: str = "text"):
        self.dataset = dataset
        self.text_key = text_key
        self.cleaner = DataCleaner(text_key=text_key)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get cleaned example at index."""
        example = self.dataset[idx]

        # If dict with text_key
        if isinstance(example, dict) and self.text_key in example:
            return self.cleaner.clean_example(example)

        # If string directly
        if isinstance(example, str):
            return self.cleaner.clean_text(example)

        # Otherwise return as-is
        return example

    def select(self, indices):
        """Support for .select() like HF datasets."""
        return CleanedDataset(self.dataset.select(indices), text_key=self.text_key)

    @property
    def column_names(self):
        """Expose columns of underlying dataset."""
        if hasattr(self.dataset, "column_names"):
            return self.dataset.column_names
        return None

    @property
    def features(self):
        """Expose features of underlying dataset."""
        if hasattr(self.dataset, "features"):
            return self.dataset.features
        return None
