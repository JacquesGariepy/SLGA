"""
Text Dataset Loader implementing IDatasetRepository.

Loads and caches text datasets from HuggingFace Hub or local disk.
Supports streaming for large datasets and memory-efficient processing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader


class TextDatasetLoader:
    """Load and cache text datasets.

    Supports HuggingFace datasets with streaming and caching.
    Automatically detects local vs remote datasets.

    Args:
        dataset_name: Dataset identifier (e.g., "openwebtext") or local path
        cache_dir: Directory for caching downloaded datasets
        streaming: Whether to use streaming mode for large datasets

    Example:
        >>> loader = TextDatasetLoader("wikipedia", streaming=True)
        >>> dataset = loader.load_dataset(split="train", subset="20220301.en")
        >>> print(f"Loaded {len(dataset)} examples")

        >>> # Local dataset
        >>> loader = TextDatasetLoader("data/mixed")
        >>> dataset = loader.load_dataset(split="train")
    """

    def __init__(
        self,
        dataset_name: str,
        cache_dir: Optional[str] = None,
        streaming: bool = False,
    ) -> None:
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.streaming = streaming
        self._is_local = os.path.exists(dataset_name)

    def load_dataset(
        self,
        split: str = "train",
        subset: Optional[str] = None,
    ) -> Dataset:
        """Load dataset split.

        Args:
            split: Split to load ("train", "validation", "test")
            subset: Subset/configuration of dataset (for HuggingFace Hub)

        Returns:
            Dataset object with examples

        Raises:
            ValueError: If split not found in local dataset
            RuntimeError: If dataset loading fails

        Example:
            >>> dataset = loader.load_dataset(split="train", subset="20220301.en")
        """
        if self._is_local:
            return self._load_local(split)
        return self._load_remote(split, subset)

    def _load_local(self, split: str) -> Dataset:
        """Load dataset from local disk."""
        print(f"Loading LOCAL dataset from: {self.dataset_name}")

        try:
            dataset = load_from_disk(self.dataset_name)

            # Handle DatasetDict (multiple splits)
            if isinstance(dataset, dict):
                if split not in dataset:
                    available = list(dataset.keys())
                    raise ValueError(
                        f"Split '{split}' not found in local dataset. "
                        f"Available splits: {available}"
                    )
                return dataset[split]
            # Single dataset without splits
            return dataset

        except Exception as e:
            raise RuntimeError(f"Failed to load local dataset from '{self.dataset_name}': {e}")

    def _load_remote(self, split: str, subset: Optional[str]) -> Dataset:
        """Load dataset from HuggingFace Hub."""
        print(f"Loading HuggingFace dataset: {self.dataset_name}")
        if subset:
            print(f"   Subset: {subset}")

        try:
            dataset = load_dataset(
                self.dataset_name,
                subset,
                split=split,
                cache_dir=self.cache_dir,
                streaming=self.streaming,
            )
            return dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset '{self.dataset_name}': {e}")

    def get_dataloader(
        self,
        dataset: Any,
        batch_size: int,
        collate_fn: Any,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> Iterator[Dict[str, Any]]:
        """Create PyTorch DataLoader for dataset.

        Args:
            dataset: Dataset object from load_dataset()
            batch_size: Batch size
            collate_fn: Collator function for batching
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes for data loading

        Returns:
            Iterator yielding batches of data

        Example:
            >>> dataloader = loader.get_dataloader(
            ...     dataset, batch_size=32, collate_fn=collator,
            ...     shuffle=True, num_workers=4
            ... )
            >>> for batch in dataloader:
            ...     input_ids = batch['input_ids']
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

    def save(self, dataset: Dataset, path: str) -> None:
        """Save dataset to disk.

        Args:
            dataset: Dataset to save
            path: Output directory path

        Example:
            >>> loader.save(dataset, "data/processed")
        """
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_path))
        print(f"Saved dataset to {path}")

    def preprocess_dataset(
        self,
        dataset: Any,
        tokenizer: Any,
        max_length: int,
        text_key: str = "text",
    ) -> Any:
        """Preprocess dataset with tokenization (optional).

        Note: This is optional. Collators can handle raw text on-the-fly.
        Pre-tokenization is faster but uses more disk space.

        Args:
            dataset: Raw dataset
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            text_key: Key in dataset dict containing text

        Returns:
            Preprocessed dataset with tokenized text
        """

        def tokenize_function(examples):
            return tokenizer(
                examples[text_key],
                max_length=max_length,
                truncation=True,
                padding=False,  # Collator will pad
            )

        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
