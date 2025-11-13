"""
Data Repository Protocol Interfaces

This module defines Protocol interfaces for data access, including dataset loading,
model checkpointing, and tokenization. These repositories abstract data sources
for clean architecture compliance.
"""

from typing import Any, Dict, Iterator, List, Optional, Protocol

import torch
from torch import Tensor


class IDatasetRepository(Protocol):
    """Protocol for dataset loading and preprocessing.

    Abstracts data sources (HuggingFace Hub, local files, custom datasets)
    and provides unified interface for training data access.
    """

    def load_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        subset: Optional[str] = None,
    ) -> Any:
        """Load dataset from source.

        Args:
            dataset_name: Dataset identifier (e.g., "openwebtext" or local path)
            split: Dataset split ("train", "validation", "test")
            subset: Optional subset/config of dataset

        Returns:
            Dataset object (e.g., HuggingFace Dataset)

        Example:
            >>> repo = HuggingFaceDatasetRepository()
            >>> train_data = repo.load_dataset("openwebtext", split="train")
            >>> print(len(train_data))
            >>>
            >>> # Local dataset
            >>> local_data = repo.load_dataset("data/mixed", split="train")

        Notes:
            - Supports both HuggingFace Hub and local datasets
            - Local datasets loaded via load_from_disk()
            - Returns raw dataset (preprocessing done by collator)
        """
        ...

    def get_dataloader(
        self,
        dataset: Any,
        batch_size: int,
        collate_fn: Any,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> Iterator[Dict[str, Tensor]]:
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
            >>> from torch.utils.data import DataLoader
            >>> dataloader = repo.get_dataloader(
            ...     dataset, batch_size=32, collate_fn=collator,
            ...     shuffle=True, num_workers=4
            ... )
            >>> for batch in dataloader:
            ...     input_ids = batch['input_ids']
            ...     labels = batch['labels']

        Notes:
            - Uses PyTorch DataLoader under the hood
            - num_workers>0 enables parallel data loading
            - collate_fn transforms raw examples to batched tensors
        """
        ...

    def preprocess_dataset(
        self,
        dataset: Any,
        tokenizer: Any,
        max_length: int,
        text_key: str = "text",
    ) -> Any:
        """Preprocess dataset with tokenization.

        Args:
            dataset: Raw dataset
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            text_key: Key in dataset dict containing text

        Returns:
            Preprocessed dataset with tokenized text

        Example:
            >>> preprocessed = repo.preprocess_dataset(
            ...     dataset, tokenizer, max_length=512, text_key="text"
            ... )

        Notes:
            - Tokenization can be done in advance (faster) or on-the-fly (collator)
            - This method is optional; collators can handle raw text
        """
        ...


class ICheckpointRepository(Protocol):
    """Protocol for model checkpointing and restoration.

    Handles saving/loading model weights, optimizer state, and training metadata
    for resuming training and inference.
    """

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        path: str,
    ) -> None:
        """Save model checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state (optional)
            epoch: Current epoch number
            step: Current training step
            metrics: Training metrics (loss, accuracy, etc.)
            path: Checkpoint file path

        Example:
            >>> repo = CheckpointRepository()
            >>> repo.save_checkpoint(
            ...     model, optimizer, epoch=5, step=10000,
            ...     metrics={'loss': 2.5, 'ppl': 12.2},
            ...     path="checkpoints/model_epoch5.pt"
            ... )

        Notes:
            - Saves model state_dict, optimizer state, and metadata
            - Use torch.save() with dict structure for compatibility
            - Recommended: Save periodically and keep best checkpoints
        """
        ...

    def load_checkpoint(
        self,
        path: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Load model checkpoint.

        Args:
            path: Checkpoint file path
            model: Model to load weights into (optional)
            optimizer: Optimizer to load state into (optional)
            device: Device to load checkpoint on

        Returns:
            Checkpoint dictionary with keys:
            - 'epoch': Epoch number
            - 'step': Training step
            - 'metrics': Training metrics
            - 'model_state_dict': Model weights (if model not provided)
            - 'optimizer_state_dict': Optimizer state (if optimizer not provided)

        Example:
            >>> # Load into existing model
            >>> checkpoint = repo.load_checkpoint(
            ...     "checkpoints/model_epoch5.pt",
            ...     model=model, optimizer=optimizer, device="cuda"
            ... )
            >>> print(f"Resuming from epoch {checkpoint['epoch']}")
            >>>
            >>> # Load state dict only
            >>> checkpoint = repo.load_checkpoint("checkpoints/model.pt")
            >>> model.load_state_dict(checkpoint['model_state_dict'])

        Notes:
            - If model/optimizer provided, loads state into them directly
            - Otherwise, returns state dicts in checkpoint dict
            - Always use map_location for device compatibility
        """
        ...

    def list_checkpoints(
        self,
        directory: str,
    ) -> List[str]:
        """List available checkpoints in directory.

        Args:
            directory: Directory to search

        Returns:
            List of checkpoint file paths, sorted by modification time

        Example:
            >>> checkpoints = repo.list_checkpoints("checkpoints/")
            >>> print(f"Found {len(checkpoints)} checkpoints")
            >>> latest = checkpoints[-1]  # Most recent
        """
        ...

    def get_best_checkpoint(
        self,
        directory: str,
        metric: str = "loss",
        mode: str = "min",
    ) -> Optional[str]:
        """Find best checkpoint by metric.

        Args:
            directory: Directory to search
            metric: Metric name to compare (e.g., "loss", "accuracy")
            mode: "min" or "max" (minimize or maximize metric)

        Returns:
            Path to best checkpoint, or None if not found

        Example:
            >>> best_path = repo.get_best_checkpoint(
            ...     "checkpoints/", metric="loss", mode="min"
            ... )
            >>> if best_path:
            ...     model = repo.load_checkpoint(best_path, model=model)

        Notes:
            - Loads each checkpoint to read metrics (can be slow)
            - Consider maintaining separate "best_model.pt" during training
        """
        ...


class ITokenizerRepository(Protocol):
    """Protocol for tokenizer management.

    Handles loading, caching, and configuration of tokenizers for text processing.
    """

    def load_tokenizer(
        self,
        tokenizer_name: str,
        cache_dir: Optional[str] = None,
    ) -> Any:
        """Load tokenizer from HuggingFace or local path.

        Args:
            tokenizer_name: Tokenizer identifier (e.g., "gpt2", "bert-base-uncased")
            cache_dir: Directory for caching tokenizer files

        Returns:
            Tokenizer instance (e.g., AutoTokenizer)

        Example:
            >>> repo = TokenizerRepository()
            >>> tokenizer = repo.load_tokenizer("gpt2")
            >>> tokens = tokenizer.encode("Hello world")

        Notes:
            - Uses HuggingFace AutoTokenizer for compatibility
            - Ensures pad_token is set (uses eos_token if missing)
            - Caches downloaded tokenizers locally
        """
        ...

    def get_special_tokens(
        self,
        tokenizer: Any,
    ) -> Dict[str, int]:
        """Get special token IDs from tokenizer.

        Args:
            tokenizer: Tokenizer instance

        Returns:
            Dict mapping special token names to IDs:
            - 'eos': End-of-sequence token
            - 'bos': Begin-of-sequence token (if exists)
            - 'pad': Padding token
            - 'unk': Unknown token (if exists)

        Example:
            >>> special = repo.get_special_tokens(tokenizer)
            >>> eos_id = special['eos']
            >>> pad_id = special['pad']

        Notes:
            - Used for loss masking (ignore pad_id in labels)
            - EOS token used for stop criteria in generation
        """
        ...

    def get_vocab_size(
        self,
        tokenizer: Any,
    ) -> int:
        """Get vocabulary size of tokenizer.

        Args:
            tokenizer: Tokenizer instance

        Returns:
            Vocabulary size (number of unique tokens)

        Example:
            >>> vocab_size = repo.get_vocab_size(tokenizer)
            >>> print(f"Vocabulary size: {vocab_size}")

        Notes:
            - Used for initializing model embedding layers
            - Includes special tokens in count
        """
        ...


class ICollatorRepository(Protocol):
    """Protocol for data collation strategies.

    Manages different collation strategies for batching examples with appropriate
    preprocessing (tokenization, padding, landmark selection).
    """

    def get_collator(
        self,
        collator_type: str,
        tokenizer: Any,
        max_length: int,
        **kwargs: Any,
    ) -> Any:
        """Get collator by type.

        Args:
            collator_type: Collator type ("local", "local_global", "tfidf")
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            **kwargs: Additional collator-specific arguments

        Returns:
            Collator instance (callable)

        Example:
            >>> repo = CollatorRepository()
            >>> # Local-only collator (learned landmarks)
            >>> collator = repo.get_collator(
            ...     "local", tokenizer, max_length=512
            ... )
            >>>
            >>> # Local-global with heuristic landmarks
            >>> collator = repo.get_collator(
            ...     "local_global", tokenizer, max_length=512,
            ...     global_every=128, max_global=48, strategy="regular"
            ... )

        Collator Types:
            - "local": No landmark selection (for learned landmarks)
            - "local_global": Heuristic landmark selection
              - strategy="regular": Uniformly spaced
              - strategy="random": Random positions
              - strategy="paragraph": Paragraph boundaries
            - "tfidf": TF-IDF based selection (requires sklearn)

        Notes:
            - Collators are called by DataLoader during batching
            - Must return dict with keys: 'input_ids', 'labels', ['cache_global_ids']
            - cache_global_ids contains positions (not tokens) for heuristic landmarks
        """
        ...


__all__ = [
    "ICheckpointRepository",
    "ICollatorRepository",
    "IDatasetRepository",
    "ITokenizerRepository",
]
