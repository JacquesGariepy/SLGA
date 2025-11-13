"""
Language Modeling Collators for batching and padding.

Includes:
- CollatorLocal: Local-only attention (learned landmarks)
- CollatorLocalGlobal: Heuristic landmark selection
- CollatorWithTFIDF: TF-IDF based landmark selection
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer


class CollatorLocal:
    """Collator for local-only attention mode.

    Used when learned_landmarks=True (model selects landmarks).
    Performs tokenization, padding, and label preparation.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        text_key: Key in examples containing text

    Example:
        >>> collator = CollatorLocal(tokenizer, max_length=512)
        >>> batch = collator(examples)
        >>> input_ids = batch['input_ids']  # (B, L)
        >>> labels = batch['labels']        # (B, L)
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
        text_key: str = "text",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples.

        Args:
            examples: List of dicts with text_key

        Returns:
            Batch dict with 'input_ids' and 'labels'
        """
        # Extract texts
        texts = [ex[self.text_key] for ex in examples]

        # Tokenize with padding
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length + 1,  # +1 for label shift
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"]  # (B, L+1)

        # Truncate to max_length + 1
        input_ids = input_ids[:, : self.max_length + 1]

        # Create labels: shift by 1 position (predict next token)
        labels = input_ids[:, 1:].clone()  # (B, L)
        input_ids = input_ids[:, :-1]  # (B, L)

        # Mask padding tokens in labels with -100
        pad_mask = labels == self.tokenizer.pad_token_id
        labels[pad_mask] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class CollatorLocalGlobal:
    """Collator with heuristic global landmark selection.

    Strategies:
    - "regular": Uniformly spaced landmarks
    - "paragraph": Paragraph boundary landmarks
    - "random": Random landmark positions

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        global_every: Spacing between landmarks (for "regular" strategy)
        max_global: Maximum number of global landmarks
        strategy: Landmark selection strategy
        text_key: Key in examples containing text

    Example:
        >>> collator = CollatorLocalGlobal(
        ...     tokenizer, max_length=512, max_global=48, strategy="regular"
        ... )
        >>> batch = collator(examples)
        >>> cache_global_ids = batch['cache_global_ids']  # (B, G)
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
        global_every: int = 128,
        max_global: int = 64,
        strategy: str = "regular",
        text_key: str = "text",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.global_every = global_every
        self.max_global = max_global
        self.strategy = strategy
        self.text_key = text_key

    def _select_landmarks_regular(self, length: int) -> List[int]:
        """Select uniformly spaced landmarks."""
        # Use linspace for exactly max_global landmarks
        positions = torch.linspace(0, length - 1, self.max_global).long().tolist()
        return positions

    def _select_landmarks_random(self, length: int) -> List[int]:
        """Select random landmark positions."""
        num_landmarks = min(self.max_global, length)
        landmarks = sorted(np.random.choice(length, num_landmarks, replace=False).tolist())
        return landmarks

    def _select_landmarks_paragraph(self, text: str, tokens: List[int]) -> List[int]:
        """Select landmarks at paragraph boundaries.

        Detects paragraph starts (after \\n\\n) and maps to token positions.
        """
        # Split by paragraph boundaries
        paragraphs = text.split("\n\n")

        # Approximate token positions for paragraph starts
        landmarks = [0]  # Always include start
        cumulative_len = 0

        for para in paragraphs[:-1]:  # Exclude last
            para_tokens = self.tokenizer.encode(para, add_special_tokens=False)
            cumulative_len += len(para_tokens)
            if cumulative_len < len(tokens):
                landmarks.append(cumulative_len)

        # Limit and sort
        landmarks = sorted(set(landmarks))[: self.max_global]
        return landmarks

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch with global landmarks.

        Returns:
            Batch dict with 'input_ids', 'labels', 'cache_global_ids'
        """
        texts = [ex[self.text_key] for ex in examples]

        # Tokenize
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length + 1,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"]

        # Truncate
        input_ids = input_ids[:, : self.max_length + 1]

        # Create labels
        labels = input_ids[:, 1:].clone()  # (B, L)
        input_ids = input_ids[:, :-1]  # (B, L)

        # Mask padding
        pad_mask = labels == self.tokenizer.pad_token_id
        labels[pad_mask] = -100

        # Select landmarks for each example
        cache_global_ids_list = []

        for i, text in enumerate(texts):
            L = self.max_length

            if self.strategy == "regular":
                landmarks = self._select_landmarks_regular(L)
            elif self.strategy == "random":
                landmarks = self._select_landmarks_random(L)
            elif self.strategy == "paragraph":
                tokens_i = input_ids[i].tolist()
                landmarks = self._select_landmarks_paragraph(text, tokens_i)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            # Pad if needed
            while len(landmarks) < self.max_global:
                landmarks.append(0)

            cache_global_ids_list.append(landmarks[: self.max_global])

        cache_global_ids = torch.tensor(cache_global_ids_list, dtype=torch.long)  # (B, G)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "cache_global_ids": cache_global_ids,  # Positions, not tokens
        }


class CollatorWithTFIDF:
    """Collator with TF-IDF based landmark selection.

    Requires sklearn. Falls back to regular spacing if not available.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        max_global: Maximum number of global landmarks
        span_length: Length of text spans for TF-IDF scoring
        text_key: Key in examples containing text
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
        max_global: int = 64,
        span_length: int = 8,
        text_key: str = "text",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_global = max_global
        self.span_length = span_length
        self.text_key = text_key

        # Initialize TF-IDF vectorizer
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        except ImportError:
            print("Warning: sklearn not found, TF-IDF collator will use fallback")
            self.tfidf = None

    def _select_landmarks_tfidf(self, text: str, tokens: List[int]) -> List[int]:
        """Select landmarks using TF-IDF scoring of spans."""
        if self.tfidf is None:
            # Fallback to regular spacing
            return list(range(0, len(tokens), len(tokens) // self.max_global))[: self.max_global]

        # Divide into spans
        num_spans = len(tokens) // self.span_length
        spans = []
        span_starts = []

        for i in range(num_spans):
            start = i * self.span_length
            end = start + self.span_length
            span_tokens = tokens[start:end]
            span_text = self.tokenizer.decode(span_tokens)
            spans.append(span_text)
            span_starts.append(start)

        if not spans:
            return [0]

        # TF-IDF scoring
        try:
            tfidf_matrix = self.tfidf.fit_transform(spans)
            scores = tfidf_matrix.sum(axis=1).A1  # (num_spans,)

            # Select top-K spans
            num_landmarks = min(self.max_global, len(scores))
            top_indices = np.argsort(scores)[-num_landmarks:][::-1]

            landmarks = sorted([span_starts[i] for i in top_indices])
        except Exception:
            # Fallback on error
            landmarks = list(range(0, len(tokens), len(tokens) // self.max_global))[
                : self.max_global
            ]

        return landmarks

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch with TF-IDF landmarks."""
        texts = [ex[self.text_key] for ex in examples]

        encoded = self.tokenizer(
            texts,
            max_length=self.max_length + 1,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"]
        input_ids = input_ids[:, : self.max_length + 1]

        labels = input_ids[:, 1:].clone()
        input_ids = input_ids[:, :-1]

        pad_mask = labels == self.tokenizer.pad_token_id
        labels[pad_mask] = -100

        # Select TF-IDF landmarks
        cache_global_ids_list = []

        for i, text in enumerate(texts):
            tokens_i = input_ids[i].tolist()
            landmarks = self._select_landmarks_tfidf(text, tokens_i)

            # Pad if needed
            while len(landmarks) < self.max_global:
                landmarks.append(0)

            cache_global_ids_list.append(landmarks[: self.max_global])

        cache_global_ids = torch.tensor(cache_global_ids_list, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "cache_global_ids": cache_global_ids,
        }
