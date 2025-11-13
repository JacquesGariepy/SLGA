"""
Tokenizer Wrapper implementing ITokenizerRepository.

Abstracts HuggingFace tokenizers with special token handling.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from transformers import AutoTokenizer


class TokenizerWrapper:
    """Wrapper for HuggingFace tokenizers.

    Handles special token configuration and provides unified interface.

    Args:
        tokenizer_name: Tokenizer identifier (e.g., "gpt2", "bert-base-uncased")
        cache_dir: Directory for caching tokenizer files

    Example:
        >>> wrapper = TokenizerWrapper("gpt2")
        >>> tokenizer = wrapper.tokenizer
        >>> tokens = tokenizer.encode("Hello world")
    """

    def __init__(
        self,
        tokenizer_name: str,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.tokenizer_name = tokenizer_name
        self.cache_dir = cache_dir
        self.tokenizer = self._load_tokenizer()

    def _load_tokenizer(self) -> AutoTokenizer:
        """Load tokenizer from HuggingFace.

        Returns:
            Configured tokenizer instance
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            cache_dir=self.cache_dir,
        )

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Warning: Set pad_token to eos_token for {self.tokenizer_name}")

        return tokenizer

    def load_tokenizer(
        self,
        tokenizer_name: str,
        cache_dir: Optional[str] = None,
    ) -> AutoTokenizer:
        """Load a different tokenizer.

        Args:
            tokenizer_name: Tokenizer identifier
            cache_dir: Cache directory

        Returns:
            Tokenizer instance
        """
        old_name = self.tokenizer_name
        self.tokenizer_name = tokenizer_name
        self.cache_dir = cache_dir or self.cache_dir
        self.tokenizer = self._load_tokenizer()
        print(f"Switched tokenizer: {old_name} -> {tokenizer_name}")
        return self.tokenizer

    def get_special_tokens(
        self,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, int]:
        """Get special token IDs from tokenizer.

        Args:
            tokenizer: Tokenizer instance (uses self.tokenizer if None)

        Returns:
            Dict mapping special token names to IDs

        Example:
            >>> special = wrapper.get_special_tokens()
            >>> eos_id = special['eos']
            >>> pad_id = special['pad']
        """
        tok = tokenizer or self.tokenizer

        special_tokens = {
            "eos": tok.eos_token_id,
            "pad": tok.pad_token_id,
        }

        # Optional tokens
        if tok.bos_token_id is not None:
            special_tokens["bos"] = tok.bos_token_id
        if tok.unk_token_id is not None:
            special_tokens["unk"] = tok.unk_token_id

        return special_tokens

    def get_vocab_size(
        self,
        tokenizer: Optional[Any] = None,
    ) -> int:
        """Get vocabulary size of tokenizer.

        Args:
            tokenizer: Tokenizer instance (uses self.tokenizer if None)

        Returns:
            Vocabulary size (number of unique tokens)

        Example:
            >>> vocab_size = wrapper.get_vocab_size()
            >>> print(f"Vocabulary size: {vocab_size}")
        """
        tok = tokenizer or self.tokenizer
        return len(tok)

    def encode(self, text: str, **kwargs) -> Any:
        """Encode text to token IDs.

        Args:
            text: Text to encode
            **kwargs: Additional arguments for tokenizer

        Returns:
            Encoded token IDs
        """
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, token_ids: Any, **kwargs) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            **kwargs: Additional arguments for tokenizer

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, **kwargs)

    def __call__(self, *args, **kwargs):
        """Forward calls to underlying tokenizer."""
        return self.tokenizer(*args, **kwargs)
