# model.py
"""
Backward compatibility layer for SLGA Transformer.

This module maintains the original API while delegating to the new modular architecture.
All new code should import from src.models instead.

Architecture now organized as:
- src/models/config.py - Model configuration with validation
- src/core/layers/embedding.py - Token and position embeddings
- src/core/layers/feedforward.py - Feed-forward networks
- src/core/layers/transformer_block.py - Transformer blocks with SLGA
- src/models/slga_model.py - Main model orchestrator
- src/models/generation.py - Text generation logic
- src/models/factory.py - Model factory with registry pattern
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import warnings

import torch
from torch import nn

from src.core.layers.feedforward import FeedForward
from src.core.layers.transformer_block import TransformerBlock

# Import from new modular structure
from src.models.config import ModelConfig as Config
from src.models.factory import ModelFactory
from src.models.generation import TextGenerator
from src.models.slga_model import SLGATransformer


class LLMTransformer(nn.Module):
    """
    Backward compatibility wrapper for SLGA Transformer.

    This class maintains the original API while delegating to the new
    modular SLGATransformer implementation.

    For new code, use src.models.SLGATransformer directly.

    Args:
        cfg: Configuration (supports both old Config and new ModelConfig)

    Example:
        >>> config = Config(vocab_size=50257, embed_dim=512)
        >>> model = LLMTransformer(config)
        >>> input_ids = torch.randint(0, 50257, (2, 10))
        >>> logits = model(input_ids)
    """

    def __init__(self, cfg: Config):
        super().__init__()

        # Issue deprecation warning
        warnings.warn(
            "LLMTransformer is deprecated. Use src.models.SLGATransformer instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Store config
        self.cfg = cfg

        # Create underlying model
        self._model = SLGATransformer(cfg)

        # Create text generator
        self._generator = TextGenerator(self._model)

        # Expose internal components for backward compatibility
        self.token_emb = self._model.embedding.token_emb.embedding
        self.pos_emb = self._model.embedding.pos_emb.embedding
        self.emb_dropout = self._model.embedding.dropout
        self.landmark_selector = self._model.landmark_selector
        self.blocks = self._model.blocks
        self.final_norm = self._model.final_norm
        self.lm_head = self._model.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_global_ids: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        global_weight: float = 1.0,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the transformer model.

        Delegates to SLGATransformer.forward().

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            cache_global_ids: Heuristic landmark indices of shape (batch, global_k)
            return_aux: If True, return auxiliary metrics
            global_weight: Weight for global attention component (0.0 to 1.0)

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
            aux: Optional auxiliary data dictionary
        """
        return self._model(
            input_ids=input_ids,
            cache_global_ids=cache_global_ids,
            return_aux=return_aux,
            global_weight=global_weight,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        cache_global_ids: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        stop_on_eos: bool = True,
        eos_token_id: int = 50256,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with various sampling strategies.

        Delegates to TextGenerator.generate().

        Args:
            input_ids: Prompt token IDs of shape (batch, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = greedy, >0.0 = stochastic)
            top_k: Top-K filtering threshold
            top_p: Nucleus (top-P) filtering threshold
            cache_global_ids: User-provided landmark indices
            seed: Random seed for reproducibility
            stop_on_eos: Stop when all sequences generate EOS
            eos_token_id: EOS token ID (default: GPT-2 EOS)
            repetition_penalty: Penalty for repeating tokens (>1.0 penalizes repeats)
            no_repeat_ngram_size: Size of n-grams to avoid repeating

        Returns:
            Generated token IDs of shape (batch, seq_len + generated_tokens)
        """
        return self._generator.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cache_global_ids=cache_global_ids,
            seed=seed,
            stop_on_eos=stop_on_eos,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count the number of model parameters."""
        return self._model.get_num_params(non_embedding=non_embedding)

    def estimate_mfu(
        self,
        fwdbwd_per_iter: int,
        dt: float,
        device: str = "cuda",
    ) -> float:
        """Estimate Model FLOPs Utilization (MFU)."""
        return self._model.estimate_mfu(
            fwdbwd_per_iter=fwdbwd_per_iter,
            dt=dt,
            device=device,
        )


# Maintain backward compatibility for direct imports
__all__ = [
    "Config",
    "LLMTransformer",
    "TransformerBlock",
    "FeedForward",
    # New modular exports (recommended)
    "SLGATransformer",
    "TextGenerator",
    "ModelFactory",
]
