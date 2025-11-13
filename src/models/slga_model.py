"""SLGA Transformer model orchestrator."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from src.core.layers.embedding import EmbeddingLayer
from src.core.layers.transformer_block import TransformerBlock
from src.legacy.landmarks import LearnableLandmarkSelector
from src.models.config import ModelConfig


class SLGATransformer(nn.Module):
    """Complete Causal Transformer LLM with SLGA (Sparse Local-Global Attention).

    Architecture Overview:
    1. Token + Position Embeddings
    2. N × TransformerBlock (SLGA Attention + FFN)
    3. Final LayerNorm
    4. LM Head (projection to vocabulary)

    Features:
    - Learned landmarks (optional) via LearnableLandmarkSelector
    - Heuristic landmarks via cache_global_ids
    - Gradient checkpointing for memory efficiency
    - Composition over inheritance for clean architecture

    Args:
        config: Model configuration

    Example:
        >>> from src.models.config import ModelConfig
        >>> config = ModelConfig(vocab_size=50257, embed_dim=512, num_heads=8)
        >>> model = SLGATransformer(config)
        >>> input_ids = torch.randint(0, 50257, (2, 10))
        >>> logits = model(input_ids)
        >>> logits.shape
        torch.Size([2, 10, 50257])
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # Validate configuration consistency
        config.validate_consistency()

        self.config = config

        # Embedding layer (token + position)
        self.embedding = EmbeddingLayer(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            embed_dim=config.embed_dim,
            dropout_rate=config.dropout_rate,
        )

        # Landmark selector (if learned landmarks are enabled)
        # Learns to select important global tokens dynamically
        if config.learned_landmarks:
            self.landmark_selector = LearnableLandmarkSelector(
                embed_dim=config.embed_dim,
                num_landmarks=config.global_k * 2,  # Select more, restrict to top-K in SLGA
            )
        else:
            self.landmark_selector = None

        # Stack of Transformer blocks
        # Each block contains SLGA attention and FFN with progressive dilation
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)]
        )

        # Final layer norm and language model head
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Tie embeddings: share weights between token_emb and lm_head
        # This reduces parameters and improves performance
        self.lm_head.weight = self.embedding.token_emb.embedding.weight

        # Initialize weights using GPT-2 style initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 style weight initialization for stable training."""
        if isinstance(module, nn.Linear):
            # Normal initialization for linear layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Normal initialization for embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # Standard initialization for layer norms
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_global_ids: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        global_weight: float = 1.0,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass through the complete transformer model.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            cache_global_ids: Heuristic landmark indices of shape (batch, global_k)
            return_aux: If True, return auxiliary metrics (landmark scores, indices, gate values)
            global_weight: Weight for global attention component (0.0 to 1.0)

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
            aux: Optional auxiliary data dictionary
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Compute embeddings
        x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)

        # Initial landmark selection
        landmark_indices = None
        landmark_scores = None

        if self.landmark_selector is not None:
            # Learned landmarks: select indices once using Gumbel-Softmax during training
            landmark_indices, _, landmark_scores = self.landmark_selector(
                x, use_gumbel=self.training
            )
            # landmark_indices: (batch, global_k)
        elif cache_global_ids is not None:
            # Heuristic landmarks: use provided indices
            landmark_indices = cache_global_ids  # (batch, global_k)

        # Pass through transformer blocks
        # Update landmarks at each layer so they evolve with the sequence
        gate_values_layers = []

        for block in self.blocks:
            # Extract current landmark states from updated embeddings
            if landmark_indices is not None:
                batch_cur, seq_cur, embed_dim = x.shape
                num_landmarks = landmark_indices.size(1)

                # Clamp indices to avoid out-of-bounds
                landmark_indices_safe = torch.clamp(landmark_indices, 0, seq_cur - 1)
                landmark_indices_exp = landmark_indices_safe.unsqueeze(-1).expand(
                    batch_cur, num_landmarks, embed_dim
                )
                landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)
            else:
                landmark_states = None

            # Forward pass through block with updated landmarks
            x = block(x, cache_global=landmark_states, global_weight=global_weight)

            # Collect gating metrics if available (for monitoring attention behavior)
            monitor = getattr(block.attn, "last_monitor", None)
            if monitor and monitor.get("gate_scalar") is not None:
                gate_values_layers.append(monitor["gate_scalar"])

        # Final normalization and language model projection
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        if return_aux:
            aux = {
                "landmark_scores": landmark_scores,  # Softmax scores (batch, seq_len)
                "landmark_indices": landmark_indices,  # Selected indices (batch, global_k)
            }

            if gate_values_layers:
                gate_stack = torch.stack(gate_values_layers, dim=0)
                aux["gate_values"] = gate_stack.detach()

            return logits, aux

        return logits

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count the number of model parameters.

        Args:
            non_embedding: If True, exclude embedding parameters

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Exclude embedding parameters (often not counted in model size)
            n_params -= self.embedding.pos_emb.embedding.weight.numel()
            n_params -= self.embedding.token_emb.embedding.weight.numel()
        return n_params

    def estimate_mfu(
        self,
        fwdbwd_per_iter: int,
        dt: float,
        device: str = "cuda",
    ) -> float:
        """Estimate Model FLOPs Utilization (MFU) as percentage of peak theoretical FLOPs.

        Args:
            fwdbwd_per_iter: Number of examples per iteration (batch_size * accum_steps)
            dt: Time per iteration in seconds
            device: Device name for peak FLOPs lookup

        Returns:
            MFU as a fraction (0.0 to 1.0)
        """
        # Approximate FLOPs per forward pass
        L = self.config.max_seq_len
        N = self.config.n_layers
        D = self.config.embed_dim

        # Rough approximation: attention + FFN FLOPs
        flops_per_token = 6 * N * D * D
        flops_per_fwdbwd = fwdbwd_per_iter * L * flops_per_token * 3  # ×3 for backward

        flops_per_sec = flops_per_fwdbwd / dt

        # Peak theoretical FLOPs based on device
        if "3090" in device or "RTX 3090" in device:
            peak_flops = 35.6e12  # RTX 3090 peak TFLOPs
        elif "4090" in device:
            peak_flops = 82.6e12  # RTX 4090 peak TFLOPs
        elif "A100" in device:
            peak_flops = 312e12  # A100 peak TFLOPs
        else:
            peak_flops = 100e12  # Default fallback

        mfu = flops_per_sec / peak_flops
        return mfu


__all__ = ["SLGATransformer"]
