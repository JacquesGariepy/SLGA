"""
Transformer Entity Protocol Interfaces

This module defines Protocol interfaces for transformer building blocks including
transformer blocks, complete models, embeddings, and position encoding.
"""

from typing import Any, Dict, Optional, Protocol, Tuple

from torch import Tensor


class TransformerBlock(Protocol):
    """Protocol for a single transformer block.

    A transformer block consists of attention (SLGA) followed by feed-forward network,
    with residual connections and layer normalization (pre-norm architecture).
    """

    layer_idx: int

    def forward(
        self,
        x: Tensor,  # [batch, seq_len, embed_dim]
        cache_global: Optional[Tensor] = None,  # [batch, num_global, embed_dim]
        global_weight: float = 1.0,
    ) -> Tensor:  # [batch, seq_len, embed_dim]
        """Forward pass through transformer block.

        Args:
            x: Input embeddings [batch, seq_len, embed_dim]
            cache_global: Global landmark states [batch, num_global, embed_dim]
            global_weight: Weight for global attention (0.0 to 1.0)

        Returns:
            Output embeddings [batch, seq_len, embed_dim]

        Example:
            >>> block = TransformerBlock(cfg, layer_idx=0)
            >>> output = block(x, cache_global=landmarks, global_weight=0.5)
            >>> assert output.shape == x.shape

        Algorithm:
            1. Attention: x = x + SLGA(LayerNorm(x), cache_global, global_weight)
            2. FFN: x = x + FFN(LayerNorm(x))
            3. Return x

        Notes:
            - Uses pre-norm architecture (norm before sub-layer)
            - Residual connections around attention and FFN
            - Optional gradient checkpointing for memory efficiency
            - Layer index affects dilation in hierarchical attention
        """
        ...


class FeedForward(Protocol):
    """Protocol for feed-forward network (FFN) in transformer blocks.

    Standard two-layer MLP with GELU activation: Linear -> GELU -> Linear
    """

    def forward(
        self,
        x: Tensor,  # [batch, seq_len, embed_dim]
    ) -> Tensor:  # [batch, seq_len, embed_dim]
        """Forward pass through FFN.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]

        Returns:
            Output tensor [batch, seq_len, embed_dim]

        Example:
            >>> ffn = FeedForward(embed_dim=512, hidden_multiplier=4, dropout=0.1)
            >>> output = ffn(x)
            >>> assert output.shape == x.shape

        Algorithm:
            1. Project to hidden dimension: h = Linear(x)  # [batch, seq_len, hidden_dim]
            2. Non-linearity: h = GELU(h)
            3. Project back: output = Dropout(Linear(h))  # [batch, seq_len, embed_dim]

        Notes:
            - hidden_dim = embed_dim * hidden_multiplier (typically 4x)
            - GELU activation (smoother than ReLU, used in GPT/BERT)
            - Dropout after final projection for regularization
        """
        ...


class TransformerModel(Protocol):
    """Protocol for complete transformer language model.

    Full architecture: Embeddings -> N x TransformerBlock -> LayerNorm -> LM Head
    """

    def forward(
        self,
        input_ids: Tensor,  # [batch, seq_len]
        cache_global_ids: Optional[Tensor] = None,  # [batch, num_global]
        return_aux: bool = False,
        global_weight: float = 1.0,
    ) -> Tensor | Tuple[Tensor, Dict[str, Any]]:
        """Forward pass through complete transformer.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            cache_global_ids: Heuristic landmark indices [batch, num_global].
                             Positions (not tokens) to use as landmarks.
            return_aux: If True, return auxiliary data (scores, indices, gates)
            global_weight: Weight for global attention (0.0 to 1.0)

        Returns:
            If return_aux=False:
                logits: Output logits [batch, seq_len, vocab_size]
            If return_aux=True:
                (logits, aux_data) where aux_data is dict with:
                - 'landmark_scores': [batch, seq_len] selection probabilities
                - 'landmark_indices': [batch, num_global] selected positions
                - 'gate_values': [num_layers, batch, num_heads, seq_len] fusion gates

        Example:
            >>> model = LLMTransformer(cfg)
            >>> logits = model(input_ids)
            >>> assert logits.shape == (batch, seq_len, vocab_size)
            >>>
            >>> # With auxiliary data
            >>> logits, aux = model(input_ids, return_aux=True)
            >>> print(aux['landmark_indices'].shape)  # (batch, num_global)

        Algorithm:
            1. Embed: x = TokenEmbed(input_ids) + PositionEmbed(positions)
            2. Select landmarks:
               - If learned: indices, states = landmark_selector(x)
               - If heuristic: indices = cache_global_ids
            3. For each layer:
               - states = gather(x, indices)  # Update landmark states
               - x = TransformerBlock(x, cache_global=states, global_weight)
            4. Final: logits = LM_Head(LayerNorm(x))

        Notes:
            - Landmarks are updated at each layer (evolving representations)
            - Token and position embeddings are learned
            - Weight tying: token_embed.weight = lm_head.weight (fewer params)
            - global_weight enables curriculum learning (progressive warmup)
        """
        ...

    def generate(
        self,
        input_ids: Tensor,  # [batch, seq_len]
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        cache_global_ids: Optional[Tensor] = None,
        seed: Optional[int] = None,
        stop_on_eos: bool = True,
        eos_token_id: int = 50256,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: Optional[int] = None,
    ) -> Tensor:  # [batch, seq_len + num_generated]
        """Autoregressive generation with advanced sampling controls.

        Args:
            input_ids: Prompt tokens [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0=greedy, >0.0=stochastic)
            top_k: Top-K filtering (keep only top K logits)
            top_p: Nucleus sampling (keep top P cumulative probability)
            cache_global_ids: User-provided landmark positions
            seed: Random seed for reproducibility
            stop_on_eos: Stop when EOS token is generated
            eos_token_id: EOS token ID (default: GPT-2)
            repetition_penalty: Penalty for repeating tokens (>1.0 = less repetition)
            no_repeat_ngram_size: Block repeating n-grams

        Returns:
            Generated sequence [batch, seq_len + num_generated]

        Example:
            >>> model.eval()
            >>> prompt = tokenizer.encode("Once upon a time")
            >>> input_ids = torch.tensor([prompt])
            >>>
            >>> # Greedy decoding
            >>> output = model.generate(input_ids, temperature=0.0)
            >>>
            >>> # Nucleus sampling with repetition penalty
            >>> output = model.generate(
            ...     input_ids, temperature=0.8, top_p=0.9,
            ...     repetition_penalty=1.2, max_new_tokens=200
            ... )
            >>>
            >>> # Deterministic with seed
            >>> output = model.generate(input_ids, seed=42, temperature=0.7)

        Algorithm:
            For step in range(max_new_tokens):
                1. Forward pass: logits = model(input_ids, cache_global_ids)
                2. Extract next token logits: logits = logits[:, -1, :]
                3. Apply penalties:
                   - Repetition penalty on seen tokens
                   - N-gram blocking
                4. Apply sampling controls (in order):
                   - Temperature scaling
                   - Top-K filtering
                   - Top-P (nucleus) filtering
                5. Sample: next_token = sample(logits)
                6. Append: input_ids = cat([input_ids, next_token], dim=1)
                7. Check EOS and update finished mask
                8. Break if all sequences finished

        Notes:
            - Temperature=0.0: Deterministic greedy sampling
            - Temperature>0.0: Stochastic sampling (higher = more random)
            - Top-K: Keep only K highest logits, set rest to -inf
            - Top-P: Keep smallest set of logits with cumulative prob >= P
            - Correct order: Temperature -> Top-K -> Top-P -> Sample
            - Repetition penalty: Divide/multiply logits of seen tokens
            - N-gram blocking: Set logits to -inf for banned continuations
            - Batch-aware EOS: Track finished sequences independently
            - Landmark recalculation: Heuristic landmarks updated each step
        """
        ...

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count model parameters.

        Args:
            non_embedding: If True, exclude embedding parameters

        Returns:
            Total number of parameters

        Example:
            >>> model = LLMTransformer(cfg)
            >>> print(f"Total params: {model.get_num_params():,}")
            >>> print(f"Non-embed params: {model.get_num_params(non_embedding=True):,}")

        Notes:
            - Embedding parameters often excluded from reported model size
            - Weight tying means token_emb and lm_head share parameters
        """
        ...

    def estimate_mfu(
        self,
        fwdbwd_per_iter: int,
        dt: float,
        device: str = "cuda",
    ) -> float:
        """Estimate Model FLOPs Utilization (MFU).

        Args:
            fwdbwd_per_iter: Total examples per iteration (batch_size * grad_accum_steps)
            dt: Time per iteration in seconds
            device: Device name for peak FLOPs lookup

        Returns:
            MFU as fraction of peak theoretical FLOPs (0.0 to 1.0)

        Example:
            >>> # After training iteration
            >>> mfu = model.estimate_mfu(
            ...     fwdbwd_per_iter=32,  # batch_size=8, grad_accum=4
            ...     dt=0.5,  # 500ms per iteration
            ...     device="A100"
            ... )
            >>> print(f"MFU: {mfu*100:.1f}%")

        Notes:
            - MFU measures hardware efficiency
            - Typical values: 0.3-0.6 (30-60%) for well-optimized training
            - Higher is better (closer to theoretical peak)
            - Depends on batch size, model size, hardware, optimization
        """
        ...


class EmbeddingLayer(Protocol):
    """Protocol for embedding layers (token and position embeddings)."""

    num_embeddings: int
    embedding_dim: int

    def forward(
        self,
        indices: Tensor,  # [batch, seq_len]
    ) -> Tensor:  # [batch, seq_len, embedding_dim]
        """Embed token/position indices.

        Args:
            indices: Integer indices [batch, seq_len]

        Returns:
            Embeddings [batch, seq_len, embedding_dim]

        Example:
            >>> token_emb = nn.Embedding(vocab_size=50257, embedding_dim=512)
            >>> embeddings = token_emb(input_ids)
            >>> assert embeddings.shape == (batch, seq_len, 512)
        """
        ...


class PositionEncoder(Protocol):
    """Protocol for positional encoding/embedding.

    Adds positional information to token embeddings. Can be learned (embeddings)
    or fixed (sinusoidal encoding).
    """

    max_seq_len: int
    embed_dim: int

    def forward(
        self,
        positions: Tensor,  # [batch, seq_len]
    ) -> Tensor:  # [batch, seq_len, embed_dim]
        """Encode positions into embeddings.

        Args:
            positions: Position indices [batch, seq_len] in range [0, max_seq_len)

        Returns:
            Position embeddings [batch, seq_len, embed_dim]

        Example:
            >>> pos_encoder = LearnedPositionEmbedding(max_seq_len=2048, embed_dim=512)
            >>> positions = torch.arange(seq_len).expand(batch, seq_len)
            >>> pos_emb = pos_encoder(positions)
            >>> combined = token_emb + pos_emb  # Add to token embeddings

        Notes:
            - Learned: Trainable embedding matrix (more flexible)
            - Sinusoidal: Fixed pattern, generalizes better to longer sequences
            - Always added to token embeddings before transformer blocks
        """
        ...


__all__ = [
    "EmbeddingLayer",
    "FeedForward",
    "PositionEncoder",
    "TransformerBlock",
    "TransformerModel",
]
