"""
Configuration Value Objects

This module defines immutable configuration dataclasses for the SLGA architecture.
Value objects ensure type safety and validation of configuration parameters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class AttentionConfig:
    """Configuration for attention mechanisms in SLGA.

    Attributes:
        window_size: Size of local sliding window (default: 128)
        num_global_tokens: Number of global landmark tokens (default: 48)
        causal: Whether to apply causal masking (default: True)
        gated_fusion: Use learned gating for local-global fusion (default: True)
        dilation: Dilation factor for windows (default: 1, no dilation)
        diverse_topk: Encourage inter-head diversity in landmark selection (default: True)
        joint_normalization: Normalize local+global scores jointly (default: False, experimental)
        attn_dropout: Dropout rate on attention weights (default: 0.1)
        proj_dropout: Dropout rate on output projection (default: 0.1)

    Example:
        >>> cfg = AttentionConfig(
        ...     window_size=128,
        ...     num_global_tokens=48,
        ...     causal=True,
        ...     gated_fusion=True,
        ...     diverse_topk=True
        ... )
        >>> print(f"Window: {cfg.window_size}, Global: {cfg.num_global_tokens}")

    Notes:
        - Immutable (frozen=True) ensures configuration doesn't change during training
        - window_size: Larger = more local context, higher memory
        - num_global_tokens: Larger = more global coverage, higher compute
        - dilation: Progressive dilation per layer for hierarchical processing
        - diverse_topk: Prevents head degeneration (all heads selecting same landmarks)
    """

    window_size: int = 128
    num_global_tokens: int = 48
    causal: bool = True
    gated_fusion: bool = True
    dilation: int = 1
    diverse_topk: bool = True
    joint_normalization: bool = False
    attn_dropout: float = 0.1
    proj_dropout: float = 0.1

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {self.window_size}")
        if self.num_global_tokens <= 0:
            raise ValueError(f"num_global_tokens must be > 0, got {self.num_global_tokens}")
        if not 0.0 <= self.attn_dropout < 1.0:
            raise ValueError(f"attn_dropout must be in [0, 1), got {self.attn_dropout}")
        if not 0.0 <= self.proj_dropout < 1.0:
            raise ValueError(f"proj_dropout must be in [0, 1), got {self.proj_dropout}")
        if self.dilation < 1:
            raise ValueError(f"dilation must be >= 1, got {self.dilation}")


@dataclass(frozen=True)
class LandmarkConfig:
    """Configuration for landmark selection mechanisms.

    Attributes:
        learned: Use learned landmark selection (True) or heuristic (False)
        num_landmarks: Number of landmarks to select (default: 48)
        hidden_dim: Hidden dimension of scorer network (default: None, uses embed_dim // 2)
        temperature: Initial Gumbel temperature (default: 1.0)
        temperature_decay: Temperature decay factor per step (default: 0.999)
        min_temperature: Minimum temperature (default: 0.3)
        use_gumbel: Use Gumbel-Softmax (True) or straight-through (False) (default: True)
        selector_type: Type of selector ("learned", "positional", "hybrid") (default: "learned")

    Example:
        >>> cfg = LandmarkConfig(
        ...     learned=True,
        ...     num_landmarks=48,
        ...     use_gumbel=True,
        ...     temperature=1.0,
        ...     temperature_decay=0.999
        ... )

    Notes:
        - learned=True: Model selects landmarks based on content
        - learned=False: Use heuristic positions (regular, random, paragraph)
        - Gumbel-Softmax recommended for training from scratch
        - Straight-through faster but less stable early in training
        - Temperature annealing: smooth gradients -> sharp selections
    """

    learned: bool = True
    num_landmarks: int = 48
    hidden_dim: Optional[int] = None
    temperature: float = 1.0
    temperature_decay: float = 0.999
    min_temperature: float = 0.3
    use_gumbel: bool = True
    selector_type: str = "learned"  # "learned", "positional", "hybrid"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_landmarks <= 0:
            raise ValueError(f"num_landmarks must be > 0, got {self.num_landmarks}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if not 0.0 < self.temperature_decay <= 1.0:
            raise ValueError(f"temperature_decay must be in (0, 1], got {self.temperature_decay}")
        if self.min_temperature <= 0:
            raise ValueError(f"min_temperature must be > 0, got {self.min_temperature}")
        if self.selector_type not in ["learned", "positional", "hybrid"]:
            raise ValueError(
                f"selector_type must be 'learned', 'positional', or 'hybrid', got {self.selector_type}"
            )


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for transformer model architecture.

    Attributes:
        vocab_size: Vocabulary size (default: 50257, GPT-2)
        max_seq_len: Maximum sequence length (default: 2048)
        embed_dim: Embedding dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        ff_hidden_multiplier: FFN hidden dimension multiplier (default: 4)
        n_layers: Number of transformer layers (default: 12)
        dropout_rate: Dropout rate for embeddings and FFN (default: 0.1)
        dilated_windows: Progressive dilation per layer (default: True)
        grad_checkpointing: Use gradient checkpointing for memory efficiency (default: False)
        attention: Attention configuration
        landmark: Landmark selection configuration

    Example:
        >>> cfg = ModelConfig(
        ...     vocab_size=50257,
        ...     max_seq_len=2048,
        ...     embed_dim=512,
        ...     num_heads=8,
        ...     n_layers=12,
        ...     attention=AttentionConfig(window_size=128, num_global_tokens=48),
        ...     landmark=LandmarkConfig(learned=True, num_landmarks=48)
        ... )

    Notes:
        - embed_dim must be divisible by num_heads (head_dim = embed_dim / num_heads)
        - ff_hidden_dim = embed_dim * ff_hidden_multiplier (typically 4x)
        - grad_checkpointing trades compute for memory (slower but less memory)
        - dilated_windows: Lower layers dense, higher layers dilated (hierarchical)
    """

    vocab_size: int = 50257
    max_seq_len: int = 2048
    embed_dim: int = 512
    num_heads: int = 8
    ff_hidden_multiplier: int = 4
    n_layers: int = 12
    dropout_rate: float = 0.1
    dilated_windows: bool = True
    grad_checkpointing: bool = False
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    landmark: LandmarkConfig = field(default_factory=LandmarkConfig)

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {self.max_seq_len}")
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {self.embed_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {self.num_heads}")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.ff_hidden_multiplier <= 0:
            raise ValueError(f"ff_hidden_multiplier must be > 0, got {self.ff_hidden_multiplier}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {self.n_layers}")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.embed_dim // self.num_heads

    @property
    def ff_hidden_dim(self) -> int:
        """FFN hidden dimension."""
        return self.embed_dim * self.ff_hidden_multiplier


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training hyperparameters.

    Attributes:
        batch_size: Batch size per device (default: 32)
        gradient_accumulation_steps: Gradient accumulation steps (default: 1)
        max_epochs: Maximum training epochs (default: 10)
        max_steps: Maximum training steps (None = train for max_epochs)
        learning_rate: Peak learning rate (default: 3e-4)
        weight_decay: Weight decay for AdamW (default: 0.1)
        betas: Adam beta coefficients (default: (0.9, 0.95))
        warmup_steps: Learning rate warmup steps (default: 2000)
        lr_decay_schedule: LR schedule ("cosine", "linear", "constant") (default: "cosine")
        grad_clip_norm: Gradient clipping norm (default: 1.0)
        log_interval: Steps between logging (default: 10)
        eval_interval: Steps between evaluation (default: 500)
        save_interval: Steps between checkpointing (default: 1000)
        mixed_precision: Use mixed precision training (fp16/bf16) (default: "bf16")
        compile_model: Use torch.compile() for speedup (default: False)
        global_weight_schedule: Curriculum for global attention weight
            - "constant": Always 1.0
            - "linear": Ramp from 0.0 to 1.0 over warmup_steps
            - "exponential": Exponential ramp

    Example:
        >>> cfg = TrainingConfig(
        ...     batch_size=32,
        ...     learning_rate=3e-4,
        ...     weight_decay=0.1,
        ...     warmup_steps=2000,
        ...     max_epochs=10,
        ...     mixed_precision="bf16"
        ... )

    Notes:
        - Effective batch size = batch_size * gradient_accumulation_steps * num_gpus
        - mixed_precision="bf16" recommended for modern GPUs (A100, 4090, etc.)
        - grad_clip_norm prevents exploding gradients (especially early in training)
        - cosine LR schedule: smooth decay to 0.1 * peak_lr
        - global_weight_schedule: Curriculum learning for SLGA (start local, add global)
    """

    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_epochs: int = 10
    max_steps: Optional[int] = None
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    warmup_steps: int = 2000
    lr_decay_schedule: str = "cosine"
    grad_clip_norm: float = 1.0
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    mixed_precision: str = "bf16"  # "fp16", "bf16", "none"
    compile_model: bool = False
    global_weight_schedule: str = "linear"  # "constant", "linear", "exponential"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"gradient_accumulation_steps must be > 0, got {self.gradient_accumulation_steps}"
            )
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be > 0, got {self.max_epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.lr_decay_schedule not in ["cosine", "linear", "constant"]:
            raise ValueError(
                f"lr_decay_schedule must be 'cosine', 'linear', or 'constant', got {self.lr_decay_schedule}"
            )
        if self.grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be > 0, got {self.grad_clip_norm}")
        if self.mixed_precision not in ["fp16", "bf16", "none"]:
            raise ValueError(
                f"mixed_precision must be 'fp16', 'bf16', or 'none', got {self.mixed_precision}"
            )
        if self.global_weight_schedule not in ["constant", "linear", "exponential"]:
            raise ValueError(
                f"global_weight_schedule must be 'constant', 'linear', or 'exponential', got {self.global_weight_schedule}"
            )

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size per update (before multi-GPU scaling)."""
        return self.batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data loading and preprocessing.

    Attributes:
        dataset_name: Dataset identifier (HuggingFace name or local path)
        dataset_subset: Optional dataset subset/config (default: None)
        tokenizer_name: Tokenizer identifier (default: "gpt2")
        max_length: Maximum sequence length for training (default: 512)
        text_key: Key in dataset containing text (default: "text")
        collator_type: Collator type ("local", "local_global") (default: "local")
        collator_kwargs: Additional kwargs for collator (default: {})
        num_workers: DataLoader worker processes (default: 4)
        prefetch_factor: Batches to prefetch per worker (default: 2)
        pin_memory: Pin memory for faster GPU transfer (default: True)

    Example:
        >>> cfg = DataConfig(
        ...     dataset_name="openwebtext",
        ...     tokenizer_name="gpt2",
        ...     max_length=512,
        ...     collator_type="local_global",
        ...     collator_kwargs={
        ...         "global_every": 128,
        ...         "max_global": 48,
        ...         "strategy": "regular"
        ...     }
        ... )

    Notes:
        - dataset_name can be HuggingFace dataset or local path
        - collator_type="local" for learned landmarks, "local_global" for heuristic
        - num_workers>0 enables parallel data loading (faster but more memory)
        - pin_memory=True recommended for GPU training
    """

    dataset_name: str = "openwebtext"
    dataset_subset: Optional[str] = None
    tokenizer_name: str = "gpt2"
    max_length: int = 512
    text_key: str = "text"
    collator_type: str = "local"
    collator_kwargs: Dict[str, Any] = field(default_factory=dict)
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if not self.dataset_name:
            raise ValueError("dataset_name must be non-empty")
        if not self.tokenizer_name:
            raise ValueError("tokenizer_name must be non-empty")
        if self.max_length <= 0:
            raise ValueError(f"max_length must be > 0, got {self.max_length}")
        if self.collator_type not in ["local", "local_global", "tfidf"]:
            raise ValueError(
                f"collator_type must be 'local', 'local_global', or 'tfidf', got {self.collator_type}"
            )
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.prefetch_factor < 1:
            raise ValueError(f"prefetch_factor must be >= 1, got {self.prefetch_factor}")


__all__ = [
    "AttentionConfig",
    "DataConfig",
    "LandmarkConfig",
    "ModelConfig",
    "TrainingConfig",
]
