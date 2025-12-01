from .model import Config, LLMTransformer, TransformerBlock, FeedForward, RMSNorm
from .slga import SLGAModule, RotaryEmbedding
from .landmarks import (
    LearnableLandmarkSelector,
    PositionalLandmarkSelector,
    HybridLandmarkSelector,
    landmark_spacing_loss,
    landmark_diversity_loss,
    landmark_sparsity_loss,
)
from .kv_cache import KVCache

# Flash Attention (optional)
try:
    from .flash_attention import (
        FLASH_ATTN_AVAILABLE,
        FLASH_ATTN_VERSION,
        flash_attention_forward,
        flash_attention_local_window,
        FlashAttentionModule,
        benchmark_flash_attention,
        get_flash_attention_info,
    )
    _FLASH_IMPORTS = [
        "FLASH_ATTN_AVAILABLE",
        "FLASH_ATTN_VERSION",
        "flash_attention_forward",
        "flash_attention_local_window",
        "FlashAttentionModule",
        "benchmark_flash_attention",
        "get_flash_attention_info",
    ]
except ImportError:
    _FLASH_IMPORTS = []

__version__ = "2.0.0"
__all__ = [
    # Model
    "Config",
    "LLMTransformer",
    "TransformerBlock",
    "FeedForward",
    "RMSNorm",
    # Attention
    "SLGAModule",
    "RotaryEmbedding",
    # Landmarks
    "LearnableLandmarkSelector",
    "PositionalLandmarkSelector",
    "HybridLandmarkSelector",
    "landmark_spacing_loss",
    "landmark_diversity_loss",
    "landmark_sparsity_loss",
    # KV Cache
    "KVCache",
] + _FLASH_IMPORTS