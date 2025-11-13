"""
Legacy Compatibility Layer

DEPRECATED: These modules provide backward compatibility for old code.
All new code should import from the new modular structure.

Migration Guide:
- src.legacy.slga.SLGAModule → src.core.attention.slga.SLGAAttention
- src.legacy.model.LLMTransformer → src.models.slga_model.SLGATransformer
- src.legacy.data → src.data.loaders, src.data.tokenizers, src.data.collators

These wrappers will be removed in version 3.0.
"""

# Re-export for backward compatibility
from .slga import SLGAModule
from .model import LLMTransformer, Config
from .data import get_tokenizer, load_text_dataset, CollatorLocal, CollatorLocalGlobal

__all__ = [
    'SLGAModule',
    'LLMTransformer',
    'Config',
    'get_tokenizer',
    'load_text_dataset',
    'CollatorLocal',
    'CollatorLocalGlobal',
]
