# Legacy Compatibility Layer

**Status:** DEPRECATED

These modules provide backward compatibility for old code that imports from the root `src/` directory.

## Deprecation Notice

All new code should import from the new modular structure:

### Migration Guide

#### SLGA Module
```python
# OLD (deprecated)
from src.slga import SLGAModule

# NEW (recommended)
from src.core.attention.slga import SLGAAttention
```

#### Model Module
```python
# OLD (deprecated)
from src.model import LLMTransformer, Config

# NEW (recommended)
from src.models.slga_model import SLGATransformer
from src.models.config import ModelConfig
```

#### Data Module
```python
# OLD (deprecated)
from src.data import get_tokenizer, load_text_dataset, CollatorLocal

# NEW (recommended)
from src.data.tokenizers.tokenizer_wrapper import TokenizerWrapper
from src.data.loaders.text_dataset import TextDatasetLoader
from src.data.collators.language_modeling_collator import LanguageModelingCollator
```

## Removal Timeline

- **Version 2.x:** Legacy wrappers maintained with deprecation warnings
- **Version 3.0:** Legacy wrappers will be removed

## Files in This Directory

- `slga.py` - Backward compatibility wrapper for SLGAModule
- `model.py` - Backward compatibility wrapper for LLMTransformer
- `data.py` - Backward compatibility wrappers for data utilities
- `__init__.py` - Package exports for convenience

For full migration documentation, see: `docs/MIGRATION_GUIDE.md`
