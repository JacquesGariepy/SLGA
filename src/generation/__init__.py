"""
SLGA Text Generation Module

Provides modular text generation capabilities for SLGA models including:
- Configurable generation parameters
- Multiple sampling strategies (temperature, top-k, top-p)
- Repetition penalties and n-gram blocking
- Flexible stopping criteria
- Checkpoint loading utilities

Example usage:
    >>> from src.generation import TextGenerator, GenerationConfig
    >>> from src.legacy.model import LLMTransformer, Config
    >>> from transformers import AutoTokenizer
    >>>
    >>> # Load model and tokenizer
    >>> model = LLMTransformer(Config(...))
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>>
    >>> # Create generator
    >>> generator = TextGenerator(model, tokenizer)
    >>>
    >>> # Configure generation
    >>> config = GenerationConfig(
    ...     max_new_tokens=100,
    ...     temperature=0.8,
    ...     top_k=40,
    ...     top_p=0.95,
    ...     repetition_penalty=1.2,
    ... )
    >>>
    >>> # Generate text
    >>> output = generator.generate("Once upon a time", config=config)
"""

from .config import GenerationConfig
from .generator import TextGenerator
from .checkpoint import load_checkpoint, get_checkpoint_info
from .sampling import (
    apply_temperature,
    apply_top_k,
    apply_top_p,
    sample_next_token,
    get_sampling_info,
)
from .penalties import (
    apply_repetition_penalty,
    apply_no_repeat_ngram,
    get_penalty_info,
)
from .stopping import (
    StoppingCriteria,
    MaxLengthStoppingCriteria,
    EOSStoppingCriteria,
    CustomTokenStoppingCriteria,
    MultiStoppingCriteria,
    create_default_stopping_criteria,
)

__all__ = [
    # Main API
    "TextGenerator",
    "GenerationConfig",
    "load_checkpoint",
    "get_checkpoint_info",
    # Sampling
    "apply_temperature",
    "apply_top_k",
    "apply_top_p",
    "sample_next_token",
    "get_sampling_info",
    # Penalties
    "apply_repetition_penalty",
    "apply_no_repeat_ngram",
    "get_penalty_info",
    # Stopping
    "StoppingCriteria",
    "MaxLengthStoppingCriteria",
    "EOSStoppingCriteria",
    "CustomTokenStoppingCriteria",
    "MultiStoppingCriteria",
    "create_default_stopping_criteria",
]
