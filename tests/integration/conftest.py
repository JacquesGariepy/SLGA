"""Shared fixtures for integration tests."""

import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from src.models import ModelConfig, SLGATransformer
from src.legacy.data import get_tokenizer, CollatorLocal, CollatorLocalGlobal


@pytest.fixture
def sample_texts():
    """Create sample text data for testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence that focuses on data.",
        "PyTorch is a popular deep learning framework used for neural network training.",
        "Natural language processing enables computers to understand human language.",
        "Transformers revolutionized NLP with their attention mechanism architecture.",
    ]


@pytest.fixture
def long_sample_texts():
    """Create longer sample texts for testing."""
    return [
        " ".join([f"This is sentence {i} of a long document." for i in range(50)]),
        " ".join([f"Another example paragraph with sentence {i}." for i in range(50)]),
        " ".join([f"Testing long context with statement {i}." for i in range(50)]),
    ]


@pytest.fixture
def tokenizer():
    """Get GPT-2 tokenizer."""
    return get_tokenizer("gpt2")


@pytest.fixture
def tiny_config():
    """Create tiny model config for fast testing."""
    return ModelConfig(
        vocab_size=50257,
        max_seq_len=256,
        embed_dim=128,
        num_heads=4,
        ff_hidden_multiplier=2,
        n_layers=2,
        dropout_rate=0.1,
        local_window=32,
        global_k=8,
        learned_landmarks=True,
    )


@pytest.fixture
def tiny_model(tiny_config):
    """Create tiny model instance."""
    return SLGATransformer(tiny_config)


@pytest.fixture
def collator_local(tokenizer):
    """Create local-only collator."""
    return CollatorLocal(tokenizer, max_length=128)


@pytest.fixture
def collator_global(tokenizer):
    """Create local+global collator with heuristic landmarks."""
    return CollatorLocalGlobal(
        tokenizer,
        max_length=128,
        global_every=32,
        max_global=8,
        strategy="regular"
    )


@pytest.fixture
def sample_dataset(sample_texts):
    """Create HuggingFace dataset from sample texts."""
    return Dataset.from_dict({"text": sample_texts})


@pytest.fixture
def device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
