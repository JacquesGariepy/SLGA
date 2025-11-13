"""
Shared pytest fixtures for all SLGA2 tests.

This conftest.py provides shared fixtures used across all test modules.
"""

import pytest
import torch
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def device():
    """Get the appropriate device (CUDA if available, else CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def cpu_device():
    """Force CPU device for specific tests."""
    return "cpu"


@pytest.fixture
def tokenizer():
    """Create a shared tokenizer instance."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("gpt2")
    except ImportError:
        pytest.skip("transformers not installed")


@pytest.fixture
def sample_config():
    """Create a sample model configuration for testing."""
    from models.config import ModelConfig

    return ModelConfig(
        vocab_size=50257,
        d_model=256,
        num_heads=4,
        n_layers=2,
        max_seq_len=128,
        dropout=0.1,
        attention_dropout=0.1
    )


@pytest.fixture
def sample_slga_config():
    """Create a sample SLGA configuration for testing."""
    return {
        "embed_dim": 256,
        "num_heads": 4,
        "local_window": 64,
        "global_k": 16,
        "attn_drop": 0.1,
        "proj_drop": 0.1,
        "causal": True,
        "gated_fusion": True
    }


@pytest.fixture
def sample_batch(tokenizer):
    """Create a sample batch of tokenized text."""
    texts = [
        "Hello world, this is a test sentence.",
        "Another test sentence for validation purposes."
    ]
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")


@pytest.fixture
def sample_input_tensor(device):
    """Create a sample input tensor for testing."""
    batch_size = 2
    seq_len = 128
    embed_dim = 256

    tensor = torch.randn(batch_size, seq_len, embed_dim)
    return tensor.to(device)


@pytest.fixture
def small_input_tensor():
    """Create a small input tensor for quick tests."""
    return torch.randn(2, 32, 128)


@pytest.fixture
def random_seed():
    """Set a random seed for reproducible tests."""
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed


@pytest.fixture
def slga_module(sample_slga_config):
    """Create a basic SLGA module for testing."""
    from legacy.slga import SLGAModule
    return SLGAModule(**sample_slga_config)


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create a temporary directory for checkpoint testing."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def sample_checkpoint_state():
    """Create a sample checkpoint state dict."""
    return {
        "model_state_dict": {
            "embedding.weight": torch.randn(50257, 256),
            "layer.0.weight": torch.randn(256, 256),
        },
        "optimizer_state_dict": {},
        "epoch": 5,
        "global_step": 1000,
        "best_val_loss": 2.5,
    }


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, 50257, (128,)),
                "attention_mask": torch.ones(128),
                "labels": torch.randint(0, 50257, (128,))
            }

    return MockDataset()


@pytest.fixture(autouse=True)
def reset_torch_state():
    """Reset torch state before each test."""
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    yield
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# Pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "regression: marks tests as regression tests for bug fixes"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "property: marks tests as property-based tests"
    )
