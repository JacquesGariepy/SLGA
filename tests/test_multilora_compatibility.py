"""Test MultiLoRA compatibility for LoRA adapters.

This test verifies that SLGA LoRA adapters can be exported in a format
compatible with the MultiLoRA inference server at D:\\ai\\MultiLoRA.

Format spec based on:
- MultiLoRA/adapters/*.json configs
- MultiLoRA/docs/lora_composition.md
"""

import os
import sys
import json
import tempfile
import shutil
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lora import (
    LoRALinear,
    LoRAConfig,
    apply_lora_to_model,
    save_multilora_adapter,
    load_multilora_adapter,
    list_multilora_adapters,
    convert_to_multilora_format,
    MultiLoRAAdapterConfig,
    save_lora_weights,
)


class DummyModel(nn.Module):
    """Simple model for testing LoRA."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 64)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'qkv_proj': nn.Linear(64, 192),
                'out_proj': nn.Linear(64, 64),
                'fc1': nn.Linear(64, 256),
                'fc2': nn.Linear(256, 64),
            })
            for _ in range(2)
        ])
        self.output = nn.Linear(64, 1000)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer['fc2'](torch.relu(layer['fc1'](x)))
        return self.output(x)


class TestMultiLoRAConfig:
    """Test MultiLoRAAdapterConfig dataclass."""

    def test_config_creation(self):
        """Test creating a config."""
        config = MultiLoRAAdapterConfig(
            adapter_name="test-adapter",
            rank=8,
            alpha=16.0,
            target_modules=["qkv_proj", "out_proj"],
            base_model="slga-v2",
        )

        assert config.adapter_name == "test-adapter"
        assert config.rank == 8
        assert config.alpha == 16.0
        assert "qkv_proj" in config.target_modules

    def test_config_to_dict(self):
        """Test config serialization."""
        config = MultiLoRAAdapterConfig(
            adapter_name="test-adapter",
            rank=8,
            alpha=16.0,
        )

        d = config.to_dict()
        assert d["adapter_name"] == "test-adapter"
        assert d["rank"] == 8
        assert d["version"] == "1.0"
        assert d["framework"] == "slga"

    def test_config_json_roundtrip(self):
        """Test JSON save/load."""
        config = MultiLoRAAdapterConfig(
            adapter_name="roundtrip-test",
            rank=16,
            alpha=32.0,
            metadata={"task": "code-generation"},
            training_steps=1000,
            final_loss=0.5,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            config.to_json(path)
            loaded = MultiLoRAAdapterConfig.from_json(path)

            assert loaded.adapter_name == "roundtrip-test"
            assert loaded.rank == 16
            assert loaded.alpha == 32.0
            assert loaded.metadata["task"] == "code-generation"
            assert loaded.training_steps == 1000
            assert loaded.final_loss == 0.5
        finally:
            os.unlink(path)


class TestMultiLoRAExport:
    """Test exporting adapters in MultiLoRA format."""

    def test_save_multilora_adapter(self):
        """Test saving adapter in MultiLoRA format."""
        # Create model with LoRA
        model = DummyModel()
        model = apply_lora_to_model(
            model,
            rank=8,
            alpha=16.0,
            target_modules=["qkv_proj", "out_proj", "fc1", "fc2"],
            dropout=0.05,
        )

        # Train a bit (just to have non-zero weights)
        for module in model.modules():
            if isinstance(module, LoRALinear):
                nn.init.normal_(module.lora_A, std=0.01)
                nn.init.normal_(module.lora_B, std=0.01)

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = save_multilora_adapter(
                model=model,
                adapter_dir=tmpdir,
                adapter_name="test-adapter",
                base_model="slga-v2",
                metadata={"task": "testing"},
                training_steps=100,
                final_loss=0.5,
                dataset="test-dataset",
            )

            # Verify structure
            assert os.path.isdir(adapter_path)
            assert os.path.exists(os.path.join(adapter_path, "config.json"))
            assert os.path.exists(os.path.join(adapter_path, "adapter_weights.pt"))

            # Load and verify config
            config = MultiLoRAAdapterConfig.from_json(
                os.path.join(adapter_path, "config.json")
            )
            assert config.adapter_name == "test-adapter"
            assert config.rank == 8
            assert config.alpha == 16.0
            assert config.base_model == "slga-v2"
            assert config.training_steps == 100
            assert config.final_loss == 0.5
            assert config.dataset == "test-dataset"
            assert config.metadata["task"] == "testing"

            # Verify weights
            weights = torch.load(os.path.join(adapter_path, "adapter_weights.pt"))
            assert len(weights) > 0
            # Should have lora_A and lora_B for each layer
            lora_a_keys = [k for k in weights.keys() if "lora_A" in k]
            lora_b_keys = [k for k in weights.keys() if "lora_B" in k]
            assert len(lora_a_keys) == len(lora_b_keys)
            assert len(lora_a_keys) == 8  # 4 modules * 2 layers

    def test_load_multilora_adapter(self):
        """Test loading adapter from MultiLoRA format."""
        # Create and save
        model = DummyModel()
        model = apply_lora_to_model(
            model,
            rank=8,
            alpha=16.0,
            target_modules=["qkv_proj", "out_proj", "fc1", "fc2"],
        )

        # Set specific weights
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.data.fill_(0.1)
                module.lora_B.data.fill_(0.2)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            adapter_path = save_multilora_adapter(
                model=model,
                adapter_dir=tmpdir,
                adapter_name="load-test",
            )

            # Create fresh model
            model2 = DummyModel()
            model2 = apply_lora_to_model(
                model2,
                rank=8,
                alpha=16.0,
                target_modules=["qkv_proj", "out_proj", "fc1", "fc2"],
            )

            # Verify weights are different
            for module in model2.modules():
                if isinstance(module, LoRALinear):
                    assert not torch.allclose(module.lora_A, torch.full_like(module.lora_A, 0.1))

            # Load
            config = load_multilora_adapter(model2, adapter_path)

            # Verify weights match
            for module in model2.modules():
                if isinstance(module, LoRALinear):
                    assert torch.allclose(module.lora_A, torch.full_like(module.lora_A, 0.1))
                    assert torch.allclose(module.lora_B, torch.full_like(module.lora_B, 0.2))

            assert config.adapter_name == "load-test"

    def test_list_multilora_adapters(self):
        """Test listing adapters in a directory."""
        model = DummyModel()
        model = apply_lora_to_model(model, rank=8, target_modules=["qkv_proj"])

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save multiple adapters
            save_multilora_adapter(model, tmpdir, "adapter-1", training_steps=100)
            save_multilora_adapter(model, tmpdir, "adapter-2", training_steps=200)
            save_multilora_adapter(model, tmpdir, "adapter-3", training_steps=300)

            # List
            adapters = list_multilora_adapters(tmpdir)

            assert len(adapters) == 3
            names = {a.adapter_name for a in adapters}
            assert names == {"adapter-1", "adapter-2", "adapter-3"}


class TestMultiLoRAConversion:
    """Test converting existing LoRA weights to MultiLoRA format."""

    def test_convert_to_multilora_format(self):
        """Test converting standard .pt weights to MultiLoRA format."""
        model = DummyModel()
        model = apply_lora_to_model(model, rank=8, target_modules=["qkv_proj", "out_proj"])

        # Set weights
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.data.fill_(0.5)
                module.lora_B.data.fill_(0.3)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save in standard format
            std_path = os.path.join(tmpdir, "standard_lora.pt")
            save_lora_weights(model, std_path)

            # Convert to MultiLoRA
            adapter_path = convert_to_multilora_format(
                lora_weights_path=std_path,
                output_dir=tmpdir,
                adapter_name="converted-adapter",
                rank=8,
                alpha=16.0,
                base_model="slga-v2",
            )

            # Verify
            assert os.path.exists(os.path.join(adapter_path, "config.json"))
            assert os.path.exists(os.path.join(adapter_path, "adapter_weights.pt"))

            config = MultiLoRAAdapterConfig.from_json(
                os.path.join(adapter_path, "config.json")
            )
            assert config.adapter_name == "converted-adapter"
            assert config.rank == 8
            assert config.alpha == 16.0


class TestMultiLoRAFormat:
    """Test that the format matches MultiLoRA server expectations."""

    def test_config_format_matches_multilora(self):
        """Verify config format matches MultiLoRA/adapters/*.json"""
        config = MultiLoRAAdapterConfig(
            adapter_name="creative-style",
            rank=8,
            alpha=16.0,
            target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
            base_model="tiny-gpt2",
            lora_dropout=0.05,
        )

        d = config.to_dict()

        # These fields are required by MultiLoRA
        assert "adapter_name" in d
        assert "rank" in d
        assert "alpha" in d
        assert "target_modules" in d
        assert "base_model" in d
        assert "lora_dropout" in d

        # Verify types
        assert isinstance(d["adapter_name"], str)
        assert isinstance(d["rank"], int)
        assert isinstance(d["alpha"], float)
        assert isinstance(d["target_modules"], list)
        assert isinstance(d["base_model"], str)
        assert isinstance(d["lora_dropout"], float)

    def test_weights_format(self):
        """Verify weights format is compatible."""
        model = DummyModel()
        model = apply_lora_to_model(model, rank=8, target_modules=["qkv_proj"])

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = save_multilora_adapter(model, tmpdir, "test")
            weights = torch.load(os.path.join(adapter_path, "adapter_weights.pt"))

            # Verify key format: layer_name.lora_A and layer_name.lora_B
            for key in weights.keys():
                assert ".lora_A" in key or ".lora_B" in key

            # Verify tensors
            for key, value in weights.items():
                assert isinstance(value, torch.Tensor)
                # lora_A: (rank, d_in), lora_B: (d_out, rank)
                assert len(value.shape) == 2


def run_tests():
    """Run all tests."""
    import traceback

    test_classes = [
        TestMultiLoRAConfig,
        TestMultiLoRAExport,
        TestMultiLoRAConversion,
        TestMultiLoRAFormat,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"  {test_class.__name__}")
        print(f"{'='*60}")

        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
                    print(f"  ✅ {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  ❌ {method_name}: {e}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
