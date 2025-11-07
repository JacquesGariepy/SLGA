"""
Test suite for model architecture
Tests model initialization, forward pass, and configuration
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
import json

# Add src to path and setup package imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path.parent))

# Import with package name
from src.model import LLMTransformer, Config as SLGAPlusConfig


class TestSLGAPlusConfig:
    """Test model configuration"""

    def test_default_config(self):
        """Test configuration par défaut"""
        config = SLGAPlusConfig()

        assert config.vocab_size == 50257
        assert config.n_layers == 12
        assert config.num_heads == 8
        assert config.embed_dim == 512

    def test_custom_config(self):
        """Test configuration personnalisée"""
        config = SLGAPlusConfig(
            vocab_size=32000,
            n_layers=6,
            num_heads=8,
            embed_dim=512
        )

        assert config.vocab_size == 32000
        assert config.n_layers == 6
        assert config.num_heads == 8
        assert config.embed_dim == 512

    def test_config_validation(self):
        """Test validation des paramètres"""
        # Config is a dataclass without validation, so we test model validation
        config = SLGAPlusConfig(embed_dim=513, num_heads=8)

        # Model should raise error during initialization
        with pytest.raises((AssertionError, ValueError, RuntimeError)):
            model = LLMTransformer(config)

    def test_config_serialization(self):
        """Test sérialisation/désérialisation"""
        config1 = SLGAPlusConfig(vocab_size=32000, n_layers=6)

        # Serialize using dataclasses
        from dataclasses import asdict
        config_dict = asdict(config1)

        # Deserialize
        config2 = SLGAPlusConfig(**config_dict)

        assert config1.vocab_size == config2.vocab_size
        assert config1.n_layers == config2.n_layers


class TestLLMTransformer:
    """Test model architecture"""

    def test_model_initialization(self):
        """Test initialisation du modèle"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)

        # Verify model was created successfully
        assert model is not None
        assert isinstance(model, LLMTransformer)

        # Verify key components exist
        assert hasattr(model, 'token_emb')
        assert hasattr(model, 'blocks')
        assert hasattr(model, 'lm_head')

    def test_model_forward(self):
        """Test forward pass basique"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)
        model.eval()

        # Input: batch_size=2, sequence_length=64
        idx = torch.randint(0, 1000, (2, 64))

        with torch.no_grad():
            logits = model(idx)

        # Output shape: (batch, seq_len, vocab_size)
        assert logits.shape == (2, 64, 1000)

    def test_model_with_targets(self):
        """Test forward pass avec targets (calcul de loss)"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)

        idx = torch.randint(0, 1000, (2, 64))
        targets = torch.randint(0, 1000, (2, 64))

        # Model doesn't take targets directly, compute loss manually
        logits = model(idx)
        assert logits.shape == (2, 64, 1000)

        # Compute cross entropy loss
        loss = F.cross_entropy(logits.view(-1, 1000), targets.view(-1))
        assert loss is not None
        assert loss.item() > 0

    def test_model_generate(self):
        """Test génération de texte"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)
        model.eval()

        # Start with a prompt
        idx = torch.randint(0, 1000, (1, 10))

        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=20)

        # Output should be longer than input
        assert output.shape[1] == 30  # 10 + 20

    def test_model_parameter_count(self):
        """Test nombre de paramètres"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params == total_params  # All params should be trainable

    def test_model_gradient_flow(self):
        """Test que les gradients se propagent"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)

        idx = torch.randint(0, 1000, (2, 64))
        targets = torch.randint(0, 1000, (2, 64))

        logits = model(idx)
        loss = F.cross_entropy(logits.view(-1, 1000), targets.view(-1))
        loss.backward()

        # Vérifier que les gradients existent pour au moins quelques paramètres
        grad_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

        assert grad_count > 0, "At least some parameters should have gradients"

    def test_model_eval_mode(self):
        """Test comportement en mode eval"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128,
            dropout_rate=0.1
        )

        model = LLMTransformer(config)
        idx = torch.randint(0, 1000, (2, 64))

        # Train mode
        model.train()
        logits_train = model(idx)

        # Eval mode
        model.eval()
        with torch.no_grad():
            logits_eval = model(idx)

        # Shapes should match
        assert logits_train.shape == logits_eval.shape


class TestModelConfiguration:
    """Test different model configurations"""

    def test_small_model(self):
        """Test configuration small"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=4,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)
        idx = torch.randint(0, 1000, (1, 64))

        with torch.no_grad():
            logits = model(idx)

        assert logits.shape == (1, 64, 1000)

    def test_medium_model(self):
        """Test configuration medium"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=8,
            num_heads=8,
            embed_dim=512,
            max_seq_len=256
        )

        model = LLMTransformer(config)
        idx = torch.randint(0, 1000, (1, 128))

        with torch.no_grad():
            logits = model(idx)

        assert logits.shape == (1, 128, 1000)

    def test_large_model(self):
        """Test configuration large (peut être lent)"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=12,
            num_heads=12,
            embed_dim=768,
            max_seq_len=512
        )

        model = LLMTransformer(config)
        idx = torch.randint(0, 1000, (1, 64))

        with torch.no_grad():
            logits = model(idx)

        assert logits.shape == (1, 64, 1000)


class TestModelPersistence:
    """Test model saving and loading"""

    def test_state_dict(self):
        """Test sauvegarde/chargement state dict"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model1 = LLMTransformer(config)
        state_dict = model1.state_dict()

        model2 = LLMTransformer(config)
        model2.load_state_dict(state_dict)

        # Les poids doivent être identiques
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(),
            model2.named_parameters()
        ):
            assert torch.equal(param1, param2)

    def test_inference_determinism(self):
        """Test que l'inférence est déterministe"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)
        model.eval()

        idx = torch.randint(0, 1000, (1, 64))

        with torch.no_grad():
            logits1 = model(idx)
            logits2 = model(idx)

        assert torch.equal(logits1, logits2)


class TestModelEdgeCases:
    """Test edge cases"""

    def test_single_token_input(self):
        """Test avec un seul token"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)
        idx = torch.randint(0, 1000, (1, 1))

        with torch.no_grad():
            logits = model(idx)

        assert logits.shape == (1, 1, 1000)

    def test_max_sequence_length(self):
        """Test avec longueur maximale"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)
        idx = torch.randint(0, 1000, (1, 128))

        with torch.no_grad():
            logits = model(idx)

        assert logits.shape == (1, 128, 1000)

    def test_large_batch_size(self):
        """Test avec grand batch"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)
        idx = torch.randint(0, 1000, (16, 64))

        with torch.no_grad():
            logits = model(idx)

        assert logits.shape == (16, 64, 1000)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
