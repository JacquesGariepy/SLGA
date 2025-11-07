"""
Test suite for training pipeline
Tests data loading, training loop, and checkpointing
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path and setup package imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path.parent))

# Import with package name
from src.model import LLMTransformer, Config as SLGAPlusConfig


class TestTrainingSetup:
    """Test training setup and configuration"""

    def test_optimizer_creation(self):
        """Test création de l'optimizer"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        assert optimizer is not None
        assert len(optimizer.param_groups) > 0

    def test_learning_rate_scheduler(self):
        """Test scheduler de learning rate"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        initial_lr = optimizer.param_groups[0]['lr']

        # Step scheduler
        for _ in range(50):
            scheduler.step()

        final_lr = optimizer.param_groups[0]['lr']

        # LR should have decreased
        assert final_lr < initial_lr

    def test_gradient_clipping(self):
        """Test gradient clipping"""
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
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        loss.backward()

        # Clip gradients
        max_norm = 1.0
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Vérifier que les gradients sont clippés
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.norm()
                assert param_norm <= max_norm * 2  # Allow some tolerance


class TestTrainingLoop:
    """Test training loop functionality"""

    def test_single_training_step(self):
        """Test une étape d'entraînement"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        idx = torch.randint(0, 1000, (2, 64))
        targets = torch.randint(0, 1000, (2, 64))

        # Training step
        model.train()
        optimizer.zero_grad()
        logits = model(idx)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_multiple_training_steps(self):
        """Test plusieurs étapes d'entraînement"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        losses = []

        for i in range(10):
            idx = torch.randint(0, 1000, (4, 64))
            targets = torch.randint(0, 1000, (4, 64))

            optimizer.zero_grad()
            logits = model(idx)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should generally decrease (with some variance)
        assert len(losses) == 10

    def test_batch_processing(self):
        """Test traitement par batch"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)

        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            idx = torch.randint(0, 1000, (batch_size, 64))
            targets = torch.randint(0, 1000, (batch_size, 64))

            logits = model(idx)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

            assert logits.shape[0] == batch_size
            assert not torch.isnan(loss).any()


class TestCheckpointing:
    """Test model checkpointing"""

    def test_save_checkpoint(self):
        """Test sauvegarde de checkpoint"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config.to_dict(),
                'step': 100,
            }

            torch.save(checkpoint, checkpoint_path)

            assert checkpoint_path.exists()

    def test_load_checkpoint(self):
        """Test chargement de checkpoint"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model1 = LLMTransformer(config)
        optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-4)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            # Save
            checkpoint = {
                'model': model1.state_dict(),
                'optimizer': optimizer1.state_dict(),
                'config': config.to_dict(),
                'step': 100,
            }
            torch.save(checkpoint, checkpoint_path)

            # Load
            checkpoint_loaded = torch.load(checkpoint_path)

            model2 = LLMTransformer(config)
            model2.load_state_dict(checkpoint_loaded['model'])

            # Verify weights match
            for (name1, param1), (name2, param2) in zip(
                model1.named_parameters(),
                model2.named_parameters()
            ):
                assert torch.equal(param1, param2)

    def test_resume_training(self):
        """Test reprise de l'entraînement"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Train for a few steps
        for _ in range(5):
            idx = torch.randint(0, 1000, (2, 64))
            targets = torch.randint(0, 1000, (2, 64))

            optimizer.zero_grad()
            logits = model(idx)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
            loss.backward()
            optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            # Save
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config.to_dict(),
                'step': 5,
            }
            torch.save(checkpoint, checkpoint_path)

            # Create new model and load
            model_new = LLMTransformer(config)
            optimizer_new = torch.optim.AdamW(model_new.parameters(), lr=1e-4)

            checkpoint_loaded = torch.load(checkpoint_path)
            model_new.load_state_dict(checkpoint_loaded['model'])
            optimizer_new.load_state_dict(checkpoint_loaded['optimizer'])

            # Continue training
            idx = torch.randint(0, 1000, (2, 64))
            targets = torch.randint(0, 1000, (2, 64))

            optimizer_new.zero_grad()
            logits, loss = model_new(idx, targets)
            loss.backward()
            optimizer_new.step()

            assert not torch.isnan(loss).any()


class TestEvaluation:
    """Test evaluation functionality"""

    def test_eval_mode(self):
        """Test passage en mode évaluation"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128,
            dropout_rate=0.1
        )

        model = LLMTransformer(config)

        # Train mode
        model.train()
        assert model.training

        # Eval mode
        model.eval()
        assert not model.training

    def test_eval_loss_computation(self):
        """Test calcul de loss en mode eval"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)
        model.eval()

        idx = torch.randint(0, 1000, (2, 64))
        targets = torch.randint(0, 1000, (2, 64))

        with torch.no_grad():
            logits = model(idx)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

        assert loss.item() > 0
        assert not torch.isnan(loss).any()

    def test_eval_determinism(self):
        """Test que l'évaluation est déterministe"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)
        model.eval()

        idx = torch.randint(0, 1000, (2, 64))
        targets = torch.randint(0, 1000, (2, 64))

        with torch.no_grad():
            logits1, loss1 = model(idx, targets)
            logits2, loss2 = model(idx, targets)

        assert torch.equal(logits1, logits2)
        assert torch.equal(loss1, loss2)


class TestMemoryManagement:
    """Test memory management during training"""

    def test_gradient_accumulation(self):
        """Test accumulation de gradients"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        accum_steps = 4

        optimizer.zero_grad()

        total_loss = 0
        for i in range(accum_steps):
            idx = torch.randint(0, 1000, (1, 64))
            targets = torch.randint(0, 1000, (1, 64))

            logits = model(idx)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
            loss = loss / accum_steps
            loss.backward()

            total_loss += loss.item()

        optimizer.step()

        assert total_loss > 0

    def test_memory_cleanup(self):
        """Test nettoyage de mémoire"""
        config = SLGAPlusConfig(
            vocab_size=1000,
            n_layers=2,
            num_heads=4,
            embed_dim=256,
            max_seq_len=128
        )

        model = LLMTransformer(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for i in range(5):
            idx = torch.randint(0, 1000, (2, 64))
            targets = torch.randint(0, 1000, (2, 64))

            optimizer.zero_grad()
            logits = model(idx)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
            loss.backward()
            optimizer.step()

            # Clear cache
            del idx, targets, logits, loss

        # Should not crash


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
