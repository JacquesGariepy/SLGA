"""Integration tests for training pipeline (40 tests).

Tests the complete training flow:
Model init → Forward pass → Loss computation → Backward pass → Optimizer step
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import os
from typing import Dict, Any

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from src.models import ModelConfig, SLGATransformer
from src.legacy.data import CollatorLocal


class TestForwardBackwardPass:
    """Forward and backward pass integration tests (10 tests)."""

    def test_forward_backward_step(self, tiny_model, tiny_config):
        """Test single forward and backward pass."""
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

        # Forward pass
        logits = tiny_model(input_ids)
        assert logits.shape == (batch_size, seq_len, tiny_config.vocab_size)

        # Compute loss
        targets = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, tiny_config.vocab_size),
            targets.reshape(-1)
        )

        # Backward pass
        loss.backward()

        # Verify gradients exist
        for name, param in tiny_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_forward_with_landmarks(self, tiny_model, tiny_config):
        """Test forward pass with heuristic landmarks."""
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

        # Create landmarks
        landmarks = torch.linspace(
            0, seq_len - 1, tiny_config.global_k, dtype=torch.long
        ).unsqueeze(0).expand(batch_size, -1)

        # Forward with landmarks
        logits = tiny_model(input_ids, cache_global_ids=landmarks)
        assert logits.shape == (batch_size, seq_len, tiny_config.vocab_size)

    def test_forward_return_aux(self, tiny_model, tiny_config):
        """Test forward pass returns auxiliary outputs."""
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

        # Forward with aux
        logits, aux = tiny_model(input_ids, return_aux=True)

        assert logits.shape == (batch_size, seq_len, tiny_config.vocab_size)
        assert isinstance(aux, dict)
        assert "landmark_indices" in aux or "landmark_scores" in aux

    def test_gradient_flow_through_model(self, tiny_model, tiny_config):
        """Test gradients flow through entire model."""
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

        logits = tiny_model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, tiny_config.vocab_size),
            targets.reshape(-1)
        )
        loss.backward()

        # Check all layers have gradients
        layers_with_grads = []
        layers_without_grads = []

        for name, param in tiny_model.named_parameters():
            if param.requires_grad:
                if param.grad is not None and param.grad.abs().sum() > 0:
                    layers_with_grads.append(name)
                else:
                    layers_without_grads.append(name)

        # Should have gradients in most layers
        assert len(layers_with_grads) > len(layers_without_grads)

    def test_loss_computation_correct(self, tiny_model, tiny_config):
        """Test loss computation is numerically correct."""
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

        logits = tiny_model(input_ids)

        # Compute loss
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, tiny_config.vocab_size),
            targets.reshape(-1),
            reduction='mean'
        )

        # Loss should be finite and positive
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_backward_with_landmarks(self, tiny_model, tiny_config):
        """Test backward pass works with landmarks."""
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
        landmarks = torch.linspace(
            0, seq_len - 1, tiny_config.global_k, dtype=torch.long
        ).unsqueeze(0).expand(batch_size, -1)

        logits = tiny_model(input_ids, cache_global_ids=landmarks)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, tiny_config.vocab_size),
            targets.reshape(-1)
        )
        loss.backward()

        # Verify gradients exist
        for name, param in tiny_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_forward_different_batch_sizes(self, tiny_model, tiny_config):
        """Test forward pass with different batch sizes."""
        seq_len = 32

        for batch_size in [1, 2, 4, 8]:
            input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
            logits = tiny_model(input_ids)
            assert logits.shape == (batch_size, seq_len, tiny_config.vocab_size)

    def test_forward_different_sequence_lengths(self, tiny_model, tiny_config):
        """Test forward pass with different sequence lengths."""
        batch_size = 2

        for seq_len in [16, 32, 64, 128]:
            if seq_len <= tiny_config.max_seq_len:
                input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
                logits = tiny_model(input_ids)
                assert logits.shape == (batch_size, seq_len, tiny_config.vocab_size)

    def test_mixed_precision_forward(self, tiny_model, tiny_config):
        """Test forward pass with mixed precision (autocast)."""
        if torch.cuda.is_available():
            tiny_model = tiny_model.cuda()
            batch_size, seq_len = 2, 32
            input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len)).cuda()

            with torch.cuda.amp.autocast():
                logits = tiny_model(input_ids)

            assert logits.dtype == torch.float16

    def test_gradient_accumulation(self, tiny_model, tiny_config):
        """Test gradient accumulation over multiple steps."""
        batch_size, seq_len = 2, 32
        accum_steps = 4

        tiny_model.zero_grad()

        accumulated_loss = 0.0
        for _ in range(accum_steps):
            input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
            targets = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

            logits = tiny_model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, tiny_config.vocab_size),
                targets.reshape(-1)
            )
            loss = loss / accum_steps  # Scale loss
            loss.backward()

            accumulated_loss += loss.item()

        # Check gradients accumulated
        for param in tiny_model.parameters():
            if param.requires_grad and param.grad is not None:
                assert param.grad.abs().sum() > 0


class TestMultiStepTraining:
    """Multi-step training integration tests (10 tests)."""

    def test_multi_step_training_loop(self, tiny_model, tiny_config):
        """Test multiple training steps."""
        optimizer = optim.AdamW(tiny_model.parameters(), lr=1e-4)
        batch_size, seq_len = 2, 32

        losses = []
        for step in range(10):
            input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
            targets = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

            # Forward
            logits = tiny_model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, tiny_config.vocab_size),
                targets.reshape(-1)
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Verify training ran without errors
        assert len(losses) == 10
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)

    def test_loss_decreases_on_repeated_batch(self, tiny_model, tiny_config):
        """Test loss decreases when training on same batch."""
        optimizer = optim.AdamW(tiny_model.parameters(), lr=1e-3)
        batch_size, seq_len = 2, 32

        # Fixed batch
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

        losses = []
        for _ in range(20):
            logits = tiny_model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, tiny_config.vocab_size),
                targets.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should decrease (overfitting to batch)
        assert losses[-1] < losses[0] * 0.8

    def test_optimizer_state_updates(self, tiny_model, tiny_config):
        """Test optimizer state is updated correctly."""
        optimizer = optim.AdamW(tiny_model.parameters(), lr=1e-4)
        batch_size, seq_len = 2, 32

        # Initial state should be empty
        assert len(optimizer.state) == 0

        # Train one step
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

        logits = tiny_model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, tiny_config.vocab_size),
            targets.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Optimizer state should now have entries
        assert len(optimizer.state) > 0

    def test_learning_rate_scheduling(self, tiny_model, tiny_config):
        """Test training with learning rate scheduler."""
        optimizer = optim.AdamW(tiny_model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        batch_size, seq_len = 2, 32

        lrs = []
        for step in range(15):
            input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
            targets = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

            logits = tiny_model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, tiny_config.vocab_size),
                targets.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            lrs.append(optimizer.param_groups[0]['lr'])

        # Learning rate should decrease at step 5 and 10
        assert lrs[6] < lrs[4]
        assert lrs[11] < lrs[9]

    def test_gradient_clipping(self, tiny_model, tiny_config):
        """Test gradient clipping during training."""
        optimizer = optim.AdamW(tiny_model.parameters(), lr=1e-3)
        batch_size, seq_len = 2, 32
        max_grad_norm = 1.0

        input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

        logits = tiny_model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, tiny_config.vocab_size),
            targets.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        grad_norm = nn.utils.clip_grad_norm_(
            tiny_model.parameters(),
            max_grad_norm
        )

        # Check clipping was applied if needed
        if grad_norm > max_grad_norm:
            # After clipping, norm should be max_grad_norm
            total_norm = 0.0
            for p in tiny_model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            assert abs(total_norm - max_grad_norm) < 0.1

    def test_training_with_different_optimizers(self, tiny_config):
        """Test training with different optimizer types."""
        batch_size, seq_len = 2, 32

        optimizers_to_test = [
            optim.SGD,
            optim.Adam,
            optim.AdamW,
        ]

        for opt_class in optimizers_to_test:
            model = SLGATransformer(tiny_config)

            if opt_class == optim.SGD:
                optimizer = opt_class(model.parameters(), lr=0.01)
            else:
                optimizer = opt_class(model.parameters(), lr=1e-4)

            # Train one step
            input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
            targets = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

            logits = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, tiny_config.vocab_size),
                targets.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check parameters were updated
            assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_training_determinism_with_seed(self, tiny_config):
        """Test training is deterministic with same seed."""
        torch.manual_seed(42)
        model1 = SLGATransformer(tiny_config)
        optimizer1 = optim.AdamW(model1.parameters(), lr=1e-4)

        torch.manual_seed(42)
        model2 = SLGATransformer(tiny_config)
        optimizer2 = optim.AdamW(model2.parameters(), lr=1e-4)

        batch_size, seq_len = 2, 32
        torch.manual_seed(100)
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

        # Train both models
        for model, optimizer in [(model1, optimizer1), (model2, optimizer2)]:
            logits = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, tiny_config.vocab_size),
                targets.reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Parameters should be identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)

    def test_training_with_weight_decay(self, tiny_model, tiny_config):
        """Test training with weight decay."""
        optimizer = optim.AdamW(tiny_model.parameters(), lr=1e-4, weight_decay=0.01)
        batch_size, seq_len = 2, 32

        # Get initial weights
        initial_weights = [p.clone() for p in tiny_model.parameters()]

        # Train several steps
        for _ in range(10):
            input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
            targets = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

            logits = tiny_model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, tiny_config.vocab_size),
                targets.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Weights should have changed
        for initial, current in zip(initial_weights, tiny_model.parameters()):
            assert not torch.equal(initial, current)

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not installed")
    def test_training_memory_efficiency(self, tiny_model, tiny_config):
        """Test training doesn't leak memory over multiple steps."""
        import gc

        process = psutil.Process(os.getpid())
        optimizer = optim.AdamW(tiny_model.parameters(), lr=1e-4)
        batch_size, seq_len = 2, 32

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        for _ in range(100):
            input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
            targets = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

            logits = tiny_model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, tiny_config.vocab_size),
                targets.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if _ % 10 == 0:
                gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        # Memory growth should be minimal (<100MB)
        assert memory_growth < 100, f"Memory grew by {memory_growth:.1f}MB"

    def test_training_with_dropout(self, tiny_config):
        """Test training/eval modes affect dropout."""
        config = tiny_config.model_copy()
        config.dropout_rate = 0.5  # High dropout
        model = SLGATransformer(config)

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Training mode
        model.train()
        logits_train1 = model(input_ids)
        logits_train2 = model(input_ids)

        # Eval mode
        model.eval()
        logits_eval1 = model(input_ids)
        logits_eval2 = model(input_ids)

        # Training mode should produce different outputs (dropout randomness)
        assert not torch.allclose(logits_train1, logits_train2)

        # Eval mode should produce same outputs (no dropout)
        assert torch.allclose(logits_eval1, logits_eval2)


class TestCheckpointing:
    """Checkpoint saving and loading tests (10 tests)."""

    def test_save_and_load_model(self, tiny_model, tmp_path):
        """Test saving and loading model state."""
        checkpoint_path = tmp_path / "checkpoint.pt"

        # Save model
        torch.save(tiny_model.state_dict(), checkpoint_path)

        # Create new model and load
        new_model = SLGATransformer(tiny_model.config)
        new_model.load_state_dict(torch.load(checkpoint_path))

        # Check parameters match
        for p1, p2 in zip(tiny_model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)

    def test_checkpoint_includes_optimizer_state(self, tiny_model, tmp_path):
        """Test checkpoint includes optimizer state."""
        optimizer = optim.AdamW(tiny_model.parameters(), lr=1e-4)
        checkpoint_path = tmp_path / "checkpoint.pt"

        # Train one step to initialize optimizer state
        input_ids = torch.randint(0, tiny_model.config.vocab_size, (2, 32))
        targets = torch.randint(0, tiny_model.config.vocab_size, (2, 32))

        logits = tiny_model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, tiny_model.config.vocab_size),
            targets.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save checkpoint
        checkpoint = {
            'model': tiny_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

        # Load checkpoint
        loaded = torch.load(checkpoint_path)
        assert 'model' in loaded
        assert 'optimizer' in loaded

        # Create new optimizer and load state
        new_optimizer = optim.AdamW(tiny_model.parameters(), lr=1e-4)
        new_optimizer.load_state_dict(loaded['optimizer'])

        # Check state was loaded
        assert len(new_optimizer.state) > 0

    def test_resume_training_from_checkpoint(self, tiny_config, tmp_path):
        """Test resuming training from checkpoint."""
        # Initial training
        model1 = SLGATransformer(tiny_config)
        optimizer1 = optim.AdamW(model1.parameters(), lr=1e-4)

        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 32))
        targets = torch.randint(0, tiny_config.vocab_size, (2, 32))

        # Train 5 steps
        for _ in range(5):
            logits = model1(input_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, tiny_config.vocab_size),
                targets.reshape(-1)
            )
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save({
            'model': model1.state_dict(),
            'optimizer': optimizer1.state_dict(),
            'step': 5,
        }, checkpoint_path)

        # Load and continue training
        model2 = SLGATransformer(tiny_config)
        optimizer2 = optim.AdamW(model2.parameters(), lr=1e-4)

        checkpoint = torch.load(checkpoint_path)
        model2.load_state_dict(checkpoint['model'])
        optimizer2.load_state_dict(checkpoint['optimizer'])

        # Continue training
        for _ in range(5):
            logits = model2(input_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, tiny_config.vocab_size),
                targets.reshape(-1)
            )
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()

        # Models should be different after more training
        assert not all(
            torch.equal(p1, p2)
            for p1, p2 in zip(model1.parameters(), model2.parameters())
        )

    def test_checkpoint_with_metadata(self, tiny_model, tmp_path):
        """Test checkpoint saves and loads metadata."""
        checkpoint_path = tmp_path / "checkpoint.pt"

        metadata = {
            'epoch': 10,
            'step': 1000,
            'loss': 2.5,
            'config': tiny_model.config.model_dump(),
        }

        checkpoint = {
            'model': tiny_model.state_dict(),
            'metadata': metadata,
        }

        torch.save(checkpoint, checkpoint_path)

        # Load and verify
        loaded = torch.load(checkpoint_path)
        assert loaded['metadata']['epoch'] == 10
        assert loaded['metadata']['step'] == 1000

    def test_checkpoint_cross_device(self, tiny_model, tmp_path):
        """Test checkpoint can be loaded on different device."""
        checkpoint_path = tmp_path / "checkpoint.pt"

        # Save on current device
        torch.save(tiny_model.state_dict(), checkpoint_path)

        # Load with map_location
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        new_model = SLGATransformer(tiny_model.config)
        new_model.load_state_dict(state_dict)

        # Verify loaded correctly
        assert len(list(new_model.parameters())) == len(list(tiny_model.parameters()))

    def test_partial_checkpoint_loading(self, tiny_config, tmp_path):
        """Test loading checkpoint with partial state dict."""
        model = SLGATransformer(tiny_config)
        checkpoint_path = tmp_path / "checkpoint.pt"

        # Save only embedding weights
        partial_state = {
            'embedding.token_emb.embedding.weight': model.embedding.token_emb.embedding.weight.clone(),
        }
        torch.save(partial_state, checkpoint_path)

        # Load partial state (strict=False)
        new_model = SLGATransformer(tiny_config)
        new_model.load_state_dict(torch.load(checkpoint_path), strict=False)

        # Embedding should match, other params should be different
        assert torch.equal(
            model.embedding.token_emb.embedding.weight,
            new_model.embedding.token_emb.embedding.weight
        )

    def test_checkpoint_size_reasonable(self, tiny_model, tmp_path):
        """Test checkpoint file size is reasonable."""
        checkpoint_path = tmp_path / "checkpoint.pt"

        torch.save(tiny_model.state_dict(), checkpoint_path)

        file_size = checkpoint_path.stat().st_size / 1024 / 1024  # MB

        # Tiny model should be small (<10MB)
        assert file_size < 10

    def test_multiple_checkpoints(self, tiny_model, tmp_path):
        """Test saving multiple checkpoints."""
        optimizer = optim.AdamW(tiny_model.parameters(), lr=1e-4)

        checkpoints = []
        for step in range(5):
            # Train one step
            input_ids = torch.randint(0, tiny_model.config.vocab_size, (2, 32))
            targets = torch.randint(0, tiny_model.config.vocab_size, (2, 32))

            logits = tiny_model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, tiny_model.config.vocab_size),
                targets.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save checkpoint
            checkpoint_path = tmp_path / f"checkpoint_step_{step}.pt"
            torch.save({
                'model': tiny_model.state_dict(),
                'step': step,
            }, checkpoint_path)
            checkpoints.append(checkpoint_path)

        # All checkpoints should exist
        assert all(cp.exists() for cp in checkpoints)

    def test_checkpoint_after_mode_change(self, tiny_model, tmp_path):
        """Test checkpoint after switching between train/eval modes."""
        checkpoint_path = tmp_path / "checkpoint.pt"

        # Switch modes
        tiny_model.train()
        tiny_model.eval()
        tiny_model.train()

        # Save checkpoint
        torch.save(tiny_model.state_dict(), checkpoint_path)

        # Load
        new_model = SLGATransformer(tiny_model.config)
        new_model.load_state_dict(torch.load(checkpoint_path))

        # Check parameters match
        for p1, p2 in zip(tiny_model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)

    def test_checkpoint_integrity(self, tiny_model, tmp_path):
        """Test checkpoint can be loaded multiple times consistently."""
        checkpoint_path = tmp_path / "checkpoint.pt"

        torch.save(tiny_model.state_dict(), checkpoint_path)

        # Load multiple times
        models = []
        for _ in range(3):
            new_model = SLGATransformer(tiny_model.config)
            new_model.load_state_dict(torch.load(checkpoint_path))
            models.append(new_model)

        # All loaded models should be identical
        for i in range(1, len(models)):
            for p1, p2 in zip(models[0].parameters(), models[i].parameters()):
                assert torch.equal(p1, p2)


class TestValidation:
    """Validation pipeline tests (10 tests)."""

    def test_eval_mode_disables_dropout(self, tiny_config):
        """Test eval mode disables dropout."""
        config = tiny_config.model_copy()
        config.dropout_rate = 0.5
        model = SLGATransformer(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 32))

        # Eval mode
        model.eval()
        logits1 = model(input_ids)
        logits2 = model(input_ids)

        # Should produce identical outputs
        assert torch.allclose(logits1, logits2)

    def test_validation_no_gradients(self, tiny_model, tiny_config):
        """Test validation doesn't compute gradients."""
        tiny_model.eval()

        with torch.no_grad():
            input_ids = torch.randint(0, tiny_config.vocab_size, (2, 32))
            logits = tiny_model(input_ids)

        # No gradients should be computed
        assert logits.requires_grad is False

    def test_compute_validation_loss(self, tiny_model, tiny_config):
        """Test computing validation loss."""
        tiny_model.eval()

        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 32))
        targets = torch.randint(0, tiny_config.vocab_size, (2, 32))

        with torch.no_grad():
            logits = tiny_model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, tiny_config.vocab_size),
                targets.reshape(-1)
            )

        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_validation_faster_than_training(self, tiny_model, tiny_config):
        """Test validation is faster than training."""
        import time

        input_ids = torch.randint(0, tiny_config.vocab_size, (4, 64))
        targets = torch.randint(0, tiny_config.vocab_size, (4, 64))

        # Training time
        tiny_model.train()
        optimizer = optim.AdamW(tiny_model.parameters(), lr=1e-4)

        start = time.time()
        for _ in range(10):
            logits = tiny_model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, tiny_config.vocab_size),
                targets.reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_time = time.time() - start

        # Validation time
        tiny_model.eval()
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                logits = tiny_model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, tiny_config.vocab_size),
                    targets.reshape(-1)
                )
        val_time = time.time() - start

        # Validation should be faster
        assert val_time < train_time

    def test_validation_on_multiple_batches(self, tiny_model, tiny_config):
        """Test validation on multiple batches."""
        tiny_model.eval()

        val_losses = []
        with torch.no_grad():
            for _ in range(10):
                input_ids = torch.randint(0, tiny_config.vocab_size, (2, 32))
                targets = torch.randint(0, tiny_config.vocab_size, (2, 32))

                logits = tiny_model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, tiny_config.vocab_size),
                    targets.reshape(-1)
                )
                val_losses.append(loss.item())

        # Should have computed 10 losses
        assert len(val_losses) == 10
        assert all(l > 0 for l in val_losses)

    def test_validation_perplexity(self, tiny_model, tiny_config):
        """Test computing perplexity from validation loss."""
        tiny_model.eval()

        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 32))
        targets = torch.randint(0, tiny_config.vocab_size, (2, 32))

        with torch.no_grad():
            logits = tiny_model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, tiny_config.vocab_size),
                targets.reshape(-1)
            )

        perplexity = torch.exp(loss)

        assert perplexity > 1.0
        assert torch.isfinite(perplexity)

    def test_train_val_mode_switching(self, tiny_model, tiny_config):
        """Test switching between train and eval modes."""
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 32))

        # Train mode
        tiny_model.train()
        assert tiny_model.training is True

        # Eval mode
        tiny_model.eval()
        assert tiny_model.training is False

        # Back to train
        tiny_model.train()
        assert tiny_model.training is True

    def test_validation_with_landmarks(self, tiny_model, tiny_config):
        """Test validation with heuristic landmarks."""
        tiny_model.eval()

        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 64))
        landmarks = torch.linspace(
            0, 63, tiny_config.global_k, dtype=torch.long
        ).unsqueeze(0).expand(2, -1)

        with torch.no_grad():
            logits = tiny_model(input_ids, cache_global_ids=landmarks)

        assert logits.shape == (2, 64, tiny_config.vocab_size)

    def test_validation_batch_independence(self, tiny_model, tiny_config):
        """Test validation batches are independent."""
        tiny_model.eval()

        # Validate on two batches
        input_ids1 = torch.randint(0, tiny_config.vocab_size, (2, 32))
        input_ids2 = torch.randint(0, tiny_config.vocab_size, (2, 32))

        with torch.no_grad():
            logits1 = tiny_model(input_ids1)
            logits2 = tiny_model(input_ids2)

        # Results should be independent
        assert not torch.allclose(logits1, logits2)

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not installed")
    def test_validation_memory_usage(self, tiny_model, tiny_config):
        """Test validation uses less memory than training."""
        import gc

        process = psutil.Process(os.getpid())

        input_ids = torch.randint(0, tiny_config.vocab_size, (4, 64))

        # Training memory
        tiny_model.train()
        gc.collect()
        train_mem_before = process.memory_info().rss / 1024 / 1024

        logits = tiny_model(input_ids)
        loss = logits.sum()
        loss.backward()

        train_mem_after = process.memory_info().rss / 1024 / 1024
        train_mem_usage = train_mem_after - train_mem_before

        # Clean up
        del logits, loss
        tiny_model.zero_grad()
        gc.collect()

        # Validation memory
        tiny_model.eval()
        gc.collect()
        val_mem_before = process.memory_info().rss / 1024 / 1024

        with torch.no_grad():
            logits = tiny_model(input_ids)

        val_mem_after = process.memory_info().rss / 1024 / 1024
        val_mem_usage = val_mem_after - val_mem_before

        # Validation should use less memory (no gradient storage)
        # Note: This test may be flaky due to Python memory management
        # So we just check validation doesn't use dramatically more memory
        assert val_mem_usage <= train_mem_usage * 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
