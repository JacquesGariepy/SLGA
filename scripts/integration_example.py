#!/usr/bin/env python3
"""
Example integration of validation utilities into training script
This demonstrates how to add validation to an existing train.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import validation utilities
from src.validation import (
    ConfigValidator,
    RuntimeValidator,
    print_validation_results,
    validate_training_step
)


def train_with_validation():
    """Example training loop with validation integrated"""

    # ==================== STEP 1: VALIDATE CONFIGURATION ====================
    print("\n" + "=" * 70)
    print("STEP 1: CONFIGURATION VALIDATION")
    print("=" * 70)

    model_config = {
        'vocab_size': 50257,
        'embed_dim': 512,
        'num_heads': 8,
        'num_layers': 6,
        'local_window': 256,
        'global_k': 64,
        'max_seq_len': 2048,
        'dropout_rate': 0.1,
        'attn_drop': 0.1,
        'proj_drop': 0.1
    }

    training_config = {
        'batch_size': 32,
        'lr': 1e-4,
        'epochs': 3,
        'seq_len_start': 128,
        'seq_len_final': 512,
        'max_grad_norm': 1.0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Validate all configurations
    passed, results = ConfigValidator.validate_all(model_config, training_config)
    print_validation_results(results)

    if not passed:
        print("\n❌ Configuration validation failed. Please fix errors.")
        return False

    print("\n✅ Configuration validation passed!")

    # ==================== STEP 2: INITIALIZE MODEL ====================
    print("\n" + "=" * 70)
    print("STEP 2: MODEL INITIALIZATION")
    print("=" * 70)

    # Create dummy model for demonstration
    class DummyModel(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.linear = nn.Linear(embed_dim, vocab_size)

        def forward(self, x, landmark_indices=None):
            emb = self.embedding(x)
            return self.linear(emb)

    device = training_config['device']
    model = DummyModel(model_config['vocab_size'], model_config['embed_dim']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config['lr'])
    criterion = nn.CrossEntropyLoss()

    print(f"✅ Model initialized on {device}")

    # ==================== STEP 3: TRAINING LOOP ====================
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING WITH VALIDATION")
    print("=" * 70)

    best_loss = float('inf')
    validation_failures = 0

    for epoch in range(training_config['epochs']):
        print(f"\n{'=' * 70}")
        print(f"EPOCH {epoch + 1}/{training_config['epochs']}")
        print(f"{'=' * 70}")

        model.train()
        total_loss = 0
        num_batches = 10  # Simulating 10 batches

        for step in range(num_batches):
            # Simulate batch data
            batch_size = training_config['batch_size']
            seq_len = 128
            input_ids = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len)).to(device)
            targets = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len)).to(device)
            landmarks = torch.randint(0, seq_len, (batch_size, 64)).to(device)

            # Forward pass
            outputs = model(input_ids, landmark_indices=landmarks)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            # ✅ VALIDATION CHECKPOINT 1: Validate loss and landmarks
            if step % 5 == 0:  # Every 5 steps
                print(f"\n[Step {step}] Validation checkpoint...")

                # Use the all-in-one validation function
                is_valid = validate_training_step(
                    model, loss, step, landmarks, seq_len
                )

                if not is_valid:
                    validation_failures += 1
                    if validation_failures >= 3:
                        print("\n❌ Too many validation failures. Stopping training.")
                        return False
                    print(f"⚠️  Validation failure #{validation_failures}")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # ✅ VALIDATION CHECKPOINT 2: Check gradients after backward
            if step % 5 == 0:
                grad_result = RuntimeValidator.check_gradients(model)

                if grad_result.severity == "error":
                    print(f"❌ Critical gradient error: {grad_result.message}")
                    return False
                elif grad_result.severity == "warning":
                    print(f"⚠️  {grad_result.message}")

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=training_config['max_grad_norm']
            )

            optimizer.step()

            total_loss += loss.item()

            # Print progress
            if step % 5 == 0:
                print(f"  Step {step}/{num_batches}: loss = {loss.item():.4f}")

        # ==================== STEP 4: EPOCH-END VALIDATION ====================
        avg_loss = total_loss / num_batches
        print(f"\n{'=' * 70}")
        print(f"EPOCH {epoch + 1} SUMMARY")
        print(f"{'=' * 70}")
        print(f"Average Loss: {avg_loss:.4f}")

        # Comprehensive validation at epoch end
        epoch_results = []

        # 1. Check loss progression
        loss_result = RuntimeValidator.check_loss(
            torch.tensor(avg_loss),
            step=epoch
        )
        epoch_results.append(loss_result)

        # 2. Check if loss is improving
        from src.validation import ValidationResult
        if avg_loss > best_loss * 1.1:
            epoch_results.append(ValidationResult(
                passed=False,
                message=f"Loss not improving: {avg_loss:.4f} > {best_loss:.4f}",
                severity="warning"
            ))
        else:
            epoch_results.append(ValidationResult(
                passed=True,
                message=f"Loss improved from {best_loss:.4f} to {avg_loss:.4f}",
                severity="info"
            ))

        # 3. Check for NaN parameters
        nan_params = sum(1 for p in model.parameters() if torch.isnan(p).any())
        if nan_params > 0:
            epoch_results.append(ValidationResult(
                passed=False,
                message=f"Found {nan_params} parameters with NaN values",
                severity="error"
            ))
        else:
            epoch_results.append(ValidationResult(
                passed=True,
                message="All parameters valid (no NaN)",
                severity="info"
            ))

        # Print epoch validation results
        print("\n📊 Epoch Validation:")
        print_validation_results(epoch_results, verbose=True)

        # Check if training should continue
        has_errors = any(r.severity == "error" and not r.passed for r in epoch_results)
        if has_errors:
            print("\n❌ Epoch validation failed with errors. Stopping training.")
            return False

        # Update best loss and save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"\n✅ New best loss: {best_loss:.4f}")
            # torch.save(model.state_dict(), "checkpoints/best_model.pt")

    # ==================== STEP 5: TRAINING COMPLETE ====================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"✅ Final best loss: {best_loss:.4f}")
    print(f"✅ Total validation failures: {validation_failures}")

    return True


def main():
    """Run training with validation"""
    print("=" * 70)
    print("TRAINING WITH VALIDATION - INTEGRATION EXAMPLE")
    print("=" * 70)

    success = train_with_validation()

    if success:
        print("\n✅ Training completed successfully!")
    else:
        print("\n❌ Training failed validation checks.")


if __name__ == "__main__":
    main()
