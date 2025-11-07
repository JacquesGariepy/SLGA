#!/usr/bin/env python3
"""
Test script for validation utilities
Demonstrates all validation features
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from src.validation import (
    ConfigValidator,
    RuntimeValidator,
    ValidationResult,
    print_validation_results,
    validate_training_step
)


def test_config_validation():
    """Test configuration validation"""
    print("\n" + "=" * 70)
    print("TEST 1: Configuration Validation")
    print("=" * 70)

    # ✅ Valid configuration
    print("\n1.1 Testing VALID configuration:")
    valid_config = {
        'embed_dim': 512,
        'num_heads': 8,
        'local_window': 256,
        'global_k': 64,
        'max_seq_len': 2048,
        'dropout_rate': 0.1,
        'attn_drop': 0.1,
        'proj_drop': 0.1
    }

    results = ConfigValidator.validate_slga_config(valid_config)
    print_validation_results(results)

    # ❌ Invalid configuration (embed_dim not divisible by num_heads)
    print("\n1.2 Testing INVALID configuration (embed_dim issue):")
    invalid_config = {
        'embed_dim': 513,  # Not divisible by 8
        'num_heads': 8,
        'local_window': 256,
        'global_k': 64,
        'max_seq_len': 2048
    }

    results = ConfigValidator.validate_slga_config(invalid_config)
    print_validation_results(results)

    # ⚠️ Configuration with warnings
    print("\n1.3 Testing configuration with WARNINGS:")
    warning_config = {
        'embed_dim': 512,
        'num_heads': 8,
        'local_window': 1024,  # Very large
        'global_k': 512,  # Larger than recommended
        'max_seq_len': 2048
    }

    results = ConfigValidator.validate_slga_config(warning_config)
    print_validation_results(results)


def test_training_config_validation():
    """Test training configuration validation"""
    print("\n" + "=" * 70)
    print("TEST 2: Training Configuration Validation")
    print("=" * 70)

    # ✅ Valid training config
    print("\n2.1 Testing VALID training configuration:")
    valid_train_config = {
        'batch_size': 32,
        'lr': 1e-4,
        'epochs': 10,
        'seq_len_start': 128,
        'seq_len_final': 512,
        'max_grad_norm': 1.0
    }

    results = ConfigValidator.validate_training_config(valid_train_config)
    print_validation_results(results)

    # ❌ Invalid training config
    print("\n2.2 Testing INVALID training configuration:")
    invalid_train_config = {
        'batch_size': 0,  # Invalid
        'lr': 1.0,  # Too high
        'seq_len_start': 512,
        'seq_len_final': 128,  # Wrong order
        'max_grad_norm': -1.0  # Invalid
    }

    results = ConfigValidator.validate_training_config(invalid_train_config)
    print_validation_results(results)


def test_gradient_validation():
    """Test gradient validation"""
    print("\n" + "=" * 70)
    print("TEST 3: Gradient Validation")
    print("=" * 70)

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(20, 10)

        def forward(self, x):
            x = self.linear1(x)
            x = torch.relu(x)
            x = self.linear2(x)
            return x

    model = SimpleModel()

    # ✅ Healthy gradients
    print("\n3.1 Testing HEALTHY gradients:")
    x = torch.randn(5, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()

    result = RuntimeValidator.check_gradients(model)
    print(f"  {result.message} [Severity: {result.severity}]")

    # ❌ No gradients (manually cleared)
    print("\n3.2 Testing NO gradients:")
    model.zero_grad()  # Clear all gradients

    result = RuntimeValidator.check_gradients(model)
    print(f"  {result.message} [Severity: {result.severity}]")

    # ⚠️ Vanishing gradients
    print("\n3.3 Testing VANISHING gradients:")
    model = SimpleModel()
    x = torch.randn(5, 10)
    y = model(x)
    loss = y.sum() * 1e-10  # Very small loss
    loss.backward()

    result = RuntimeValidator.check_gradients(model, warn_threshold=1e-7)
    print(f"  {result.message} [Severity: {result.severity}]")

    # ⚠️ Exploding gradients
    print("\n3.4 Testing EXPLODING gradients:")
    model = SimpleModel()
    x = torch.randn(5, 10)
    y = model(x)
    loss = y.sum() * 1e10  # Very large loss
    loss.backward()

    result = RuntimeValidator.check_gradients(model, error_threshold=100.0)
    print(f"  {result.message} [Severity: {result.severity}]")


def test_loss_validation():
    """Test loss validation"""
    print("\n" + "=" * 70)
    print("TEST 4: Loss Validation")
    print("=" * 70)

    # ✅ Valid loss
    print("\n4.1 Testing VALID loss:")
    loss = torch.tensor(2.5)
    result = RuntimeValidator.check_loss(loss, step=100)
    print(f"  {result.message} [Severity: {result.severity}]")

    # ❌ NaN loss
    print("\n4.2 Testing NaN loss:")
    loss = torch.tensor(float('nan'))
    result = RuntimeValidator.check_loss(loss, step=100)
    print(f"  {result.message} [Severity: {result.severity}]")

    # ❌ Inf loss
    print("\n4.3 Testing Inf loss:")
    loss = torch.tensor(float('inf'))
    result = RuntimeValidator.check_loss(loss, step=100)
    print(f"  {result.message} [Severity: {result.severity}]")

    # ❌ Negative loss
    print("\n4.4 Testing NEGATIVE loss:")
    loss = torch.tensor(-1.5)
    result = RuntimeValidator.check_loss(loss, step=100)
    print(f"  {result.message} [Severity: {result.severity}]")

    # ⚠️ Loss too high
    print("\n4.5 Testing loss EXCEEDING maximum:")
    loss = torch.tensor(15.0)
    result = RuntimeValidator.check_loss(loss, step=100, max_loss=10.0)
    print(f"  {result.message} [Severity: {result.severity}]")


def test_landmark_validation():
    """Test landmark validation"""
    print("\n" + "=" * 70)
    print("TEST 5: Landmark Validation")
    print("=" * 70)

    seq_len = 512
    B, G = 4, 64

    # ✅ Valid landmarks
    print("\n5.1 Testing VALID landmarks:")
    landmarks = torch.randint(0, seq_len, (B, G))
    result = RuntimeValidator.check_landmarks(landmarks, seq_len)
    print(f"  {result.message} [Severity: {result.severity}]")

    # ❌ Out of range landmarks
    print("\n5.2 Testing OUT OF RANGE landmarks:")
    landmarks = torch.randint(-10, seq_len + 10, (B, G))
    result = RuntimeValidator.check_landmarks(landmarks, seq_len)
    print(f"  {result.message} [Severity: {result.severity}]")

    # ⚠️ Duplicate landmarks
    print("\n5.3 Testing DUPLICATE landmarks:")
    landmarks = torch.zeros(B, G, dtype=torch.long)  # All zeros
    result = RuntimeValidator.check_landmarks(landmarks, seq_len, min_unique_ratio=0.9)
    print(f"  {result.message} [Severity: {result.severity}]")

    # ℹ️ No landmarks (heuristic)
    print("\n5.4 Testing NO landmarks (heuristic):")
    result = RuntimeValidator.check_landmarks(None, seq_len)
    print(f"  {result.message} [Severity: {result.severity}]")


def test_model_output_validation():
    """Test model output validation"""
    print("\n" + "=" * 70)
    print("TEST 6: Model Output Validation")
    print("=" * 70)

    vocab_size = 50257
    B, T = 4, 512

    # ✅ Valid logits
    print("\n6.1 Testing VALID logits:")
    outputs = torch.randn(B, T, vocab_size)
    result = RuntimeValidator.check_model_outputs(outputs, vocab_size, check_logits=True)
    print(f"  {result.message} [Severity: {result.severity}]")

    # ❌ NaN outputs
    print("\n6.2 Testing NaN outputs:")
    outputs = torch.randn(B, T, vocab_size)
    outputs[0, 0, 0] = float('nan')
    result = RuntimeValidator.check_model_outputs(outputs, vocab_size)
    print(f"  {result.message} [Severity: {result.severity}]")

    # ❌ Wrong vocab size
    print("\n6.3 Testing WRONG vocab size:")
    outputs = torch.randn(B, T, vocab_size - 1)
    result = RuntimeValidator.check_model_outputs(outputs, vocab_size)
    print(f"  {result.message} [Severity: {result.severity}]")

    # ✅ Valid probabilities
    print("\n6.4 Testing VALID probabilities:")
    outputs = torch.softmax(torch.randn(B, T, vocab_size), dim=-1)
    result = RuntimeValidator.check_model_outputs(outputs, vocab_size, check_logits=False)
    print(f"  {result.message} [Severity: {result.severity}]")


def test_full_validation():
    """Test complete validation workflow"""
    print("\n" + "=" * 70)
    print("TEST 7: Complete Validation Workflow")
    print("=" * 70)

    # Create model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()

    # Simulate training step
    print("\n7.1 Testing COMPLETE training step validation:")
    x = torch.randn(5, 10)
    y = model(x)
    loss = y.sum()

    # Validate before backward
    landmarks = torch.randint(0, 512, (4, 64))
    success = validate_training_step(model, loss, step=100, landmarks=landmarks, seq_len=512)

    print(f"\n  Validation result: {'✅ PASSED' if success else '❌ FAILED'}")

    # Do backward
    loss.backward()

    # Validate after backward
    success = validate_training_step(model, loss, step=100, landmarks=landmarks, seq_len=512)

    print(f"  Validation with gradients: {'✅ PASSED' if success else '❌ FAILED'}")


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("SLGA VALIDATION UTILITIES - TEST SUITE")
    print("=" * 70)

    try:
        test_config_validation()
        test_training_config_validation()
        test_gradient_validation()
        test_loss_validation()
        test_landmark_validation()
        test_model_output_validation()
        test_full_validation()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
