"""
Model summary and analysis utilities.

This module provides functions for displaying detailed information
about PyTorch models, including parameter counts and module breakdown.
"""

from typing import Optional

import torch

from .metrics import count_parameters


def print_model_summary(
    model: torch.nn.Module,
    input_shape: Optional[tuple] = None
) -> None:
    """
    Print a detailed summary of a PyTorch model.

    This function displays:
    - Total parameter count
    - Trainable parameter count
    - Non-trainable parameter count
    - Parameter breakdown by module

    Args:
        model: The PyTorch model to summarize.
        input_shape: Optional input shape for FLOP calculation.
                    Currently not used, reserved for future enhancement.

    Example:
        >>> model = MyTransformer(d_model=512, num_layers=6)
        >>> print_model_summary(model)
        ================================================================================
        MODEL SUMMARY
        ================================================================================
        Total parameters: 44,234,176 (44.23M)
        Trainable parameters: 44,234,176 (44.23M)
        Non-trainable parameters: 0

        Module breakdown:
          embedding: 8,388,608 (8.39M)
          encoder: 31,457,280 (31.46M)
          decoder: 4,194,304 (4.19M)
          output_projection: 193,984 (0.19M)
        ================================================================================

    Note:
        The input_shape parameter is included for future compatibility
        with FLOP counting but is not currently used.
    """
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)

    # Count parameters
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    non_trainable_params = total_params - trainable_params

    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"Non-trainable parameters: {non_trainable_params:,}")

    # Module breakdown
    print("\nModule breakdown:")
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {num_params:,} ({num_params / 1e6:.2f}M)")

    print("=" * 80)
