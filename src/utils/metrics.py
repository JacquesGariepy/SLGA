"""
Model and training metrics utilities.

This module provides utilities for tracking metrics, counting parameters,
and monitoring memory usage during training.
"""

from typing import Dict

import torch


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.

    This class is useful for tracking running averages during training,
    such as loss, accuracy, or other metrics.

    Attributes:
        name: Name of the metric being tracked.
        val: Most recent value.
        avg: Running average of all values.
        sum: Sum of all values added.
        count: Number of values added.

    Example:
        >>> loss_meter = AverageMeter("loss")
        >>> for batch in dataloader:
        ...     loss = compute_loss(batch)
        ...     loss_meter.update(loss.item())
        >>> print(loss_meter)
        loss: 0.4523 (current: 0.4401)
    """

    def __init__(self, name: str = ""):
        """
        Initialize the AverageMeter.

        Args:
            name: Optional name for the metric (used in string representation).
        """
        self.name = name
        self.reset()

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update the meter with a new value.

        Args:
            val: The value to add (typically a loss or metric value).
            n: The weight/count for this value (default: 1).
                  Use batch_size when averaging per-sample metrics.

        Example:
            >>> meter = AverageMeter("accuracy")
            >>> # Update with batch accuracy and batch size
            >>> meter.update(acc, n=batch_size)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        """Return a string representation of the meter."""
        return f"{self.name}: {self.avg:.4f} (current: {self.val:.4f})"


def count_parameters(
    model: torch.nn.Module,
    trainable_only: bool = False
) -> int:
    """
    Count the number of parameters in a PyTorch model.

    Args:
        model: The PyTorch model to analyze.
        trainable_only: If True, count only trainable parameters
                       (parameters with requires_grad=True).
                       If False, count all parameters.

    Returns:
        num_params: Total number of parameters.

    Example:
        >>> total = count_parameters(model)
        >>> trainable = count_parameters(model, trainable_only=True)
        >>> print(f"Total: {total:,} ({total/1e6:.2f}M)")
        >>> print(f"Trainable: {trainable:,} ({trainable/1e6:.2f}M)")
        Total: 125,234,176 (125.23M)
        Trainable: 125,234,176 (125.23M)
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage statistics.

    Returns a dictionary containing memory usage information in gigabytes (GB).
    If CUDA is not available, returns zeros for all values.

    Returns:
        memory_info: Dictionary with the following keys:
            - 'allocated': Memory allocated by PyTorch tensors (GB)
            - 'reserved': Memory reserved by PyTorch's caching allocator (GB)
            - 'free': Estimated free memory on the GPU (GB)
            - 'total': Total GPU memory capacity (GB)

    Note:
        The 'free' memory is calculated as total - reserved, which gives
        an estimate of memory available for allocation. The actual available
        memory may differ due to other processes using the GPU.

    Example:
        >>> mem = get_memory_usage()
        >>> print(f"Using {mem['allocated']:.2f}GB / {mem['total']:.2f}GB")
        >>> print(f"Free: {mem['free']:.2f}GB")
        Using 4.23GB / 16.00GB
        Free: 11.77GB
    """
    if not torch.cuda.is_available():
        return {
            "allocated": 0.0,
            "reserved": 0.0,
            "free": 0.0,
            "total": 0.0
        }

    # Memory currently allocated to tensors (GB)
    allocated = torch.cuda.memory_allocated() / 1e9

    # Memory reserved by PyTorch's caching allocator (GB)
    reserved = torch.cuda.memory_reserved() / 1e9

    # Total GPU memory capacity
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 1e9

    # Free memory = total - reserved (reserved includes allocated)
    free = total - reserved

    return {
        "allocated": allocated,
        "reserved": reserved,
        "free": free,
        "total": total,
    }
