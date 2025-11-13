"""Warmup cosine learning rate scheduler."""

from __future__ import annotations
import torch
from typing import Dict, Any
from transformers import get_cosine_schedule_with_warmup


def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
) -> Any:
    """
    Create cosine learning rate scheduler with warmup.

    The scheduler uses optimizer steps (not forward passes):
    - If accum_steps=4 and max_steps=100000, scheduler sees 25000 steps
    - If warmup_steps=2000, scheduler warmup is 500 steps

    Args:
        optimizer: Optimizer to schedule
        config: Training configuration dictionary

    Returns:
        Cosine scheduler with warmup
    """
    total_steps = config["train"]["max_steps"]
    warmup_steps = config["train"]["warmup_steps"]
    accum_steps = config["train"]["accum_steps"]

    # CRITICAL: Scheduler must use OPTIMIZER steps (not forward passes)
    # Forward passes are accumulated before optimizer.step()
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps // accum_steps,
        num_training_steps=total_steps // accum_steps,
    )

    return scheduler
