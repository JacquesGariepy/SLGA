"""Optimizer factory with parameter group configuration."""

from __future__ import annotations
import torch
from typing import Dict, Any, List, Optional
from torch.optim.lr_scheduler import _LRScheduler


def get_optimizer_groups(
    model: torch.nn.Module,
    base_lr: float,
    scorer_lr_multiplier: float = 5.0,
) -> List[Dict[str, Any]]:
    """
    Create optimizer parameter groups with separate LR for landmark scorer.

    Args:
        model: Model to create parameter groups for
        base_lr: Base learning rate
        scorer_lr_multiplier: Multiplier for landmark scorer LR (default: 5x)

    Returns:
        List of parameter group dictionaries
    """
    # Separate parameters: scorer vs rest of model
    scorer_params = []
    other_params = []

    if hasattr(model, "landmark_selector") and model.landmark_selector is not None:
        scorer_params = list(model.landmark_selector.scorer.parameters())
        # All other parameters
        scorer_param_ids = {id(p) for p in scorer_params}
        other_params = [p for p in model.parameters() if id(p) not in scorer_param_ids]
    else:
        other_params = list(model.parameters())

    # Create param groups
    param_groups = [
        {
            "params": other_params,
            "lr": base_lr,
            "name": "model",
        },
    ]

    if scorer_params:
        param_groups.append({
            "params": scorer_params,
            "lr": base_lr * scorer_lr_multiplier,
            "name": "scorer",
        })

    return param_groups


def create_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any],
    scorer_lr_multiplier: float = 5.0,
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer with parameter groups.

    Args:
        model: Model to optimize
        config: Training configuration dictionary
        scorer_lr_multiplier: Multiplier for landmark scorer LR

    Returns:
        Configured AdamW optimizer
    """
    base_lr = config["train"]["lr"]
    param_groups = get_optimizer_groups(model, base_lr, scorer_lr_multiplier)

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=tuple(config["train"]["betas"]),
        eps=config["train"]["eps"],
        weight_decay=config["train"]["weight_decay"],
    )

    return optimizer


def _unwrap_scheduler(scheduler: Optional[_LRScheduler]) -> Optional[_LRScheduler]:
    """
    Unwrap nested Accelerate schedulers to reach the underlying PyTorch scheduler.

    Accelerate wraps schedulers multiple times, this function peels back the layers
    to get to the actual torch scheduler for direct manipulation.

    Args:
        scheduler: Potentially wrapped scheduler

    Returns:
        Unwrapped scheduler or None
    """
    current = scheduler
    visited = set()
    while current is not None and hasattr(current, "scheduler") and id(current) not in visited:
        visited.add(id(current))
        current = getattr(current, "scheduler")
    return current


def apply_lr_from_config(
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[_LRScheduler],
    base_lr: float,
    scorer_lr_multiplier: float,
) -> None:
    """
    Re-sync optimizer and scheduler learning rates with config values after wrapping/resuming.

    This ensures that after loading checkpoints or wrapping with Accelerate,
    the learning rates match the intended values from the config.

    Args:
        optimizer: Optimizer to update
        scheduler: Scheduler to update (may be wrapped by Accelerate)
        base_lr: Base learning rate
        scorer_lr_multiplier: Multiplier for scorer learning rate
    """
    real_scheduler = _unwrap_scheduler(scheduler)

    target_lrs = []
    for group in optimizer.param_groups:
        name = group.get("name", "")
        lr = base_lr * scorer_lr_multiplier if name == "scorer" else base_lr
        group["lr"] = lr
        if "initial_lr" in group:
            group["initial_lr"] = lr
        target_lrs.append(lr)

    if real_scheduler is not None:
        if hasattr(real_scheduler, "base_lrs"):
            real_scheduler.base_lrs = list(target_lrs)
        if hasattr(real_scheduler, "_last_lr"):
            real_scheduler._last_lr = list(target_lrs)
