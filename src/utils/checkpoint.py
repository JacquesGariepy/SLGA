"""
Model checkpoint management utilities.

This module provides functions for saving and loading model checkpoints,
including model weights, optimizer state, and scheduler state.
"""

import os
import json
import shutil
from typing import Any, Optional

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    out_dir: str,
    step: int,
    accelerator: Any,
    keep_last_n: Optional[int] = None,
    extra_state: Optional[dict] = None,
    custom_dir: Optional[str] = None,
) -> None:
    """
    Save a checkpoint of the model and training state.

    This function saves:
    - Model state dict (unwrapped from DDP/FSDP)
    - Model configuration (if available)
    - Training state (step, optimizer, scheduler)
    - Extra state (best losses, etc.)

    Args:
        model: The PyTorch model to save.
        optimizer: The optimizer state to save.
        scheduler: The learning rate scheduler state to save.
        out_dir: Directory where checkpoints will be saved.
        step: Current training step number.
        accelerator: Accelerator object (used to unwrap DDP/FSDP models).
        keep_last_n: If specified, only keep the N most recent checkpoints.
                    Older checkpoints will be automatically deleted.
        extra_state: Optional dictionary with additional state (e.g., best losses).
        custom_dir: Optional custom directory name (e.g., "best_overall").
                   If not provided, uses "ckpt_{step}".

    Example:
        >>> save_checkpoint(
        ...     model=my_model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     out_dir="./checkpoints",
        ...     step=1000,
        ...     accelerator=accelerator,
        ...     keep_last_n=3  # Keep only last 3 checkpoints
        ... )

    Directory structure:
        out_dir/
            ckpt_1000/
                model.pt              # Model weights
                model_config.json     # Model configuration
                trainer_state.pt      # Training state
    """
    # Use custom directory name or default to ckpt_{step}
    checkpoint_dir = os.path.join(out_dir, custom_dir or f"ckpt_{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Unwrap model (if DDP/FSDP wrapped)
    unwrapped_model = accelerator.unwrap_model(model)

    # Save model state dict
    torch.save(
        unwrapped_model.state_dict(),
        os.path.join(checkpoint_dir, "model.pt"),
    )

    # Save model config (for reconstruction)
    if hasattr(unwrapped_model, 'config'):
        config_dict = vars(unwrapped_model.config)
        with open(os.path.join(checkpoint_dir, "model_config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)

    # Save training state
    training_state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
    }

    # Add extra state if provided (best losses, etc.)
    if extra_state:
        training_state.update(extra_state)

    torch.save(
        training_state,
        os.path.join(checkpoint_dir, "trainer_state.pt"),
    )

    print(f"\n{'='*60}")
    print(f"✓ CHECKPOINT SAVED at step {step}")
    print(f"  Location: {checkpoint_dir}")
    print(f"  Files: model.pt, trainer_state.pt")
    print(f"{'='*60}\n")

    # Checkpoint rotation: keep only the N most recent checkpoints
    if keep_last_n is not None:
        all_ckpts = sorted(
            [d for d in os.listdir(out_dir) if d.startswith("ckpt_")],
            key=lambda x: int(x.split("_")[1])
        )
        if len(all_ckpts) > keep_last_n:
            for old in all_ckpts[:-keep_last_n]:
                old_path = os.path.join(out_dir, old)
                shutil.rmtree(old_path)
                print(f"  🗑️ Removed old checkpoint: {old}")


def load_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cuda",
    extra_state: Optional[dict] = None,
) -> int:
    """
    Load a checkpoint from disk.

    This function loads:
    - Model state dict
    - Optimizer state (if provided)
    - Scheduler state (if provided)
    - Extra state (if provided)

    Args:
        checkpoint_dir: Directory containing the checkpoint files.
        model: Model to load the weights into.
        optimizer: Optimizer to load state into (optional).
        scheduler: Scheduler to load state into (optional).
        device: Device to load tensors onto ('cuda' or 'cpu').
        extra_state: Dictionary to populate with extra state (modified in-place).

    Returns:
        step: The training step number from the checkpoint.

    Raises:
        FileNotFoundError: If model checkpoint file is not found.

    Example:
        >>> extra_state = {}
        >>> step = load_checkpoint(
        ...     checkpoint_dir="./checkpoints/ckpt_1000",
        ...     model=my_model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     device="cuda",
        ...     extra_state=extra_state
        ... )
        >>> print(f"Resumed from step {step}")
        >>> print(f"Best loss: {extra_state.get('best_overall')}")
    """
    # Load model (try multiple file names for compatibility)
    model_path = os.path.join(checkpoint_dir, "model.pt")
    if not os.path.exists(model_path):
        # Fallback: try HuggingFace format
        model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[✓] Model loaded from {model_path}")
    else:
        raise FileNotFoundError(
            f"Could not find model.pt or pytorch_model.bin in {checkpoint_dir}"
        )

    # Load training state
    step = 0
    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.pt")

    if os.path.exists(trainer_state_path):
        trainer_state = torch.load(trainer_state_path, map_location=device)

        step = trainer_state.get("step", 0)

        if optimizer and "optimizer" in trainer_state:
            optimizer.load_state_dict(trainer_state["optimizer"])
            print(f"[✓] Optimizer state loaded")

        if scheduler and "scheduler" in trainer_state and trainer_state["scheduler"]:
            scheduler.load_state_dict(trainer_state["scheduler"])
            print(f"[✓] Scheduler state loaded")

        # Load extra state if dictionary provided
        if extra_state is not None:
            # Standard keys that are not part of extra_state
            standard_keys = {"step", "optimizer", "scheduler"}
            for key, value in trainer_state.items():
                if key not in standard_keys:
                    extra_state[key] = value
            if extra_state:
                print(f"[✓] Extra state loaded: {list(extra_state.keys())}")

    print(f"[✓] Checkpoint loaded (step {step})")

    return step


def load_latest_checkpoint(
    out_dir: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cuda"
) -> int:
    """
    Load the most recent checkpoint from a directory.

    This function scans the output directory for checkpoint folders,
    identifies the most recent one by step number, and loads it.

    Args:
        out_dir: Directory containing checkpoint subdirectories.
        model: Model to load weights into.
        optimizer: Optimizer to load state into (optional).
        scheduler: Scheduler to load state into (optional).
        device: Device to load tensors onto ('cuda' or 'cpu').

    Returns:
        step: The training step number from the checkpoint.
              Returns 0 if no checkpoints are found.

    Example:
        >>> step = load_latest_checkpoint(
        ...     out_dir="./checkpoints",
        ...     model=my_model,
        ...     optimizer=optimizer,
        ...     device="cuda"
        ... )
        >>> if step > 0:
        ...     print(f"Resumed from step {step}")
        ... else:
        ...     print("No checkpoint found, starting fresh")
    """
    checkpoints = [d for d in os.listdir(out_dir) if d.startswith("ckpt_")]
    if not checkpoints:
        print(f"No checkpoints found in {out_dir}")
        return 0

    # Find checkpoint with highest step number
    latest = max(checkpoints, key=lambda x: int(x.split("_")[1]))
    latest_path = os.path.join(out_dir, latest)

    print(f"Loading latest checkpoint: {latest}")
    return load_checkpoint(latest_path, model, optimizer, scheduler, device)
