"""
Training utilities for SLGA.

Includes:
- Seed setting for reproducibility
- Checkpoint saving/loading
- Model weight utilities
"""

from __future__ import annotations
import os
import random
import shutil
import torch
import numpy as np
from typing import Optional, Dict, Any


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Note: These can slow down training but ensure reproducibility
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    out_dir: str,
    step: int,
    accelerator: Any = None,
    keep_last_n: Optional[int] = None,
    extra_state: Optional[Dict[str, Any]] = None,
    custom_dir: Optional[str] = None,
) -> str:
    """
    Save a training checkpoint.
    
    Args:
        model: PyTorch model (may be wrapped by Accelerator)
        optimizer: Optimizer
        scheduler: LR scheduler
        out_dir: Base output directory
        step: Current training step
        accelerator: Accelerate accelerator (optional)
        keep_last_n: Keep only last N checkpoints (None = keep all)
        extra_state: Additional state to save
        custom_dir: Custom subdirectory name (default: ckpt_{step})
    
    Returns:
        Path to saved checkpoint directory
    """
    # Determine checkpoint directory
    if custom_dir:
        ckpt_dir = os.path.join(out_dir, custom_dir)
    else:
        ckpt_dir = os.path.join(out_dir, f"ckpt_{step}")
    
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Unwrap model if using Accelerator
    if accelerator is not None:
        model_to_save = accelerator.unwrap_model(model)
    else:
        model_to_save = model
    
    # Save model weights
    model_path = os.path.join(ckpt_dir, "model.pt")
    torch.save(model_to_save.state_dict(), model_path)
    
    # Save optimizer state
    optimizer_path = os.path.join(ckpt_dir, "optimizer.pt")
    torch.save(optimizer.state_dict(), optimizer_path)
    
    # Save scheduler state
    if scheduler is not None:
        scheduler_path = os.path.join(ckpt_dir, "scheduler.pt")
        torch.save(scheduler.state_dict(), scheduler_path)
    
    # Save training state
    state = {
        "step": step,
        "model_config": model_to_save.cfg.__dict__ if hasattr(model_to_save, 'cfg') else {},
    }
    
    if extra_state:
        state.update(extra_state)
    
    state_path = os.path.join(ckpt_dir, "state.pt")
    torch.save(state, state_path)
    
    print(f"💾 Saved checkpoint to {ckpt_dir}")
    
    # Rotation: keep only last N checkpoints
    if keep_last_n is not None and keep_last_n > 0 and custom_dir is None:
        _rotate_checkpoints(out_dir, keep_last_n)
    
    return ckpt_dir


def load_checkpoint(
    ckpt_dir: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
    extra_state: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Load a training checkpoint.
    
    Args:
        ckpt_dir: Checkpoint directory
        model: PyTorch model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load tensors to
        extra_state: Dict to populate with extra saved state
    
    Returns:
        Training step from checkpoint
    """
    print(f"📂 Loading checkpoint from {ckpt_dir}")
    
    # Load model weights
    model_path = os.path.join(ckpt_dir, "model.pt")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"  ✓ Loaded model weights")
    else:
        print(f"  ⚠ Model weights not found at {model_path}")
    
    # Load optimizer state
    if optimizer is not None:
        optimizer_path = os.path.join(ckpt_dir, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
            print(f"  ✓ Loaded optimizer state")
    
    # Load scheduler state
    if scheduler is not None:
        scheduler_path = os.path.join(ckpt_dir, "scheduler.pt")
        if os.path.exists(scheduler_path):
            scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))
            print(f"  ✓ Loaded scheduler state")
    
    # Load training state
    state_path = os.path.join(ckpt_dir, "state.pt")
    step = 0
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=device)
        step = state.get("step", 0)
        
        # Populate extra_state if provided
        if extra_state is not None:
            for key in state:
                if key not in ("step", "model_config"):
                    extra_state[key] = state[key]
        
        print(f"  ✓ Loaded training state (step={step})")
    
    return step


def _rotate_checkpoints(out_dir: str, keep_last_n: int) -> None:
    """
    Keep only the last N checkpoints, delete older ones.
    
    Args:
        out_dir: Directory containing checkpoints
        keep_last_n: Number of checkpoints to keep
    """
    # Find all checkpoint directories
    ckpt_dirs = []
    for name in os.listdir(out_dir):
        if name.startswith("ckpt_"):
            try:
                step = int(name.split("_")[1])
                ckpt_dirs.append((step, name))
            except (ValueError, IndexError):
                continue
    
    # Sort by step (newest first)
    ckpt_dirs.sort(key=lambda x: x[0], reverse=True)
    
    # Delete old checkpoints
    for step, name in ckpt_dirs[keep_last_n:]:
        path = os.path.join(out_dir, name)
        try:
            shutil.rmtree(path)
            print(f"🗑️ Deleted old checkpoint: {name}")
        except Exception as e:
            print(f"⚠️ Could not delete {name}: {e}")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_grad_norm(model: torch.nn.Module, norm_type: float = 2.0) -> float:
    """
    Compute gradient norm across all parameters.
    
    Args:
        model: PyTorch model
        norm_type: Type of norm (default: L2)
    
    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    return total_norm ** (1.0 / norm_type)


__all__ = [
    "set_seed",
    "save_checkpoint",
    "load_checkpoint",
    "count_parameters",
    "get_grad_norm",
]
