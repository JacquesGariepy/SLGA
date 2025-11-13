"""
Checkpoint loading utilities for SLGA models.
Handles loading model weights and extracting metadata.
"""

import os
import pickle
import torch
from datetime import datetime
from typing import Dict, Any, Tuple


def _format_metric(value: float, decimal_places: int = 4) -> str:
    """Display tiny magnitudes without dropping them to zero.

    Args:
        value: Metric value to format
        decimal_places: Number of decimal places

    Returns:
        Formatted string
    """
    if value is None:
        return "N/A"
    if value != 0.0 and abs(value) < 10 ** (-decimal_places):
        return f"{value:.{decimal_places + 2}e}"
    return f"{value:.{decimal_places}f}"


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    device: str = "cpu",
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load a checkpoint and return model with metadata.

    Supports two formats:
    1. Directory: out_slga/ckpt_11000/ (contains model.pt and trainer_state.pt)
    2. File: path/to/model.pt (direct state dict)

    Args:
        checkpoint_path: Path to checkpoint (directory or file)
        model: Model to load weights into
        device: Device to map tensors to (default: cpu)

    Returns:
        Tuple of (model with loaded weights, metadata dict)

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    print(f"Loading checkpoint from {checkpoint_path}...")

    metadata = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_type": None,
        "step": None,
        "loss": None,
        "timestamp": None,
    }

    if os.path.isdir(checkpoint_path):
        # Directory format: out_slga/ckpt_11000/
        metadata["checkpoint_type"] = "directory"

        # Extract step from directory name
        dir_name = os.path.basename(checkpoint_path)
        if dir_name.startswith("ckpt_"):
            try:
                metadata["step"] = int(dir_name.split("_")[1])
            except (ValueError, IndexError) as e:
                print(f"⚠ Warning: Could not parse step from '{dir_name}': {e}")

        model_path = os.path.join(checkpoint_path, "model.pt")
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.pt")

        if not os.path.exists(model_path):
            available = os.listdir(checkpoint_path)
            raise FileNotFoundError(
                f"❌ model.pt not found in {checkpoint_path}\n"
                f"   Available files: {available}\n"
                f"   Expected: model.pt (state dict of the model)"
            )

        print(f"  Loading state dict from {model_path}...")
        state_dict = _load_state_dict(model_path, device)

        # Load trainer metadata if available
        if os.path.exists(trainer_state_path):
            try:
                trainer_state = torch.load(trainer_state_path, map_location=device)
                metadata["step"] = trainer_state.get("step", metadata["step"])
                metadata["loss"] = trainer_state.get("loss", None)
                metadata["timestamp"] = datetime.fromtimestamp(
                    os.path.getmtime(trainer_state_path)
                ).isoformat()
            except Exception as e:
                print(f"⚠ Warning: Could not load trainer state: {e}")

    else:
        # Direct file format: model.pt
        metadata["checkpoint_type"] = "file"

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ Checkpoint file not found: {checkpoint_path}")

        print(f"  Loading state dict from file...")
        state_dict = _load_state_dict(checkpoint_path, device)

        try:
            metadata["timestamp"] = datetime.fromtimestamp(
                os.path.getmtime(checkpoint_path)
            ).isoformat()
        except OSError as e:
            print(f"⚠ Warning: Could not get file modification time: {e}")

    # Load weights into model
    _load_weights(model, state_dict, metadata)

    return model, metadata


def _load_state_dict(path: str, device: str) -> Dict[str, torch.Tensor]:
    """Load state dict from file with error handling.

    Args:
        path: Path to state dict file
        device: Device to map tensors to

    Returns:
        State dictionary

    Raises:
        RuntimeError: If loading fails
    """
    try:
        return torch.load(path, map_location=device)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load checkpoint file (PyTorch error): {e}\n"
            f"   File may be corrupted or from incompatible PyTorch version"
        ) from e
    except pickle.UnpicklingError as e:
        raise RuntimeError(
            f"Failed to unpickle checkpoint file: {e}\n"
            f"   File appears to be corrupted"
        ) from e
    except EOFError as e:
        raise RuntimeError(
            f"Checkpoint file is incomplete (truncated): {e}\n"
            f"   File may have been interrupted during save"
        ) from e


def _load_weights(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    metadata: Dict[str, Any],
) -> None:
    """Load state dict into model with validation.

    Args:
        model: Model to load weights into
        state_dict: State dictionary
        metadata: Metadata dict to update

    Raises:
        RuntimeError: If loading fails
    """
    try:
        model.load_state_dict(state_dict)
        print("✓ Checkpoint loaded successfully")
        print(f"  Loaded {len(state_dict)} parameter tensors")

        # Sanity check - verify weights are not random
        first_param = next(iter(state_dict.values()))
        param_mean = first_param.float().mean().item()
        print(
            "  Sanity check - first param mean: "
            f"{_format_metric(param_mean, decimal_places=6)}"
        )

        metadata["num_parameters"] = len(state_dict)
        metadata["first_param_mean"] = param_mean

        if metadata["step"] is not None:
            print(f"  Step: {metadata['step']}")
        if metadata["loss"] is not None:
            print(f"  Loss: {_format_metric(metadata['loss'])}")

    except (RuntimeError, TypeError, KeyError) as e:
        print(f"❌ Error loading state dict into model: {e}")
        print(f"   This usually means checkpoint architecture mismatch.")
        print(f"   Available keys in state dict: {list(state_dict.keys())[:5]}...")
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e


def get_checkpoint_info(metadata: Dict[str, Any]) -> str:
    """Format checkpoint metadata as human-readable string.

    Args:
        metadata: Checkpoint metadata dictionary

    Returns:
        Formatted string with checkpoint information
    """
    lines = [
        "Checkpoint Information:",
        f"  Path: {metadata['checkpoint_path']}",
        f"  Type: {metadata['checkpoint_type']}",
    ]

    if metadata['step'] is not None:
        lines.append(f"  Step: {metadata['step']}")

    if metadata['loss'] is not None:
        lines.append(f"  Loss: {_format_metric(metadata['loss'])}")

    if metadata['timestamp']:
        lines.append(f"  Saved: {metadata['timestamp']}")

    if 'num_parameters' in metadata:
        lines.append(f"  Parameters: {metadata['num_parameters']} tensors")

    if 'first_param_mean' in metadata:
        lines.append(
            f"  First param mean: "
            f"{_format_metric(metadata['first_param_mean'], decimal_places=6)}"
        )

    return "\n".join(lines)
