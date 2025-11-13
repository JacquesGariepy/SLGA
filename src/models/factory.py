"""Model factory for creating models with registry pattern."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch

from src.models.config import ModelConfig, get_config
from src.models.slga_model import SLGATransformer


class ModelFactory:
    """Factory for creating SLGA models with registry pattern.

    Supports:
    - Creating models from config
    - Creating models from presets
    - Loading models from checkpoints
    - Registering custom model variants

    Example:
        >>> factory = ModelFactory()
        >>> model = factory.create("small")
        >>> model = factory.create_from_config(config)
        >>> model = factory.load_checkpoint("path/to/checkpoint.pt")
    """

    # Registry for custom model variants
    _registry: Dict[str, Callable[[ModelConfig], SLGATransformer]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        builder: Callable[[ModelConfig], SLGATransformer],
    ) -> None:
        """Register a custom model variant.

        Args:
            name: Variant name
            builder: Function that takes ModelConfig and returns model

        Example:
            >>> def build_custom(config: ModelConfig) -> SLGATransformer:
            ...     config.embed_dim = 1024
            ...     return SLGATransformer(config)
            >>> ModelFactory.register("custom", build_custom)
        """
        cls._registry[name] = builder

    @classmethod
    def create(
        cls,
        variant: str = "small",
        **config_overrides: Any,
    ) -> SLGATransformer:
        """Create model from preset or custom variant.

        Args:
            variant: Preset name ("tiny", "small", "medium", "large") or custom variant
            **config_overrides: Override specific config values

        Returns:
            Initialized model

        Example:
            >>> model = ModelFactory.create("small", n_layers=16)
        """
        # Check if it's a custom variant
        if variant in cls._registry:
            config = get_config("small")  # Start with default
            for key, value in config_overrides.items():
                setattr(config, key, value)
            return cls._registry[variant](config)

        # Otherwise use preset
        config = get_config(variant)

        # Apply overrides
        for key, value in config_overrides.items():
            if not hasattr(config, key):
                raise ValueError(f"Unknown config parameter: {key}")
            setattr(config, key, value)

        return cls.create_from_config(config)

    @classmethod
    def create_from_config(cls, config: ModelConfig) -> SLGATransformer:
        """Create model from configuration.

        Args:
            config: Model configuration

        Returns:
            Initialized model

        Example:
            >>> config = ModelConfig(vocab_size=50257, embed_dim=512)
            >>> model = ModelFactory.create_from_config(config)
        """
        return SLGATransformer(config)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        strict: bool = True,
    ) -> SLGATransformer:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on
            strict: Whether to strictly enforce state dict matching

        Returns:
            Loaded model

        Example:
            >>> model = ModelFactory.load_checkpoint("checkpoint.pt", device="cuda")
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract config and state dict
        if isinstance(checkpoint, dict):
            if "config" in checkpoint:
                # Checkpoint with config
                config_dict = checkpoint["config"]
                config = ModelConfig(**config_dict)
                state_dict = checkpoint["model_state_dict"]
            elif "model_state_dict" in checkpoint:
                # Checkpoint without config - use default
                config = get_config("small")
                state_dict = checkpoint["model_state_dict"]
            else:
                # Direct state dict
                config = get_config("small")
                state_dict = checkpoint
        else:
            raise ValueError("Checkpoint format not recognized")

        # Create model and load weights
        model = cls.create_from_config(config)
        model.load_state_dict(state_dict, strict=strict)
        model.to(device)

        return model

    @classmethod
    def save_checkpoint(
        cls,
        model: SLGATransformer,
        checkpoint_path: str,
        include_optimizer: bool = False,
        optimizer_state: Optional[Dict] = None,
        **metadata: Any,
    ) -> None:
        """Save model checkpoint.

        Args:
            model: Model to save
            checkpoint_path: Path to save checkpoint
            include_optimizer: Whether to include optimizer state
            optimizer_state: Optimizer state dict (if include_optimizer=True)
            **metadata: Additional metadata to save

        Example:
            >>> ModelFactory.save_checkpoint(
            ...     model,
            ...     "checkpoint.pt",
            ...     include_optimizer=True,
            ...     optimizer_state=optimizer.state_dict(),
            ...     epoch=10,
            ... )
        """
        checkpoint = {
            "config": model.config.model_dump(),
            "model_state_dict": model.state_dict(),
            **metadata,
        }

        if include_optimizer and optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state

        torch.save(checkpoint, checkpoint_path)

    @classmethod
    def list_variants(cls) -> Dict[str, str]:
        """List available model variants.

        Returns:
            Dictionary of variant names and descriptions
        """
        variants = {
            "tiny": "Tiny model (256 dim, 6 layers)",
            "small": "Small model (512 dim, 12 layers)",
            "medium": "Medium model (768 dim, 16 layers)",
            "large": "Large model (1024 dim, 24 layers)",
        }

        # Add custom variants
        for name in cls._registry:
            variants[name] = f"Custom variant: {name}"

        return variants


__all__ = ["ModelFactory"]
