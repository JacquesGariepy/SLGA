"""Base trainer class with core training loop logic."""

from __future__ import annotations
import os
import time
import math
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm


@dataclass
class TrainingState:
    """Training state tracking."""
    step: int = 0
    optimizer_step: int = 0
    epoch: int = 0
    best_overall: Dict[str, float] = field(default_factory=lambda: {"loss": float("inf"), "step": -1})
    best_loss_per_sequence: Dict[int, Dict[str, float]] = field(default_factory=dict)


class BaseTrainer:
    """
    Base trainer class with core training loop functionality.

    Handles:
    - Training loop management
    - Gradient accumulation
    - AMP (Automatic Mixed Precision)
    - Checkpointing
    - Logging and metrics
    - Validation

    Attributes:
        config: Training configuration dictionary
        model: PyTorch model to train
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        train_loader: Training data loader
        val_loader: Validation data loader
        accelerator: Accelerate accelerator for distributed training
        state: Training state tracker
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        accelerator: Accelerator,
        tokenizer: Any,
    ):
        """
        Initialize the base trainer.

        Args:
            config: Training configuration dictionary
            model: Model to train
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            train_loader: Training data loader
            val_loader: Validation data loader
            accelerator: Accelerate accelerator
            tokenizer: Tokenizer (for pad_token_id)
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.accelerator = accelerator
        self.tokenizer = tokenizer

        # Training state
        self.state = TrainingState()

        # Training configuration
        self.max_steps = config["train"]["max_steps"]
        self.accum_steps = config["train"]["accum_steps"]
        self.grad_clip = config["train"]["grad_clip"]
        self.log_every = config["train"].get("log_every", 100)
        self.eval_every = config["train"].get("eval_every", 1000)
        self.save_every = config["train"].get("save_every", 5000)
        self.out_dir = config["save"]["out_dir"]

        # AMP setup
        self.amp_enabled = config["train"].get("amp", True)
        amp_dtype_str = config["train"].get("amp_dtype", "bf16").lower()

        if amp_dtype_str == "bf16" and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16

        # Performance tracking
        self.step_start_time = time.time()
        self.steps_since_log = 0
        self.last_grad_norm = 0.0

        # Callbacks
        self.callbacks: Dict[str, Callable] = {}

        # Create output directory
        os.makedirs(self.out_dir, exist_ok=True)

    def add_callback(self, name: str, callback: Callable) -> None:
        """Add a callback function."""
        self.callbacks[name] = callback

    def remove_callback(self, name: str) -> None:
        """Remove a callback function."""
        if name in self.callbacks:
            del self.callbacks[name]

    def trigger_callback(self, event: str, **kwargs) -> None:
        """Trigger all callbacks for a specific event."""
        for callback in self.callbacks.values():
            if hasattr(callback, event):
                getattr(callback, event)(**kwargs)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Execute a single training step.

        Args:
            batch: Batch of training data

        Returns:
            Dictionary containing loss and metrics

        Note:
            This should be overridden by subclasses to implement
            model-specific training logic.
        """
        raise NotImplementedError("Subclasses must implement train_step()")

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the given logits and labels.

        Args:
            logits: Model output logits (B, L, V)
            labels: Target labels (B, L)

        Returns:
            Loss scalar tensor

        Note:
            Should be overridden by subclasses for custom loss computation.
        """
        raise NotImplementedError("Subclasses must implement compute_loss()")

    def should_log(self) -> bool:
        """Check if we should log at this step."""
        return self.state.step % self.log_every == 0

    def should_validate(self) -> bool:
        """Check if we should validate at this step."""
        return self.state.step % self.eval_every == 0

    def should_save(self) -> bool:
        """Check if we should save a checkpoint at this step."""
        return self.state.step % self.save_every == 0 and self.state.step > 0

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log training metrics.

        Args:
            metrics: Dictionary of metrics to log
        """
        # Trigger logging callbacks
        self.trigger_callback("on_log", metrics=metrics, step=self.state.step)

        # Default console logging
        if self.accelerator.is_main_process:
            loss = metrics.get("loss", 0.0)
            ppl = metrics.get("perplexity", 0.0)
            lr = metrics.get("lr")  # Can be None between optimizer steps

            # Format lr safely (handle None)
            lr_str = f"{lr:.2e}" if lr is not None else "N/A"

            print(
                f"Step {self.state.step:6d} (opt {self.state.optimizer_step:5d}) | "
                f"Loss: {loss:.4f} | PPL: {ppl:7.2f} | LR: {lr_str}"
            )

    def train_epoch(self) -> None:
        """Train for one epoch (or until max_steps reached)."""
        self.model.train()

        for batch in self.train_loader:
            # Execute training step
            step_output = self.train_step(batch)

            # Gradient accumulation
            if (self.state.step + 1) % self.accum_steps == 0:
                # Calculate gradient norm
                if self.accelerator.is_main_process:
                    grad_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            grad_norm += param_norm.item() ** 2
                    grad_norm = grad_norm ** 0.5
                    self.last_grad_norm = grad_norm

                # Gradient clipping
                if self.grad_clip > 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                # Increment optimizer step
                self.state.optimizer_step += 1

            # Increment forward pass counter
            self.state.step += 1
            self.steps_since_log += 1

            # Logging
            if self.should_log():
                self.log_metrics(step_output)
                self.step_start_time = time.time()
                self.steps_since_log = 0

            # Validation
            if self.should_validate():
                self.validate()

            # Checkpointing
            if self.should_save():
                self.save_checkpoint()

            # Check if training is complete
            if self.state.step >= self.max_steps:
                break

    def train(self) -> None:
        """Main training loop."""
        print(f"\n{'='*80}")
        print(f"Starting training for {self.max_steps} steps")
        print(f"{'='*80}\n")

        while self.state.step < self.max_steps:
            self.state.epoch += 1
            self.train_epoch()

            if self.state.step >= self.max_steps:
                break

        # Final checkpoint
        if self.accelerator.is_main_process:
            self.save_checkpoint()
            print("\n=== Training Complete ===")

    def validate(self) -> Dict[str, float]:
        """
        Run validation loop.

        Returns:
            Dictionary containing validation metrics

        Note:
            Should be overridden by subclasses for custom validation logic.
        """
        raise NotImplementedError("Subclasses must implement validate()")

    def save_checkpoint(self, custom_dir: Optional[str] = None) -> None:
        """
        Save a checkpoint.

        Args:
            custom_dir: Optional custom directory name for checkpoint
        """
        if not self.accelerator.is_main_process:
            return

        self.accelerator.wait_for_everyone()

        # Trigger checkpoint callbacks
        self.trigger_callback(
            "on_checkpoint",
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.state.step,
            state=self.state,
            out_dir=self.out_dir,
            custom_dir=custom_dir,
        )

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Step number from checkpoint
        """
        # This will be implemented by checkpoint callback
        raise NotImplementedError("Use checkpoint callback to load checkpoints")
