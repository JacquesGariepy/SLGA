"""Metrics logging callback for W&B and TensorBoard."""

from __future__ import annotations
from typing import Dict, Any, Optional
import torch


class MetricsCallback:
    """
    Callback for logging metrics to W&B and TensorBoard.

    Handles:
    - W&B logging
    - TensorBoard logging
    - Metric tracking and aggregation
    """

    def __init__(self, config: Dict[str, Any], writer: Optional[Any] = None):
        """
        Initialize metrics callback.

        Args:
            config: Training configuration dictionary
            writer: Optional TensorBoard SummaryWriter
        """
        self.config = config
        self.writer = writer
        self.use_wandb = config["log"].get("wandb", False)
        self.use_tensorboard = writer is not None

        # Initialize W&B if enabled
        if self.use_wandb:
            import wandb
            run_name = config["log"].get("run_name") or f"slga_{config['model']['embed_dim']}d_{config['model']['n_layers']}l"
            wandb.init(
                project=config["log"]["project"],
                name=run_name,
                config=config,
            )

    def on_log(self, metrics: Dict[str, Any], step: int, **kwargs) -> None:
        """
        Log metrics callback.

        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
        """
        # W&B logging
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

        # TensorBoard logging
        if self.use_tensorboard and self.writer is not None:
            self._log_to_tensorboard(metrics, step)

    def on_validation(self, metrics: Dict[str, float], step: int, **kwargs) -> None:
        """
        Log validation metrics.

        Args:
            metrics: Validation metrics dictionary
            step: Current training step
        """
        # W&B
        if self.use_wandb:
            import wandb
            wandb.log(
                {
                    "val_loss": metrics["loss"],
                    "val_perplexity": metrics["perplexity"],
                },
                step=step,
            )

        # TensorBoard
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_scalar("val/loss", metrics["loss"], step)
            self.writer.add_scalar("val/perplexity", metrics["perplexity"], step)

    def _log_to_tensorboard(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Log metrics to TensorBoard.

        Args:
            metrics: Dictionary of metrics
            step: Current training step
        """
        # Basic metrics
        if "loss" in metrics:
            self.writer.add_scalar("train/loss", metrics["loss"], step)
        if "perplexity" in metrics:
            self.writer.add_scalar("train/perplexity", metrics["perplexity"], step)
        if "lr" in metrics and metrics["lr"] is not None:
            self.writer.add_scalar("train/learning_rate", metrics["lr"], step)

        # Curriculum learning
        if "seq_len" in metrics:
            self.writer.add_scalar("train/seq_len", metrics["seq_len"], step)
        if "global_weight" in metrics:
            self.writer.add_scalar("train/global_weight", metrics["global_weight"], step)

        # Gradient metrics
        if "grad_norm" in metrics:
            self.writer.add_scalar("train/grad_norm", metrics["grad_norm"], step)

        # Landmark metrics
        if "num_landmarks" in metrics and metrics["num_landmarks"] > 0:
            self.writer.add_scalar("landmarks/num_selected", metrics["num_landmarks"], step)
        if "spacing_loss" in metrics and metrics["spacing_loss"] > 0:
            self.writer.add_scalar("train/loss_spacing", metrics["spacing_loss"], step)
        if "sparsity_loss" in metrics and metrics["sparsity_loss"] > 0:
            self.writer.add_scalar("train/loss_sparsity", metrics["sparsity_loss"], step)
        if "mass_ratio" in metrics and metrics["mass_ratio"] > 0:
            self.writer.add_scalar("landmarks/mass_in_top_g", metrics["mass_ratio"], step)

        # Gate metrics
        if "gate_entropy" in metrics:
            self.writer.add_scalar("train/gate_entropy", metrics["gate_entropy"], step)
        if "gate_local_mean" in metrics:
            self.writer.add_scalar("train/pct_tokens_local", metrics["gate_local_mean"], step)
            self.writer.add_scalar("train/pct_tokens_global", 1.0 - metrics["gate_local_mean"], step)
        if "gate_local_majority" in metrics:
            self.writer.add_scalar("train/pct_tokens_local_majority", metrics["gate_local_majority"], step)

        # Landmark coverage
        if "landmark_coverage" in metrics:
            self.writer.add_scalar("train/landmark_coverage", metrics["landmark_coverage"], step)

        # Performance metrics
        if "throughput" in metrics:
            self.writer.add_scalar("perf/tokens_per_sec", metrics["throughput"], step)
        if "gpu_mem_gb" in metrics:
            self.writer.add_scalar("perf/gpu_memory_allocated_gb", metrics["gpu_mem_gb"], step)

    def close(self) -> None:
        """Cleanup and close logging resources."""
        if self.use_wandb:
            import wandb
            wandb.finish()

        if self.writer is not None:
            self.writer.close()
