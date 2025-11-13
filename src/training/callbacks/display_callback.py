"""Real-time display callback for training progress."""

from __future__ import annotations
import math
import time
from typing import Dict, Any, Optional
from src.monitoring.realtime_display import RealtimeTrainingDisplay


class DisplayCallback:
    """
    Callback for real-time training display.

    Provides:
    - Live metrics display (updated every step)
    - Detailed periodic logs
    - Progress tracking
    """

    def __init__(
        self,
        max_steps: int,
        detail_every: int = 50,
        width: int = 100,
        enabled: bool = True,
    ):
        """
        Initialize display callback.

        Args:
            max_steps: Total training steps
            detail_every: Print detailed logs every N steps
            width: Display width in characters
            enabled: Whether to enable real-time display
        """
        self.enabled = enabled
        self.detail_every = detail_every

        if self.enabled:
            self.display = RealtimeTrainingDisplay(
                max_steps=max_steps,
                detail_every=detail_every,
                width=width,
            )
        else:
            self.display = None

        # Performance tracking
        self.step_start_time = time.time()
        self.steps_since_log = 0

    def update_live(self, step: int, metrics: Dict[str, Any]) -> None:
        """
        Update live display with current metrics.

        Args:
            step: Current training step
            metrics: Current metrics dictionary
        """
        if not self.enabled or self.display is None:
            return

        self.display.update_live(
            step=step,
            loss=metrics.get("loss"),
            ppl=metrics.get("perplexity"),
            lr=metrics.get("lr"),
            throughput=metrics.get("throughput"),
            gpu_mem_gb=metrics.get("gpu_mem_gb"),
            gpu_total_gb=metrics.get("gpu_total_gb"),
            seq_len=metrics.get("seq_len"),
            global_weight=metrics.get("global_weight"),
            num_landmarks=metrics.get("num_landmarks"),
            active_landmarks=metrics.get("num_landmarks"),  # Same as num_landmarks for SLGA
        )

    def on_log(self, metrics: Dict[str, Any], step: int, **kwargs) -> None:
        """
        Handle logging callback - print detailed metrics.

        Args:
            metrics: Dictionary of metrics to display
            step: Current training step
        """
        if not self.enabled or self.display is None:
            # Fallback to basic console logging
            self._print_basic(metrics, step)
            return

        # Calculate throughput
        elapsed_time = time.time() - self.step_start_time
        if elapsed_time > 0:
            steps_per_sec = self.steps_since_log / elapsed_time
            batch_size = kwargs.get("batch_size", 1)
            seq_len = metrics.get("seq_len", 512)
            throughput = steps_per_sec * batch_size * seq_len
            metrics["throughput"] = throughput

        # Print detailed display
        self.display.print_detailed(
            step=step,
            loss=metrics.get("loss", 0.0),
            ppl=metrics.get("perplexity", 0.0),
            lr=metrics.get("lr", 0.0),
            grad_norm=metrics.get("grad_norm", 0.0),
            seq_len=metrics.get("seq_len", 512),
            global_weight=metrics.get("global_weight", 0.0),
            throughput=metrics.get("throughput", 0.0),
            gpu_mem_gb=metrics.get("gpu_mem_gb", 0.0),
            gpu_total_gb=metrics.get("gpu_total_gb", 0.0),
            num_landmarks=metrics.get("num_landmarks", 0),
            spacing_loss=metrics.get("spacing_loss", 0.0),
            sparsity_loss=metrics.get("sparsity_loss", 0.0),
        )

        # Reset timing
        self.step_start_time = time.time()
        self.steps_since_log = 0

    def on_validation(self, metrics: Dict[str, float], step: int, **kwargs) -> None:
        """
        Display validation metrics.

        Args:
            metrics: Validation metrics dictionary
            step: Current training step
        """
        if not self.enabled or self.display is None:
            print(f"Val Loss: {metrics['loss']:.4f} | Val PPL: {metrics['perplexity']:.2f}")
            return

        self.display.print_validation(
            step=step,
            val_loss=metrics["loss"],
            val_ppl=metrics["perplexity"],
        )

    def increment_steps(self) -> None:
        """Increment step counter for throughput calculation."""
        self.steps_since_log += 1

    def _print_basic(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Fallback basic console logging.

        Args:
            metrics: Metrics dictionary
            step: Current training step
        """
        loss = metrics.get("loss", 0.0)
        ppl = metrics.get("perplexity", 0.0)
        lr = metrics.get("lr")  # Can be None between optimizer steps
        grad_norm = metrics.get("grad_norm", 0.0)
        optimizer_step = metrics.get("optimizer_step", 0)

        # Format lr safely (handle None)
        lr_str = f"{lr:.2e}" if lr is not None else "N/A"

        print(
            f"Step {step:6d} (opt {optimizer_step:5d}) | "
            f"Loss: {loss:.4f} | PPL: {ppl:7.2f} | "
            f"LR: {lr_str} | GradNorm: {grad_norm:5.2f}"
        )

        # Second line with additional metrics
        seq_len = metrics.get("seq_len", 512)
        global_weight = metrics.get("global_weight", 0.0)
        num_landmarks = metrics.get("num_landmarks", 0)
        gpu_mem = metrics.get("gpu_mem_gb", 0.0)
        throughput = metrics.get("throughput", 0.0)

        print(
            f"           | SeqLen: {seq_len:4d} | GW: {global_weight:.2f} | "
            f"LM: {num_landmarks} | GPU: {gpu_mem:4.1f}GB | "
            f"Tok/s: {throughput:5.0f}"
        )
