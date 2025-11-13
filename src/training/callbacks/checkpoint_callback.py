"""Checkpoint saving and loading callback."""

from __future__ import annotations
import os
import shutil
import subprocess
import sys
from typing import Optional, Dict, Any
from scripts.utils import save_checkpoint, load_checkpoint


class CheckpointCallback:
    """
    Callback for checkpoint management.

    Handles:
    - Periodic checkpoint saving
    - Best model tracking (overall and per-sequence)
    - Checkpoint rotation (keep_last_n)
    - Resume from checkpoint
    - Generation preview after checkpointing
    """

    def __init__(
        self,
        config: Dict[str, Any],
        keep_last_n: Optional[int] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize checkpoint callback.

        Args:
            config: Training configuration dictionary
            keep_last_n: Number of recent checkpoints to keep (None = keep all)
            config_path: Path to config file for generation preview
        """
        self.config = config
        self.keep_last_n = keep_last_n or config["train"].get("keep_last_checkpoints")
        self.config_path = config_path
        self.out_dir = config["save"]["out_dir"]

        # Create directories
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "best_sequence"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "best_overall"), exist_ok=True)

    def on_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        step: int,
        state: Any,
        out_dir: str,
        custom_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Save checkpoint callback.

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            step: Current training step
            state: Training state object
            out_dir: Output directory
            custom_dir: Optional custom directory name
        """
        # Build extra state
        extra_state = self._build_extra_state(state)

        # Save checkpoint
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            out_dir,
            step,
            accelerator=kwargs.get("accelerator"),
            keep_last_n=self.keep_last_n,
            extra_state=extra_state,
            custom_dir=custom_dir,
        )

        print(f"✅ Checkpoint step {step} saved successfully!")

        # Run generation preview if config path provided
        if self.config_path:
            checkpoint_dir = os.path.join(out_dir, custom_dir or f"ckpt_{step}")
            self._run_quick_generate(checkpoint_dir)

    def on_validation(
        self,
        metrics: Dict[str, float],
        step: int,
        model,
        optimizer,
        scheduler,
        state: Any,
        accelerator,
        **kwargs,
    ) -> None:
        """
        Handle validation callback - save best checkpoints.

        Args:
            metrics: Validation metrics dictionary
            step: Current training step
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            state: Training state object
            accelerator: Accelerator instance
        """
        val_loss = metrics["loss"]
        current_seq_len = kwargs.get("current_seq_len", 512)

        # Best per sequence length
        prev_seq_best = state.best_loss_per_sequence.get(
            current_seq_len, {"loss": float("inf"), "step": -1}
        )

        if val_loss < prev_seq_best["loss"]:
            state.best_loss_per_sequence[current_seq_len] = {"loss": val_loss, "step": step}

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                seq_dir = os.path.join("best_sequence", f"seq_{current_seq_len}")
                full_seq_dir = os.path.join(self.out_dir, seq_dir)

                # Remove old checkpoint
                if os.path.isdir(full_seq_dir):
                    shutil.rmtree(full_seq_dir)

                print(f"\n💾 New best for seq_len={current_seq_len}: {val_loss:.4f} (step {step})")

                # Save checkpoint
                self.on_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    state=state,
                    out_dir=self.out_dir,
                    custom_dir=seq_dir,
                    accelerator=accelerator,
                )

                # Run generation preview
                if self.config_path:
                    self._run_quick_generate(full_seq_dir)

        # Best overall
        if val_loss < state.best_overall["loss"]:
            state.best_overall = {"loss": val_loss, "step": step}

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                overall_dir = "best_overall"
                full_overall_dir = os.path.join(self.out_dir, overall_dir)

                if os.path.isdir(full_overall_dir):
                    shutil.rmtree(full_overall_dir)

                print(f"\n💾 New best overall validation loss: {val_loss:.4f} (step {step})")

                # Save checkpoint
                self.on_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    state=state,
                    out_dir=self.out_dir,
                    custom_dir=overall_dir,
                    accelerator=accelerator,
                )

                # Run generation preview
                if self.config_path:
                    self._run_quick_generate(full_overall_dir)

    def load_from_checkpoint(
        self,
        checkpoint_path: str,
        model,
        optimizer,
        scheduler,
        device: str,
    ) -> tuple[int, Dict[str, Any]]:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            model: Model to load into
            optimizer: Optimizer to restore
            scheduler: Scheduler to restore
            device: Device string

        Returns:
            Tuple of (step, extra_state)
        """
        extra_state: Dict[str, Any] = {}
        step = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            device=device,
            extra_state=extra_state,
        )

        return step, extra_state

    def _build_extra_state(self, state: Any) -> Dict[str, Any]:
        """
        Build extra state dictionary for checkpoint.

        Args:
            state: Training state object

        Returns:
            Dictionary with best losses and steps
        """
        return {
            "best_overall": dict(state.best_overall),
            "best_loss_per_sequence": {
                int(k): {"loss": float(v["loss"]), "step": int(v.get("step", -1))}
                for k, v in state.best_loss_per_sequence.items()
            },
        }

    def _run_quick_generate(self, checkpoint_dir: str) -> None:
        """
        Run quick generation preview.

        Args:
            checkpoint_dir: Path to checkpoint directory
        """
        if not self.config_path:
            return

        script_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "generate.py")
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        config_path = os.path.abspath(self.config_path)

        if not os.path.isdir(checkpoint_dir):
            print(f"⚠️  Generation skipped, checkpoint dir missing: {checkpoint_dir}")
            return

        cmd = [
            sys.executable,
            script_path,
            "--config",
            config_path,
            "--prompt",
            "Albert Einstein (14 March 1879 – 18 April 1955) was",
            "--temperature",
            "0.7",
            "--repetition-penalty",
            "1.25",
            "--no-repeat-ngram-size",
            "4",
            "--max-tokens",
            "50",
            "--checkpoint",
            checkpoint_dir,
        ]

        print(f"▶ Quick generate from {checkpoint_dir}...")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout.strip())
            if result.stderr:
                print(result.stderr.strip())
        except Exception as exc:
            print(f"⚠️  Generation preview failed: {exc}")
