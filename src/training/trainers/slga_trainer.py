"""SLGA-specific trainer with landmark losses."""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from contextlib import nullcontext

from .base_trainer import BaseTrainer
from src.legacy.landmarks import (
    landmark_spacing_loss,
    landmark_sparsity_loss,
    landmark_diversity_loss,
)


class SLGATrainer(BaseTrainer):
    """
    Trainer specialized for SLGA models with landmark-based attention.

    Adds:
    - Landmark auxiliary losses (spacing, sparsity, diversity)
    - Curriculum learning for sequence length
    - Progressive global attention warmup
    - Landmark metrics tracking
    """

    def __init__(self, *args, **kwargs):
        """Initialize SLGA trainer with auxiliary loss weights."""
        super().__init__(*args, **kwargs)

        # Landmark loss weights
        self.lambda_spacing = self.config["train"].get("lambda_spacing", 0.0)
        self.lambda_sparsity = self.config["train"].get("lambda_sparsity", 0.0)
        self.lambda_diversity = self.config["train"].get("lambda_diversity", 0.0)

        # Curriculum learning
        self.seq_len_start = self.config["train"].get("seq_len_start", 512)
        self.seq_len_mid = self.config["train"].get("seq_len_mid", 1024)
        self.seq_len_final = self.config["train"].get("seq_len_final", 2048)
        self.seq_len_warmup_steps = self.config["train"].get("seq_len_warmup_steps", 15000)

        # Global attention warmup
        self.global_warmup_start = self.config["train"].get("global_warmup_start", 30000)
        self.global_warmup_end = self.config["train"].get("global_warmup_end", 50000)

        # Persistent metrics
        self.last_num_landmarks = 0
        self.last_spacing_loss = 0.0
        self.last_spar_loss = 0.0
        self.last_mass_ratio = 0.0

    def get_current_seq_len(self) -> int:
        """
        Calculate current sequence length according to curriculum.

        Returns:
            Current sequence length for this training step
        """
        step = self.state.step
        warmup_steps = self.seq_len_warmup_steps

        if step < warmup_steps // 2:
            # Phase 1: start -> mid
            progress = step / (warmup_steps // 2)
            seq_len = self.seq_len_start + progress * (self.seq_len_mid - self.seq_len_start)
        elif step < warmup_steps:
            # Phase 2: mid -> final
            progress = (step - warmup_steps // 2) / (warmup_steps // 2)
            seq_len = self.seq_len_mid + progress * (self.seq_len_final - self.seq_len_mid)
        else:
            # Phase 3: final
            seq_len = self.seq_len_final

        return int(seq_len)

    def get_global_warmup_weight(self) -> float:
        """
        Calculate warmup weight for global attention.

        Returns:
            Weight between 0.0 and 1.0 for global attention contribution
        """
        step = self.state.step

        if step < self.global_warmup_start:
            return 0.0
        elif step < self.global_warmup_end:
            progress = (step - self.global_warmup_start) / (self.global_warmup_end - self.global_warmup_start)
            return progress
        else:
            return 1.0

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        aux: Dict[str, Any],
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute loss with auxiliary landmark losses.

        Args:
            logits: Model output logits (B, L, V)
            labels: Target labels (B, L) - already shifted, with -100 for padding
            aux: Auxiliary outputs from model

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Main cross-entropy loss (with proper shifting)
        # IMPORTANT: Labels are already shifted by collator!
        logits_shifted = logits[:, :-1, :].contiguous()  # (B, L-1, V)
        labels_shifted = labels[:, :-1].contiguous()     # (B, L-1)

        loss_ce = F.cross_entropy(
            logits_shifted.view(-1, logits_shifted.size(-1)),
            labels_shifted.view(-1),
            ignore_index=-100,
        )

        # Scale for gradient accumulation
        loss = loss_ce / self.accum_steps

        # Metrics dictionary
        metrics = {
            "loss_ce": loss_ce.item(),
            "spacing_loss": 0.0,
            "sparsity_loss": 0.0,
            "num_landmarks": 0,
            "mass_ratio": 0.0,
        }

        # Extract landmark outputs
        landmark_indices = aux.get("landmark_indices", None)
        landmark_scores = aux.get("landmark_scores", None)

        # Count selected landmarks
        if landmark_indices is not None and landmark_indices.numel() > 0:
            num_landmarks = landmark_indices.size(1)
            metrics["num_landmarks"] = num_landmarks
            self.last_num_landmarks = num_landmarks
        else:
            num_landmarks = 0

        # Auxiliary losses
        if landmark_indices is not None and landmark_scores is not None:
            seq_len = logits.size(1)

            # Spacing loss: encourages even distribution
            if self.lambda_spacing > 0 and num_landmarks > 1:
                spacing_loss = landmark_spacing_loss(
                    landmark_indices=landmark_indices,
                    seq_len=seq_len,
                    lambda_reg=self.lambda_spacing,
                    selection_scores=landmark_scores,
                )
                metrics["spacing_loss"] = spacing_loss.item()
                self.last_spacing_loss = spacing_loss.item()
                loss = loss + spacing_loss / self.accum_steps

            # Sparsity loss: encourages concentration on top landmarks
            if self.lambda_sparsity > 0 and num_landmarks > 0:
                spar_loss = landmark_sparsity_loss(
                    selection_scores=landmark_scores,
                    num_landmarks=num_landmarks,
                    lambda_reg=self.lambda_sparsity,
                )
                metrics["sparsity_loss"] = spar_loss.item()
                self.last_spar_loss = spar_loss.item()
                loss = loss + spar_loss / self.accum_steps

            # Legacy diversity loss (backward compatibility)
            if self.lambda_diversity > 0:
                div_loss = landmark_diversity_loss(landmark_scores, self.lambda_diversity)
                loss = loss + div_loss / self.accum_steps

            # Calculate mass concentration metric
            if self.lambda_sparsity > 0 and num_landmarks > 0:
                probs = F.softmax(landmark_scores, dim=-1)
                _, top_g_idx = torch.topk(landmark_scores, k=num_landmarks, dim=-1)
                top_g_probs = torch.gather(probs, dim=1, index=top_g_idx)
                mass_in_top_g = top_g_probs.sum(dim=-1).mean().item()
                metrics["mass_ratio"] = mass_in_top_g
                self.last_mass_ratio = mass_in_top_g

        return loss, metrics

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Execute a single SLGA training step.

        Args:
            batch: Batch of training data

        Returns:
            Dictionary containing loss and metrics
        """
        # Get current curriculum parameters
        current_seq_len = self.get_current_seq_len()
        global_weight = self.get_global_warmup_weight()

        # Prepare inputs
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        cache_ids = batch.get("cache_global_ids")

        # Truncate on CPU to reduce GPU memory
        if input_ids.size(1) > current_seq_len:
            input_ids = input_ids[:, :current_seq_len]
            labels = labels[:, :current_seq_len]

            if cache_ids is not None:
                cache_ids = torch.clamp(cache_ids, 0, current_seq_len - 1)

        # Move to device
        device = self.accelerator.device
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        cache_ids = cache_ids.to(device) if cache_ids is not None else None

        # Autocast context
        use_autocast = self.amp_enabled and device.type == "cuda"
        if use_autocast:
            autocast_ctx = torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        else:
            autocast_ctx = nullcontext()

        # Forward pass
        with autocast_ctx:
            logits, aux = self.model(
                input_ids,
                cache_global_ids=cache_ids,
                return_aux=True,
                global_weight=global_weight,
            )

            # Compute loss
            loss, metrics = self.compute_loss(logits, labels, aux)

            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n{'='*80}")
                print(f"❌ DIVERGENCE DETECTED at step {self.state.step}!")
                print(f"{'='*80}")
                print(f"   Total loss: {loss.item()}")
                print(f"   CE loss: {metrics['loss_ce']:.6f}")
                print(f"   Spacing loss: {metrics['spacing_loss']:.6f}")
                print(f"   Sparsity loss: {metrics['sparsity_loss']:.6f}")
                raise ValueError(f"Training diverged at step {self.state.step}")

        # Backward pass
        self.accelerator.backward(loss)

        # Prepare metrics for logging
        output_metrics = {
            "loss": metrics["loss_ce"],
            "perplexity": math.exp(min(metrics["loss_ce"], 10)),
            "lr": self.scheduler.get_last_lr()[0] if (self.state.step + 1) % self.accum_steps == 0 else None,
            "seq_len": current_seq_len,
            "global_weight": global_weight,
            "num_landmarks": metrics["num_landmarks"],
            "spacing_loss": metrics["spacing_loss"],
            "sparsity_loss": metrics["sparsity_loss"],
            "mass_ratio": metrics["mass_ratio"],
            "grad_norm": self.last_grad_norm,
        }

        return output_metrics

    def validate(self) -> Dict[str, float]:
        """
        Run validation loop for SLGA model.

        Returns:
            Dictionary containing validation metrics
        """
        if not self.accelerator.is_main_process:
            return {}

        print("\n=== Validation ===")

        self.accelerator.wait_for_everyone()
        self.model.eval()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        total_loss = 0.0
        total_tokens = 0
        max_batches = 10  # Limited validation for speed

        device = self.accelerator.device
        pad_id = self.tokenizer.pad_token_id

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= max_batches:
                    break

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                cache_ids = batch.get("cache_global_ids")
                cache_ids = cache_ids.to(device) if cache_ids is not None else None

                # Forward pass
                logits = self.model(input_ids, cache_global_ids=cache_ids)

                # Compute loss
                logits_shifted = logits[:, :-1, :].contiguous()
                labels_shifted = labels[:, :-1].contiguous()

                loss = F.cross_entropy(
                    logits_shifted.view(-1, logits_shifted.size(-1)),
                    labels_shifted.view(-1),
                    ignore_index=-100,
                )

                # Count valid tokens
                num_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

                # Free memory
                del input_ids, labels, cache_ids, logits, loss

                if (i + 1) % 5 == 0:
                    print(f"\r  Validation: {i+1}/{max_batches} batches...", end='', flush=True)

        print()  # New line

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 10))

        metrics = {"loss": avg_loss, "perplexity": perplexity}

        # Log validation metrics
        print(f"Val Loss: {avg_loss:.4f} | Val PPL: {perplexity:.2f}")

        # Trigger validation callbacks
        self.trigger_callback("on_validation", metrics=metrics, step=self.state.step)

        # Restore training mode
        self.model.train()

        return metrics
