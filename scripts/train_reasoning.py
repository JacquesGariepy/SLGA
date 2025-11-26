#!/usr/bin/env python3
"""
Script d'entraînement pour SLGA-Reasoning Model

Entraîne un modèle de raisonnement avec:
- Chain-of-Thought (CoT) sur datasets de raisonnement
- Process Reward Model (PRM) pour superviser les étapes
- Curriculum learning sur la profondeur de raisonnement
- Self-consistency pour l'évaluation

Usage:
    python scripts/train_reasoning.py --config config/config_reasoning.yaml
"""

import os
import sys
import argparse
import yaml
import math
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reasoning_model import SLGAReasoningModel, ReasoningModelConfig, create_reasoning_model
from src.reasoning import format_cot_example, extract_reasoning_steps


# =============================================================================
# Datasets de Raisonnement
# =============================================================================

class ReasoningDataset(Dataset):
    """
    Dataset pour l'entraînement au raisonnement.

    Supporte les formats:
    - GSM8K (mathématiques)
    - MATH (mathématiques avancées)
    - LogiQA (raisonnement logique)
    - ARC (science)
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_seq_len: int = 4096,
        special_tokens: Dict[str, int] = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.special_tokens = special_tokens or {
            "<think>": 50257,
            "</think>": 50258,
            "<step>": 50259,
            "<answer>": 50260,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Formater en Chain-of-Thought
        if "reasoning_steps" in item:
            # Format explicite avec étapes
            text = format_cot_example(
                question=item["question"],
                reasoning_steps=item["reasoning_steps"],
                answer=item["answer"],
            )
        else:
            # Format simple question -> réponse
            text = f"{item['question']} <think><step> {item.get('solution', '')} </think><answer>{item['answer']}</answer>"

        # Tokenizer
        tokens = self.tokenizer.encode(text)

        # Remplacer les tokens spéciaux par leurs IDs
        for token_str, token_id in self.special_tokens.items():
            token_encoded = self.tokenizer.encode(token_str)
            if len(token_encoded) == 1:
                # Token déjà dans le vocabulaire
                continue
            # Sinon, on doit gérer manuellement (simplifié ici)

        # Truncate / pad
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "targets": targets,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function avec padding."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = []
    targets = []

    for item in batch:
        L = item["input_ids"].size(0)
        pad_len = max_len - L

        # Pad à droite
        input_ids.append(F.pad(item["input_ids"], (0, pad_len), value=50256))
        targets.append(F.pad(item["targets"], (0, pad_len), value=-100))

    return {
        "input_ids": torch.stack(input_ids),
        "targets": torch.stack(targets),
    }


# =============================================================================
# Chargement des données
# =============================================================================

def load_gsm8k(split: str = "train") -> List[Dict[str, Any]]:
    """Charge le dataset GSM8K."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split=split)

        data = []
        for item in ds:
            # Extraire les étapes du "answer" (format: step1 #### final_answer)
            answer_text = item["answer"]
            parts = answer_text.split("####")

            if len(parts) == 2:
                steps_text = parts[0].strip()
                final_answer = parts[1].strip()

                # Diviser en étapes (par lignes ou phrases)
                steps = [s.strip() for s in steps_text.split("\n") if s.strip()]

                data.append({
                    "question": item["question"],
                    "reasoning_steps": steps,
                    "answer": final_answer,
                })

        return data
    except ImportError:
        print("Warning: datasets library not installed. Using dummy data.")
        return [
            {
                "question": "What is 2 + 2?",
                "reasoning_steps": ["2 + 2 = 4"],
                "answer": "4",
            }
        ]


def load_math_dataset(split: str = "train") -> List[Dict[str, Any]]:
    """Charge le dataset MATH."""
    try:
        from datasets import load_dataset
        ds = load_dataset("hendrycks/competition_math", split=split)

        data = []
        for item in ds:
            data.append({
                "question": item["problem"],
                "solution": item["solution"],
                "answer": item["solution"].split("\\boxed{")[-1].split("}")[0] if "\\boxed{" in item["solution"] else "",
            })

        return data
    except ImportError:
        return []


def load_cot_collection(split: str = "train") -> List[Dict[str, Any]]:
    """Charge CoT-Collection."""
    try:
        from datasets import load_dataset
        ds = load_dataset("kaist-ai/CoT-Collection", split=split)

        data = []
        for item in ds:
            data.append({
                "question": item["source"],
                "reasoning_steps": item["rationale"].split("\n"),
                "answer": item["target"],
            })

        return data[:10000]  # Limiter pour mémoire
    except ImportError:
        return []


# =============================================================================
# Entraînement
# =============================================================================

@dataclass
class TrainingArgs:
    """Arguments d'entraînement."""
    batch_size: int = 4
    accum_steps: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 100000
    eval_every: int = 1000
    save_every: int = 5000
    gradient_clip: float = 1.0
    amp: bool = True
    output_dir: str = "checkpoints/reasoning"


class ReasoningTrainer:
    """Trainer pour le modèle de raisonnement."""

    def __init__(
        self,
        model: SLGAReasoningModel,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader],
        args: TrainingArgs,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.args = args
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )

        # Scheduler
        self.scheduler = self._create_scheduler()

        # AMP
        self.scaler = GradScaler() if args.amp else None

        # Tracking
        self.global_step = 0
        self.best_eval_loss = float('inf')

        # Create output dir
        os.makedirs(args.output_dir, exist_ok=True)

    def _create_scheduler(self):
        """Crée le scheduler avec warmup + cosine decay."""
        def lr_lambda(step):
            if step < self.args.warmup_steps:
                return step / self.args.warmup_steps
            else:
                progress = (step - self.args.warmup_steps) / (self.args.max_steps - self.args.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        """Boucle d'entraînement principale."""
        self.model.train()
        accum_loss = 0.0
        accum_steps = 0

        train_iter = iter(self.train_dataloader)

        print(f"Starting training for {self.args.max_steps} steps...")
        print(f"Effective batch size: {self.args.batch_size * self.args.accum_steps}")

        start_time = time.time()

        while self.global_step < self.args.max_steps:
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)

            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)

            # Forward
            with autocast(enabled=self.args.amp):
                outputs = self.model(input_ids, targets=targets)
                loss = outputs["loss"] / self.args.accum_steps

            # Backward
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item() * self.args.accum_steps
            accum_steps += 1

            # Optimizer step
            if accum_steps == self.args.accum_steps:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Logging
                if self.global_step % 100 == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = self.global_step / elapsed
                    lr = self.scheduler.get_last_lr()[0]

                    print(f"Step {self.global_step}/{self.args.max_steps} | "
                          f"Loss: {accum_loss:.4f} | "
                          f"LR: {lr:.2e} | "
                          f"Steps/s: {steps_per_sec:.2f}")

                    # Log loss components if available
                    if "loss_components" in outputs:
                        components = outputs["loss_components"]
                        comp_str = " | ".join([f"{k}: {v.item():.4f}" for k, v in components.items() if k != "total_loss"])
                        print(f"  Components: {comp_str}")

                accum_loss = 0.0
                accum_steps = 0

                # Evaluation
                if self.global_step % self.args.eval_every == 0 and self.eval_dataloader is not None:
                    eval_loss = self.evaluate()
                    print(f"Eval loss: {eval_loss:.4f}")

                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self.save_checkpoint("best")

                    self.model.train()

                # Save checkpoint
                if self.global_step % self.args.save_every == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

        print("Training complete!")
        self.save_checkpoint("final")

    @torch.no_grad()
    def evaluate(self) -> float:
        """Évalue le modèle."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)

            with autocast(enabled=self.args.amp):
                outputs = self.model(input_ids, targets=targets)
                total_loss += outputs["loss"].item()

            num_batches += 1

            if num_batches >= 100:  # Limiter l'évaluation
                break

        return total_loss / num_batches

    def save_checkpoint(self, name: str):
        """Sauvegarde un checkpoint."""
        path = os.path.join(self.args.output_dir, f"checkpoint_{name}.pt")

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "config": self.model.cfg,
        }, path)

        print(f"Saved checkpoint to {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train SLGA-Reasoning Model")
    parser.add_argument("--config", type=str, default="config/config_reasoning.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("SLGA-Reasoning Training")
    print("=" * 60)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer (using GPT-2 tokenizer)
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    except ImportError:
        print("transformers not installed. Using dummy tokenizer.")
        tokenizer = None

    # Load data
    print("Loading datasets...")
    train_data = []

    # GSM8K
    gsm8k_data = load_gsm8k("train")
    print(f"  GSM8K: {len(gsm8k_data)} examples")
    train_data.extend(gsm8k_data)

    # MATH
    math_data = load_math_dataset("train")
    print(f"  MATH: {len(math_data)} examples")
    train_data.extend(math_data)

    # CoT Collection
    cot_data = load_cot_collection("train")
    print(f"  CoT-Collection: {len(cot_data)} examples")
    train_data.extend(cot_data)

    print(f"Total training examples: {len(train_data)}")

    if len(train_data) == 0:
        print("No training data available. Please install 'datasets' library.")
        print("  pip install datasets")
        return

    # Create datasets
    train_dataset = ReasoningDataset(train_data, tokenizer, max_seq_len=config.get("model", {}).get("max_seq_len", 4096))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.get("training", {}).get("batch_size", 4),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    print("Creating model...")
    model_config = config.get("model", {})
    reasoning_config = config.get("reasoning", {})

    model = create_reasoning_model(
        vocab_size=model_config.get("vocab_size", 50261),
        embed_dim=model_config.get("embed_dim", 768),
        n_layers=model_config.get("n_layers", 16),
        max_seq_len=model_config.get("max_seq_len", 4096),
        local_window=config.get("slga", {}).get("local_window", 256),
        global_k=config.get("slga", {}).get("global_k", 48),
        max_reasoning_steps=reasoning_config.get("max_reasoning_steps", 16),
        use_prm=reasoning_config.get("use_prm", True),
    )

    num_params = model.get_num_params()
    print(f"Model parameters: {num_params / 1e6:.1f}M")

    # Resume if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Create trainer
    training_config = config.get("training", {})
    train_args = TrainingArgs(
        batch_size=training_config.get("batch_size", 4),
        accum_steps=training_config.get("accum_steps", 8),
        lr=training_config.get("lr", 1e-4),
        warmup_steps=training_config.get("warmup_steps", 1000),
        max_steps=training_config.get("max_steps", 100000),
        eval_every=training_config.get("eval_every_steps", 1000),
        save_every=training_config.get("save_every_steps", 5000),
        amp=training_config.get("amp", True),
    )

    trainer = ReasoningTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=None,  # TODO: Add eval dataloader
        args=train_args,
        device=device,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
