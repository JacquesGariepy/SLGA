#!/usr/bin/env python3
"""
🧠 SLGA-Reasoning: Script d'Entraînement Complet
================================================

Ce script est AUTONOME - tout est inclus pour entraîner un reasoning model.

Usage:
    python train_reasoning_simple.py                    # Entraînement par défaut
    python train_reasoning_simple.py --small            # Version légère (test)
    python train_reasoning_simple.py --dataset gsm8k    # Dataset spécifique
    python train_reasoning_simple.py --resume ckpt.pt   # Reprendre

Prérequis:
    pip install torch datasets transformers tqdm
"""

import os
import sys
import argparse
import json
import math
import time
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class TrainConfig:
    """Configuration complète d'entraînement."""
    # Modèle
    vocab_size: int = 50261          # GPT-2 + 4 tokens spéciaux
    embed_dim: int = 512             # 768 pour plus de capacité
    num_heads: int = 8
    n_layers: int = 12               # 16 pour plus de capacité
    max_seq_len: int = 1024          # 2048+ pour raisonnement long
    dropout: float = 0.1

    # SLGA
    local_window: int = 128
    global_k: int = 32

    # Reasoning
    max_reasoning_steps: int = 12
    use_prm: bool = True

    # Entraînement
    batch_size: int = 4
    accum_steps: int = 4             # Batch effectif = 16
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 10000
    eval_every: int = 500
    save_every: int = 1000

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True

    # Chemins
    output_dir: str = "checkpoints/reasoning"

    @classmethod
    def small(cls) -> 'TrainConfig':
        """Config légère pour test."""
        return cls(
            embed_dim=256,
            num_heads=4,
            n_layers=4,
            max_seq_len=512,
            batch_size=2,
            max_steps=1000,
            eval_every=100,
        )


# ==============================================================================
# TOKENS SPÉCIAUX
# ==============================================================================

SPECIAL_TOKENS = {
    "<think>": 50257,
    "</think>": 50258,
    "<step>": 50259,
    "<answer>": 50260,
}

# ==============================================================================
# MODÈLE SIMPLIFIÉ (tout-en-un)
# ==============================================================================

class SLGAAttention(nn.Module):
    """Sparse Local-Global Attention simplifié."""

    def __init__(self, embed_dim: int, num_heads: int, local_window: int = 128, global_k: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.local_window = local_window
        self.global_k = global_k
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(0.1)

        # Gate pour fusion local/global
        self.gate = nn.Linear(2 * self.head_dim, self.head_dim)

    def forward(self, x: torch.Tensor, landmarks: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape

        # QKV
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # === Local Attention (fenêtre glissante) ===
        # Simplifié: attention causale standard pour les petites séquences
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Masque causal
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        # Masque fenêtre locale (optionnel pour efficacité)
        if L > self.local_window * 2:
            window_mask = torch.ones(L, L, device=x.device).bool()
            for i in range(L):
                start = max(0, i - self.local_window)
                window_mask[i, start:i+1] = False
            scores = scores.masked_fill(window_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        ctx_local = torch.matmul(attn, v)  # (B, H, L, Dh)

        # === Global Attention (sur landmarks) ===
        if landmarks is not None:
            G = landmarks.size(1)
            # Project landmarks
            lm_qkv = self.qkv(landmarks).reshape(B, G, 3, self.num_heads, self.head_dim)
            lm_qkv = lm_qkv.permute(2, 0, 3, 1, 4)
            lm_k, lm_v = lm_qkv[1], lm_qkv[2]

            # Global scores
            global_scores = torch.matmul(q, lm_k.transpose(-2, -1)) * self.scale
            global_attn = F.softmax(global_scores, dim=-1)
            ctx_global = torch.matmul(global_attn, lm_v)

            # Fusion avec gate
            gate_input = torch.cat([ctx_local, ctx_global], dim=-1)
            gate_input = gate_input.permute(0, 2, 1, 3).reshape(B * L, self.num_heads, 2 * self.head_dim)
            gate_weights = torch.sigmoid(self.gate(gate_input))
            gate_weights = gate_weights.reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            ctx = gate_weights * ctx_local + (1 - gate_weights) * ctx_global
        else:
            ctx = ctx_local

        # Output
        ctx = ctx.permute(0, 2, 1, 3).reshape(B, L, D)
        return self.out_proj(ctx)


class TransformerBlock(nn.Module):
    """Bloc Transformer avec SLGA."""

    def __init__(self, embed_dim: int, num_heads: int, local_window: int, global_k: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SLGAAttention(embed_dim, num_heads, local_window, global_k)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, landmarks: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), landmarks)
        x = x + self.ffn(self.norm2(x))
        return x


class ReasoningModel(nn.Module):
    """Modèle de raisonnement complet."""

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.embed_dim,
                config.num_heads,
                config.local_window,
                config.global_k,
                config.dropout
            ) for _ in range(config.n_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Tie weights

        # Landmark selector (simple)
        self.landmark_scorer = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Linear(config.embed_dim // 2, 1),
        )

        # Process Reward Model (optionnel)
        if config.use_prm:
            self.prm_head = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim // 2),
                nn.GELU(),
                nn.Linear(config.embed_dim // 2, 1),
            )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)

    def _select_landmarks(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Sélectionne les K positions les plus importantes."""
        B, L, D = x.shape
        scores = self.landmark_scorer(x).squeeze(-1)  # (B, L)
        _, indices = torch.topk(scores, k=min(k, L), dim=-1)  # (B, K)

        # Gather
        indices_exp = indices.unsqueeze(-1).expand(-1, -1, D)
        landmarks = torch.gather(x, dim=1, index=indices_exp)
        return landmarks

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B, L = input_ids.shape
        device = input_ids.device

        # Embeddings
        tok_emb = self.token_emb(input_ids)
        pos = torch.arange(L, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        x = self.dropout(tok_emb + pos_emb)

        # Transformer avec landmarks
        landmarks = None
        for i, block in enumerate(self.blocks):
            # Sélectionner landmarks après quelques couches
            if i == self.config.n_layers // 3:
                landmarks = self._select_landmarks(x, self.config.global_k)
            x = block(x, landmarks)
            # Mettre à jour landmarks
            if landmarks is not None and i % 2 == 0:
                landmarks = self._select_landmarks(x, self.config.global_k)

        # Output
        x = self.norm(x)
        logits = self.lm_head(x)

        output = {"logits": logits}

        # Loss
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

            # PRM loss sur les tokens <step>
            if self.config.use_prm:
                step_mask = (input_ids == SPECIAL_TOKENS["<step>"])
                if step_mask.any():
                    step_states = x[step_mask]
                    step_rewards = self.prm_head(step_states).squeeze(-1)
                    # Récompense positive par défaut (sera affinée avec labels)
                    prm_loss = F.binary_cross_entropy_with_logits(
                        step_rewards,
                        torch.ones_like(step_rewards)
                    )
                    output["prm_loss"] = prm_loss
                    output["loss"] = output["loss"] + 0.1 * prm_loss

        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Génère du texte avec raisonnement."""
        self.eval()

        for _ in range(max_new_tokens):
            # Truncate si nécessaire
            if input_ids.size(1) >= self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]

            # Forward
            outputs = self(input_ids)
            logits = outputs["logits"][:, -1, :]

            # Sampling
            if temperature > 0:
                logits = logits / temperature

                # Top-p
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                sorted_logits[sorted_indices_to_remove] = float('-inf')

                probs = F.softmax(sorted_logits, dim=-1)
                idx = torch.multinomial(probs, num_samples=1)
                next_token = torch.gather(sorted_indices, -1, idx)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop sur </think> ou <answer>
            if next_token.item() in [SPECIAL_TOKENS["</think>"], SPECIAL_TOKENS["<answer>"], 50256]:
                break

        return input_ids


# ==============================================================================
# DATASET
# ==============================================================================

class CoTDataset(Dataset):
    """Dataset Chain-of-Thought."""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 1024
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Formater en CoT
        text = self._format_cot(item)

        # Tokenizer
        tokens = self.tokenizer.encode(text)

        # Truncate
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        # Créer input/target
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)

        return {"input_ids": input_ids, "targets": targets}

    def _format_cot(self, item: Dict[str, Any]) -> str:
        """Formate un exemple en Chain-of-Thought."""
        question = item.get("question", item.get("problem", ""))

        # Si des étapes sont fournies
        if "reasoning_steps" in item:
            steps = item["reasoning_steps"]
            answer = item.get("answer", "")

            steps_text = " ".join([f"<step> {s}" for s in steps])
            return f"Question: {question}\n<think>{steps_text}</think>\n<answer>{answer}</answer>"

        # Si solution brute
        elif "solution" in item:
            solution = item["solution"]
            answer = item.get("answer", solution.split("=")[-1].strip() if "=" in solution else "")
            return f"Question: {question}\n<think><step> {solution}</think>\n<answer>{answer}</answer>"

        # Fallback
        else:
            answer = item.get("answer", "")
            return f"Question: {question}\n<think><step> Let me solve this.</think>\n<answer>{answer}</answer>"


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate avec padding."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = []
    targets = []

    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids.append(F.pad(item["input_ids"], (0, pad_len), value=50256))
        targets.append(F.pad(item["targets"], (0, pad_len), value=-100))

    return {
        "input_ids": torch.stack(input_ids),
        "targets": torch.stack(targets),
    }


# ==============================================================================
# CHARGEMENT DES DONNÉES
# ==============================================================================

def load_gsm8k() -> List[Dict[str, Any]]:
    """Charge GSM8K."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")

        data = []
        for item in ds:
            answer_text = item["answer"]
            parts = answer_text.split("####")

            if len(parts) == 2:
                steps = [s.strip() for s in parts[0].split("\n") if s.strip()]
                final = parts[1].strip()
                data.append({
                    "question": item["question"],
                    "reasoning_steps": steps,
                    "answer": final,
                })

        print(f"✓ GSM8K: {len(data)} exemples chargés")
        return data

    except Exception as e:
        print(f"⚠ GSM8K non disponible: {e}")
        return []


def load_synthetic_data(n: int = 1000) -> List[Dict[str, Any]]:
    """Génère des données synthétiques simples."""
    data = []

    for _ in range(n):
        # Arithmétique simple
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        op = random.choice(["+", "-", "*"])

        if op == "+":
            result = a + b
            steps = [f"{a} + {b} = {result}"]
        elif op == "-":
            result = a - b
            steps = [f"{a} - {b} = {result}"]
        else:
            result = a * b
            steps = [f"{a} × {b} = {result}"]

        data.append({
            "question": f"What is {a} {op} {b}?",
            "reasoning_steps": steps,
            "answer": str(result),
        })

    print(f"✓ Synthétique: {len(data)} exemples générés")
    return data


def load_data(dataset_name: str = "all") -> List[Dict[str, Any]]:
    """Charge les données selon le choix."""
    data = []

    if dataset_name in ["gsm8k", "all"]:
        data.extend(load_gsm8k())

    if dataset_name in ["synthetic", "all"] or len(data) == 0:
        data.extend(load_synthetic_data(2000 if len(data) == 0 else 500))

    random.shuffle(data)
    return data


# ==============================================================================
# TOKENIZER SIMPLE
# ==============================================================================

class SimpleTokenizer:
    """Tokenizer simple basé sur GPT-2."""

    def __init__(self):
        try:
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except:
            self.tokenizer = None
            print("⚠ transformers non installé, utilisation d'un tokenizer basique")

    def encode(self, text: str) -> List[int]:
        if self.tokenizer:
            # Remplacer les tokens spéciaux avant encodage
            for token, idx in SPECIAL_TOKENS.items():
                text = text.replace(token, f" {token} ")

            tokens = self.tokenizer.encode(text)

            # Post-process: remplacer les tokens spéciaux
            result = []
            i = 0
            while i < len(tokens):
                # Chercher les tokens spéciaux dans le texte décodé
                found = False
                for token, idx in SPECIAL_TOKENS.items():
                    token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                    if tokens[i:i+len(token_ids)] == token_ids:
                        result.append(idx)
                        i += len(token_ids)
                        found = True
                        break

                if not found:
                    result.append(tokens[i])
                    i += 1

            return result
        else:
            # Fallback très basique
            return [ord(c) % 50000 for c in text]

    def decode(self, tokens: List[int]) -> str:
        if self.tokenizer:
            # Filtrer les tokens spéciaux pour le décodage
            regular_tokens = [t for t in tokens if t < 50257]
            text = self.tokenizer.decode(regular_tokens)

            # Réinsérer les tokens spéciaux
            for token, idx in SPECIAL_TOKENS.items():
                if idx in tokens:
                    text = text  # Simplification

            return text
        else:
            return "".join([chr(t) for t in tokens if t < 128])


# ==============================================================================
# ENTRAÎNEMENT
# ==============================================================================

def train(config: TrainConfig, dataset_name: str = "all", resume: Optional[str] = None):
    """Boucle d'entraînement principale."""

    print("=" * 60)
    print("🧠 SLGA-Reasoning Training")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Embed dim: {config.embed_dim}")
    print(f"Layers: {config.n_layers}")
    print(f"Batch size: {config.batch_size} x {config.accum_steps} = {config.batch_size * config.accum_steps}")
    print("=" * 60)

    # Créer le dossier de sortie
    os.makedirs(config.output_dir, exist_ok=True)

    # Tokenizer
    tokenizer = SimpleTokenizer()

    # Données
    print("\n📚 Chargement des données...")
    data = load_data(dataset_name)

    if len(data) == 0:
        print("❌ Aucune donnée disponible!")
        return

    # Split train/val
    split_idx = int(len(data) * 0.95)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Datasets
    train_dataset = CoTDataset(train_data, tokenizer, config.max_seq_len)
    val_dataset = CoTDataset(val_data, tokenizer, config.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
    )

    # Modèle
    print("\n🏗️ Création du modèle...")
    model = ReasoningModel(config).to(config.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres: {num_params / 1e6:.1f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    # Scheduler
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP
    scaler = torch.cuda.amp.GradScaler() if config.amp and config.device == "cuda" else None

    # Resume
    start_step = 0
    if resume:
        print(f"\n📂 Reprise depuis {resume}")
        ckpt = torch.load(resume, map_location=config.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)

    # Training loop
    print("\n🚀 Démarrage de l'entraînement...")
    model.train()

    train_iter = iter(train_loader)
    accum_loss = 0.0
    accum_steps = 0
    best_val_loss = float('inf')

    start_time = time.time()

    for step in range(start_step, config.max_steps):
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(config.device)
        targets = batch["targets"].to(config.device)

        # Forward
        with torch.cuda.amp.autocast(enabled=config.amp and config.device == "cuda"):
            outputs = model(input_ids, targets)
            loss = outputs["loss"] / config.accum_steps

        # Backward
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_loss += loss.item() * config.accum_steps
        accum_steps += 1

        # Optimizer step
        if accum_steps == config.accum_steps:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            # Log
            if (step + 1) % 50 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1 - start_step) / elapsed
                lr = scheduler.get_last_lr()[0]

                print(f"Step {step+1}/{config.max_steps} | "
                      f"Loss: {accum_loss:.4f} | "
                      f"LR: {lr:.2e} | "
                      f"Speed: {steps_per_sec:.1f} steps/s")

            accum_loss = 0.0
            accum_steps = 0

            # Eval
            if (step + 1) % config.eval_every == 0:
                val_loss = evaluate(model, val_loader, config)
                print(f"  → Val loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, step, config, "best")

                model.train()

            # Save
            if (step + 1) % config.save_every == 0:
                save_checkpoint(model, optimizer, step, config, f"step_{step+1}")

    # Final save
    save_checkpoint(model, optimizer, config.max_steps, config, "final")
    print("\n✅ Entraînement terminé!")

    return model


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, config: TrainConfig) -> float:
    """Évalue le modèle."""
    model.eval()
    total_loss = 0.0
    count = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(config.device)
        targets = batch["targets"].to(config.device)

        with torch.cuda.amp.autocast(enabled=config.amp and config.device == "cuda"):
            outputs = model(input_ids, targets)
            total_loss += outputs["loss"].item()

        count += 1
        if count >= 50:
            break

    return total_loss / count


def save_checkpoint(model: nn.Module, optimizer, step: int, config: TrainConfig, name: str):
    """Sauvegarde un checkpoint."""
    path = os.path.join(config.output_dir, f"checkpoint_{name}.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config,
    }, path)
    print(f"  💾 Saved: {path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train SLGA-Reasoning Model")
    parser.add_argument("--small", action="store_true", help="Use small config for testing")
    parser.add_argument("--dataset", type=str, default="all", choices=["gsm8k", "synthetic", "all"])
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--steps", type=int, default=None, help="Override max_steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    args = parser.parse_args()

    # Config
    if args.small:
        config = TrainConfig.small()
        print("📦 Using SMALL config (for testing)")
    else:
        config = TrainConfig()

    # Overrides
    if args.steps:
        config.max_steps = args.steps
    if args.batch_size:
        config.batch_size = args.batch_size

    # Train
    train(config, args.dataset, args.resume)


if __name__ == "__main__":
    main()
