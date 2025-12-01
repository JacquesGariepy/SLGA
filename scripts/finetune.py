"""Fine-tuning script for SLGA with LoRA/QLoRA support.

This script enables efficient fine-tuning of pretrained SLGA models using:
- LoRA: Low-Rank Adaptation (reduces trainable params by 100-1000x)
- QLoRA: Quantized LoRA with 4-bit base weights (reduces memory by 4x)
- MultiLoRA: Export adapters compatible with MultiLoRA inference server

Usage:
    # Standard LoRA fine-tuning
    python scripts/finetune.py --config config.yaml --checkpoint out/ckpt_10000 --lora-rank 8

    # QLoRA with 4-bit quantization
    python scripts/finetune.py --config config.yaml --checkpoint out/ckpt_10000 --qlora --lora-rank 16

    # Resume LoRA training
    python scripts/finetune.py --config config.yaml --checkpoint out/ckpt_10000 --lora-checkpoint lora_weights.pt

    # Export to MultiLoRA format (compatible with D:\\ai\\MultiLoRA)
    python scripts/finetune.py --config config.yaml --checkpoint out/ckpt_10000 \\
        --adapter-name code-assistant \\
        --multilora-dir /path/to/MultiLoRA/adapters \\
        --base-model-name slga-v2

Example:
    Fine-tune 116M model on custom data:
    - Base model: 116M params (frozen)
    - LoRA params: ~300K params (0.26% of model)
    - Memory: ~2GB (vs 8GB for full fine-tuning)
    - Speed: 2-3x faster than full fine-tuning

MultiLoRA Integration:
    Adapters are saved in MultiLoRA-compatible format:
    - adapters/<adapter-name>/config.json     # Adapter configuration
    - adapters/<adapter-name>/adapter_weights.pt       # PyTorch weights
    - adapters/<adapter-name>/adapter_weights.safetensors  # SafeTensors (if available)

    Use with MultiLoRA server:
    curl -X POST http://localhost:8080/v1/completions -d '{
        "prompt": "...",
        "adapter": "code-assistant"
    }'
"""

from __future__ import annotations
import os
import sys
import yaml
import math
import time
import torch
import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import Config, LLMTransformer
from src.data import get_tokenizer, load_text_dataset, CollatorLocal, CollatorInstruction, INSTRUCTION_DATASETS
from src.lora import (
    apply_lora_to_model,
    get_lora_parameters,
    count_trainable_parameters,
    save_lora_weights,
    load_lora_weights,
    merge_lora_weights,
    # MultiLoRA compatibility
    save_multilora_adapter,
    load_multilora_adapter,
)
from src.evaluation import GenerationEvaluator
from scripts.utils import set_seed, load_checkpoint

# Default evaluation prompts for fine-tuning
FINETUNE_EVAL_PROMPTS = [
    "Who was Albert Einstein?",
    "What is the capital of France?",
    "Explain machine learning in simple terms:",
    "The theory of relativity states that",
    "Python is a programming language that",
]


# ======================================================================
# Utility Functions
# ======================================================================

def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    device: str = "cuda",
) -> str:
    """Generate text from a prompt."""
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(generated)
            next_token_logits = outputs[:, -1, :] / temperature

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    next_token_logits[0, token_id] /= repetition_penalty

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def save_generation_history(
    history: list,
    out_dir: str,
    filename: str = "generation_history.jsonl",
):
    """Save generation history to JSONL file."""
    import json
    from datetime import datetime

    filepath = os.path.join(out_dir, filename)

    with open(filepath, "a", encoding="utf-8") as f:
        for entry in history:
            entry["timestamp"] = datetime.now().isoformat()
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def evaluate_and_log_generation(
    model: torch.nn.Module,
    tokenizer,
    evaluator: GenerationEvaluator,
    prompts: list,
    step: int,
    out_dir: str,
    device: str = "cuda",
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    writer=None,
) -> dict:
    """Evaluate generation quality and log results."""
    import time

    model.eval()
    results = []
    history_entries = []

    print("\n--- Generation Evaluation ---")

    for prompt in prompts:
        start_time = time.time()

        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device=device,
        )

        gen_time = time.time() - start_time

        # Evaluate quality
        metrics = evaluator.evaluate(prompt, generated, gen_time, compute_perplexity=False)
        results.append(metrics)

        # Print sample
        generated_only = generated[len(prompt):].strip() if generated.startswith(prompt) else generated
        print(f"  [{metrics.quality_grade}] {metrics.overall_score:.0f}/100 | {prompt[:30]}...")
        print(f"      -> {generated_only[:80]}...")

        # Save to history
        history_entries.append({
            "step": step,
            "prompt": prompt,
            "generated": generated,
            "generated_only": generated_only,
            "score": metrics.overall_score,
            "grade": metrics.quality_grade,
            "metrics": {
                "distinct_2": metrics.diversity.distinct_2,
                "rep_2": metrics.repetition.rep_2,
                "coherence": metrics.coherence.coherence_score,
            },
            "generation_time": gen_time,
        })

    # Save history
    save_generation_history(history_entries, out_dir)

    # Compute averages
    avg_score = sum(m.overall_score for m in results) / len(results)
    avg_distinct_2 = sum(m.diversity.distinct_2 for m in results) / len(results)
    avg_rep_2 = sum(m.repetition.rep_2 for m in results) / len(results)

    print(f"  Average: {avg_score:.1f}/100 | Distinct-2: {avg_distinct_2:.3f} | Rep-2: {avg_rep_2:.3f}")

    # Log to TensorBoard
    if writer is not None:
        writer.add_scalar("generation/avg_score", avg_score, step)
        writer.add_scalar("generation/distinct_2", avg_distinct_2, step)
        writer.add_scalar("generation/rep_2", avg_rep_2, step)

        # Log sample texts
        sample_text = "\n\n".join([
            f"**[{m.quality_grade}] {m.overall_score:.0f}/100**\n"
            f"Prompt: {m.prompt}\n"
            f"Output: {m.generated_only[:200]}"
            for m in results
        ])
        writer.add_text("generation/samples", sample_text, step)

    return {
        "avg_score": avg_score,
        "avg_distinct_2": avg_distinct_2,
        "avg_rep_2": avg_rep_2,
        "results": results,
    }


def cross_entropy_shifted(
    logits: torch.Tensor, labels: torch.Tensor, pad_id: int
) -> torch.Tensor:
    """Compute cross-entropy with proper shifting for causal LM."""
    logits_shifted = logits[:, :-1, :].contiguous()
    labels_shifted = labels[:, :-1].contiguous()

    loss = F.cross_entropy(
        logits_shifted.view(-1, logits_shifted.size(-1)),
        labels_shifted.view(-1),
        ignore_index=-100,
    )

    return loss


def validate(
    model: LLMTransformer,
    val_loader: DataLoader,
    pad_id: int,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> dict:
    """Evaluate model on validation set."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    max_b = max_batches if max_batches else len(val_loader)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches and i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)
            loss = cross_entropy_shifted(logits, labels, pad_id)

            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            num_batches += 1

            if (i + 1) % 5 == 0:
                print(f"\r  Validation: {i+1}/{max_b} batches...", end='', flush=True)

    print()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 10))

    return {"loss": avg_loss, "perplexity": perplexity}


# ======================================================================
# Main Fine-tuning Loop
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune SLGA with LoRA/QLoRA")

    # Basic config
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained checkpoint")
    parser.add_argument("--out-dir", type=str, default="out/lora_finetune", help="Output directory")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps from config")

    # LoRA config
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha scaling")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora-target", type=str, nargs='+',
                        default=["qkv_proj", "out_proj", "fc1", "fc2"],
                        help="Target modules for LoRA")
    parser.add_argument("--qlora", action="store_true", help="Use QLoRA (4-bit quantization)")
    parser.add_argument("--lora-checkpoint", type=str, default=None,
                        help="Load existing LoRA weights")

    # MultiLoRA compatibility
    parser.add_argument("--adapter-name", type=str, default=None,
                        help="Adapter name for MultiLoRA export (enables MultiLoRA format)")
    parser.add_argument("--multilora-dir", type=str, default=None,
                        help="Output directory for MultiLoRA adapters (default: out-dir/adapters)")
    parser.add_argument("--base-model-name", type=str, default="slga-v2",
                        help="Base model name for MultiLoRA metadata")

    # Training config
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length for fine-tuning")

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override config with args
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Save config
    config_save_path = os.path.join(out_dir, "finetune_config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump({
            'base_config': cfg,
            'lora': {
                'rank': args.lora_rank,
                'alpha': args.lora_alpha,
                'dropout': args.lora_dropout,
                'target_modules': args.lora_target,
                'use_qlora': args.qlora,
            },
            'training': {
                'lr': args.lr,
                'seq_len': args.seq_len,
            }
        }, f)

    # Setup
    set_seed(cfg.get("seed", 42))
    accelerator = Accelerator()
    device = accelerator.device

    print(f"\n{'='*60}")
    print(f"  SLGA LoRA Fine-tuning")
    print(f"{'='*60}")
    print(f"Base checkpoint: {args.checkpoint}")
    print(f"Output directory: {out_dir}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"QLoRA: {'Yes (4-bit)' if args.qlora else 'No'}")
    print(f"Target modules: {args.lora_target}")
    print(f"{'='*60}\n")

    # Tokenizer
    tokenizer = get_tokenizer(cfg["tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    actual_vocab_size = len(tokenizer)
    cfg["model"]["vocab_size"] = actual_vocab_size

    # Load pretrained model
    print("Loading pretrained model...")
    model_cfg = Config(**cfg["model"])
    model = LLMTransformer(model_cfg)

    # Load checkpoint weights
    checkpoint_path = args.checkpoint
    if os.path.isdir(checkpoint_path):
        # Find model.pt in directory
        model_file = os.path.join(checkpoint_path, "model.pt")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"model.pt not found in {checkpoint_path}")
        checkpoint_path = model_file

    state_dict = torch.load(checkpoint_path, map_location='cpu')

    # Handle wrapped state dict
    if 'model' in state_dict:
        state_dict = state_dict['model']

    model.load_state_dict(state_dict, strict=True)
    print(f"✅ Loaded pretrained weights from {args.checkpoint}")

    # Freeze all base model parameters BEFORE applying LoRA
    for param in model.parameters():
        param.requires_grad = False
    print("✅ Frozen all base model parameters")

    # Apply LoRA
    print("\nApplying LoRA to model...")
    model = apply_lora_to_model(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        target_modules=args.lora_target,
        dropout=args.lora_dropout,
        use_qlora=args.qlora,
        quantize_base=args.qlora,
    )

    # Load LoRA checkpoint if provided
    if args.lora_checkpoint is not None:
        print(f"\nLoading LoRA weights from {args.lora_checkpoint}...")
        load_lora_weights(model, args.lora_checkpoint)

    # Count parameters
    param_counts = count_trainable_parameters(model)
    print(f"\n{'='*60}")
    print(f"  Parameter Counts")
    print(f"{'='*60}")
    print(f"Total parameters:     {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,} ({param_counts['trainable_pct']:.2f}%)")
    print(f"Frozen parameters:    {param_counts['frozen']:,}")
    print(f"{'='*60}\n")

    # Data
    print("Loading datasets...")
    streaming = cfg["data"].get("streaming", False)

    ds_train = load_text_dataset(
        cfg["data"]["dataset"],
        cfg["data"].get("subset"),
        cfg["data"]["split_train"],
        streaming=streaming,
        shuffle_buffer_size=cfg["data"].get("shuffle_buffer_size", 10000),
        seed=cfg.get("seed", 42),
    )

    # Handle validation split
    split_val = cfg["data"].get("split_val")
    val_split_ratio = cfg["data"].get("val_split_ratio", 0.05)
    max_val = cfg["data"].get("max_val_samples", 500)

    if split_val:
        # Use explicit validation split
        ds_val = load_text_dataset(
            cfg["data"]["dataset"],
            cfg["data"].get("subset"),
            split_val,
            streaming=False,
        )
    else:
        # Create validation from train split
        print(f"No validation split - creating {val_split_ratio*100:.0f}% from train...")
        if streaming:
            # For streaming, load a small non-streaming subset for val
            ds_val_full = load_text_dataset(
                cfg["data"]["dataset"],
                cfg["data"].get("subset"),
                cfg["data"]["split_train"],
                streaming=False,
            )
            # Use last N samples as validation
            total_samples = len(ds_val_full)
            val_size = min(max_val, int(total_samples * val_split_ratio))
            ds_val = ds_val_full.select(range(total_samples - val_size, total_samples))
            print(f"  Created validation set with {len(ds_val)} samples")
        else:
            # For non-streaming, split the dataset
            from datasets import Dataset
            total_samples = len(ds_train)
            val_size = min(max_val, int(total_samples * val_split_ratio))
            train_size = total_samples - val_size

            # Split: train gets first N, val gets last M
            ds_val = ds_train.select(range(train_size, total_samples))
            ds_train = ds_train.select(range(train_size))
            print(f"  Train: {len(ds_train)} samples, Val: {len(ds_val)} samples")

    # Limit validation size
    if hasattr(ds_val, '__len__') and max_val and len(ds_val) > max_val:
        ds_val = ds_val.select(range(max_val))

    # Collators - use instruction collator for instruction datasets
    dataset_name = cfg["data"]["dataset"]
    is_instruction_dataset = dataset_name in INSTRUCTION_DATASETS or any(
        kw in dataset_name.lower() for kw in ["oasst", "alpaca", "dolly", "guanaco", "ultrachat", "instruct"]
    )

    if is_instruction_dataset:
        print(f"Using instruction collator for {dataset_name}")
        collate_train = CollatorInstruction(tokenizer, args.seq_len, dataset_name)
    else:
        collate_train = CollatorLocal(tokenizer, args.seq_len, dataset_name)

    def collate_val(examples):
        max_len_val = 512
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        ex0 = examples[0]

        if isinstance(ex0, dict) and "input_ids" in ex0:
            ids_list = [ex["input_ids"] for ex in examples]
        else:
            # Find text field
            text_key = None
            if isinstance(ex0, dict):
                for k in ("text", "content", "document"):
                    if k in ex0 and isinstance(ex0[k], str):
                        text_key = k
                        break

            if text_key:
                texts = [ex[text_key] for ex in examples]
            elif isinstance(ex0, str):
                texts = list(examples)
            else:
                for k, v in ex0.items():
                    if isinstance(v, str) and len(v) > 10:
                        texts = [ex[k] for ex in examples]
                        break
                else:
                    raise KeyError(f"Cannot find text field in {list(ex0.keys())}")

            encoded = tokenizer(
                texts,
                max_length=max_len_val + 1,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            ids_list = encoded["input_ids"]

        # Stack and process
        tens = []
        for ids in ids_list:
            if not torch.is_tensor(ids):
                ids = torch.as_tensor(ids, dtype=torch.long)
            ids = ids.view(-1)
            if ids.numel() >= max_len_val + 1:
                ids = ids[:max_len_val + 1]
            else:
                pad = torch.full((max_len_val + 1 - ids.numel(),), pad_id, dtype=torch.long)
                ids = torch.cat([ids, pad])
            tens.append(ids)

        input_ids = torch.stack(tens, dim=0)
        input_ids_final = input_ids[:, :-1]
        labels = input_ids[:, 1:].clone()
        labels[labels == pad_id] = -100

        return {
            "input_ids": input_ids_final,
            "labels": labels,
        }

    # DataLoaders
    batch_size = cfg["train"]["batch_size"]

    if streaming:
        train_loader = DataLoader(
            ds_train,
            batch_size=batch_size,
            collate_fn=collate_train,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_train,
            num_workers=2,
            pin_memory=True,
        )

    val_loader = DataLoader(
        ds_val,
        batch_size=max(1, batch_size // 2),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_val,
        num_workers=0,
        pin_memory=False,
    )

    # Print batch info (handle streaming datasets)
    try:
        train_len = len(train_loader)
        print(f"Train batches: {train_len}")
    except TypeError:
        print(f"Train batches: streaming")
    print(f"Val batches: {len(val_loader)}")

    # Optimizer (only LoRA parameters)
    lora_params = get_lora_parameters(model)
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.lr,
        betas=tuple(cfg["train"].get("betas", [0.9, 0.95])),
        eps=cfg["train"].get("eps", 1e-8),
        weight_decay=cfg["train"].get("weight_decay", 0.01),
    )

    # Scheduler
    total_steps = cfg["train"]["max_steps"]
    warmup_steps = cfg["train"].get("warmup_steps", 500)
    accum_steps = cfg["train"].get("accum_steps", 1)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps // accum_steps,
        num_training_steps=total_steps // accum_steps,
    )

    # Prepare with Accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # AMP
    amp_enabled = cfg["train"].get("amp", True)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # TensorBoard
    writer = None
    if cfg.get("log", {}).get("tensorboard", True) and accelerator.is_main_process:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(out_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"TensorBoard: {tb_dir}")

    # Generation evaluator
    gen_evaluator = GenerationEvaluator(tokenizer=tokenizer, device=str(device))
    eval_prompts = cfg.get("evaluation", {}).get("prompts", FINETUNE_EVAL_PROMPTS)
    eval_gen_every = cfg.get("evaluation", {}).get("generate_every", 500)
    print(f"Generation evaluation every {eval_gen_every} steps")

    # Training loop
    model.train()
    pad_id = tokenizer.pad_token_id

    step = 0
    best_val_loss = float("inf")

    progress_bar = tqdm(total=total_steps, desc="Fine-tuning", disable=not accelerator.is_main_process)

    epoch = 0
    step_start_time = time.time()
    steps_since_log = 0

    print(f"\n{'='*60}")
    print(f"  Starting Fine-tuning")
    print(f"{'='*60}\n")

    while step < total_steps:
        epoch += 1

        # Reshuffle for streaming
        if streaming and hasattr(ds_train, 'set_epoch'):
            ds_train.set_epoch(epoch)

        for batch in train_loader:
            if step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Truncate to seq_len
            if input_ids.size(1) > args.seq_len:
                input_ids = input_ids[:, :args.seq_len]
                labels = labels[:, :args.seq_len]

            # Forward
            use_autocast = amp_enabled and device.type == "cuda"
            ctx = torch.autocast("cuda", dtype=amp_dtype) if use_autocast else torch.enable_grad()

            with ctx:
                logits = model(input_ids)
                loss = cross_entropy_shifted(logits, labels, pad_id)
                loss = loss / accum_steps

            # Check NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  NaN/Inf loss at step {step}, skipping batch")
                optimizer.zero_grad()
                continue

            # Backward
            accelerator.backward(loss)

            # Optimizer step
            if (step + 1) % accum_steps == 0:
                grad_clip = cfg["train"].get("grad_clip", 1.0)
                if grad_clip > 0:
                    accelerator.clip_grad_norm_(lora_params, grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1
            steps_since_log += 1
            progress_bar.update(1)

            # Logging
            log_every = cfg["train"].get("log_every", 50)
            if accelerator.is_main_process and step % log_every == 0:
                loss_val = accelerator.gather(loss.detach() * accum_steps).mean().item()
                lr = scheduler.get_last_lr()[0]
                ppl = math.exp(min(loss_val, 10))

                elapsed = time.time() - step_start_time
                tok_per_sec = (steps_since_log * batch_size * args.seq_len) / elapsed

                if torch.cuda.is_available():
                    mem = torch.cuda.memory_allocated() / 1e9
                else:
                    mem = 0

                print(f"Step {step:6d} | Loss: {loss_val:.4f} | PPL: {ppl:7.2f} | "
                      f"LR: {lr:.2e} | GPU: {mem:.1f}GB | Tok/s: {tok_per_sec:.0f}")

                # TensorBoard logging
                if writer is not None:
                    writer.add_scalar("train/loss", loss_val, step)
                    writer.add_scalar("train/perplexity", ppl, step)
                    writer.add_scalar("train/lr", lr, step)
                    writer.add_scalar("train/tokens_per_sec", tok_per_sec, step)

                step_start_time = time.time()
                steps_since_log = 0

            # Validation
            eval_every = cfg["train"].get("eval_every", 1000)
            if accelerator.is_main_process and step % eval_every == 0:
                print("\n=== Validation ===")
                model.eval()

                val_metrics = validate(
                    accelerator.unwrap_model(model),
                    val_loader, pad_id, device,
                    max_batches=10,
                )

                print(f"Val Loss: {val_metrics['loss']:.4f} | Val PPL: {val_metrics['perplexity']:.2f}\n")

                # TensorBoard validation logging
                if writer is not None:
                    writer.add_scalar("val/loss", val_metrics['loss'], step)
                    writer.add_scalar("val/perplexity", val_metrics['perplexity'], step)

                # Save best
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    lora_save_path = os.path.join(out_dir, f"lora_best.pt")
                    save_lora_weights(accelerator.unwrap_model(model), lora_save_path)
                    print(f"💾 Saved best LoRA checkpoint (val_loss={best_val_loss:.4f})\n")

                model.train()

            # Generation evaluation
            if accelerator.is_main_process and step % eval_gen_every == 0 and step > 0:
                model.eval()

                gen_metrics = evaluate_and_log_generation(
                    model=accelerator.unwrap_model(model),
                    tokenizer=tokenizer,
                    evaluator=gen_evaluator,
                    prompts=eval_prompts,
                    step=step,
                    out_dir=out_dir,
                    device=str(device),
                    max_new_tokens=cfg.get("evaluation", {}).get("max_new_tokens", 100),
                    temperature=cfg.get("evaluation", {}).get("temperature", 0.7),
                    top_k=cfg.get("evaluation", {}).get("top_k", 50),
                    top_p=cfg.get("evaluation", {}).get("top_p", 0.9),
                    repetition_penalty=cfg.get("evaluation", {}).get("repetition_penalty", 1.2),
                    writer=writer,
                )

                model.train()

            # Checkpointing
            save_every = cfg["train"].get("save_every", 2000)
            if accelerator.is_main_process and step % save_every == 0 and step > 0:
                lora_save_path = os.path.join(out_dir, f"lora_step_{step}.pt")
                save_lora_weights(accelerator.unwrap_model(model), lora_save_path)
                print(f"✅ Saved LoRA checkpoint at step {step}\n")

    # Final save
    progress_bar.close()

    if accelerator.is_main_process:
        print("\n=== Fine-tuning Complete ===")

        # Save final LoRA weights (standard format)
        lora_save_path = os.path.join(out_dir, "lora_final.pt")
        save_lora_weights(accelerator.unwrap_model(model), lora_save_path)

        # Save in MultiLoRA format if adapter name provided
        multilora_adapter_path = None
        if args.adapter_name:
            multilora_dir = args.multilora_dir or os.path.join(out_dir, "adapters")

            # Get dataset name from config
            dataset_name = cfg["data"].get("dataset", "unknown")

            multilora_adapter_path = save_multilora_adapter(
                model=accelerator.unwrap_model(model),
                adapter_dir=multilora_dir,
                adapter_name=args.adapter_name,
                base_model=args.base_model_name,
                metadata={
                    "task": "fine-tuning",
                    "qlora": args.qlora,
                    "target_modules": args.lora_target,
                    "learning_rate": args.lr,
                    "seq_len": args.seq_len,
                },
                training_steps=step,
                final_loss=best_val_loss if best_val_loss < float("inf") else 0.0,
                dataset=dataset_name,
            )
            print(f"\n✅ MultiLoRA adapter saved: {multilora_adapter_path}")
            print(f"   Compatible with: MultiLoRA inference server")
            print(f"   Use with: --adapter {args.adapter_name}")

        # Save merged model
        print("\nMerging LoRA weights into base model...")
        merge_lora_weights(accelerator.unwrap_model(model))

        merged_save_path = os.path.join(out_dir, "model_merged.pt")
        torch.save(accelerator.unwrap_model(model).state_dict(), merged_save_path)
        print(f"✅ Saved merged model to {merged_save_path}")

        # Close TensorBoard writer
        if writer is not None:
            writer.close()

        print(f"\n{'='*60}")
        print(f"  Fine-tuning complete!")
        print(f"{'='*60}")
        print(f"  LoRA weights: {lora_save_path}")
        if multilora_adapter_path:
            print(f"  MultiLoRA adapter: {multilora_adapter_path}")
        print(f"  Merged model: {merged_save_path}")
        print(f"  Generation history: {os.path.join(out_dir, 'generation_history.jsonl')}")
        print(f"  TensorBoard: {os.path.join(out_dir, 'tensorboard')}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
