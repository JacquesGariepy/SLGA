"""
Main training script for SLGA (Sparse Landmark Global Attention)

Includes:
- Automatic Mixed Precision (AMP)
- Gradient accumulation
- Learning rate warmup + cosine decay
- Auxiliary losses (spacing, sparsity)
- Progressive global attention warmup
- Periodic validation
- Checkpointing with rotation
- Optional W&B / TensorBoard logging
"""

from __future__ import annotations
import os
import sys
import math
import time
import yaml
import torch
import shutil
import threading
import subprocess
import gc

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
from typing import Optional, Dict, Tuple
from torch.optim.lr_scheduler import _LRScheduler

from src.model import Config, LLMTransformer
from src.data import get_tokenizer, load_text_dataset, CollatorLocal, CollatorLocalGlobal
from src.landmarks import landmark_spacing_loss, landmark_sparsity_loss, landmark_diversity_loss
from scripts.utils import set_seed, save_checkpoint, load_checkpoint

# Generation evaluation
try:
    from src.evaluation import GenerationEvaluator, GenerationQualityMetrics
    HAS_EVALUATION = True
except ImportError:
    HAS_EVALUATION = False
    GenerationEvaluator = None

# Optional imports
try:
    from src.live_metrics import LiveMetricsDisplay
except ImportError:
    LiveMetricsDisplay = None

try:
    from src.realtime_display import RealtimeTrainingDisplay
except ImportError:
    RealtimeTrainingDisplay = None

try:
    from src.monitoring import compute_long_dependency_recall
except ImportError:
    compute_long_dependency_recall = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


# ======================================================================
# Utility Functions
# ======================================================================

def update_checkpointing(model: LLMTransformer, seq_len: int):
    """Enable gradient checkpointing for longer sequences."""
    model.cfg.grad_checkpointing = (seq_len > 512)
    for block in model.blocks:
        block.cfg.grad_checkpointing = model.cfg.grad_checkpointing


def _unwrap_scheduler(scheduler: Optional[_LRScheduler]) -> Optional[_LRScheduler]:
    """Unwrap nested Accelerate schedulers."""
    current = scheduler
    visited = set()
    while current is not None and hasattr(current, "scheduler") and id(current) not in visited:
        visited.add(id(current))
        current = getattr(current, "scheduler")
    return current


def apply_lr_from_config(
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[_LRScheduler],
    base_lr: float,
    scorer_lr_multiplier: float,
) -> None:
    """Re-sync optimizer and scheduler LRs after resume."""
    real_scheduler = _unwrap_scheduler(scheduler)
    
    target_lrs = []
    for group in optimizer.param_groups:
        name = group.get("name", "")
        lr = base_lr * scorer_lr_multiplier if name == "scorer" else base_lr
        group["lr"] = lr
        if "initial_lr" in group:
            group["initial_lr"] = lr
        target_lrs.append(lr)
    
    if real_scheduler is not None:
        if hasattr(real_scheduler, "base_lrs"):
            real_scheduler.base_lrs = list(target_lrs)
        if hasattr(real_scheduler, "_last_lr"):
            real_scheduler._last_lr = list(target_lrs)


def build_extra_state(
    best_overall: dict,
    best_loss_per_sequence: dict,
) -> dict:
    """Create serializable state for checkpoint metadata."""
    return {
        "best_overall": dict(best_overall),
        "best_loss_per_sequence": {
            int(k): {"loss": float(v["loss"]), "step": int(v.get("step", -1))}
            for k, v in best_loss_per_sequence.items()
        },
    }


def get_current_seq_len(step: int, cfg: dict) -> int:
    """
    Calculate current sequence length per curriculum schedule.
    
    Progression: seq_len_start -> seq_len_mid -> seq_len_final
    """
    warmup_steps = cfg["train"].get("seq_len_warmup_steps", 15000)
    start_len = cfg["train"].get("seq_len_start", 512)
    mid_len = cfg["train"].get("seq_len_mid", 1024)
    final_len = cfg["train"].get("seq_len_final", 2048)
    
    if step < warmup_steps // 2:
        progress = step / (warmup_steps // 2)
        seq_len = start_len + progress * (mid_len - start_len)
    elif step < warmup_steps:
        progress = (step - warmup_steps // 2) / (warmup_steps // 2)
        seq_len = mid_len + progress * (final_len - mid_len)
    else:
        seq_len = final_len
    
    return int(seq_len)


def get_global_warmup_weight(step: int, cfg: dict) -> float:
    """Calculate warmup weight for global attention (0→1)."""
    warmup_start = cfg["train"].get("global_warmup_start", 30000)
    warmup_end = cfg["train"].get("global_warmup_end", 50000)
    
    if step < warmup_start:
        return 0.0
    elif step < warmup_end:
        progress = (step - warmup_start) / (warmup_end - warmup_start)
        return progress
    else:
        return 1.0


def cross_entropy_shifted(
    logits: torch.Tensor, labels: torch.Tensor, pad_id: int
) -> torch.Tensor:
    """
    Compute cross-entropy with proper shifting for causal LM.
    
    Note: Collator has already shifted labels, so logits[i] predicts labels[i].
    """
    logits_shifted = logits[:, :-1, :].contiguous()
    labels_shifted = labels[:, :-1].contiguous()
    
    loss = F.cross_entropy(
        logits_shifted.view(-1, logits_shifted.size(-1)),
        labels_shifted.view(-1),
        ignore_index=-100,
    )
    
    return loss


def run_quick_generate(config_path: str, checkpoint_dir: str) -> None:
    """Launch quick generation to inspect checkpoint quality."""
    script_path = os.path.join(os.path.dirname(__file__), "generate.py")
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    config_path = os.path.abspath(config_path)
    
    if not os.path.isdir(checkpoint_dir):
        print(f"⚠️  Generation skipped, checkpoint dir missing: {checkpoint_dir}")
        return
    
    cmd = [
        sys.executable, script_path,
        "--config", config_path,
        "--prompt", "Albert Einstein (14 March 1879 – 18 April 1955) was",
        "--temperature", "0.7",
        "--repetition-penalty", "1.25",
        "--no-repeat-ngram-size", "4",
        "--max-tokens", "50",
        "--checkpoint", checkpoint_dir,
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


# ======================================================================
# Generation Evaluation During Training
# ======================================================================

# Default evaluation prompts for training monitoring
EVAL_PROMPTS = [
    "The cat sat on the",
    "Albert Einstein was a",
    "In the year 2024,",
    "The quick brown fox",
    "Machine learning is",
]


def evaluate_generation_quality(
    model: LLMTransformer,
    tokenizer,
    device: torch.device,
    evaluator: Optional["GenerationEvaluator"],
    prompts: list = None,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
) -> Dict[str, any]:
    """
    Evaluate generation quality during training.

    Returns:
        Dict with average metrics across all prompts
    """
    if not HAS_EVALUATION or evaluator is None:
        return {}

    prompts = prompts or EVAL_PROMPTS
    model.eval()

    all_metrics = []
    generated_samples = []

    eos_token_id = tokenizer.eos_token_id

    for prompt in prompts:
        try:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            start_time = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=eos_token_id,
                    stop_on_eos=True,
                )
            gen_time = time.time() - start_time

            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Evaluate
            metrics = evaluator.evaluate(
                prompt=prompt,
                generated_text=output_text,
                generation_time=gen_time,
                compute_perplexity=True,
            )
            all_metrics.append(metrics)
            generated_samples.append({
                "prompt": prompt,
                "output": output_text,
                "score": metrics.overall_score,
                "grade": metrics.quality_grade,
            })

        except Exception as e:
            print(f"  Warning: Generation failed for prompt '{prompt[:30]}...': {e}")
            continue

    if not all_metrics:
        return {}

    # Aggregate metrics
    avg_score = sum(m.overall_score for m in all_metrics) / len(all_metrics)
    avg_ppl = sum(m.perplexity.sequence_perplexity for m in all_metrics if m.perplexity.sequence_perplexity > 0)
    avg_ppl = avg_ppl / len([m for m in all_metrics if m.perplexity.sequence_perplexity > 0]) if avg_ppl > 0 else 0
    avg_distinct_2 = sum(m.diversity.distinct_2 for m in all_metrics) / len(all_metrics)
    avg_rep_2 = sum(m.repetition.rep_2 for m in all_metrics) / len(all_metrics)
    avg_coherence = sum(m.coherence.coherence_score for m in all_metrics) / len(all_metrics)
    avg_gibberish = sum(m.coherence.gibberish_score for m in all_metrics) / len(all_metrics)

    # Count grades
    grades = [m.quality_grade for m in all_metrics]
    grade_counts = {g: grades.count(g) for g in ["A", "B", "C", "D", "F"]}

    # Count issues
    loop_count = sum(1 for m in all_metrics if m.repetition.loop_detected)
    gibberish_count = sum(1 for m in all_metrics if m.coherence.gibberish_score > 0.5)

    return {
        "overall_score": avg_score,
        "gen_perplexity": avg_ppl,
        "distinct_2": avg_distinct_2,
        "rep_2": avg_rep_2,
        "coherence": avg_coherence,
        "gibberish": avg_gibberish,
        "grade_counts": grade_counts,
        "loop_count": loop_count,
        "gibberish_count": gibberish_count,
        "samples": generated_samples,
        "num_evaluated": len(all_metrics),
    }


def log_generation_metrics(
    writer,
    metrics: Dict[str, any],
    step: int,
    prefix: str = "generation",
):
    """Log generation metrics to TensorBoard."""
    if writer is None or not metrics:
        return

    writer.add_scalar(f"{prefix}/overall_score", metrics.get("overall_score", 0), step)
    writer.add_scalar(f"{prefix}/perplexity", metrics.get("gen_perplexity", 0), step)
    writer.add_scalar(f"{prefix}/distinct_2", metrics.get("distinct_2", 0), step)
    writer.add_scalar(f"{prefix}/repetition_2", metrics.get("rep_2", 0), step)
    writer.add_scalar(f"{prefix}/coherence", metrics.get("coherence", 0), step)
    writer.add_scalar(f"{prefix}/gibberish", metrics.get("gibberish", 0), step)
    writer.add_scalar(f"{prefix}/loop_count", metrics.get("loop_count", 0), step)

    # Grade distribution
    grade_counts = metrics.get("grade_counts", {})
    for grade in ["A", "B", "C", "D", "F"]:
        writer.add_scalar(f"{prefix}/grades/{grade}", grade_counts.get(grade, 0), step)


def print_generation_summary(metrics: Dict[str, any], step: int):
    """Print generation evaluation summary."""
    if not metrics:
        return

    print(f"\n=== Generation Quality (step {step}) ===")
    print(f"  Score: {metrics.get('overall_score', 0):.1f}/100")
    print(f"  Gen PPL: {metrics.get('gen_perplexity', 0):.2f}")
    print(f"  Distinct-2: {metrics.get('distinct_2', 0):.3f}")
    print(f"  Rep-2: {metrics.get('rep_2', 0):.3f}")
    print(f"  Coherence: {metrics.get('coherence', 0):.3f}")
    print(f"  Gibberish: {metrics.get('gibberish', 0):.3f}")

    grades = metrics.get("grade_counts", {})
    grade_str = " | ".join(f"{g}:{grades.get(g, 0)}" for g in ["A", "B", "C", "D", "F"])
    print(f"  Grades: {grade_str}")

    if metrics.get("loop_count", 0) > 0:
        print(f"  ⚠️  Loops detected: {metrics['loop_count']}/{metrics['num_evaluated']}")
    if metrics.get("gibberish_count", 0) > 0:
        print(f"  ⚠️  High gibberish: {metrics['gibberish_count']}/{metrics['num_evaluated']}")

    # Show samples
    samples = metrics.get("samples", [])
    if samples:
        print(f"\n  Sample generations:")
        for i, s in enumerate(samples[:3]):  # Show first 3
            print(f"    [{s['grade']}] {s['prompt'][:20]}... → {s['output'][len(s['prompt']):len(s['prompt'])+40]}...")

    print("=" * 40)


def log_system_info(tag: str = "SNAPSHOT") -> None:
    """Emit system snapshot for monitoring."""
    print(f"\n===== SYSTEM SNAPSHOT ({tag}) =====", flush=True)
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = props.total_memory / 1e9
            print(f"GPU {i}: {props.name} | {allocated:.2f}GB / {total:.2f}GB allocated", flush=True)
    
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"RAM: {vm.used/1e9:.2f}GB / {vm.total/1e9:.2f}GB", flush=True)
    except ImportError:
        pass
    
    print("===== END SNAPSHOT =====\n", flush=True)


# ======================================================================
# Data Loading
# ======================================================================

def build_loaders(cfg: dict, tokenizer=None) -> tuple:
    """
    Build tokenizer and data loaders with streaming support.
    
    Following HuggingFace best practices:
    https://huggingface.co/docs/datasets/stream
    
    Returns:
        tokenizer, train_loader, val_loader, train_dataset (for set_epoch)
    """
    tokenizer = tokenizer or get_tokenizer(cfg["tokenizer"])
    
    # Check streaming mode
    streaming = cfg["data"].get("streaming", False)
    shuffle_buffer = cfg["data"].get("shuffle_buffer_size", 10000)
    seed = cfg.get("seed", 42)
    
    # Load datasets
    try:
        ds_train = load_text_dataset(
            cfg["data"]["dataset"],
            cfg["data"].get("subset"),
            cfg["data"]["split_train"],
            streaming=streaming,
            shuffle_buffer_size=shuffle_buffer,
            seed=seed,
        )
        
        # Validation: always non-streaming for metrics
        ds_val = load_text_dataset(
            cfg["data"]["dataset"],
            cfg["data"].get("subset"),
            cfg["data"]["split_val"],
            streaming=False,
            shuffle_buffer_size=0,
            seed=seed,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Limit validation size
    max_val = cfg["data"].get("max_val_samples", 500)
    if hasattr(ds_val, '__len__') and max_val and len(ds_val) > max_val:
        ds_val = ds_val.select(range(max_val))
    
    # Report sizes
    if hasattr(ds_train, '__len__'):
        print(f"Train samples: {len(ds_train):,}")
    else:
        print(f"Train: streaming mode (buffer={shuffle_buffer})")
    print(f"Val samples: {len(ds_val) if hasattr(ds_val, '__len__') else 'N/A'}")
    
    use_learned = cfg["model"].get("learned_landmarks", True)
    seq_len_final = cfg["train"].get("seq_len_final", 1024)
    dataset_name = cfg["data"]["dataset"]
    
    def build_collator(max_length: int):
        if use_learned:
            return CollatorLocal(tokenizer, max_length, dataset_name)
        return CollatorLocalGlobal(
            tokenizer, max_length,
            cfg["train"].get("global_every", 64),
            cfg["train"].get("max_global", 48),
            dataset_name,
        )
    
    collate_train = build_collator(seq_len_final)
    
    # Validation collator
    def collate_val_reduced(examples):
        max_len_val = 512
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        # Handle different formats
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
                # Last resort
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
        
        # Don't include cache_global_ids: None - Accelerate can't concatenate None
        return {
            "input_ids": input_ids_final,
            "labels": labels,
        }
    
    # Train DataLoader - different for streaming
    if streaming:
        # Streaming: no shuffle in DataLoader (done via buffer on dataset)
        # num_workers=0 recommended for streaming
        train_loader = DataLoader(
            ds_train,
            batch_size=cfg["train"]["batch_size"],
            collate_fn=collate_train,
            num_workers=cfg["data"].get("num_workers", 0),
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            ds_train,
            batch_size=cfg["train"]["batch_size"],
            shuffle=True,
            drop_last=True,
            collate_fn=collate_train,
            num_workers=cfg["data"].get("num_workers", 2),
            pin_memory=True,
        )
    
    val_batch_size = max(1, cfg["train"]["batch_size"] // 2)
    val_loader = DataLoader(
        ds_val,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_val_reduced,
        num_workers=0,
        pin_memory=False,
    )
    
    # Return dataset too for set_epoch() in streaming mode
    return tokenizer, train_loader, val_loader, ds_train


# ======================================================================
# Validation
# ======================================================================

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
            cache_ids = batch.get("cache_global_ids")
            cache_ids = cache_ids.to(device) if cache_ids is not None else None
            
            logits = model(input_ids, cache_global_ids=cache_ids)
            loss = cross_entropy_shifted(logits, labels, pad_id)
            
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            num_batches += 1
            
            del input_ids, labels, cache_ids, logits, loss
            
            if i % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if (i + 1) % 5 == 0:
                print(f"\r  Validation: {i+1}/{max_b} batches...", end='', flush=True)
    
    print()
    
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 10))
    
    return {"loss": avg_loss, "perplexity": perplexity}


# ======================================================================
# Main Training Loop
# ======================================================================

def main():
    """Main training function."""
    import argparse
    parser = argparse.ArgumentParser(description="Train SLGA model")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, nargs='?', const='auto')
    args = parser.parse_args()
    
    config_file_path = os.path.abspath(args.config)
    
    # CUDA config
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Load config
    with open(config_file_path) as f:
        cfg = yaml.safe_load(f)
    
    out_dir = cfg["train"].get("out_dir", cfg["save"].get("out_dir", "out_slga"))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "best_sequence"), exist_ok=True)
    
    log_cfg = cfg.get("log", {})
    
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    
    # TensorBoard
    writer = None
    if log_cfg.get("tensorboard", False) and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=f"{out_dir}/tensorboard")
    
    # Setup
    set_seed(cfg["seed"])
    accelerator = Accelerator()
    device = accelerator.device
    
    if accelerator.is_main_process:
        log_system_info("STARTUP")
    
    # W&B
    if log_cfg.get("wandb", False):
        import wandb
        run_name = log_cfg.get("run_name") or f"slga_{cfg['model']['embed_dim']}d"
        wandb.init(project=log_cfg["project"], name=run_name, config=cfg)
    
    # Tokenizer
    tokenizer = get_tokenizer(cfg["tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # CORRECTION: Utiliser len(tokenizer) pour vocab_size
    actual_vocab_size = len(tokenizer)
    print(f"Tokenizer vocab_size: {tokenizer.vocab_size} | len(tokenizer): {actual_vocab_size}")
    cfg["model"]["vocab_size"] = actual_vocab_size
    
    # Model
    model_cfg = Config(**cfg["model"])
    if "grad_checkpointing" in cfg["train"]:
        model_cfg.grad_checkpointing = cfg["train"]["grad_checkpointing"]
    
    model = LLMTransformer(model_cfg)
    
    if accelerator.is_main_process:
        print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")
        print(f"Learned landmarks: {model_cfg.learned_landmarks}")
    
    # Data
    tokenizer, train_loader, val_loader, train_dataset = build_loaders(cfg, tokenizer=tokenizer)
    pad_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id  # CORRECTION: Récupérer EOS du tokenizer
    
    # Check if streaming mode
    is_streaming = cfg["data"].get("streaming", False)
    
    # Optimizer
    scorer_lr_multiplier = cfg["train"].get("scorer_lr_multiplier", 5.0)
    base_lr = cfg["train"]["lr"]
    
    scorer_params = []
    other_params = []
    
    if model.landmark_selector is not None:
        scorer_params = list(model.landmark_selector.scorer.parameters())
        scorer_param_ids = {id(p) for p in scorer_params}
        other_params = [p for p in model.parameters() if id(p) not in scorer_param_ids]
    else:
        other_params = list(model.parameters())
    
    param_groups = [{"params": other_params, "lr": base_lr, "name": "model"}]
    if scorer_params:
        param_groups.append({
            "params": scorer_params,
            "lr": base_lr * scorer_lr_multiplier,
            "name": "scorer",
        })
    
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=tuple(cfg["train"]["betas"]),
        eps=cfg["train"]["eps"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    
    # Scheduler
    total_steps = cfg["train"]["max_steps"]
    warmup_steps = cfg["train"]["warmup_steps"]
    accum_steps = cfg["train"]["accum_steps"]
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps // accum_steps,
        num_training_steps=total_steps // accum_steps,
    )
    
    # Prepare with Accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    apply_lr_from_config(optimizer, scheduler, base_lr, scorer_lr_multiplier)
    
    # AMP
    amp_enabled = cfg["train"].get("amp", True)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Resume
    step = 0
    optimizer_step = 0
    best_overall = {"loss": float("inf"), "step": -1}
    best_loss_per_sequence = {}
    
    if args.resume:
        checkpoint_path = None
        if args.resume != 'auto':
            checkpoint_path = args.resume
        else:
            checkpoints = [d for d in os.listdir(out_dir) if d.startswith("ckpt_")]
            if checkpoints:
                latest = max(checkpoints, key=lambda x: int(x.split("_")[1]))
                checkpoint_path = os.path.join(out_dir, latest)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            resume_state = {}
            unwrapped = accelerator.unwrap_model(model)
            step = load_checkpoint(checkpoint_path, unwrapped, optimizer, scheduler, str(device), resume_state)
            optimizer_step = step // accum_steps
            
            if "best_overall" in resume_state:
                best_overall = resume_state["best_overall"]
            if "best_loss_per_sequence" in resume_state:
                best_loss_per_sequence = resume_state["best_loss_per_sequence"]
            
            apply_lr_from_config(optimizer, scheduler, base_lr, scorer_lr_multiplier)
            print(f"✅ Resumed from step {step}")
    
    model.train()

    # Generation evaluator for quality monitoring
    gen_evaluator = None
    if HAS_EVALUATION:
        try:
            gen_evaluator = GenerationEvaluator(
                model=accelerator.unwrap_model(model),
                tokenizer=tokenizer,
                device=str(device),
            )
            print("Generation evaluator initialized")
        except Exception as e:
            print(f"Warning: Could not initialize generation evaluator: {e}")

    # Custom eval prompts from config
    eval_prompts = cfg.get("evaluation", {}).get("prompts", EVAL_PROMPTS)
    eval_gen_every = cfg.get("evaluation", {}).get("generate_every", cfg["train"].get("eval_every", 1000))

    # Training state
    grad_clip = cfg["train"]["grad_clip"]
    last_grad_norm = 0.0
    last_spacing_loss = 0.0
    last_spar_loss = 0.0
    last_num_landmarks = 0
    
    epoch = 0
    step_start_time = time.time()
    steps_since_log = 0
    
    # Progress
    progress_bar = tqdm(total=total_steps, desc="Training", disable=not accelerator.is_main_process)
    progress_bar.update(step)
    
    # Training loop
    while step < total_steps:
        epoch += 1
        
        # HF streaming: reshuffle between epochs
        # https://huggingface.co/docs/datasets/stream#reshuffle
        if is_streaming and hasattr(train_dataset, 'set_epoch'):
            train_dataset.set_epoch(epoch)
        
        for batch in train_loader:
            if step >= total_steps:
                break
            
            # Curriculum
            current_seq_len = get_current_seq_len(step, cfg)
            global_weight = get_global_warmup_weight(step, cfg)
            
            # Dynamic checkpointing
            model.cfg.grad_checkpointing = (current_seq_len > 1024)
            for block in model.blocks:
                block.cfg.grad_checkpointing = model.cfg.grad_checkpointing
            
            # Data prep
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            cache_ids = batch.get("cache_global_ids")
            
            # Truncate
            if input_ids.size(1) > current_seq_len:
                input_ids = input_ids[:, :current_seq_len]
                labels = labels[:, :current_seq_len]
                if cache_ids is not None:
                    cache_ids = torch.clamp(cache_ids, 0, current_seq_len - 1)
            
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            cache_ids = cache_ids.to(device) if cache_ids is not None else None
            
            # Forward
            use_autocast = amp_enabled and device.type == "cuda"
            ctx = torch.autocast("cuda", dtype=amp_dtype) if use_autocast else torch.enable_grad()
            
            with ctx:
                logits, aux = model(input_ids, cache_global_ids=cache_ids, return_aux=True, global_weight=global_weight)
                loss_ce = cross_entropy_shifted(logits, labels, pad_id)
                loss = loss_ce / accum_steps
                
                # Auxiliary losses
                landmark_indices = aux.get("landmark_indices")
                landmark_scores = aux.get("landmark_scores")
                
                if landmark_indices is not None:
                    last_num_landmarks = landmark_indices.size(1)
                
                if landmark_indices is not None and landmark_scores is not None:
                    seq_len = input_ids.size(1)
                    
                    lambda_spacing = cfg["train"].get("lambda_spacing", 0.0)
                    if lambda_spacing > 0 and last_num_landmarks > 1:
                        spacing_loss = landmark_spacing_loss(
                            landmark_indices, seq_len, lambda_spacing, landmark_scores
                        )
                        last_spacing_loss = spacing_loss.item()
                        loss = loss + spacing_loss / accum_steps
                    
                    lambda_spar = cfg["train"].get("lambda_sparsity", 0.0)
                    if lambda_spar > 0 and last_num_landmarks > 0:
                        spar_loss = landmark_sparsity_loss(
                            landmark_scores, last_num_landmarks, lambda_spar
                        )
                        last_spar_loss = spar_loss.item()
                        loss = loss + spar_loss / accum_steps
            
            # Check NaN
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"Training diverged at step {step}")
            
            # Backward
            accelerator.backward(loss)
            
            # Optimizer step
            if (step + 1) % accum_steps == 0:
                # Grad norm
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                last_grad_norm = grad_norm
                
                if grad_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
            
            step += 1
            steps_since_log += 1
            progress_bar.update(1)
            
            # Logging
            if accelerator.is_main_process and step % cfg["train"].get("log_every", 100) == 0:
                loss_val = accelerator.gather(loss_ce.detach()).mean().item()
                lr = scheduler.get_last_lr()[0]
                ppl = math.exp(min(loss_val, 10))
                
                elapsed = time.time() - step_start_time
                tok_per_sec = (steps_since_log * cfg["train"]["batch_size"] * current_seq_len) / elapsed
                
                if torch.cuda.is_available():
                    mem = torch.cuda.memory_allocated() / 1e9
                else:
                    mem = 0
                
                print(f"Step {step:6d} | Loss: {loss_val:.4f} | PPL: {ppl:7.2f} | "
                      f"LR: {lr:.2e} | GradNorm: {last_grad_norm:5.2f} | "
                      f"SeqLen: {current_seq_len} | GW: {global_weight:.2f} | "
                      f"GPU: {mem:.1f}GB | Tok/s: {tok_per_sec:.0f}")
                
                if writer is not None:
                    writer.add_scalar("train/loss", loss_val, step)
                    writer.add_scalar("train/perplexity", ppl, step)
                    writer.add_scalar("train/lr", lr, step)
                    writer.add_scalar("train/grad_norm", last_grad_norm, step)
                    writer.add_scalar("train/seq_len", current_seq_len, step)
                    writer.add_scalar("train/global_weight", global_weight, step)
                    if last_spacing_loss > 0:
                        writer.add_scalar("train/spacing_loss", last_spacing_loss, step)
                    if last_spar_loss > 0:
                        writer.add_scalar("train/sparsity_loss", last_spar_loss, step)
                
                step_start_time = time.time()
                steps_since_log = 0
            
            # Validation
            if accelerator.is_main_process and step % cfg["train"].get("eval_every", 1000) == 0:
                print("\n=== Validation ===")
                accelerator.wait_for_everyone()
                model.eval()
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                val_metrics = validate(
                    accelerator.unwrap_model(model),
                    val_loader, pad_id, device,
                    max_batches=10,
                )
                
                print(f"Val Loss: {val_metrics['loss']:.4f} | Val PPL: {val_metrics['perplexity']:.2f}")
                
                if writer is not None:
                    writer.add_scalar("val/loss", val_metrics["loss"], step)
                    writer.add_scalar("val/perplexity", val_metrics["perplexity"], step)

                # Generation quality evaluation
                if gen_evaluator is not None and step % eval_gen_every == 0:
                    print("\n--- Evaluating generation quality ---")
                    gen_metrics = evaluate_generation_quality(
                        model=accelerator.unwrap_model(model),
                        tokenizer=tokenizer,
                        device=device,
                        evaluator=gen_evaluator,
                        prompts=eval_prompts,
                        max_new_tokens=cfg.get("evaluation", {}).get("max_new_tokens", 50),
                        temperature=cfg.get("evaluation", {}).get("temperature", 0.7),
                        top_k=cfg.get("evaluation", {}).get("top_k", 50),
                        top_p=cfg.get("evaluation", {}).get("top_p", 0.9),
                        repetition_penalty=cfg.get("evaluation", {}).get("repetition_penalty", 1.2),
                    )

                    if gen_metrics:
                        print_generation_summary(gen_metrics, step)
                        log_generation_metrics(writer, gen_metrics, step)

                        # Log samples as text to TensorBoard
                        if writer is not None:
                            samples = gen_metrics.get("samples", [])
                            sample_text = "\n\n".join([
                                f"**[{s['grade']}] {s['score']:.0f}/100**\n"
                                f"Prompt: {s['prompt']}\n"
                                f"Output: {s['output']}"
                                for s in samples
                            ])
                            writer.add_text("generation/samples", sample_text, step)

                val_loss = val_metrics["loss"]
                
                # Best checkpoints
                if val_loss < best_overall["loss"]:
                    best_overall = {"loss": val_loss, "step": step}
                    save_checkpoint(
                        model, optimizer, scheduler, out_dir, step, accelerator,
                        extra_state=build_extra_state(best_overall, best_loss_per_sequence),
                        custom_dir="best_overall",
                    )
                    print(f"💾 New best overall: {val_loss:.4f}")
                
                current_sl = get_current_seq_len(step, cfg)
                prev = best_loss_per_sequence.get(current_sl, {"loss": float("inf")})
                if val_loss < prev["loss"]:
                    best_loss_per_sequence[current_sl] = {"loss": val_loss, "step": step}
                    save_checkpoint(
                        model, optimizer, scheduler, out_dir, step, accelerator,
                        extra_state=build_extra_state(best_overall, best_loss_per_sequence),
                        custom_dir=f"best_sequence/seq_{current_sl}",
                    )
                    print(f"💾 New best for seq_len={current_sl}: {val_loss:.4f}")
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                model.train()
            
            # Checkpointing
            save_every = cfg["train"].get("save_every", 5000)
            if accelerator.is_main_process and step % save_every == 0 and step > 0:
                accelerator.wait_for_everyone()
                save_checkpoint(
                    model, optimizer, scheduler, out_dir, step, accelerator,
                    keep_last_n=cfg["train"].get("keep_last_checkpoints"),
                    extra_state=build_extra_state(best_overall, best_loss_per_sequence),
                )
                print(f"✅ Checkpoint saved at step {step}")
                run_quick_generate(config_file_path, os.path.join(out_dir, f"ckpt_{step}"))

                # Save generation evaluation report periodically
                if gen_evaluator is not None and gen_evaluator.evaluation_history:
                    try:
                        report_path = os.path.join(out_dir, "generation_eval_report")
                        gen_evaluator.generate_report(report_path, format="json")
                    except Exception as e:
                        print(f"Warning: Could not save generation report: {e}")
    
    # Final
    progress_bar.close()
    
    if accelerator.is_main_process:
        accelerator.wait_for_everyone()
        save_checkpoint(
            model, optimizer, scheduler, out_dir, step, accelerator,
            extra_state=build_extra_state(best_overall, best_loss_per_sequence),
        )

        # Final generation evaluation report
        if gen_evaluator is not None and gen_evaluator.evaluation_history:
            print("\n=== Final Generation Quality Report ===")
            summary = gen_evaluator.get_summary_statistics()
            if summary:
                print(f"Total evaluations: {summary.get('num_evaluations', 0)}")
                score_stats = summary.get('overall_score', {})
                print(f"Average score: {score_stats.get('mean', 0):.1f}/100")
                print(f"Score range: {score_stats.get('min', 0):.1f} - {score_stats.get('max', 0):.1f}")
                print(f"Grade distribution: {summary.get('grade_distribution', {})}")

            # Save final reports
            try:
                report_path = os.path.join(out_dir, "final_generation_report")
                report_files = gen_evaluator.generate_report(report_path, format="all")
                print(f"Final reports saved:")
                for fmt, path in report_files.items():
                    print(f"  {fmt}: {path}")
            except Exception as e:
                print(f"Warning: Could not save final report: {e}")

        print("\n=== Training Complete ===")
        log_system_info("COMPLETE")
    
    if log_cfg.get("wandb", False):
        import wandb
        wandb.finish()
    
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
