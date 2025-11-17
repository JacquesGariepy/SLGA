# train.py
"""
Main training script for SLGA (Sparse Landmark Global Attention)

Includes:
- Automatic Mixed Precision (AMP)
- Gradient accumulation
- Learning rate warmup + cosine decay
- Auxiliary losses (diversity, sparsity)
- Progressive global attention warmup
- Periodic validation
- Checkpointing
- Optional W&B logging
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

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
from typing import Optional
from torch.optim.lr_scheduler import _LRScheduler

from src.model import Config, LLMTransformer
from src.data import get_tokenizer, load_text_dataset, CollatorLocal, CollatorLocalGlobal
from src.landmarks import landmark_spacing_loss, landmark_sparsity_loss, landmark_diversity_loss
from scripts.utils import set_seed, save_checkpoint, load_checkpoint
from src.live_metrics import LiveMetricsDisplay
from src.realtime_display import RealtimeTrainingDisplay
from src.monitoring import compute_long_dependency_recall

from torch.utils.tensorboard import SummaryWriter

def update_checkpointing(model: LLMTransformer, seq_len: int):
    model.cfg.grad_checkpointing = (seq_len > 1024)
    for block in model.blocks:
        block.cfg.grad_checkpointing = model.cfg.grad_checkpointing

def _unwrap_scheduler(scheduler: Optional[_LRScheduler]) -> Optional[_LRScheduler]:
    """
    Unwrap nested Accelerate schedulers to reach the underlying PyTorch scheduler.

    Accelerate wraps schedulers multiple times, this function peels back the layers
    to get to the actual torch scheduler for direct manipulation.
    """
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
    """
    Re-sync optimizer and scheduler learning rates with config values after wrapping/resuming.

    This ensures that after loading checkpoints or wrapping with Accelerate,
    the learning rates match the intended values from the config.
    """
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
    best_overall: dict[str, float | int],
    best_loss_per_sequence: dict[int, dict[str, float | int]],
) -> dict:
    """
    Create a serializable state dictionary for checkpoint metadata.

    This includes best validation losses and their corresponding steps,
    which are saved with checkpoints for tracking training progress.
    """
    return {
        "best_overall": dict(best_overall),
        "best_loss_per_sequence": {
            int(k): {"loss": float(v["loss"]), "step": int(v.get("step", -1))}
            for k, v in best_loss_per_sequence.items()
        },
    }


def run_quick_generate(config_path: str, checkpoint_dir: str) -> None:
    """
    Launch a quick text generation run to inspect the latest checkpoint quality.

    This runs the generate.py script with minimal parameters to produce sample text,
    helping verify that checkpoints are working and generating reasonable output.
    """
    script_path = os.path.join(os.path.dirname(__file__), "generate.py")
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    config_path = os.path.abspath(config_path)

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
    except Exception as exc:  # pylint: disable=broad-except
        print(f"⚠️  Generation preview failed: {exc}")

def get_current_seq_len(step: int, cfg: dict) -> int:
    """
    Calculate the current sequence length according to curriculum learning schedule.

    Progression: seq_len_start -> seq_len_mid -> seq_len_final

    This implements a gradual increase in sequence length during training to improve
    stability and allow the model to learn both short and long-range dependencies.

    Args:
        step: Forward pass counter (NOT optimizer step)
        cfg: Training configuration dictionary

    Returns:
        Current sequence length for this training step

    Note: This uses forward pass steps, not optimizer steps!
    """
    warmup_steps = cfg["train"].get("seq_len_warmup_steps", 15000)
    start_len = cfg["train"].get("seq_len_start", 512)
    mid_len = cfg["train"].get("seq_len_mid", 1024)
    final_len = cfg["train"].get("seq_len_final", 2048)
    
    if step < warmup_steps // 2:
        # Phase 1: start -> mid
        progress = step / (warmup_steps // 2)
        seq_len = start_len + progress * (mid_len - start_len)
    elif step < warmup_steps:
        # Phase 2: mid -> final
        progress = (step - warmup_steps // 2) / (warmup_steps // 2)
        seq_len = mid_len + progress * (final_len - mid_len)
    else:
        # Phase 3: final
        seq_len = final_len
    
    return int(seq_len)


def get_global_warmup_weight(step: int, cfg: dict) -> float:
    """
    Calculate the warmup weight for global attention.

    This allows progressive activation of global attention to avoid instability
    during early training when the model hasn't learned good landmark selection.

    Args:
        step: Forward pass counter (NOT optimizer step)
        cfg: Training configuration dictionary

    Returns:
        Weight between 0.0 and 1.0 for global attention contribution

    Note: This uses forward pass steps, not optimizer steps!
    """
    warmup_start = cfg["train"].get("global_warmup_start", 30000)
    warmup_end = cfg["train"].get("global_warmup_end", 50000)
    
    if step < warmup_start:
        return 0.0
    elif step < warmup_end:
        progress = (step - warmup_start) / (warmup_end - warmup_start)
        return progress
    else:
        return 1.0


def _run_command(cmd: list[str]) -> str:
    """Run a shell command and return its output or error message."""
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except Exception as exc:  # pylint: disable=broad-except
        return f"[command failed] {' '.join(cmd)}\n{exc}\n"


def log_system_info(tag: str = "SNAPSHOT") -> None:
    """
    Emit a system snapshot including GPU/RAM/FS information for logging.

    This collects various system metrics that are useful for monitoring
    training stability and resource usage over time.
    """
    commands = {
        "date": ["date"],
        "uname": ["uname", "-a"],
        "uptime": ["uptime"],
        "nvidia-smi": ["nvidia-smi"],
        "lscpu": ["bash", "-lc", "lscpu | head -n 20"],
        "free": ["bash", "-lc", "free -h"],
        "df": ["bash", "-lc", "df -h"],
        "top": ["bash", "-lc", "top -b -n1 | head -n 5"],
    }

    print(f"\n===== SYSTEM SNAPSHOT ({tag}) =====", flush=True)
    for label, cmd in commands.items():
        print(f"$ {' '.join(cmd)}  # {label}", flush=True)
        output = _run_command(cmd)
        print(output.rstrip(), flush=True)

    print("-- Python system summary --", flush=True)
    try:
        import platform
        cpu_count = os.cpu_count()
        platform_str = platform.platform()
        print(f"platform: {platform_str}", flush=True)
        print(f"cpu_count: {cpu_count}", flush=True)

        try:
            import psutil  # type: ignore

            cpu_freq = psutil.cpu_freq()
            cpu_percent = psutil.cpu_percent(interval=None)
            load_avg = psutil.getloadavg() if hasattr(psutil, "getloadavg") else None
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()

            if cpu_freq is not None:
                print(f"cpu_freq: current={cpu_freq.current:.1f}MHz max={cpu_freq.max:.1f}MHz", flush=True)
            print(f"cpu_percent: {cpu_percent:.1f}%", flush=True)
            if load_avg is not None:
                print(f"load_avg (1m,5m,15m): {load_avg}", flush=True)
            print(
                "memory: used={:.2f}GB free={:.2f}GB total={:.2f}GB".format(
                    vm.used / 1e9, vm.available / 1e9, vm.total / 1e9
                ),
                flush=True,
            )
            print(
                "swap: used={:.2f}GB free={:.2f}GB total={:.2f}GB".format(
                    swap.used / 1e9, swap.free / 1e9, swap.total / 1e9
                ),
                flush=True,
            )
        except ImportError:
            print("psutil not available; install it for detailed CPU/memory stats", flush=True)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[python system summary failed] {exc}", flush=True)

    print("===== END SYSTEM SNAPSHOT =====\n", flush=True)


def cross_entropy_shifted(
    logits: torch.Tensor, labels: torch.Tensor, pad_id: int
) -> torch.Tensor:
    """
    Compute cross-entropy loss with proper shifting for causal language modeling.

    Args:
        logits: Model output logits (B, L, V)
        labels: Target labels (B, L) - already shifted by collator, with -100 for padding
        pad_id: DEPRECATED - padding is now handled with -100 in labels

    Returns:
        Loss scalar tensor

    Note: The collator has already shifted labels, so logits[i] predicts labels[i],
    not labels[i+1]. We just remove the last position since there's no target for it.
    """
    # IMPORTANT: The collator has ALREADY shifted the labels!
    # labels[i] contains the token following input_ids[i]
    # So logits[i] should predict labels[i], NOT labels[i+1]!
    # We just remove the last position (no target for it)
    logits_shifted = logits[:, :-1, :].contiguous()  # (B, L-1, V)
    labels_shifted = labels[:, :-1].contiguous()     # (B, L-1)

    # Flatten
    loss = F.cross_entropy(
        logits_shifted.view(-1, logits_shifted.size(-1)),
        labels_shifted.view(-1),
        ignore_index=-100,  # ignore padding targets
    )

    return loss


def build_loaders(cfg: dict, tokenizer=None) -> tuple:
    """
    Build tokenizer and data loaders for training and validation.

    Handles dataset loading, automatic downloading if needed, and creates
    appropriate collators based on whether learned landmarks are used.

    Returns:
        tuple: (tokenizer, train_loader, val_loader)
    """
    tokenizer = tokenizer or get_tokenizer(cfg["tokenizer"])
    
    # Load datasets
    try:
        ds_train = load_text_dataset(
            cfg["data"]["dataset"],
            cfg["data"].get("subset"),
            cfg["data"]["split_train"],
        )
        ds_val = load_text_dataset(
            cfg["data"]["dataset"],
            cfg["data"].get("subset"),
            cfg["data"]["split_val"],
        )
    except Exception as e:
        print(f"Warning: Could not load dataset: {e}")
        print("Attempting to download dataset automatically...")
        
        # Auto-download dataset
        download_script = os.path.join(os.path.dirname(__file__), "download_dataset.py")
        dataset_name = cfg["data"]["dataset"]
        subset = cfg["data"].get("subset")
        split_train = cfg["data"]["split_train"]
        
        # Download training split
        cmd = [sys.executable, download_script, "--dataset", dataset_name, "--split", split_train]
        if subset:
            cmd.extend(["--subset", subset])
        
        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Download output:")
            print(result.stdout)
            if result.stderr:
                print("Download stderr:")
                print(result.stderr)
        except subprocess.CalledProcessError as download_error:
            print(f"❌ Failed to download dataset: {download_error}")
            print("Please download the dataset manually or check your configuration.")
            raise download_error
        
        # Retry loading after download
        try:
            ds_train = load_text_dataset(
                cfg["data"]["dataset"],
                cfg["data"].get("subset"),
                cfg["data"]["split_train"],
            )
            ds_val = load_text_dataset(
                cfg["data"]["dataset"],
                cfg["data"].get("subset"),
                cfg["data"]["split_val"],
            )
        except Exception as retry_error:
            print(f"Warning: Could not load validation split after download: {retry_error}")
            print("Using subset of training data for validation")
            ds_all = load_text_dataset(
                cfg["data"]["dataset"],
                cfg["data"].get("subset"),
                cfg["data"]["split_train"],
            )
            # Manual split
            split_idx = int(len(ds_all) * 0.95)
            ds_train = ds_all.select(range(split_idx))
            ds_val = ds_all.select(range(split_idx, len(ds_all)))
    
    # Limit dataset size if specified
    max_train = cfg["data"].get("max_train_samples")
    max_val = cfg["data"].get("max_val_samples")
    
    if max_train and len(ds_train) > max_train:
        ds_train = ds_train.select(range(max_train))
    if max_val and len(ds_val) > max_val:
        ds_val = ds_val.select(range(max_val))
    
    print(f"Train samples: {len(ds_train)}")
    print(f"Val samples: {len(ds_val)}")
    
    use_learned = cfg["model"].get("learned_landmarks", True)
    
    seq_len_start = cfg["train"].get("seq_len_start", 512)
    seq_len_final = cfg["train"].get("seq_len_final", seq_len_start)

    def build_collator(max_length: int):
        if use_learned:
            return CollatorLocal(tokenizer, max_length)
        return CollatorLocalGlobal(
            tokenizer,
            max_length,
            cfg["train"]["global_every"],
            cfg["train"]["max_global"],
        )

    collate_train = build_collator(seq_len_final)
    collate_val = build_collator(cfg["train"].get("seq_len_final", seq_len_final))
    
    # DataLoaders
    train_loader = DataLoader(
        ds_train,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=collate_train,
        num_workers=cfg["data"].get("num_workers", 2),
        pin_memory=True,
    )
    
    # Validation collator: accepts raw text or pre-tokenized datasets
    def collate_val_reduced(examples):
        """
        Robust validation collator that handles both raw text and pre-tokenized datasets.
        - Truncates/pads to 512 (+1 for label shifting).
        - Auto-detects format: looks for 'input_ids' or text fields.
        """
        import torch
        max_len_val = 512
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        def _stack_input_ids(ids_list):
            tens = []
            for ids in ids_list:
                if not torch.is_tensor(ids):
                    ids = torch.as_tensor(ids, dtype=torch.long)
                ids = ids.view(-1)
                # truncate/pad to max_len_val+1
                if ids.numel() >= max_len_val + 1:
                    ids = ids[: max_len_val + 1]
                else:
                    pad = torch.full((max_len_val + 1 - ids.numel(),), pad_id, dtype=torch.long)
                    ids = torch.cat([ids, pad], dim=0)
                tens.append(ids)
            input_ids = torch.stack(tens, dim=0)  # (B, L+1)

            # Create input_ids and labels BEFORE masking
            input_ids_final = input_ids[:, :-1]  # (B, L)
            labels = input_ids[:, 1:].clone()  # (B, L)

            # Mask padding tokens with -100 after setup
            pad_mask = (labels == pad_id)
            labels[pad_mask] = -100

            return {
                "input_ids": input_ids_final,  # (B, L)
                "labels": labels,  # (B, L) with -100 for pads
                "cache_global_ids": None,
            }

        # 1) Pre-tokenized case: each example contains 'input_ids'
        ex0 = examples[0]
        if isinstance(ex0, dict) and "input_ids" in ex0:
            return _stack_input_ids([ex["input_ids"] for ex in examples])

        # 2) Raw text case: find a text key
        text_key = None
        if isinstance(ex0, dict):
            for k in ("text", "content", "document", "raw", "prompt"):
                if k in ex0 and isinstance(ex0[k], str):
                    text_key = k
                    break
        # 2b) Example is directly a string
        if text_key is None and isinstance(ex0, str):
            texts = list(examples)
        elif text_key is not None:
            texts = [ex[text_key] for ex in examples]
        else:
            # Last resort: take first string field found
            if isinstance(ex0, dict):
                candidates = [k for k, v in ex0.items() if isinstance(v, str)]
                if candidates:
                    texts = [ex[candidates[0]] for ex in examples]
                else:
                    raise KeyError(f"Unable to find text field. Keys: {list(ex0.keys())}")
            else:
                raise KeyError(f"Unsupported example format: {type(ex0).__name__}")

        # Tokenize text → input_ids (L+1)
        encoded = tokenizer(
            texts,
            max_length=max_len_val + 1,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]

        # Create input_ids and labels BEFORE masking
        input_ids_final = input_ids[:, :-1]  # (B, L)
        labels = input_ids[:, 1:].clone()  # (B, L)

        # Mask padding tokens with -100
        pad_mask = (labels == pad_id)
        labels[pad_mask] = -100

        return {
            "input_ids": input_ids_final,
            "labels": labels,
            "cache_global_ids": None,
        }

    # 🔍 DEBUG: Display validation dataset format
    if len(ds_val) > 0:
        ex0 = ds_val[0]
        if isinstance(ex0, dict):
            print(f"🔍 DEBUG Dataset format: keys={list(ex0.keys())}, types={[(k, type(v).__name__) for k, v in list(ex0.items())[:5]]}")
        else:
            print(f"🔍 DEBUG Dataset format: type={type(ex0).__name__}")

    # Validation with reduced batch_size to avoid OOM
    val_batch_size = max(1, cfg["train"]["batch_size"] // 2)  # Half of train batch size
    print(f"Validation config: batch_size={val_batch_size} (train: {cfg['train']['batch_size']}), seq_len=512 (train: up to 2048)")

    # 🔧 DEBUG ASTUCE: num_workers=0 temporarily to see full stacktrace
    # If validation works, revert to num_workers=2 for performance
    val_loader = DataLoader(
        ds_val,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_val_reduced,  # 🔧 Robust auto-detection collator
        num_workers=0,  # 🔧 TEMPORARY: 0 for debug, revert to 2 when stable
        pin_memory=False,  # 🔧 TEMPORARY: disabled for debug
    )
    
    return tokenizer, train_loader, val_loader


def validate(
    model: LLMTransformer,
    val_loader: DataLoader,
    pad_id: int,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> dict:
    """
    Evaluate the model on the validation set.

    Returns:
        dict: Metrics containing 'loss' and 'perplexity'
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    # Progress bar for validation
    import sys
    max_b = max_batches if max_batches else len(val_loader)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches and i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            cache_ids = batch.get("cache_global_ids")
            cache_ids = cache_ids.to(device) if cache_ids is not None else None

            # Check labels before forward to detect out-of-vocab values
            if hasattr(model, 'cfg') and hasattr(model.cfg, 'vocab_size'):
                vocab_size = model.cfg.vocab_size
            elif hasattr(model, 'lm_head') and hasattr(model.lm_head, 'out_features'):
                vocab_size = model.lm_head.out_features
            else:
                vocab_size = 50257  # Fallback to GPT-2 default

            # Keep these operations outside the graph to avoid CUDA retention
            invalid_mask = (labels != -100) & ((labels < 0) | (labels >= vocab_size))
            invalid_count = invalid_mask.sum().item()
            if invalid_count > 0:
                print(f"\n❌ VALIDATION BATCH {i} HAS INVALID LABELS!")
                print(f"   Invalid count: {invalid_count}")
                invalid_values = labels[invalid_mask][:20].detach().cpu().tolist()
                print(f"   Invalid values: {invalid_values}")
                positions = invalid_mask.nonzero(as_tuple=False).detach().cpu()
                batch_idx = positions[:10, 0].tolist()
                seq_idx = positions[:10, 1].tolist()
                print(f"   First 10 positions (batch, seq): {list(zip(batch_idx, seq_idx))}")
                print(f"   Corresponding input_ids: {input_ids[positions[:10, 0], positions[:10, 1]].detach().cpu().tolist()}")
                # Force stop to prevent CUDA crash
                raise ValueError(f"Invalid labels detected in validation batch {i}")

            # Forward pass
            logits = model(input_ids, cache_global_ids=cache_ids)

            # Loss calculation
            loss = cross_entropy_shifted(logits, labels, pad_id)

            # Count valid tokens (ignore -100)
            num_tokens = (labels != -100).sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            num_batches += 1

            # Explicitly free tensors to prevent memory accumulation
            del input_ids, labels, cache_ids, logits, loss, invalid_mask

            if i % 5 == 0 and torch.cuda.is_available():  # Every 5 batches
                torch.cuda.empty_cache()

            # Progress indicator
            if (i + 1) % 5 == 0 or (i + 1) == max_b:
                print(f"\r  Validation: {i+1}/{max_b} batches...", end='', flush=True)

    print()  # New line after progress
    
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 10))  # Cap for stability
    
    return {"loss": avg_loss, "perplexity": perplexity}


def main():
    """Main training loop function"""

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Train SLGA model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    config_file_path = os.path.abspath(args.config)

    # Reduce CUDA fragmentation (new recommended variable, keep old for compatibility)
    if os.environ.get("PYTORCH_ALLOC_CONF") is None:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Load config
    with open(config_file_path) as f:
        cfg = yaml.safe_load(f)

    # Create best_sequence folder from the beginning
    out_dir = cfg["train"].get("out_dir", "out_slga")
    best_sequence_root = os.path.join(out_dir, "best_sequence")
    os.makedirs(best_sequence_root, exist_ok=True)

    log_cfg = cfg.get("log", {})

    # Override max_steps if provided
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
        # Adjust warmup proportionally if max_steps is reduced
        if args.max_steps < cfg["train"]["warmup_steps"]:
            original_warmup = cfg["train"]["warmup_steps"]
            adjusted_warmup = max(100, args.max_steps // 2)  # 50% of total or min 100
            cfg["train"]["warmup_steps"] = adjusted_warmup
            print(f"⚠️  Warmup adjusted: {original_warmup} → {adjusted_warmup} (max_steps={args.max_steps})")

    # TensorBoard writer (created after config loading)
    if log_cfg.get("tensorboard", False):
        writer = SummaryWriter(log_dir=f"{cfg['save']['out_dir']}/tensorboard")
    else:
        writer = None
    
    # Setup
    set_seed(cfg["seed"])
    accelerator = Accelerator()
    system_info_every = log_cfg.get("system_info_every", 500)
    if system_info_every is not None and system_info_every <= 0:
        system_info_every = None

    system_info_interval_seconds = log_cfg.get("system_info_interval_seconds", 120)
    if system_info_interval_seconds is not None and system_info_interval_seconds <= 0:
        system_info_interval_seconds = None

    system_monitor_stop: Optional[threading.Event] = None
    system_monitor_thread: Optional[threading.Thread] = None

    if accelerator.is_main_process:
        log_system_info("STARTUP")

        if system_info_interval_seconds is not None:
            system_monitor_stop = threading.Event()

            def _system_monitor_loop() -> None:
                while not system_monitor_stop.wait(system_info_interval_seconds):
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    log_system_info(f"PERIODIC {timestamp}")

            system_monitor_thread = threading.Thread(
                target=_system_monitor_loop,
                name="SystemMonitor",
                daemon=True,
            )
            system_monitor_thread.start()
    device = accelerator.device
    
    # W&B logging (optional)
    if log_cfg.get("wandb", False):
        import wandb
        run_name = cfg["log"].get("run_name") or f"slga_{cfg['model']['embed_dim']}d_{cfg['model']['n_layers']}l"
        wandb.init(
            project=cfg["log"]["project"],
            name=run_name,
            config=cfg,
        )
    
    # Tokenizer
    tokenizer = get_tokenizer(cfg["tokenizer"])
    if tokenizer.vocab_size is not None:
        cfg["model"]["vocab_size"] = tokenizer.vocab_size

    # Model
    model_cfg = Config(**cfg["model"])
    if "grad_checkpointing" in cfg["train"]:
        model_cfg.grad_checkpointing = cfg["train"]["grad_checkpointing"]

    model = LLMTransformer(model_cfg)
    
    if accelerator.is_main_process:
        print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")
        print(f"Non-embedding params: {model.get_num_params(non_embedding=True) / 1e6:.2f}M")
        print(f"🔍 Learned landmarks: {model_cfg.learned_landmarks}")
        print(f"🔍 Landmark selector: {model.landmark_selector is not None}")
        if model.landmark_selector is not None:
            print(f"   → Num landmarks: {model.landmark_selector.num_landmarks}")
            print(f"   → Temperature: {model.landmark_selector.temperature}")
            print(f"   → Temperature decay: {model.landmark_selector.temperature_decay}")

    # Data
    tokenizer, train_loader, val_loader = build_loaders(cfg, tokenizer=tokenizer)
    pad_id = tokenizer.pad_token_id
    
    # Optimizer with separate LR for landmark scorer
    scorer_lr_multiplier = cfg["train"].get("scorer_lr_multiplier", 5.0)  # 5× default
    base_lr = cfg["train"]["lr"]

    # Separate parameters: scorer vs rest of model
    scorer_params = []
    other_params = []

    if model.landmark_selector is not None:
        scorer_params = list(model.landmark_selector.scorer.parameters())
        # All other parameters
        scorer_param_ids = {id(p) for p in scorer_params}
        other_params = [p for p in model.parameters() if id(p) not in scorer_param_ids]
    else:
        other_params = list(model.parameters())

    # Create param groups
    param_groups = [
        {
            "params": other_params,
            "lr": base_lr,
            "name": "model",
        },
    ]

    if scorer_params:
        param_groups.append({
            "params": scorer_params,
            "lr": base_lr * scorer_lr_multiplier,  # Higher LR for scorer
            "name": "scorer",
        })
        if accelerator.is_main_process:
            print(f"🔧 Scorer LR: {base_lr * scorer_lr_multiplier:.2e} (×{scorer_lr_multiplier} vs model)")

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

    # 🔧 CRITICAL: UNDERSTANDING STEP COUNTING
    # ==========================================
    # Two different step counters are used in training:
    #
    # 1. `step` = Forward pass counter (0 → max_steps)
    #    - Increments on EVERY  pass
    #    - Used for: logging, checkpointing, curriculum, warmup schedules
    #    - With accum_steps=4: 0, 1, 2, 3, 4, 5, ..., 100000
    #
    # 2. `optimizer_step` = Optimizer update counter (0 → max_steps // accum_steps)
    #    - Only increments when optimizer.step() is called
    #    - Used for: scheduler.step() (LR schedule)
    #    - With accum_steps=4: 0, 0, 0, 0, 1, 1, 1, 1, 2, ..., 25000
    #
    # Config file interpretation:
    #   - max_steps: 100000      → 100K  passes = 25K optimizer updates (with accum=4)
    #   - warmup_steps: 2000     → 2000  passes = 500 optimizer updates (with accum=4)
    #
    # Scheduler must use OPTIMIZER steps (not  passes):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps // accum_steps,    # 2000 → 500 (with accum=4)
        num_training_steps=total_steps // accum_steps,    # 100000 → 25000 (with accum=4)
    )
    
    # Prepare with Accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    apply_lr_from_config(optimizer, scheduler, base_lr, scorer_lr_multiplier)
    
    # AMP setup
    amp_enabled = cfg["train"].get("amp", True)
    amp_dtype_str = cfg["train"].get("amp_dtype", "bf16").lower()
    
    if amp_dtype_str == "bf16" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    
    if accelerator.is_main_process:
        print(f"AMP enabled: {amp_enabled}, dtype: {amp_dtype}")
    
    # Training loop
    out_dir = cfg["save"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    accum_steps = cfg["train"]["accum_steps"]
    grad_clip = cfg["train"]["grad_clip"]

    # Resume from checkpoint if requested
    step = 0  # Forward passes (actual training iterations)
    optimizer_step = 0  # Optimizer updates (step // accum_steps)
    if args.resume:
        # Find latest checkpoint
        checkpoints = [d for d in os.listdir(out_dir) if d.startswith("ckpt_")]
        if checkpoints:
            # Sort by step number (ckpt_1000 -> 1000)
            latest_ckpt = max(checkpoints, key=lambda x: int(x.split("_")[1]))
            checkpoint_path = os.path.join(out_dir, latest_ckpt)

            if accelerator.is_main_process:
                print(f"\n{'='*80}")
                print(f"🔄 RESUMING FROM CHECKPOINT: {latest_ckpt}")
                print(f"{'='*80}\n")

            # Unwrap model for loading (if wrapped by Accelerator)
            unwrapped_model = accelerator.unwrap_model(model)

            # Load checkpoint
            resume_state: dict = {}
            step = load_checkpoint(
                checkpoint_path,
                unwrapped_model,
                optimizer,
                scheduler,
                device=str(device),
                extra_state=resume_state,
            )
            # Calculate optimizer_step from loaded step
            optimizer_step = step // accum_steps

            if "best_overall" in resume_state:
                loaded_overall = resume_state["best_overall"]
                best_overall = {
                    "loss": float(loaded_overall.get("loss", float("inf"))),
                    "step": int(loaded_overall.get("step", -1)),
                }
            elif "best_overall_loss" in resume_state:
                best_overall = {
                    "loss": float(resume_state["best_overall_loss"]),
                    "step": -1,
                }
            elif "best_val_loss" in resume_state:
                best_overall = {
                    "loss": float(resume_state["best_val_loss"]),
                    "step": -1,
                }
            if "best_loss_per_sequence" in resume_state:
                best_loss_per_sequence = {}
                for k, v in resume_state["best_loss_per_sequence"].items():
                    if isinstance(v, dict):
                        best_loss_per_sequence[int(k)] = {
                            "loss": float(v.get("loss", float("inf"))),
                            "step": int(v.get("step", -1)),
                        }
                    else:
                        best_loss_per_sequence[int(k)] = {
                            "loss": float(v),
                            "step": -1,
                        }

            apply_lr_from_config(optimizer, scheduler, base_lr, scorer_lr_multiplier)

            if accelerator.is_main_process:
                print(f"\n✅ Resumed from step {step} (optimizer_step {optimizer_step})\n")
                log_system_info(f"RESUME STEP {step}")
        else:
            if accelerator.is_main_process:
                print(f"\n⚠️ --resume specified but no checkpoints found in {out_dir}")
                print(f"Starting from scratch...\n")

    model.train()

    # Performance metrics tracking
    step_start_time = time.time()
    steps_since_log = 0

    # Persistent variables for metrics (avoid 0.00 in logs)
    last_grad_norm = 0.0
    last_num_landmarks = 0
    last_spacing_loss = 0.0  # replaces old last_div_loss
    last_spar_loss = 0.0
    last_mass_ratio = 0.0  # Mass concentration in top-G landmarks
    best_overall: dict[str, float | int] = {"loss": float("inf"), "step": -1}
    best_loss_per_sequence: dict[int, dict[str, float | int]] = {}

    # Infinite loop over dataloader
    epoch = 0

    # Real-time display (updates every step)
    if accelerator.is_main_process:
        realtime_display = RealtimeTrainingDisplay(
            max_steps=total_steps,
            detail_every=cfg["train"].get("log_every", 50),
            width=100,
        )
    else:
        realtime_display = None

    # Disable tqdm (we use realtime_display)
    progress_bar = tqdm(
        total=total_steps,
        desc="Training",
        disable=True,  # Always disabled if realtime_display
    )

    # Small CPU caches detached for logging, avoid retaining graph
    log_landmark_indices = None
    log_gate_values = None

    while step < total_steps:
        epoch += 1  # Track epochs (though we use infinite loop with step-based termination)
        
        for batch in train_loader:
            # === CORE TRAINING STEP ===
            # Curriculum sequence length (update collator if needed)
            current_seq_len = get_current_seq_len(step, cfg)
            
            # === NOUVEAU : checkpointing dynamique ===
            model.cfg.grad_checkpointing = (current_seq_len > 1024)
            for block in model.blocks:
                block.cfg.grad_checkpointing = model.cfg.grad_checkpointing
            # =========================================

            # Global warmup weight
            global_weight = get_global_warmup_weight(step, cfg)
            
            # === DATA PREPARATION ===
            # Load from CPU, then apply truncation BEFORE GPU transfer
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            cache_ids = batch.get("cache_global_ids")

            # Truncate on CPU to reduce GPU memory footprint
            if input_ids.size(1) > current_seq_len:
                input_ids = input_ids[:, :current_seq_len]
                labels = labels[:, :current_seq_len]

                # Adapt heuristic landmarks to new length
                if cache_ids is not None and not model.cfg.learned_landmarks:
                    B = cache_ids.size(0)
                    G = cache_ids.size(1)
                    cache_ids = torch.linspace(
                        0, current_seq_len - 1, G, device=input_ids.device if input_ids.is_cuda else torch.device("cpu")
                    ).long().unsqueeze(0).expand(B, -1)
                elif cache_ids is not None and model.cfg.learned_landmarks:
                    cache_ids = torch.clamp(cache_ids, 0, current_seq_len - 1)

            # GPU transfer AFTER truncation
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            cache_ids = cache_ids.to(device) if cache_ids is not None else None
            
            # === FORWARD PASS ===
            # Initialize lambda_spar before autocast block for all configurations
            lambda_spar = cfg["train"].get("lambda_sparsity", 0.0)

            # Autocast only when CUDA is available
            use_autocast = amp_enabled and device.type == "cuda"
            if use_autocast:
                autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
            else:
                from contextlib import nullcontext
                autocast_ctx = nullcontext()

            with autocast_ctx:
                # === MODEL FORWARD PASS ===
                logits, aux = model(input_ids, cache_global_ids=cache_ids, return_aux=True, global_weight=global_weight)
                
                # === LOSS CALCULATION ===
                loss_ce = cross_entropy_shifted(logits, labels, pad_id)
                loss = loss_ce / accum_steps  # Scale for gradient accumulation

                # === AUXILIARY LOSSES INITIALIZATION ===
                spacing_loss_val = 0.0
                spar_loss_val = 0.0
                num_landmarks_selected = 0
                gate_entropy_val: Optional[float] = None
                gate_local_mean: Optional[float] = None
                gate_local_majority: Optional[float] = None

                # Extract auxiliary outputs for landmark analysis
                landmark_indices = aux.get("landmark_indices", None)
                landmark_scores = aux.get("landmark_scores", None)

                # Count selected landmarks
                if landmark_indices is not None and landmark_indices.numel() > 0:
                    num_landmarks_selected = landmark_indices.size(1)

                # === AUXILIARY LOSSES ===
                if landmark_indices is not None and landmark_scores is not None:
                    seq_len = input_ids.size(1)

                    # Spacing loss: encourages even distribution of landmarks
                    lambda_spacing = cfg["train"].get("lambda_spacing", 0.0)
                    if lambda_spacing > 0 and num_landmarks_selected > 1:
                        spacing_loss = landmark_spacing_loss(
                            landmark_indices=landmark_indices,
                            seq_len=seq_len,
                            lambda_reg=lambda_spacing,
                            selection_scores=landmark_scores
                        )
                        spacing_loss_val = spacing_loss.item()
                        loss = loss + spacing_loss / accum_steps

                    # Sparsity loss: encourages concentration on top landmarks
                    if lambda_spar > 0 and num_landmarks_selected > 0:
                        spar_loss = landmark_sparsity_loss(
                            selection_scores=landmark_scores,
                            num_landmarks=num_landmarks_selected,  # NEW: adaptive parameter
                            lambda_reg=lambda_spar
                        )
                        spar_loss_val = spar_loss.item()
                        loss = loss + spar_loss / accum_steps

                    # Legacy diversity loss (optional, for backward compatibility)
                    lambda_div = cfg["train"].get("lambda_diversity", 0.0)
                    if lambda_div > 0:
                        # NOTE: Use lambda_spacing instead (recommended)
                        div_loss = landmark_diversity_loss(landmark_scores, lambda_div)
                        loss = loss + div_loss / accum_steps

                # === MONITORING METRICS ===
                # Calculate mass_in_top_g for monitoring (even if loss constant)
                mass_in_top_g = 0.0
                if lambda_spar > 0 and landmark_scores is not None and num_landmarks_selected > 0:
                    # Recalculate for logging (cheap)
                    import torch.nn.functional as F
                    probs = F.softmax(landmark_scores, dim=-1)
                    _, top_g_idx = torch.topk(landmark_scores, k=num_landmarks_selected, dim=-1)
                    top_g_probs = torch.gather(probs, dim=1, index=top_g_idx)
                    mass_in_top_g = top_g_probs.sum(dim=-1).mean().item()

                # Save for logging (persistent values)
                last_spacing_loss = spacing_loss_val  # NEW: replaces div_loss
                last_spar_loss = spar_loss_val
                last_num_landmarks = num_landmarks_selected
                last_mass_ratio = mass_in_top_g  # New metric

            # Detach large aux tensors and keep only useful for logs
            if landmark_indices is not None:
                try:
                    log_landmark_indices = landmark_indices.detach().to("cpu")
                except Exception:
                    log_landmark_indices = None
            else:
                log_landmark_indices = None

            if isinstance(aux, dict) and ("gate_values" in aux) and (aux["gate_values"] is not None):
                try:
                    log_gate_values = aux["gate_values"].detach().to("cpu")
                except Exception:
                    log_gate_values = None
            else:
                log_gate_values = None

            # Calculate gate metrics if available
            if log_gate_values is not None:
                gate_tensor = log_gate_values.float()  # (layers, B, H, L)
                gate_flat = gate_tensor.reshape(-1)
                if gate_flat.numel() > 0:
                    gate_clamped = torch.clamp(gate_flat, min=1e-4, max=1 - 1e-4)
                    gate_entropy = -(gate_clamped * torch.log(gate_clamped) + (1 - gate_clamped) * torch.log(1 - gate_clamped))
                    gate_entropy_val = gate_entropy.mean().item()
                    gate_local_mean = gate_flat.mean().item()
                    gate_local_majority = (gate_flat > 0.5).float().mean().item()

            # Free references to avoid retaining graph during backward
            landmark_scores = None
            landmark_indices = None
            aux = None

            # Detect NaN/Inf before backward
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n{'='*80}")
                print(f"❌ DIVERGENCE DETECTED at step {step}!")
                print(f"{'='*80}")
                print(f"   Loss totale: {loss.item()}")
                print(f"   Loss CE: {loss_ce.item()}")
                if spacing_loss_val > 0:
                    print(f"   Spacing loss: {spacing_loss_val:.6f}")
                if spar_loss_val > 0:
                    print(f"   Sparsity loss: {spar_loss_val:.6f}")
                print(f"   Learning rate: {scheduler.get_last_lr()[0]:.2e}")
                print(f"   Gradient norm (last): {last_grad_norm:.4f}")

                # Save debug checkpoint
                if accelerator.is_main_process:
                    accelerator.wait_for_everyone()
                    debug_dir = f"ckpt_{step}_diverged"
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        out_dir,
                        step,
                        accelerator,
                        extra_state=build_extra_state(best_overall, best_loss_per_sequence),
                    )
                    print(f"   💾 Debug checkpoint sauvegardé: {out_dir}/{debug_dir}")

                raise ValueError(f"Training diverged with NaN/Inf at step {step}")

            # === BACKWARD PASS ===
            accelerator.backward(loss)
            
            # === GRADIENT ACCUMULATION ===
            if (step + 1) % accum_steps == 0:
                # Calculate gradient norm BEFORE clipping (for monitoring)
                grad_norm = 0.0
                if accelerator.is_main_process:
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            grad_norm += param_norm.item() ** 2
                    grad_norm = grad_norm ** 0.5
                    last_grad_norm = grad_norm  # Save for logging

                    # Gradient flow monitoring (every 500 steps)
                    if step % 500 == 0:
                        grad_norms_per_layer = {}

                        # Use named_parameters() to avoid crash
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                layer_norm = param.grad.data.norm(2).item()
                                grad_norms_per_layer[name] = layer_norm

                        # Log the largest gradients
                        top_grads = sorted(grad_norms_per_layer.items(), key=lambda x: x[1], reverse=True)[:5]
                        print(f"\n  Top gradient norms:")
                        for name, norm in top_grads:
                            print(f"    {name}: {norm:.4f}")

                # Gradient clipping
                if grad_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)

                # === OPTIMIZER STEP ===
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # Increment optimizer step counter after update
                optimizer_step += 1

            # === STEP MANAGEMENT ===
            step += 1  # Increment forward pass counter
            steps_since_log += 1
            progress_bar.update(1)

            # Real-time update (EVERY step)
            if realtime_display and accelerator.is_main_process:
                # Calculate current metrics
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1e9
                    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                else:
                    mem_allocated = 0
                    mem_total = 0

                # Calculate active landmarks (number of landmarks actually selected)
                active_lm = 0
                if landmark_indices is not None:
                    # Selected landmarks are those in landmark_indices
                    active_lm = landmark_indices.numel() // landmark_indices.size(0)  # G landmarks per batch
                    # Or simply use num_landmarks_selected which is already calculated
                    active_lm = num_landmarks_selected

                # Update live line
                realtime_display.update_live(
                    step=step,
                    loss=loss_ce.item() if step % accum_steps == 0 else None,  # Only after accum
                    ppl=math.exp(min(loss_ce.item(), 10)) if step % accum_steps == 0 else None,
                    lr=scheduler.get_last_lr()[0] if step % accum_steps == 0 else None,
                    throughput=None,  # Calculated in detailed section
                    gpu_mem_gb=mem_allocated,
                    gpu_total_gb=mem_total,
                    seq_len=current_seq_len,
                    global_weight=global_weight,
                    num_landmarks=num_landmarks_selected,
                    active_landmarks=active_lm,
                )

            # Logging
            if accelerator.is_main_process and step % cfg["train"].get("log_every", 100) == 0:
                if system_info_every and step % system_info_every == 0:
                    log_system_info(f"STEP {step}")

                # Gather loss (multi-GPU)
                loss_gathered = accelerator.gather(loss_ce.detach()).mean().item()
                lr_current = scheduler.get_last_lr()[0]
                ppl = math.exp(min(loss_gathered, 10))

                # Performance metrics
                elapsed_time = time.time() - step_start_time
                steps_per_sec = steps_since_log / elapsed_time if elapsed_time > 0 else 0
                tokens_per_sec = steps_per_sec * cfg["train"]["batch_size"] * current_seq_len

                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1e9  # GB
                    mem_reserved = torch.cuda.memory_reserved() / 1e9    # GB
                    mem_cached = torch.cuda.memory_cached() / 1e9 if hasattr(torch.cuda, 'memory_cached') else 0
                    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
                else:
                    mem_allocated = 0
                    mem_reserved = 0
                    mem_cached = 0
                    mem_total = 0

                # Calculate landmark coverage (portion of sequence at distance < window)
                coverage_val = None
                landmark_pos_hist = None
                if log_landmark_indices is not None:
                    valid_landmarks = log_landmark_indices[log_landmark_indices >= 0]
                    if valid_landmarks.numel() > 0:
                        seq_len = int(input_ids.size(1))
                        local_window = cfg["model"].get("local_window", 0)
                        if local_window and local_window > 0:
                            coverage_scores = []
                            positions = torch.arange(seq_len)
                            for sample_landmarks in log_landmark_indices:
                                landmarks_sample = sample_landmarks[sample_landmarks >= 0]
                                if landmarks_sample.numel() == 0:
                                    continue
                                distances = torch.abs(landmarks_sample.unsqueeze(1) - positions.unsqueeze(0))
                                min_dist, _ = torch.min(distances, dim=0)
                                coverage_scores.append((min_dist <= local_window).float().mean().item())
                            if coverage_scores:
                                coverage_val = sum(coverage_scores) / len(coverage_scores)
                        landmark_pos_hist = valid_landmarks.float() / max(seq_len - 1, 1)

                log_dict = {
                    "step": step,
                    "optimizer_step": optimizer_step,  # expose optimizer step for logging
                    "epoch": epoch,
                    "loss": loss_gathered,
                    "perplexity": ppl,
                    "lr": lr_current,
                    "seq_len": current_seq_len,
                    "global_weight": global_weight,
                    "grad_norm": last_grad_norm,  # Use persistent value
                }

                if gate_entropy_val is not None:
                    log_dict["gate_entropy"] = gate_entropy_val
                if gate_local_mean is not None:
                    log_dict["gate_local_mean"] = gate_local_mean
                    log_dict["gate_global_mean"] = 1.0 - gate_local_mean
                if gate_local_majority is not None:
                    log_dict["gate_local_majority"] = gate_local_majority
                if coverage_val is not None:
                    log_dict["landmark_coverage"] = coverage_val

                if log_gate_values is not None:
                    gate_mean = log_gate_values.mean().item()
                    gate_std = log_gate_values.std().item()

                    log_dict['gate_mean'] = gate_mean
                    log_dict['gate_std'] = gate_std

                    if writer is not None:
                        writer.add_scalar("train/gate_mean", gate_mean, step)
                        writer.add_scalar("train/gate_std", gate_std, step)

                # W&B
                if cfg["log"].get("wandb", False):
                    wandb.log(log_dict, step=step)

                # TensorBoard - Basic metrics
                if writer is not None:
                    writer.add_scalar("train/loss", loss_gathered, step)
                    writer.add_scalar("train/perplexity", ppl, step)
                    writer.add_scalar("train/learning_rate", lr_current, step)
                    writer.add_scalar("train/seq_len", current_seq_len, step)
                    writer.add_scalar("train/global_weight", global_weight, step)

                    # Gradient norm (persistent value)
                    writer.add_scalar("train/grad_norm", last_grad_norm, step)

                    if last_spacing_loss > 0:
                        writer.add_scalar("train/loss_spacing", last_spacing_loss, step)  
                    if last_spar_loss > 0:
                        writer.add_scalar("train/loss_sparsity", last_spar_loss, step)
                        # 📊 Logging mass_in_top_g to track scorer evolution

                    if gate_entropy_val is not None:
                        writer.add_scalar("train/gate_entropy", gate_entropy_val, step)
                    if gate_local_mean is not None:
                        writer.add_scalar("train/pct_tokens_local", gate_local_mean, step)
                        writer.add_scalar("train/pct_tokens_global", 1.0 - gate_local_mean, step)
                    if gate_local_majority is not None:
                        writer.add_scalar("train/pct_tokens_local_majority", gate_local_majority, step)
                    if coverage_val is not None:
                        writer.add_scalar("train/landmark_coverage", coverage_val, step)
                    if landmark_pos_hist is not None:
                        writer.add_histogram("train/landmark_position_norm", landmark_pos_hist, step)
                        # Even if loss is constant at first, mass increases when scorer learns
                        if last_mass_ratio > 0:
                            writer.add_scalar("landmarks/mass_in_top_g", last_mass_ratio, step)

                    # Landmark statistics (persistent value)
                    if last_num_landmarks > 0:
                        writer.add_scalar("landmarks/num_selected", last_num_landmarks, step)

                    # ✅ AJOUT #2: Landmark spacing metrics
                    if log_landmark_indices is not None:
                        landmark_indices_detached = log_landmark_indices  # (B, G) already CPU/detached

                        # Calculate spacing between landmarks
                        sorted_idx = torch.sort(landmark_indices_detached, dim=-1)[0]
                        gaps = sorted_idx[:, 1:] - sorted_idx[:, :-1]

# Use .item() to break the graph
                        spacing_mean = gaps.float().mean().item()
                        spacing_std = gaps.float().std().item()

                        writer.add_scalar("landmarks/spacing_mean", spacing_mean, step)
                        writer.add_scalar("landmarks/spacing_std", spacing_std, step)

                    # Performance metrics
                    writer.add_scalar("perf/steps_per_sec", steps_per_sec, step)
                    writer.add_scalar("perf/tokens_per_sec", tokens_per_sec, step)
                    writer.add_scalar("perf/gpu_memory_allocated_gb", mem_allocated, step)
                    writer.add_scalar("perf/gpu_memory_reserved_gb", mem_reserved, step)
                    writer.add_scalar("perf/gpu_memory_cached_gb", mem_cached, step)

                # Real-time Display: Detailed display
                if realtime_display:
                    realtime_display.print_detailed(
                        step=step,
                        loss=loss_gathered,
                        ppl=ppl,
                        lr=lr_current,
                        grad_norm=last_grad_norm,
                        seq_len=current_seq_len,
                        global_weight=global_weight,
                        throughput=tokens_per_sec,
                        gpu_mem_gb=mem_allocated,
                        gpu_total_gb=mem_total,
                        num_landmarks=last_num_landmarks,
                        spacing_loss=last_spacing_loss,
                        sparsity_loss=last_spar_loss,
                    )
                else:
                    # Fallback: Basic console if realtime_display disabled
                    global_k_cfg = cfg["model"].get("global_k", 24)
# Display both step (forward passes) and optimizer_step for clarity
                    print(
                        f"Step {step:6d} (opt {optimizer_step:5d}) | Loss: {loss_gathered:.4f} | PPL: {ppl:7.2f} | "
                        f"LR: {lr_current:.2e} | GradNorm: {last_grad_norm:5.2f}"
                    )
                    print(
                        f"           | SeqLen: {current_seq_len:4d} | GW: {global_weight:.2f} | "
                        f"LM: {last_num_landmarks}→{global_k_cfg} | GPU: {mem_allocated:4.1f}GB | "
                        f"Tok/s: {tokens_per_sec:5.0f}"
                    )

                # Reset timing
                step_start_time = time.time()
                steps_since_log = 0

                # Cleanup CPU caches after use
                log_landmark_indices = None
                log_gate_values = None
            
            # === VALIDATION ===
            if accelerator.is_main_process and step % cfg["train"].get("eval_every", 1000) == 0:
                print("\n=== Validation ===")

                if hasattr(accelerator, 'wait_for_everyone'):
                    accelerator.wait_for_everyone()

                model.eval()

                import gc
                gc.collect()  # Force Python garbage collection

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Wait for CUDA operations to finish
                    mem_before = torch.cuda.memory_allocated() / 1e9
                    mem_reserved = torch.cuda.memory_reserved() / 1e9
                    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"  GPU Memory: {mem_before:.2f} GB allocated, {mem_reserved:.2f} GB reserved, {mem_total:.2f} GB total")

                # === RUN VALIDATION ===
                val_metrics = validate(
                    accelerator.unwrap_model(model),
                    val_loader,
                    pad_id,
                    device,
                    max_batches=10,  # 100 → 10 (10x plus rapide, ~80 exemples suffisants)
                )
                # Update realtime display with validation metrics
                if realtime_display:
                    realtime_display.print_validation(
                        step=step,
                        val_loss=val_metrics['loss'],
                        val_ppl=val_metrics['perplexity'],
                    )
                else:
                    print(
                        f"Val Loss: {val_metrics['loss']:.4f} | "
                        f"Val PPL: {val_metrics['perplexity']:.2f}"
                    )

                # Synthetic recall monitoring
                recall_k = (
                    max(1, cfg["model"].get("global_k", 24) // 2),
                    cfg["model"].get("global_k", 24),
                )
                recall_metrics = compute_long_dependency_recall(
                    accelerator.unwrap_model(model),
                    device,
                    cfg,
                    k_values=recall_k,
                    batch_size=cfg["train"].get("monitor_batch_size", 4),
                    num_batches=1,
                    global_weight=global_weight,
                )

                if recall_metrics:
                    print("Synthetic long-range recall:")
                    for name, value in recall_metrics.items():
                        print(f"  {name}: {value:.3f}")

                    if cfg["log"].get("wandb", False):
                        wandb.log({f"synthetic/{k}": v for k, v in recall_metrics.items()}, step=step)

                    if writer is not None:
                        for key, value in recall_metrics.items():
                            writer.add_scalar(f"synthetic/{key}", value, step)

                if cfg["log"].get("wandb", False):
                    wandb.log(
                        {
                            "val_loss": val_metrics["loss"],
                            "val_perplexity": val_metrics["perplexity"],
                        },
                        step=step,
                    )

                # TensorBoard validation
                if writer is not None:
                    writer.add_scalar("val/loss", val_metrics["loss"], step)
                    writer.add_scalar("val/perplexity", val_metrics["perplexity"], step)

                val_loss = val_metrics["loss"]

                # Best per current training sequence length
                current_seq_len = get_current_seq_len(step, cfg)
                prev_seq_best = best_loss_per_sequence.get(
                    current_seq_len, {"loss": float("inf"), "step": -1}
                )
                if val_loss < prev_seq_best["loss"]:
                    best_loss_per_sequence[current_seq_len] = {"loss": val_loss, "step": step}
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        seq_dir = os.path.join("best_sequence", f"seq_{current_seq_len}")
                        full_seq_dir = os.path.join(out_dir, seq_dir)
                        # Explicit creation of best_sequence/seq_xxx folder
                        os.makedirs(full_seq_dir, exist_ok=True)
                        if os.path.isdir(full_seq_dir):
                            shutil.rmtree(full_seq_dir)
                        print(
                            f"\n💾 New best for seq_len={current_seq_len}: {val_loss:.4f} (step {step})"
                        )
                        save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            out_dir,
                            step,
                            accelerator,
                            keep_last_n=None,
                            extra_state=build_extra_state(best_overall, best_loss_per_sequence),
                            custom_dir=seq_dir,
                        )
                        # Run generate on the best sequence
                        run_quick_generate(config_file_path, full_seq_dir)

                # Best overall across the full training run
                if val_loss < best_overall["loss"]:
                    best_overall = {"loss": val_loss, "step": step}
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        overall_dir = "best_overall"
                        full_overall_dir = os.path.join(out_dir, overall_dir)
                        if os.path.isdir(full_overall_dir):
                            shutil.rmtree(full_overall_dir)
                        print(
                            f"\n💾 New best overall validation loss: {val_loss:.4f} (step {step})"
                        )
                        save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            out_dir,
                            step,
                            accelerator,
                            keep_last_n=None,
                            extra_state=build_extra_state(best_overall, best_loss_per_sequence),
                            custom_dir=overall_dir,
                        )
                        # Run generate on the best overall
                        run_quick_generate(config_file_path, full_overall_dir)

                del val_metrics  # Explicitly free memory
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                model.train()
            
            # === CHECKPOINTING ===
            save_every = cfg["train"].get("save_every", 5000)
            is_save_step = step % save_every == 0
            is_main = accelerator.is_main_process

            debug_ckpt = cfg["train"].get("debug_checkpoints", False)
            if debug_ckpt and (step <= 10 or (step % 100 == 0)):
                print(f"\n[DEBUG Checkpoint] step={step}, save_every={save_every}, is_save_step={is_save_step}, is_main_process={is_main}")

            if is_main and is_save_step and step > 0:
                # Synchronize all GPUs before saving (prevents corruption)
                accelerator.wait_for_everyone()
                print(f"\n🔵 Attempting to save checkpoint step {step}...")
                try:
                    # Use keep_last_n for rotation
                    keep_last = cfg["train"].get("keep_last_checkpoints")
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        out_dir,
                        step,
                        accelerator,
                        keep_last_n=keep_last,
                        extra_state=build_extra_state(best_overall, best_loss_per_sequence),
                    )
                    print(f"✅ Checkpoint step {step} saved successfully!")
                    checkpoint_dir = os.path.join(out_dir, f"ckpt_{step}")
                    run_quick_generate(config_file_path, checkpoint_dir)
                except Exception as e:
                    print(f"❌ ERROR during checkpoint save step {step}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # === TRAINING TERMINATION CHECK ===
            # Stop if max steps reached
            if step >= total_steps:
                break
        
        if step >= total_steps:
            break
    
    # === FINAL CHECKPOINT ===
    # Synchronize before final checkpoint as well
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            out_dir,
            step,
            accelerator,
            extra_state=build_extra_state(best_overall, best_loss_per_sequence),
        )
        checkpoint_dir = os.path.join(out_dir, f"ckpt_{step}")
        run_quick_generate(config_file_path, checkpoint_dir)
        print("\n=== Training Complete ===")
        log_system_info("TRAINING COMPLETE")

        if system_monitor_stop is not None:
            system_monitor_stop.set()
            if system_monitor_thread is not None:
                system_monitor_thread.join(timeout=5.0)
    
    # === CLEANUP ===
    progress_bar.close()

    if cfg["log"].get("wandb", False):
        wandb.finish()

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
