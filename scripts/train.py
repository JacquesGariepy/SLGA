# train.py
"""
Script d'entraînement principal pour SLGA

Inclut:
- AMP (Automatic Mixed Precision)
- Gradient accumulation
- Learning rate warmup + cosine decay
- Loss auxiliaires (diversity, sparsity)
- Warmup progressif du global attention
- Validation périodique
- Checkpointing
- Logging W&B optionnel
"""

from __future__ import annotations
import os
import sys
import math
import time
import yaml
import torch
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

from src.model import Config, LLMTransformer
from src.data import get_tokenizer, load_text_dataset, CollatorLocal, CollatorLocalGlobal
from src.landmarks import landmark_spacing_loss, landmark_sparsity_loss, landmark_diversity_loss
from scripts.utils import set_seed, save_checkpoint, load_checkpoint
from src.live_metrics import LiveMetricsDisplay
from src.realtime_display import RealtimeTrainingDisplay
from src.monitoring import compute_long_dependency_recall

from torch.utils.tensorboard import SummaryWriter

def get_current_seq_len(step: int, cfg: dict) -> int:
    """
    Calcule la longueur de séquence actuelle selon curriculum.

    Progression: seq_len_start -> seq_len_mid -> seq_len_final

    Args:
        step: Forward pass counter (NOT optimizer step)
        cfg: Config dict

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
    Calcule le poids de warmup pour attention globale.

    Permet d'activer progressivement le global pour éviter instabilités.

    Args:
        step: Forward pass counter (NOT optimizer step)
        cfg: Config dict

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
    """Emit a GPU/RAM/FS snapshot so it lands in the training log."""
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
    Cross-entropy loss avec shift pour causal LM.

    Args:
        logits: (B, L, V)
        labels: (B, L) - DÉJÀ shiftés par le collator, avec -100 pour pads
        pad_id: DEPRECATED - on ignore -100 maintenant

    Returns:
        loss: Scalaire
    """
    # IMPORTANT: Le collator a DÉJÀ shifté les labels!
    # labels[i] contient le token suivant pour input_ids[i]
    # Donc logits[i] doit prédire labels[i], PAS labels[i+1]!
    # On retire juste la dernière position (pas de target pour elle)
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
    Construit tokenizer et dataloaders.
    
    Returns:
        tokenizer, train_loader, val_loader
    """
    tokenizer = tokenizer or get_tokenizer(cfg["tokenizer"])
    
    # Datasets
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
        print(f"Warning: Could not load validation split: {e}")
        print("Using subset of training data for validation")
        ds_all = load_text_dataset(
            cfg["data"]["dataset"],
            cfg["data"].get("subset"),
            cfg["data"]["split_train"],
        )
        # Split manuel
        split_idx = int(len(ds_all) * 0.95)
        ds_train = ds_all.select(range(split_idx))
        ds_val = ds_all.select(range(split_idx, len(ds_all)))
    
    # Limiter taille si spécifié
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
    
    # Dataloaders
    train_loader = DataLoader(
        ds_train,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=collate_train,
        num_workers=cfg["data"].get("num_workers", 2),
        pin_memory=True,
    )
    
    # Collator validation : accepte texte brut ou jeux déjà tokenisés
    def collate_val_reduced(examples):
        """
        Collator validation robuste (gère texte brut OU jeux déjà tokenisés).
        - Tronque/pad à 512 (+1 pour le shift labels).
        - Ne dépend d'aucune clé spécifique: détecte 'input_ids' ou un champ texte.
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
                # tronque/pad à max_len_val+1
                if ids.numel() >= max_len_val + 1:
                    ids = ids[: max_len_val + 1]
                else:
                    pad = torch.full((max_len_val + 1 - ids.numel(),), pad_id, dtype=torch.long)
                    ids = torch.cat([ids, pad], dim=0)
                tens.append(ids)
            input_ids = torch.stack(tens, dim=0)  # (B, L+1)

            # Créer input_ids et labels AVANT masking
            input_ids_final = input_ids[:, :-1]  # (B, L)
            labels = input_ids[:, 1:].clone()  # (B, L)

            # Masquer les tokens de padding avec -100 après mise en forme
            pad_mask = (labels == pad_id)
            labels[pad_mask] = -100

            return {
                "input_ids": input_ids_final,  # (B, L)
                "labels": labels,  # (B, L) with -100 for pads
                "cache_global_ids": None,
            }

        # 1) Cas pré-tokenisé: chaque exemple contient 'input_ids'
        ex0 = examples[0]
        if isinstance(ex0, dict) and "input_ids" in ex0:
            return _stack_input_ids([ex["input_ids"] for ex in examples])

        # 2) Cas texte brut: chercher une clé texte plausible
        text_key = None
        if isinstance(ex0, dict):
            for k in ("text", "content", "document", "raw", "prompt"):
                if k in ex0 and isinstance(ex0[k], str):
                    text_key = k
                    break
        # 2b) Exemple est directement une string
        if text_key is None and isinstance(ex0, str):
            texts = list(examples)
        elif text_key is not None:
            texts = [ex[text_key] for ex in examples]
        else:
            # Dernier recours: prendre le premier champ string trouvé
            if isinstance(ex0, dict):
                candidates = [k for k, v in ex0.items() if isinstance(v, str)]
                if candidates:
                    texts = [ex[candidates[0]] for ex in examples]
                else:
                    raise KeyError(f"Impossible de trouver un champ texte. Clés: {list(ex0.keys())}")
            else:
                raise KeyError(f"Format d'exemple non supporté: {type(ex0)}")

        # Tokenization texte → input_ids (L+1)
        encoded = tokenizer(
            texts,
            max_length=max_len_val + 1,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]

        # Créer input_ids et labels AVANT masking
        input_ids_final = input_ids[:, :-1]  # (B, L)
        labels = input_ids[:, 1:].clone()  # (B, L)

        # Masquer les tokens de padding avec -100
        pad_mask = (labels == pad_id)
        labels[pad_mask] = -100

        return {
            "input_ids": input_ids_final,
            "labels": labels,
            "cache_global_ids": None,
        }

    # 🔍 DEBUG: Afficher le format du premier exemple de validation
    if len(ds_val) > 0:
        ex0 = ds_val[0]
        if isinstance(ex0, dict):
            print(f"🔍 DEBUG Dataset format: keys={list(ex0.keys())}, types={[(k, type(v).__name__) for k, v in list(ex0.items())[:5]]}")
        else:
            print(f"🔍 DEBUG Dataset format: type={type(ex0).__name__}")

    # Validation avec batch_size réduit pour éviter OOM
    val_batch_size = max(1, cfg["train"]["batch_size"] // 2)  # Divisé par 2 pour la mémoire
    print(f"Validation config: batch_size={val_batch_size} (train: {cfg['train']['batch_size']}), seq_len=512 (train: up to 2048)")

    # 🔧 ASTUCE DEBUG: num_workers=0 temporairement pour voir stacktrace complète
    # Si validation fonctionne, remettre num_workers=2 pour performance
    val_loader = DataLoader(
        ds_val,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_val_reduced,  # 🔧 Collator robuste auto-détection
        num_workers=0,  # 🔧 TEMPORAIRE: 0 pour debug, remettre 2 une fois stable
        pin_memory=False,  # 🔧 TEMPORAIRE: désactivé pour debug
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
    Évalue le modèle sur validation set.

    Returns:
        metrics: Dict avec 'loss' et 'perplexity'
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    # Progress bar pour validation
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

            # Vérifier les labels avant le forward pour détecter les valeurs hors vocabulaire
            if hasattr(model, 'cfg') and hasattr(model.cfg, 'vocab_size'):
                vocab_size = model.cfg.vocab_size
            elif hasattr(model, 'lm_head') and hasattr(model.lm_head, 'out_features'):
                vocab_size = model.lm_head.out_features
            else:
                vocab_size = 50257  # Fallback to GPT-2 default

            # Garder ces opérations hors du graphe pour éviter les rétentions CUDA
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

            # Forward
            logits = model(input_ids, cache_global_ids=cache_ids)

            # Loss
            loss = cross_entropy_shifted(logits, labels, pad_id)

            # Compter les tokens valides (ignore -100)
            num_tokens = (labels != -100).sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            num_batches += 1

            # Libérer explicitement les tenseurs pour éviter l'accumulation mémoire
            del input_ids, labels, cache_ids, logits, loss, invalid_mask

            if i % 5 == 0 and torch.cuda.is_available():  # Tous les 5 batches
                torch.cuda.empty_cache()

            # Progress indicator
            if (i + 1) % 5 == 0 or (i + 1) == max_b:
                print(f"\r  Validation: {i+1}/{max_b} batches...", end='', flush=True)

    print()  # New line après progress
    
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 10))  # Cap pour stabilité
    
    return {"loss": avg_loss, "perplexity": perplexity}


def main():
    """Boucle d'entraînement principale"""

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Train SLGA model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    # Réduire la fragmentation CUDA (nouvelle variable recommandée). Garder l'ancienne pour compat.
    if os.environ.get("PYTORCH_ALLOC_CONF") is None:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Charger config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_cfg = cfg.get("log", {})

    # Override max_steps if provided
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
        # Ajuster le warmup proportionnellement si max_steps est réduit
        if args.max_steps < cfg["train"]["warmup_steps"]:
            original_warmup = cfg["train"]["warmup_steps"]
            adjusted_warmup = max(100, args.max_steps // 2)  # 50% du total ou min 100
            cfg["train"]["warmup_steps"] = adjusted_warmup
            print(f"⚠️  Warmup ajusté: {original_warmup} → {adjusted_warmup} (max_steps={args.max_steps})")

    # TensorBoard writer (créé après lecture de la config)
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
    
    # W&B logging (optionnel)
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

    # Modèle
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
    scorer_lr_multiplier = cfg["train"].get("scorer_lr_multiplier", 5.0)  # 5× par défaut
    base_lr = cfg["train"]["lr"]

    # Séparer paramètres: scorer vs reste du modèle
    scorer_params = []
    other_params = []

    if model.landmark_selector is not None:
        scorer_params = list(model.landmark_selector.scorer.parameters())
        # Tous les autres paramètres
        scorer_param_ids = {id(p) for p in scorer_params}
        other_params = [p for p in model.parameters() if id(p) not in scorer_param_ids]
    else:
        other_params = list(model.parameters())

    # Créer param groups
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
            "lr": base_lr * scorer_lr_multiplier,  # LR plus élevé pour scorer
            "name": "scorer",
        })
        if accelerator.is_main_process:
            print(f"🔧 Scorer LR: {base_lr * scorer_lr_multiplier:.2e} (×{scorer_lr_multiplier} vs modèle)")

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
    #    - Increments on EVERY forward pass
    #    - Used for: logging, checkpointing, curriculum, warmup schedules
    #    - With accum_steps=4: 0, 1, 2, 3, 4, 5, ..., 100000
    #
    # 2. `optimizer_step` = Optimizer update counter (0 → max_steps // accum_steps)
    #    - Only increments when optimizer.step() is called
    #    - Used for: scheduler.step() (LR schedule)
    #    - With accum_steps=4: 0, 0, 0, 0, 1, 1, 1, 1, 2, ..., 25000
    #
    # Config file interpretation:
    #   - max_steps: 100000      → 100K forward passes = 25K optimizer updates (with accum=4)
    #   - warmup_steps: 2000     → 2000 forward passes = 500 optimizer updates (with accum=4)
    #
    # Scheduler must use OPTIMIZER steps (not forward passes):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps // accum_steps,    # 2000 → 500 (with accum=4)
        num_training_steps=total_steps // accum_steps,    # 100000 → 25000 (with accum=4)
    )
    
    # Prepare avec Accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
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

            # Unwrap model for loading (si wrapped par Accelerator)
            unwrapped_model = accelerator.unwrap_model(model)

            # Load checkpoint
            step = load_checkpoint(
                checkpoint_path,
                unwrapped_model,
                optimizer,
                scheduler,
                device=str(device)
            )
            # Calculate optimizer_step from loaded step
            optimizer_step = step // accum_steps

            if accelerator.is_main_process:
                print(f"\n✅ Resumed from step {step} (optimizer_step {optimizer_step})\n")
                log_system_info(f"RESUME STEP {step}")
        else:
            if accelerator.is_main_process:
                print(f"\n⚠️ --resume specified but no checkpoints found in {out_dir}")
                print(f"Starting from scratch...\n")

    model.train()

    # Tracking pour métriques de performance
    step_start_time = time.time()
    steps_since_log = 0

    # Variables persistantes pour métriques (éviter 0.00 dans logs)
    last_grad_norm = 0.0
    last_num_landmarks = 0
    last_spacing_loss = 0.0  # remplace l'ancienne last_div_loss
    last_spar_loss = 0.0
    last_mass_ratio = 0.0  # Mass concentration dans top-G landmarks

    # Infinite loop sur dataloader
    epoch = 0

    # Real-time display (mise à jour chaque step)
    if accelerator.is_main_process:
        realtime_display = RealtimeTrainingDisplay(
            max_steps=total_steps,
            detail_every=cfg["train"].get("log_every", 50),
            width=100,
        )
    else:
        realtime_display = None

    # Disable tqdm (on utilise realtime_display)
    progress_bar = tqdm(
        total=total_steps,
        desc="Training",
        disable=True,  # Toujours désactivé si realtime_display
    )

    # Petits caches CPU détachés pour logging, évitent de retenir le graphe
    log_landmark_indices = None
    log_gate_values = None

    while step < total_steps:
        epoch += 1
        
        for batch in train_loader:
            # Curriculum seq_len (mettre à jour collator si besoin)
            current_seq_len = get_current_seq_len(step, cfg)
            
            # Global warmup weight
            global_weight = get_global_warmup_weight(step, cfg)
            
            # Charger depuis CPU, puis appliquer la troncature AVANT transfert GPU
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            cache_ids = batch.get("cache_global_ids")

            # Tronquer sur CPU pour réduire l'empreinte GPU
            if input_ids.size(1) > current_seq_len:
                input_ids = input_ids[:, :current_seq_len]
                labels = labels[:, :current_seq_len]

                # Adapter les landmarks heuristiques à la nouvelle longueur
                if cache_ids is not None and not model.cfg.learned_landmarks:
                    B = cache_ids.size(0)
                    G = cache_ids.size(1)
                    cache_ids = torch.linspace(
                        0, current_seq_len - 1, G, device=input_ids.device if input_ids.is_cuda else torch.device("cpu")
                    ).long().unsqueeze(0).expand(B, -1)
                elif cache_ids is not None and model.cfg.learned_landmarks:
                    cache_ids = torch.clamp(cache_ids, 0, current_seq_len - 1)

            # Transfert GPU APRÈS troncature
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            cache_ids = cache_ids.to(device) if cache_ids is not None else None
            
            # Initialiser lambda_spar avant le bloc autocast pour toute configuration
            lambda_spar = cfg["train"].get("lambda_sparsity", 0.0)

            # Autocast uniquement lorsque CUDA est disponible
            use_autocast = amp_enabled and device.type == "cuda"
            if use_autocast:
                autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
            else:
                from contextlib import nullcontext
                autocast_ctx = nullcontext()

            with autocast_ctx:
                logits, aux = model(input_ids, cache_global_ids=cache_ids, return_aux=True, global_weight=global_weight)
                loss_ce = cross_entropy_shifted(logits, labels, pad_id)
                loss = loss_ce / accum_steps

                spacing_loss_val = 0.0
                spar_loss_val = 0.0
                num_landmarks_selected = 0
                gate_entropy_val: Optional[float] = None
                gate_local_mean: Optional[float] = None
                gate_local_majority: Optional[float] = None

                landmark_indices = aux.get("landmark_indices", None)
                landmark_scores = aux.get("landmark_scores", None)

                if landmark_indices is not None and landmark_indices.numel() > 0:
                    num_landmarks_selected = landmark_indices.size(1)

                if landmark_indices is not None and landmark_scores is not None:
                    seq_len = input_ids.size(1)

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

                    if lambda_spar > 0 and num_landmarks_selected > 0:
                        spar_loss = landmark_sparsity_loss(
                            selection_scores=landmark_scores,
                            num_landmarks=num_landmarks_selected,  # NEW: paramètre adaptatif
                            lambda_reg=lambda_spar
                        )
                        spar_loss_val = spar_loss.item()
                        loss = loss + spar_loss / accum_steps

                    # Legacy diversity loss (optionnel, pour backward compatibility)
                    lambda_div = cfg["train"].get("lambda_diversity", 0.0)
                    if lambda_div > 0:
                        # NOTE: Utiliser lambda_spacing à la place (recommandé)
                        div_loss = landmark_diversity_loss(landmark_scores, lambda_div)
                        loss = loss + div_loss / accum_steps

                # Calculer mass_in_top_g pour monitoring (même si loss constante)
                mass_in_top_g = 0.0
                if lambda_spar > 0 and landmark_scores is not None and num_landmarks_selected > 0:
                    # Recalculer pour logging (peu coûteux)
                    import torch.nn.functional as F
                    probs = F.softmax(landmark_scores, dim=-1)
                    _, top_g_idx = torch.topk(landmark_scores, k=num_landmarks_selected, dim=-1)
                    top_g_probs = torch.gather(probs, dim=1, index=top_g_idx)
                    mass_in_top_g = top_g_probs.sum(dim=-1).mean().item()

                # Sauvegarder pour logging (valeurs persistantes)
                last_spacing_loss = spacing_loss_val  # NEW: remplace div_loss
                last_spar_loss = spar_loss_val
                last_num_landmarks = num_landmarks_selected
                last_mass_ratio = mass_in_top_g  # Nouvelle métrique

            # Détacher les gros tenseurs d'aux et conserver uniquement ce qui est utile pour logs
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

            # Calculer métriques de gate si disponibles
            if log_gate_values is not None:
                gate_tensor = log_gate_values.float()  # (layers, B, H, L)
                gate_flat = gate_tensor.reshape(-1)
                if gate_flat.numel() > 0:
                    gate_clamped = torch.clamp(gate_flat, min=1e-4, max=1 - 1e-4)
                    gate_entropy = -(gate_clamped * torch.log(gate_clamped) + (1 - gate_clamped) * torch.log(1 - gate_clamped))
                    gate_entropy_val = gate_entropy.mean().item()
                    gate_local_mean = gate_flat.mean().item()
                    gate_local_majority = (gate_flat > 0.5).float().mean().item()

            # Libérer références pour ne pas retenir le graphe pendant backward
            landmark_scores = None
            landmark_indices = None
            aux = None

# Détection NaN/Inf avant backward
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n{'='*80}")
                print(f"❌ DIVERGENCE DÉTECTÉE au step {step}!")
                print(f"{'='*80}")
                print(f"   Loss totale: {loss.item()}")
                print(f"   Loss CE: {loss_ce.item()}")
                if spacing_loss_val > 0:
                    print(f"   Spacing loss: {spacing_loss_val:.6f}")
                if spar_loss_val > 0:
                    print(f"   Sparsity loss: {spar_loss_val:.6f}")
                print(f"   Learning rate: {scheduler.get_last_lr()[0]:.2e}")
                print(f"   Gradient norm (last): {last_grad_norm:.4f}")

                # Sauver checkpoint de debug
                if accelerator.is_main_process:
                    accelerator.wait_for_everyone()
                    debug_dir = f"ckpt_{step}_diverged"
                    save_checkpoint(model, optimizer, scheduler, out_dir, step, accelerator)
                    print(f"   💾 Debug checkpoint sauvegardé: {out_dir}/{debug_dir}")

                raise ValueError(f"Training diverged with NaN/Inf at step {step}")

            # Backward
            accelerator.backward(loss)
            
            # Gradient accumulation
            if (step + 1) % accum_steps == 0:
                # Calculate gradient norm BEFORE clipping (pour monitoring)
                grad_norm = 0.0
                if accelerator.is_main_process:
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            grad_norm += param_norm.item() ** 2
                    grad_norm = grad_norm ** 0.5
                    last_grad_norm = grad_norm  # Sauvegarder pour logging

                    # ✅ AJOUT #3: Gradient flow monitoring (tous les 500 steps)
                    if step % 500 == 0:
                        grad_norms_per_layer = {}

# Utiliser named_parameters() pour éviter crash
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                layer_norm = param.grad.data.norm(2).item()
                                grad_norms_per_layer[name] = layer_norm

                        # Logger les plus gros gradients
                        top_grads = sorted(grad_norms_per_layer.items(), key=lambda x: x[1], reverse=True)[:5]
                        print(f"\n  Top gradient norms:")
                        for name, norm in top_grads:
                            print(f"    {name}: {norm:.4f}")

                # Gradient clipping
                if grad_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # Incrémente le compteur d'étapes optimiseur après la mise à jour
                optimizer_step += 1

            step += 1
            steps_since_log += 1
            progress_bar.update(1)

            # Real-time update (CHAQUE step)
            if realtime_display and accelerator.is_main_process:
                # Calculer métriques courantes
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1e9
                    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                else:
                    mem_allocated = 0
                    mem_total = 0

                # Calculer active landmarks (nombre de landmarks réellement sélectionnés)
                active_lm = 0
                if landmark_indices is not None:
                    # Les landmarks sélectionnés sont ceux dans landmark_indices
                    active_lm = landmark_indices.numel() // landmark_indices.size(0)  # G landmarks par batch
                    # Ou simplement utiliser num_landmarks_selected qui est déjà calculé
                    active_lm = num_landmarks_selected

                # Update ligne live
                realtime_display.update_live(
                    step=step,
                    loss=loss_ce.item() if step % accum_steps == 0 else None,  # Seulement après accum
                    ppl=math.exp(min(loss_ce.item(), 10)) if step % accum_steps == 0 else None,
                    lr=scheduler.get_last_lr()[0] if step % accum_steps == 0 else None,
                    throughput=None,  # Calculé dans section détaillée
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

                # Calculer couverture des landmarks (portion de la séquence à distance < window)
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
                    "grad_norm": last_grad_norm,  # Utiliser valeur persistante
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

                # TensorBoard - Métriques de base
                if writer is not None:
                    writer.add_scalar("train/loss", loss_gathered, step)
                    writer.add_scalar("train/perplexity", ppl, step)
                    writer.add_scalar("train/learning_rate", lr_current, step)
                    writer.add_scalar("train/seq_len", current_seq_len, step)
                    writer.add_scalar("train/global_weight", global_weight, step)

                    # Gradient norm (valeur persistante)
                    writer.add_scalar("train/grad_norm", last_grad_norm, step)

                    if last_spacing_loss > 0:
                        writer.add_scalar("train/loss_spacing", last_spacing_loss, step)  
                    if last_spar_loss > 0:
                        writer.add_scalar("train/loss_sparsity", last_spar_loss, step)
                        # 📊 Logging mass_in_top_g pour suivre évolution scorer

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
                        # Même si loss constante au début, mass augmente quand scorer apprend
                        if last_mass_ratio > 0:
                            writer.add_scalar("landmarks/mass_in_top_g", last_mass_ratio, step)

                    # Landmark statistics (valeur persistante)
                    if last_num_landmarks > 0:
                        writer.add_scalar("landmarks/num_selected", last_num_landmarks, step)

                    # ✅ AJOUT #2: Landmark spacing metrics
                    if log_landmark_indices is not None:
                        landmark_indices_detached = log_landmark_indices  # (B, G) déjà CPU/detach

                        # Calculer spacing entre landmarks
                        sorted_idx = torch.sort(landmark_indices_detached, dim=-1)[0]
                        gaps = sorted_idx[:, 1:] - sorted_idx[:, :-1]

# Utiliser .item() pour couper le graph
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

                # Real-time Display: Affichage détaillé
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
                    # Fallback: Console basique si realtime_display désactivé
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

                # Nettoyage des caches CPU après usage
                log_landmark_indices = None
                log_gate_values = None
            
            # Validation
            if accelerator.is_main_process and step % cfg["train"].get("eval_every", 1000) == 0:
                print("\n=== Validation ===")

                if hasattr(accelerator, 'wait_for_everyone'):
                    accelerator.wait_for_everyone()

                model.eval()

                import gc
                gc.collect()  # Force Python garbage collection

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Attendre fin des opérations CUDA
                    mem_before = torch.cuda.memory_allocated() / 1e9
                    mem_reserved = torch.cuda.memory_reserved() / 1e9
                    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"  GPU Memory: {mem_before:.2f} GB allocated, {mem_reserved:.2f} GB reserved, {mem_total:.2f} GB total")

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

                del val_metrics  # Libérer explicitement
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                model.train()
            
            # Checkpoint
            save_every = cfg["train"].get("save_every", 5000)
            is_save_step = step % save_every == 0
            is_main = accelerator.is_main_process

            debug_ckpt = cfg["train"].get("debug_checkpoints", False)
            if debug_ckpt and (step <= 10 or (step % 100 == 0)):
                print(f"\n[DEBUG Checkpoint] step={step}, save_every={save_every}, is_save_step={is_save_step}, is_main_process={is_main}")

            if is_main and is_save_step and step > 0:
# Synchroniser tous les GPUs avant sauvegarde (évite corruption)
                accelerator.wait_for_everyone()
                print(f"\n🔵 Tentative de sauvegarde checkpoint step {step}...")
                try:
# Utiliser keep_last_n pour rotation
                    keep_last = cfg["train"].get("keep_last_checkpoints")
                    save_checkpoint(model, optimizer, scheduler, out_dir, step, accelerator, keep_last_n=keep_last)
                    print(f"✅ Checkpoint step {step} sauvegardé avec succès!")
                except Exception as e:
                    print(f"❌ ERREUR lors de la sauvegarde checkpoint step {step}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Stop si max steps atteint
            if step >= total_steps:
                break
        
        if step >= total_steps:
            break
    
    # Final checkpoint
# Synchroniser avant checkpoint final aussi
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(model, optimizer, scheduler, out_dir, step, accelerator)
        print("\n=== Training Complete ===")
        log_system_info("TRAINING COMPLETE")

        if system_monitor_stop is not None:
            system_monitor_stop.set()
            if system_monitor_thread is not None:
                system_monitor_thread.join(timeout=5.0)
    
    progress_bar.close()

    if cfg["log"].get("wandb", False):
        wandb.finish()

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
