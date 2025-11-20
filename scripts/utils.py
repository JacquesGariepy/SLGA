# utils.py
"""
Utilitaires pour SLGA: seed, checkpointing, helpers
"""

import os
import random
import numpy as np
import torch
from typing import Any


def set_seed(seed: int):
    """
    Fixe la seed pour reproductibilité.
    
    Args:
        seed: Seed à utiliser
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Comportement déterministe (peut ralentir)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    out_dir: str,
    step: int,
    accelerator: Any,
    keep_last_n: int = None,
    extra_state: dict = None,
    custom_dir: str = None,
):
    """
    Sauvegarde un checkpoint du modèle et de l'état d'entraînement.

    Args:
        model: Modèle à sauvegarder
        optimizer: Optimizer
        scheduler: LR scheduler
        out_dir: Répertoire de sortie
        step: Numéro d'étape actuel
        accelerator: Accelerator (pour unwrap)
        keep_last_n: Si spécifié, garde seulement les N derniers checkpoints (rotation)
        extra_state: Dictionnaire d'état supplémentaire à sauvegarder (optionnel)
        custom_dir: Nom de répertoire personnalisé au lieu de ckpt_{step} (optionnel)
    """
    # Use custom directory name if provided, otherwise use default
    dir_name = custom_dir if custom_dir else f"ckpt_{step}"
    checkpoint_dir = os.path.join(out_dir, dir_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Unwrap model (si DDP/FSDP)
    unwrapped_model = accelerator.unwrap_model(model)

    # Sauvegarder modèle (state_dict)
    torch.save(
        unwrapped_model.state_dict(),
        os.path.join(checkpoint_dir, "model.pt"),
    )

    # Sauvegarder config du modèle (pour reconstruction)
    if hasattr(unwrapped_model, 'config'):
        import json
        config_dict = vars(unwrapped_model.config)
        with open(os.path.join(checkpoint_dir, "model_config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)

    # Sauvegarder état d'entraînement
    training_state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
    }

    # Add extra state if provided
    if extra_state is not None:
        training_state.update(extra_state)

    torch.save(
        training_state,
        os.path.join(checkpoint_dir, "trainer_state.pt"),
    )

    print(f"\n{'='*60}")
    print(f"✓ CHECKPOINT SAVED at step {step}")
    print(f"  Location: {checkpoint_dir}")
    print(f"  Files: model.pt, trainer_state.pt")
    print(f"{'='*60}\n")

    # 🔧 FIX: Rotation des checkpoints (garde seulement les N derniers)
    if keep_last_n is not None:
        import shutil
        all_ckpts = sorted([d for d in os.listdir(out_dir) if d.startswith("ckpt_")],
                          key=lambda x: int(x.split("_")[1]))
        if len(all_ckpts) > keep_last_n:
            for old in all_ckpts[:-keep_last_n]:
                old_path = os.path.join(out_dir, old)
                shutil.rmtree(old_path)
                print(f"  🗑️ Supprimé ancien checkpoint: {old}")

def load_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: Any = None,
    device: str = "cuda",
    extra_state: dict = None,
) -> int:
    """
    Charge un checkpoint.

    Args:
        checkpoint_dir: Répertoire du checkpoint
        model: Modèle à charger
        optimizer: Optimizer (optionnel)
        scheduler: Scheduler (optionnel)
        device: Device
        extra_state: Dictionnaire pour stocker l'état supplémentaire chargé (optionnel, modifié in-place)

    Returns:
        step: Numéro d'étape du checkpoint
    """
    # Charger modèle (utilise le bon nom de fichier!)
    model_path = os.path.join(checkpoint_dir, "model.pt")
    if not os.path.exists(model_path):
        # Fallback: essayer pytorch_model.bin (format HF)
        model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[✓] Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Could not find model.pt or pytorch_model.bin in {checkpoint_dir}")

    # Charger état d'entraînement
    step = 0
    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.pt")

    if os.path.exists(trainer_state_path):
        trainer_state = torch.load(trainer_state_path, map_location=device)

        step = trainer_state.get("step", 0)

        if optimizer and "optimizer" in trainer_state:
            optimizer.load_state_dict(trainer_state["optimizer"])
            print(f"[✓] Optimizer state loaded")

        if scheduler and "scheduler" in trainer_state and trainer_state["scheduler"]:
            scheduler.load_state_dict(trainer_state["scheduler"])
            print(f"[✓] Scheduler state loaded")

        # Load extra state if requested
        if extra_state is not None:
            # Copy all extra keys (not optimizer, scheduler, step) to extra_state dict
            reserved_keys = {"optimizer", "scheduler", "step"}
            for key, value in trainer_state.items():
                if key not in reserved_keys:
                    extra_state[key] = value

            if extra_state:
                print(f"[✓] Extra state loaded: {list(extra_state.keys())}")

    print(f"[✓] Checkpoint loaded (step {step})")

    return step


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """
    Compte le nombre de paramètres d'un modèle.
    
    Args:
        model: Modèle PyTorch
        trainable_only: Si True, compte seulement les paramètres entraînables
    
    Returns:
        num_params: Nombre de paramètres
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_memory_usage() -> dict:
    """
    Retourne l'usage mémoire GPU actuel.
    
    Returns:
        memory_info: Dict avec 'allocated', 'reserved', 'free' (en GB)
    """
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "free": 0}
    
    allocated = torch.cuda.memory_allocated() / 1e9  # GB
    reserved = torch.cuda.memory_reserved() / 1e9  # GB

    # Mémoire totale du GPU
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 1e9  # GB
    # 🔧 FIX: Utiliser reserved, pas allocated pour calculer free
    free = total - reserved  # ✅ CORRECT: free = total - reserved
    
    return {
        "allocated": allocated,
        "reserved": reserved,
        "free": free,
        "total": total,
    }


def format_time(seconds: float) -> str:
    """
    Formate un temps en secondes en format lisible.
    
    Args:
        seconds: Temps en secondes
    
    Returns:
        formatted: String formaté (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def estimate_training_time(
    total_steps: int,
    current_step: int,
    elapsed_time: float,
) -> dict:
    """
    Estime le temps restant d'entraînement.
    
    Args:
        total_steps: Nombre total d'étapes
        current_step: Étape actuelle
        elapsed_time: Temps écoulé (secondes)
    
    Returns:
        estimates: Dict avec 'remaining_time', 'eta', 'steps_per_sec'
    """
    if current_step == 0:
        return {
            "remaining_time": 0,
            "eta": "Unknown",
            "steps_per_sec": 0,
        }
    
    steps_per_sec = current_step / elapsed_time
    remaining_steps = total_steps - current_step
    remaining_time = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
    
    return {
        "remaining_time": remaining_time,
        "eta": format_time(remaining_time),
        "steps_per_sec": steps_per_sec,
    }


class AverageMeter:
    """Calcule et stocke la moyenne et la valeur actuelle"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f} (current: {self.val:.4f})"


def print_model_summary(model: torch.nn.Module, input_shape: tuple = None):
    """
    Affiche un résumé du modèle.
    
    Args:
        model: Modèle PyTorch
        input_shape: Forme de l'entrée (optionnel, pour calculer FLOPs)
    """
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    
    # Paramètres
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Breakdown par module
    print("\nModule breakdown:")
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    print("=" * 80)


def load_latest_checkpoint(out_dir: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, scheduler: Any = None, device: str = "cuda") -> int:
    """
    Charge le dernier checkpoint disponible.

    Args:
        out_dir: Répertoire des checkpoints
        model: Modèle à charger
        optimizer: Optimizer (optionnel)
        scheduler: Scheduler (optionnel)
        device: Device

    Returns:
        step: Numéro d'étape du checkpoint (0 si aucun checkpoint trouvé)
    """
    checkpoints = [d for d in os.listdir(out_dir) if d.startswith("ckpt_")]
    if not checkpoints:
        print(f"No checkpoints found in {out_dir}")
        return 0

    latest = max(checkpoints, key=lambda x: int(x.split("_")[1]))
    latest_path = os.path.join(out_dir, latest)

    print(f"Loading latest checkpoint: {latest}")
    return load_checkpoint(latest_path, model, optimizer, scheduler, device)


if __name__ == "__main__":
    # Tests
    print("=== Utils Tests ===\n")
    
    # Test seed
    set_seed(42)
    print(f"Random number (seed 42): {random.random():.6f}")
    
    # Test time formatting
    print(f"Time format test: {format_time(7265)}")  # 2h 1m 5s
    
    # Test memory
    if torch.cuda.is_available():
        mem = get_memory_usage()
        print(f"GPU Memory: {mem['allocated']:.2f}GB allocated, {mem['free']:.2f}GB free")
    
    # Test average meter
    meter = AverageMeter("loss")
    for i in range(5):
        meter.update(1.0 / (i + 1))
    print(f"Average meter: {meter}")
    
    print("\n✓ All tests passed!")
