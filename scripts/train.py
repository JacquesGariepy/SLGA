#!/usr/bin/env python3
"""
Main training script for SLGA (Sparse Landmark Global Attention)

This is a thin wrapper around src.training modules.
All training logic has been modularized into src/training/.
"""

from __future__ import annotations
import os
import sys
import yaml
import argparse
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from accelerate import Accelerator
from src.models import SLGATransformer, Config
from scripts.utils import set_seed
from src.training import (
    SLGATrainer,
    CheckpointCallback,
    MetricsCallback,
    DisplayCallback,
    create_optimizer,
    apply_lr_from_config,
    create_warmup_cosine_scheduler,
    build_loaders,
    log_system_info,
    SystemMonitor,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SLGA model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    config_file_path = os.path.abspath(args.config)

    # Reduce CUDA fragmentation
    if os.environ.get("PYTORCH_ALLOC_CONF") is None:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Load config
    with open(config_file_path) as f:
        cfg = yaml.safe_load(f)

    # Override max_steps if provided
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
        if args.max_steps < cfg["train"]["warmup_steps"]:
            original_warmup = cfg["train"]["warmup_steps"]
            adjusted_warmup = max(100, args.max_steps // 2)
            cfg["train"]["warmup_steps"] = adjusted_warmup
            print(f"⚠️  Warmup adjusted: {original_warmup} → {adjusted_warmup} (max_steps={args.max_steps})")

    # Setup
    set_seed(cfg["seed"])
    accelerator = Accelerator()

    # System monitoring
    log_cfg = cfg.get("log", {})
    system_monitor = None
    if accelerator.is_main_process:
        log_system_info("STARTUP")

        # Start periodic system monitoring if enabled
        system_info_interval = log_cfg.get("system_info_interval_seconds", 120)
        if system_info_interval and system_info_interval > 0:
            system_monitor = SystemMonitor(interval_seconds=system_info_interval)
            system_monitor.start()

    # TensorBoard writer
    writer = None
    if log_cfg.get("tensorboard", False):
        writer = SummaryWriter(log_dir=f"{cfg['save']['out_dir']}/tensorboard")

    # Data
    tokenizer, train_loader, val_loader = build_loaders(cfg)
    if tokenizer.vocab_size is not None:
        cfg["model"]["vocab_size"] = tokenizer.vocab_size

    # Model
    model_cfg = Config(**cfg["model"])
    if "grad_checkpointing" in cfg["train"]:
        model_cfg.grad_checkpointing = cfg["train"]["grad_checkpointing"]

    model = SLGATransformer(model_cfg)

    if accelerator.is_main_process:
        print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")
        print(f"Non-embedding params: {model.get_num_params(non_embedding=True) / 1e6:.2f}M")
        print(f"🔍 Learned landmarks: {model_cfg.learned_landmarks}")
        if model.landmark_selector is not None:
            print(f"   → Num landmarks: {model.landmark_selector.num_landmarks}")
            print(f"   → Temperature: {model.landmark_selector.temperature}")

    # Optimizer
    scorer_lr_multiplier = cfg["train"].get("scorer_lr_multiplier", 5.0)
    optimizer = create_optimizer(model, cfg, scorer_lr_multiplier)

    if accelerator.is_main_process and model.landmark_selector is not None:
        base_lr = cfg["train"]["lr"]
        print(f"🔧 Scorer LR: {base_lr * scorer_lr_multiplier:.2e} (×{scorer_lr_multiplier} vs model)")

    # Scheduler
    scheduler = create_warmup_cosine_scheduler(optimizer, cfg)

    # Prepare with Accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # Apply learning rates after wrapping
    apply_lr_from_config(optimizer, scheduler, cfg["train"]["lr"], scorer_lr_multiplier)

    # Create trainer
    trainer = SLGATrainer(
        config=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        accelerator=accelerator,
        tokenizer=tokenizer,
    )

    # Add callbacks
    checkpoint_callback = CheckpointCallback(
        config=cfg,
        config_path=config_file_path,
    )
    trainer.add_callback("checkpoint", checkpoint_callback)

    metrics_callback = MetricsCallback(config=cfg, writer=writer)
    trainer.add_callback("metrics", metrics_callback)

    display_callback = DisplayCallback(
        max_steps=cfg["train"]["max_steps"],
        detail_every=cfg["train"].get("log_every", 50),
        enabled=accelerator.is_main_process,
    )
    trainer.add_callback("display", display_callback)

    # Resume from checkpoint if requested
    if args.resume:
        checkpoints = [d for d in os.listdir(cfg["save"]["out_dir"]) if d.startswith("ckpt_")]
        if checkpoints:
            latest_ckpt = max(checkpoints, key=lambda x: int(x.split("_")[1]))
            checkpoint_path = os.path.join(cfg["save"]["out_dir"], latest_ckpt)

            if accelerator.is_main_process:
                print(f"\n{'='*80}")
                print(f"🔄 RESUMING FROM CHECKPOINT: {latest_ckpt}")
                print(f"{'='*80}\n")

            unwrapped_model = accelerator.unwrap_model(model)
            step, extra_state = checkpoint_callback.load_from_checkpoint(
                checkpoint_path,
                unwrapped_model,
                optimizer,
                scheduler,
                device=str(accelerator.device),
            )

            # Restore training state
            trainer.state.step = step
            trainer.state.optimizer_step = step // cfg["train"]["accum_steps"]

            if "best_overall" in extra_state:
                trainer.state.best_overall = extra_state["best_overall"]
            if "best_loss_per_sequence" in extra_state:
                trainer.state.best_loss_per_sequence = extra_state["best_loss_per_sequence"]

            # Re-apply learning rates after loading
            apply_lr_from_config(optimizer, scheduler, cfg["train"]["lr"], scorer_lr_multiplier)

            if accelerator.is_main_process:
                print(f"\n✅ Resumed from step {step}\n")
                log_system_info(f"RESUME STEP {step}")
        else:
            if accelerator.is_main_process:
                print(f"\n⚠️ --resume specified but no checkpoints found")
                print(f"Starting from scratch...\n")

    # Train
    try:
        trainer.train()
    finally:
        # Cleanup
        if accelerator.is_main_process:
            if system_monitor is not None:
                system_monitor.stop()

            metrics_callback.close()

            log_system_info("TRAINING COMPLETE")


if __name__ == "__main__":
    main()
