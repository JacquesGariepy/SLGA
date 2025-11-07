#!/usr/bin/env python3
"""
Resume training avec nouveau dataset et options de reset optimizer.

Usage:
    python scripts/resume_with_new_dataset.py \\
        --checkpoint out_slga/ckpt_16000/model.pt \\
        --config config_new_dataset.yaml \\
        --reset-optimizer \\
        --start-step 0

Options de resume:
    1. --reset-optimizer: Crée un nouvel optimizer (RECOMMANDÉ pour nouveau dataset)
    2. --load-optimizer: Charge l'optimizer du checkpoint (risqué si dataset différent)
    3. --reduce-lr-factor: Réduit le LR d'un facteur (ex: 0.5 pour diviser par 2)
"""

import argparse
import os
import sys
import yaml
import torch
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def parse_args():
    parser = argparse.ArgumentParser(description="Resume training with new dataset")

    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (model.pt)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to NEW config.yaml with new dataset")

    # Optimizer options (mutually exclusive group)
    opt_group = parser.add_mutually_exclusive_group(required=True)
    opt_group.add_argument("--reset-optimizer", action="store_true",
                          help="Create fresh optimizer (RECOMMENDED for new dataset)")
    opt_group.add_argument("--load-optimizer", type=str,
                          help="Load optimizer from trainer_state.pt")

    # Training control
    parser.add_argument("--start-step", type=int, default=0,
                        help="Starting step (0 for reset, checkpoint step otherwise)")
    parser.add_argument("--reduce-lr-factor", type=float, default=1.0,
                        help="Multiply LR by this factor (e.g., 0.5 to halve)")

    # Optional overrides
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Override warmup steps (default: from config)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")

    # Safety
    parser.add_argument("--force", action="store_true",
                        help="Skip confirmation prompts")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load config.yaml"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def analyze_checkpoint(checkpoint_path: str) -> dict:
    """Analyze checkpoint and extract info"""
    print(f"\n📦 Analyzing checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")

    # Count parameters
    num_params = sum(p.numel() for p in state.values())

    # Get some sample weights
    first_param = next(iter(state.values()))
    param_mean = first_param.float().mean().item()
    param_std = first_param.float().std().item()

    info = {
        "num_tensors": len(state),
        "num_params": num_params,
        "param_mean": param_mean,
        "param_std": param_std,
    }

    print(f"  ✓ Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"  ✓ Tensors: {info['num_tensors']}")
    print(f"  ✓ Sanity check - mean: {param_mean:.6f}, std: {param_std:.6f}")

    return info, state


def analyze_optimizer_state(trainer_state_path: str) -> dict:
    """Analyze optimizer state"""
    print(f"\n🔧 Analyzing optimizer state: {trainer_state_path}")

    if not os.path.exists(trainer_state_path):
        raise FileNotFoundError(f"Trainer state not found: {trainer_state_path}")

    state = torch.load(trainer_state_path, map_location="cpu")

    info = {
        "step": state.get("step", 0),
        "lr": None,
    }

    if "optimizer" in state and "param_groups" in state["optimizer"]:
        lr = state["optimizer"]["param_groups"][0]["lr"]
        info["lr"] = lr
        print(f"  ✓ Current step: {info['step']}")
        print(f"  ✓ Current LR: {lr:.6e}")

        # Analyze momentum
        if "state" in state["optimizer"]:
            opt_state = state["optimizer"]["state"]
            num_params_tracked = len(opt_state)
            print(f"  ✓ Parameters tracked: {num_params_tracked}")

            # Sample first param
            if num_params_tracked > 0:
                first_key = list(opt_state.keys())[0]
                first_state = opt_state[first_key]
                if "exp_avg" in first_state:
                    momentum_norm = first_state["exp_avg"].norm().item()
                    print(f"  ✓ Sample momentum norm: {momentum_norm:.6f}")

    return info, state


def confirm_action(message: str, force: bool = False) -> bool:
    """Ask user confirmation"""
    if force:
        return True

    response = input(f"\n{message} (y/N): ").strip().lower()
    return response in ["y", "yes"]


def create_training_plan(args, config: dict, ckpt_info: dict, opt_info: dict = None):
    """Create and display training plan"""
    print("\n" + "=" * 80)
    print("📋 TRAINING RESUME PLAN")
    print("=" * 80)

    print("\n🗂️  Data Configuration:")
    print(f"  Old dataset: Wikipedia (assumed)")
    print(f"  New dataset: {config['data']['dataset']}")
    if 'subset' in config['data']:
        print(f"  Subset: {config['data']['subset']}")
    print(f"  Train split: {config['data']['split_train']}")
    print(f"  Val split: {config['data']['split_val']}")

    print("\n🧠 Model Configuration:")
    print(f"  Parameters: {ckpt_info['num_params']/1e6:.2f}M")
    print(f"  Embed dim: {config['model']['embed_dim']}")
    print(f"  Layers: {config['model']['n_layers']}")
    print(f"  Learned landmarks: {config['model']['learned_landmarks']}")

    print("\n⚙️  Optimizer Configuration:")
    if args.reset_optimizer:
        print("  Mode: ✅ RESET OPTIMIZER (fresh start)")
        print(f"  Initial LR: {config['train']['lr']:.6e}")
        warmup = args.warmup_steps or config['train']['warmup_steps']
        print(f"  Warmup steps: {warmup}")
        print(f"  Starting step: {args.start_step}")
    else:
        print("  Mode: ⚠️  LOAD OPTIMIZER (from checkpoint)")
        print(f"  Loaded LR: {opt_info['lr']:.6e}")
        if args.reduce_lr_factor != 1.0:
            new_lr = opt_info['lr'] * args.reduce_lr_factor
            print(f"  Adjusted LR: {new_lr:.6e} (×{args.reduce_lr_factor})")
        print(f"  Resume step: {opt_info['step']}")

    print("\n🎯 Training Schedule:")
    print(f"  Start step: {args.start_step}")
    print(f"  Max steps: {config['train']['max_steps']}")
    print(f"  Batch size: {config['train']['batch_size']}")
    print(f"  Accum steps: {config['train']['accum_steps']}")
    print(f"  Effective batch: {config['train']['batch_size'] * config['train']['accum_steps']}")

    print("\n💾 Output:")
    out_dir = args.output_dir or config['save']['out_dir']
    print(f"  Directory: {out_dir}")
    print(f"  Save every: {config['train']['save_every']} steps")
    print(f"  Eval every: {config['train']['eval_every']} steps")

    print("\n" + "=" * 80)


def prepare_resume_command(args, config: dict) -> str:
    """Prepare the actual training command"""

    # Base command
    cmd = "python scripts/train.py"

    # Model checkpoint
    cmd += f" --resume-model {args.checkpoint}"

    # Optimizer
    if args.load_optimizer:
        cmd += f" --resume-optimizer {args.load_optimizer}"
        if args.reduce_lr_factor != 1.0:
            cmd += f" --reduce-lr {args.reduce_lr_factor}"

    # Start step
    if args.start_step != 0:
        cmd += f" --start-step {args.start_step}"

    # Config
    cmd += f" --config {args.config}"

    # Output dir
    if args.output_dir:
        cmd += f" --output-dir {args.output_dir}"

    return cmd


def main():
    args = parse_args()

    print("=" * 80)
    print("=== SLGA Training Resume Tool ===")
    print("=" * 80)

    # Load config
    config = load_config(args.config)

    # Analyze model checkpoint
    ckpt_info, model_state = analyze_checkpoint(args.checkpoint)

    # Analyze optimizer if loading
    opt_info = None
    if args.load_optimizer:
        opt_info, opt_state = analyze_optimizer_state(args.load_optimizer)

    # Create training plan
    create_training_plan(args, config, ckpt_info, opt_info)

    # Warnings
    print("\n⚠️  IMPORTANT WARNINGS:")

    if args.reset_optimizer:
        print("  ✅ Reset optimizer: Good choice for new dataset!")
        print("  ✅ Fresh momentum/variance will adapt to new data distribution")
        print("  ⚠️  Initial ~500 steps may have higher loss (normal)")
    else:
        print("  ⚠️  Loading old optimizer: Risky for different dataset!")
        print("  ⚠️  Momentum/variance from old data may cause instability")
        print("  ⚠️  Consider --reset-optimizer instead")

    if config['model']['learned_landmarks']:
        print("  ⚠️  learned_landmarks=true: May be suboptimal for new dataset")
        print("  💡 Consider setting to 'false' initially")

    # Confirmation
    if not confirm_action("Proceed with this configuration?", args.force):
        print("\n❌ Aborted by user")
        return

    # Prepare modified config
    print("\n📝 Preparing training configuration...")

    # Create output directory
    out_dir = args.output_dir or config['save']['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    # Save modified config
    config_out = os.path.join(out_dir, "resume_config.yaml")
    with open(config_out, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"  ✓ Saved config to: {config_out}")

    # Save training plan
    plan_file = os.path.join(out_dir, "resume_plan.txt")
    with open(plan_file, "w") as f:
        f.write("SLGA Training Resume Plan\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Reset optimizer: {args.reset_optimizer}\n")
        f.write(f"Start step: {args.start_step}\n")
        f.write(f"New dataset: {config['data']['dataset']}\n")
        f.write("\nParameters:\n")
        f.write(f"  Model params: {ckpt_info['num_params']/1e6:.2f}M\n")
        if opt_info:
            f.write(f"  Current LR: {opt_info['lr']:.6e}\n")
            f.write(f"  Current step: {opt_info['step']}\n")
    print(f"  ✓ Saved plan to: {plan_file}")

    # Show command
    print("\n🚀 Ready to resume training!")
    print("\nNext steps:")
    print("  1. Review the configuration above")
    print("  2. Modify your train.py to support --resume-model and --resume-optimizer")
    print("  3. Run your training script with appropriate arguments")

    print("\n💡 Example training script modifications needed:")
    print("""
    # In train.py, add arguments:
    parser.add_argument("--resume-model", type=str, help="Model checkpoint to load")
    parser.add_argument("--resume-optimizer", type=str, help="Optimizer state to load")
    parser.add_argument("--start-step", type=int, default=0, help="Starting step")

    # Load model checkpoint
    if args.resume_model:
        model_state = torch.load(args.resume_model)
        model.load_state_dict(model_state)

    # Load optimizer (optional)
    if args.resume_optimizer:
        trainer_state = torch.load(args.resume_optimizer)
        optimizer.load_state_dict(trainer_state['optimizer'])
        scheduler.load_state_dict(trainer_state['scheduler'])
        start_step = trainer_state['step']
    else:
        start_step = args.start_step
    """)

    print("\n✅ Configuration prepared successfully!")
    print(f"   Resume files saved to: {out_dir}")


if __name__ == "__main__":
    main()
