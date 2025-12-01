#!/usr/bin/env python
"""Merge LoRA weights into base model and save as a single checkpoint.

Usage:
    python scripts/merge_lora.py \
        --base-checkpoint out/out_slga/best_overall \
        --lora-weights out/lora_finetune/lora_best.pt \
        --output out/merged_model \
        --config config/config.yaml
"""

import os
import sys
import yaml
import torch
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import Config, LLMTransformer
from src.data import get_tokenizer
from src.lora import apply_lora_to_model, load_lora_weights, merge_lora_weights


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA into base model")
    parser.add_argument("--base-checkpoint", type=str, required=True,
                        help="Path to base model checkpoint")
    parser.add_argument("--lora-weights", type=str, required=True,
                        help="Path to LoRA weights (.pt file)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for merged model")
    parser.add_argument("--config", type=str, required=True,
                        help="Config file")

    # LoRA config (should match training)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-target", type=str, nargs='+',
                        default=["qkv_proj", "out_proj", "fc1", "fc2"])

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Check for finetune config format
    if 'lora' in cfg:
        args.lora_rank = cfg['lora'].get('rank', args.lora_rank)
        args.lora_alpha = cfg['lora'].get('alpha', args.lora_alpha)
        args.lora_target = cfg['lora'].get('target_modules', args.lora_target)

    # Load tokenizer to get vocab size
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(cfg["tokenizer"])
    cfg["model"]["vocab_size"] = len(tokenizer)

    # Load base model
    print("Loading base model...")
    model_cfg = Config(**cfg["model"])
    model = LLMTransformer(model_cfg)

    checkpoint_path = args.base_checkpoint
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "model.pt")

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded base weights from {args.base_checkpoint}")

    # Freeze base and apply LoRA
    for param in model.parameters():
        param.requires_grad = False

    print(f"Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    model = apply_lora_to_model(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        target_modules=args.lora_target,
        dropout=0.0,
    )

    # Load LoRA weights
    print(f"Loading LoRA weights from {args.lora_weights}...")
    load_lora_weights(model, args.lora_weights)

    # Merge LoRA into base model
    print("Merging LoRA weights into base model...")
    merge_lora_weights(model)

    # Save merged model
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "model.pt")

    # Get clean state dict (without LoRA parameters)
    state_dict = model.state_dict()
    # Remove LoRA-specific keys
    clean_state = {k: v for k, v in state_dict.items()
                   if not any(x in k for x in ['lora_A', 'lora_B', 'rank', 'alpha', 'scaling'])}

    torch.save(clean_state, output_path)
    print(f"\nSaved merged model to {output_path}")

    # Copy config
    config_out = os.path.join(args.output, "config.yaml")
    with open(config_out, 'w') as f:
        yaml.dump(cfg, f)
    print(f"Saved config to {config_out}")

    print(f"\nDone! You can now use:")
    print(f"  python scripts/generate.py --checkpoint {args.output} --prompt \"Your prompt\"")


if __name__ == "__main__":
    main()
