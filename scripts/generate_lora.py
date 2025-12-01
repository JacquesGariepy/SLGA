#!/usr/bin/env python
"""Generate text using a LoRA fine-tuned model."""

import os
import sys
import yaml
import torch
import argparse
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import Config, LLMTransformer
from src.data import get_tokenizer
from src.lora import apply_lora_to_model, load_lora_weights


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


def main():
    parser = argparse.ArgumentParser(description="Generate text with LoRA model")
    parser.add_argument("--base-checkpoint", type=str, required=True,
                        help="Path to base model checkpoint")
    parser.add_argument("--lora-weights", type=str, required=True,
                        help="Path to LoRA weights (.pt file)")
    parser.add_argument("--config", type=str, default=None,
                        help="Config file (optional)")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt to generate from")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--device", type=str, default="cuda")

    # LoRA config
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-target", type=str, nargs='+',
                        default=["qkv_proj", "out_proj", "fc1", "fc2"])

    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        # Handle nested config from finetune
        if 'base_config' in cfg:
            base_cfg = cfg['base_config']
            lora_cfg = cfg.get('lora', {})
            args.lora_rank = lora_cfg.get('rank', args.lora_rank)
            args.lora_alpha = lora_cfg.get('alpha', args.lora_alpha)
            args.lora_target = lora_cfg.get('target_modules', args.lora_target)
            cfg = base_cfg
    else:
        # Load config from base checkpoint directory
        config_path = os.path.join(os.path.dirname(args.base_checkpoint), "config.yaml")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        else:
            raise ValueError("No config found. Provide --config")

    device = args.device

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(cfg["tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Update vocab size
    cfg["model"]["vocab_size"] = len(tokenizer)

    # Load base model
    print("Loading base model...")
    model_cfg = Config(**cfg["model"])
    model = LLMTransformer(model_cfg)

    # Load base weights
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

    model = model.to(device)
    model.eval()

    print(f"\nPrompt: {args.prompt}")
    print("Generating...\n")

    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=device,
    )

    print("Output:")
    print(output)


if __name__ == "__main__":
    main()
