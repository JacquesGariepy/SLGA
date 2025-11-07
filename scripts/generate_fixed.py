from __future__ import annotations
import os
import sys
import yaml
import torch
import argparse
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import Config, LLMTransformer


def generate_text(
    model: LLMTransformer,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = None,
    device: str = "cuda",
) -> str:
    """Génère du texte à partir d'un prompt."""
    model.eval()
    
    # Encoder prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"Prompt length: {input_ids.size(1)} tokens")
    print(f"Generating {max_new_tokens} new tokens...")
    print()
    
    # Générer
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    
    # Décoder
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text


def load_checkpoint(checkpoint_path: str, model: LLMTransformer) -> LLMTransformer:
    """
    Charge un checkpoint CORRECTEMENT.
    
    Args:
        checkpoint_path: Chemin vers checkpoint (dir ou fichier)
        model: Modèle à charger
    
    Returns:
        model: Modèle avec poids chargés
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    if os.path.isdir(checkpoint_path):
        # Format: out_slga/ckpt_11000/
        model_path = os.path.join(checkpoint_path, "model.pt")
        
        if not os.path.exists(model_path):
            available = os.listdir(checkpoint_path)
            raise FileNotFoundError(
                f"❌ model.pt not found in {checkpoint_path}\n"
                f"   Available files: {available}\n"
                f"   Expected: model.pt (state dict of the model)"
            )
        
        print(f"  Loading state dict from {model_path}...")
        state_dict = torch.load(model_path, map_location="cpu")
        
    else:
        # Format direct: model.pt
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ Checkpoint file not found: {checkpoint_path}")
        
        print(f"  Loading state dict from file...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # Charger les poids
    try:
        model.load_state_dict(state_dict)
        print("✓ Checkpoint loaded successfully")
        print(f"  Loaded {len(state_dict)} parameter tensors")
        
        # Vérifier que les poids ne sont pas random
        first_param = next(iter(state_dict.values()))
        print(f"  Sanity check - first param mean: {first_param.float().mean().item():.6f}")
        
    except Exception as e:
        print(f"❌ Error loading state dict: {e}")
        raise
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Generate text with SLGA model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config")
    parser.add_argument("--prompt", type=str, default="The future of AI is", help="Prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K filtering")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    print("=" * 80)
    print("=== SLGA Text Generation (FIXED VERSION) ===")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print()
    
    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")
    print()
    
    # Create model
    print("Creating model...")
    model_cfg = Config(**cfg["model"])
    model = LLMTransformer(model_cfg)
    print(f"✓ Model created: {model.get_num_params() / 1e6:.2f}M parameters")
    print()
    
    # Load checkpoint (CORRECTED!)
    model = load_checkpoint(args.checkpoint, model)
    print()
    
    model = model.to(args.device)
    model.eval()
    
    # Generation settings
    print("Generation settings:")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-K: {args.top_k if args.top_k > 0 else 'disabled'}")
    print(f"  Top-P: {args.top_p if args.top_p else 'disabled'}")
    print()
    
    # Generate
    print("=" * 80)
    print(f"PROMPT: {args.prompt}")
    print("=" * 80)
    print()
    
    output = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p,
        device=args.device,
    )
    
    print("=" * 80)
    print("GENERATED TEXT:")
    print("=" * 80)
    print(output)
    print("=" * 80)
    
    # Save
    output_dir = cfg["save"]["out_dir"]
    output_path = os.path.join(output_dir, "generated_sample.txt")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Prompt: {args.prompt}\n\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Top-K: {args.top_k}\n")
        f.write(f"Top-P: {args.top_p}\n\n")
        f.write(f"Generated:\n{output}\n")
    
    print(f"\n✓ Output saved to {output_path}")


if __name__ == "__main__":
    main()