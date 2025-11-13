"""
Text generation CLI for SLGA models.
Provides a command-line interface for generating text using trained SLGA models.
"""

from __future__ import annotations
import os
import sys
import yaml
import torch
import argparse
from transformers import AutoTokenizer
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.legacy.model import Config, LLMTransformer
from src.generation import (
    TextGenerator,
    GenerationConfig,
    load_checkpoint,
    get_checkpoint_info,
)


def _format_metric(value: float, decimal_places: int = 4) -> str:
    """Display tiny magnitudes without dropping them to zero."""
    if value is None:
        return "N/A"
    if value != 0.0 and abs(value) < 10 ** (-decimal_places):
        return f"{value:.{decimal_places + 2}e}"
    return f"{value:.{decimal_places}f}"


def validate_generation_params(args):
    """Validate command-line generation parameters.

    Args:
        args: Parsed command-line arguments

    Raises:
        SystemExit: If any parameter validation fails
    """
    errors = []
    warnings = []

    # Validation
    if args.temperature < 0:
        errors.append(f"Temperature must be >= 0, got {args.temperature}")

    if args.top_k is not None and args.top_k < 0:
        errors.append(f"top_k must be >= 0, got {args.top_k}")

    if args.top_p is not None and not (0 < args.top_p <= 1):
        errors.append(f"top_p must be in (0, 1], got {args.top_p}")

    if args.max_tokens <= 0:
        errors.append(f"max_tokens must be > 0, got {args.max_tokens}")

    if args.repetition_penalty is not None and args.repetition_penalty < 1.0:
        errors.append(
            f"repetition_penalty must be >= 1.0 (1.0 disables it), got {args.repetition_penalty}"
        )

    if args.no_repeat_ngram_size is not None and args.no_repeat_ngram_size < 0:
        errors.append(
            f"no_repeat_ngram_size must be >= 0 (0 disables it), got {args.no_repeat_ngram_size}"
        )

    # Warning for using both top_k and top_p
    user_set_top_k = any(arg.startswith('--top-k') or arg.startswith('--top_k')
                         for arg in sys.argv)
    user_set_top_p = any(arg.startswith('--top-p') or arg.startswith('--top_p')
                         for arg in sys.argv)

    if (user_set_top_k and user_set_top_p and
        args.top_k is not None and args.top_k > 0 and
        args.top_p is not None and args.top_p < 1.0):
        warnings.append(
            f"⚠️  Using both top_k={args.top_k} and top_p={args.top_p} simultaneously.\n"
            f"    This applies BOTH filters sequentially (top_k THEN top_p).\n"
            f"    For most use cases, using only one is recommended:\n"
            f"    - Creative text: Use top_p=0.9 (nucleus sampling)\n"
            f"    - Focused generation: Use top_k=40\n"
            f"    - Greedy decoding: Set temperature=0.0 (disables both)"
        )

    # Display errors and exit if any
    if errors:
        print("=" * 80)
        print("❌ PARAMETER VALIDATION ERRORS")
        print("=" * 80)
        for err in errors:
            print(f"  • {err}")
        print("=" * 80)
        print("\nPlease fix the parameters and try again.")
        sys.exit(1)

    # Display warnings (non-blocking)
    if warnings:
        print("=" * 80)
        print("⚠️  PARAMETER WARNINGS")
        print("=" * 80)
        for warn in warnings:
            print(f"  {warn}")
        print("=" * 80)
        print()


def save_generation_output(
    output: str,
    prompt: str,
    args: argparse.Namespace,
    config_dict: dict,
    checkpoint_metadata: dict,
    generation_time: float,
    model_params: float,
):
    """Save generated text with complete metadata.

    Args:
        output: Generated text
        prompt: Original prompt
        args: Command-line arguments
        config_dict: Model configuration
        checkpoint_metadata: Checkpoint information
        generation_time: Time taken for generation
        model_params: Number of model parameters (in millions)
    """
    # Prepare metadata
    generation_metadata = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "generated_text": output,
        "generation_params": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k if args.top_k > 0 else None,
            "top_p": args.top_p if args.top_p is not None and args.top_p < 1.0 else None,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": (
                None if args.no_repeat_ngram_size in (None, 0)
                else args.no_repeat_ngram_size
            ),
        },
        "model_config": config_dict["model"],
        "checkpoint": checkpoint_metadata,
        "device": args.device,
        "generation_time_seconds": generation_time,
        "model_params_millions": model_params,
    }

    # Determine output directory
    try:
        output_dir = config_dict["save"]["out_dir"]
    except KeyError:
        print(f"\n⚠ Warning: 'save.out_dir' not found in config")
        output_dir = "."
        print(f"   Using current directory: {output_dir}")

    try:
        os.makedirs(output_dir, exist_ok=True)
    except (PermissionError, OSError) as e:
        print(f"\n⚠ Warning: Cannot create output directory '{output_dir}': {e}")
        output_dir = "."

    # 1. Timestamped file with full details
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    step_str = (
        f"_step{checkpoint_metadata['step']}"
        if checkpoint_metadata.get('step') is not None
        else ""
    )
    unique_output_path = os.path.join(
        output_dir,
        f"generation_{timestamp_str}{step_str}.txt"
    )

    try:
        with open(unique_output_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("SLGA GENERATION LOG\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Timestamp: {generation_metadata['timestamp']}\n")
            f.write(f"Generation time: {generation_time:.2f}s\n\n")

            f.write("--- CHECKPOINT INFO ---\n")
            f.write(f"Path: {checkpoint_metadata['checkpoint_path']}\n")
            if checkpoint_metadata.get('step') is not None:
                f.write(f"Step: {checkpoint_metadata['step']}\n")
            if checkpoint_metadata.get('loss') is not None:
                f.write(f"Loss: {_format_metric(checkpoint_metadata['loss'])}\n")
            if checkpoint_metadata.get('timestamp'):
                f.write(f"Checkpoint saved: {checkpoint_metadata['timestamp']}\n")
            if checkpoint_metadata.get('num_parameters'):
                f.write(f"Parameters: {checkpoint_metadata['num_parameters']} tensors\n")
            if checkpoint_metadata.get('first_param_mean'):
                f.write(
                    f"First param mean: "
                    f"{_format_metric(checkpoint_metadata['first_param_mean'], decimal_places=6)}\n"
                )
            f.write("\n")

            f.write("--- MODEL CONFIG ---\n")
            for key, value in config_dict["model"].items():
                f.write(f"{key}: {value}\n")
            f.write(f"\nTotal parameters: {model_params:.2f}M\n\n")

            f.write("--- GENERATION PARAMS ---\n")
            f.write(f"Temperature: {args.temperature}\n")
            f.write(f"Top-K: {args.top_k if args.top_k > 0 else 'disabled'}\n")
            f.write(
                f"Top-P: {args.top_p if args.top_p is not None and args.top_p < 1.0 else 'disabled'}\n"
            )
            f.write(
                f"Repetition penalty: "
                f"{args.repetition_penalty if args.repetition_penalty and args.repetition_penalty > 1.0 else 'disabled'}\n"
            )
            f.write(
                f"No-repeat n-gram: "
                f"{args.no_repeat_ngram_size if args.no_repeat_ngram_size and args.no_repeat_ngram_size > 0 else 'disabled'}\n"
            )
            f.write(f"Max tokens: {args.max_tokens}\n")
            f.write(f"Device: {args.device}\n\n")

            f.write("=" * 80 + "\n")
            f.write(f"PROMPT:\n{prompt}\n")
            f.write("=" * 80 + "\n\n")

            f.write("=" * 80 + "\n")
            f.write(f"GENERATED TEXT:\n{output}\n")
            f.write("=" * 80 + "\n")

        print(f"\n✓ Full log saved to: {unique_output_path}")
    except (PermissionError, OSError, UnicodeEncodeError) as e:
        print(f"\n❌ Error writing output file: {e}")

    # 2. Append to history log (JSONL)
    log_path = os.path.join(output_dir, "generation_history.jsonl")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(generation_metadata, ensure_ascii=False) + "\n")
        print(f"✓ History logged to: {log_path}")
    except (PermissionError, OSError, TypeError, ValueError) as e:
        print(f"\n⚠ Warning: Could not log to history: {e}")

    # 3. Legacy compatibility file
    legacy_path = os.path.join(output_dir, "generated_sample.txt")
    try:
        with open(legacy_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"✓ Quick view saved to: {legacy_path}")
    except (PermissionError, OSError, UnicodeEncodeError) as e:
        print(f"\n⚠ Warning: Could not save legacy file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate text with SLGA model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config")
    parser.add_argument("--prompt", type=str, default="The future of AI is", help="Prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature")
    parser.add_argument(
        "--top-k",
        type=int,
        default=80,
        help="Top-K filtering (use 0 to disable)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling (set >=1.0 to disable)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.15,
        help="Penalty for repeated tokens (1.0 disables)",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=4,
        help="Ban repeated n-grams of this size (0 disables)",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--eos-token-id",
        type=int,
        default=None,
        help="EOS token ID (defaults to tokenizer.eos_token_id)",
    )

    eos_group = parser.add_mutually_exclusive_group()
    eos_group.add_argument(
        "--stop-on-eos",
        dest="stop_on_eos",
        action="store_true",
        default=True,
        help="Stop generation when EOS is encountered (default)",
    )
    eos_group.add_argument(
        "--no-stop-on-eos",
        dest="stop_on_eos",
        action="store_false",
        help="Continue generation after EOS token",
    )

    args = parser.parse_args()

    # Validate parameters
    validate_generation_params(args)

    # Load config
    try:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing config file: {e}")
        sys.exit(1)

    print("=" * 80)
    print("=== SLGA Text Generation ===")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded")
        print()
    except (OSError, ValueError, KeyError) as e:
        print(f"❌ Error loading tokenizer: {e}")
        sys.exit(1)

    # Handle null vocab_size in config
    if cfg["model"].get("vocab_size") is None:
        if hasattr(tokenizer, 'vocab_size') and tokenizer.vocab_size is not None:
            cfg["model"]["vocab_size"] = tokenizer.vocab_size
            print(f"ℹ️  vocab_size was null, using tokenizer vocab_size: {tokenizer.vocab_size}")
        else:
            print(f"⚠️  Warning: vocab_size is null and tokenizer unavailable")
            print(f"   Using default: 50257 (GPT-2)")
            cfg["model"]["vocab_size"] = 50257

    # Create model
    print("Creating model...")
    try:
        model_cfg = Config(**cfg["model"])
        model = LLMTransformer(model_cfg)
        print(f"✓ Model created: {model.get_num_params() / 1e6:.2f}M parameters")
        print()
    except (TypeError, KeyError, ValueError) as e:
        print(f"❌ Error creating model: {e}")
        sys.exit(1)

    # Load checkpoint
    try:
        model, checkpoint_metadata = load_checkpoint(args.checkpoint, model, device="cpu")
        print()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Move to device
    try:
        model = model.to(args.device)
        model.eval()
    except RuntimeError as e:
        print(f"❌ Error moving model to device '{args.device}': {e}")
        sys.exit(1)

    # Create generator
    generator = TextGenerator(model, tokenizer, device=args.device)

    # Create generation config
    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p if args.top_p is not None and args.top_p < 1.0 else None,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=(
            None if args.no_repeat_ngram_size in (None, 0)
            else args.no_repeat_ngram_size
        ),
        eos_token_id=args.eos_token_id,
        stop_on_eos=args.stop_on_eos,
    )

    # Display generation settings
    print("Generation settings:")
    gen_info = generator.get_generation_info(gen_config)
    for key, value in gen_info.items():
        print(f"  {key}: {value}")
    print()

    # Generate
    print("=" * 80)
    print(f"PROMPT: {args.prompt}")
    print("=" * 80)
    print()

    generation_start = datetime.now()
    try:
        output = generator.generate(args.prompt, config=gen_config, verbose=True)
        generation_time = (datetime.now() - generation_start).total_seconds()
    except (RuntimeError, ValueError) as e:
        print(f"\n❌ Error during generation: {e}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("GENERATED TEXT:")
    print("=" * 80)
    print(output)
    print("=" * 80)

    # Save output
    save_generation_output(
        output,
        args.prompt,
        args,
        cfg,
        checkpoint_metadata,
        generation_time,
        model.get_num_params() / 1e6,
    )


if __name__ == "__main__":
    main()
