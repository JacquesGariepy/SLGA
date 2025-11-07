# generate_fixed.py
"""
Génération de texte avec un modèle SLGA entraîné - VERSION CORRIGÉE.
Logs toutes les générations avec métadonnées complètes.
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
import pickle


def _format_metric(value: float, decimal_places: int = 4) -> str:
    """Display tiny magnitudes without dropping them to zero."""
    if value is None:
        return "N/A"
    if value != 0.0 and abs(value) < 10 ** (-decimal_places):
        return f"{value:.{decimal_places + 2}e}"
    return f"{value:.{decimal_places}f}"

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import Config, LLMTransformer


def generate_text(
    model: LLMTransformer,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.6,
    top_k: int = 80,
    top_p: float | None = 0.95,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int | None = 4,
    eos_token_id: int | None = None,
    stop_on_eos: bool = True,
    device: str = "cuda",
) -> str:
    """Génère du texte à partir d'un prompt."""
    # Validate generation parameters
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")

    if temperature < 0:
        raise ValueError(f"temperature must be ≥ 0, got {temperature}")

    if top_k is not None and top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}")

    if top_p is not None and not (0 < top_p <= 1):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    if repetition_penalty is not None and repetition_penalty < 1.0:
        raise ValueError(
            f"repetition_penalty must be ≥ 1.0 (1.0 disables it), got {repetition_penalty}"
        )

    if no_repeat_ngram_size is not None and no_repeat_ngram_size < 0:
        raise ValueError(
            f"no_repeat_ngram_size must be ≥ 0 (0 disables it), got {no_repeat_ngram_size}"
        )

    model.eval()

    # Encoder prompt
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to encode prompt: {e}") from e

    print(f"Prompt length: {input_ids.size(1)} tokens")
    print(f"Generating {max_new_tokens} new tokens...")
    print()

    # ✅ FIX: Passer eos_token_id depuis tokenizer si non fourni
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    # Générer
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                eos_token_id=eos_token_id,
                stop_on_eos=stop_on_eos,
            )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise RuntimeError(
                f"GPU out of memory during generation. "
                f"Try reducing max_new_tokens or using CPU."
            ) from e
        raise RuntimeError(f"Model generation failed: {e}") from e
    except ValueError as e:
        raise ValueError(f"Invalid generation parameters: {e}") from e

    # Décoder
    try:
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        raise RuntimeError(f"Failed to decode output: {e}") from e

    return generated_text


def load_checkpoint(checkpoint_path: str, model: LLMTransformer) -> tuple[LLMTransformer, dict]:
    """
    Charge un checkpoint CORRECTEMENT et retourne les métadonnées.

    Args:
        checkpoint_path: Chemin vers checkpoint (dir ou fichier)
        model: Modèle à charger

    Returns:
        model: Modèle avec poids chargés
        metadata: Dictionnaire avec métadonnées du checkpoint
    """
    print(f"Loading checkpoint from {checkpoint_path}...")

    metadata = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_type": None,
        "step": None,
        "loss": None,
        "timestamp": None,
    }

    if os.path.isdir(checkpoint_path):
        # Format: out_slga/ckpt_11000/
        metadata["checkpoint_type"] = "directory"

        # Extraire step depuis le nom du dossier
        dir_name = os.path.basename(checkpoint_path)
        if dir_name.startswith("ckpt_"):
            try:
                metadata["step"] = int(dir_name.split("_")[1])
            except ValueError as e:
                print(f"⚠ Warning: Invalid step number in directory name '{dir_name}': {e}")
                print(f"   Expected format: ckpt_<number>")
                # Continue without step metadata
            except IndexError as e:
                print(f"⚠ Warning: Malformed directory name '{dir_name}': {e}")
                print(f"   Expected format: ckpt_<number>")
                # Continue without step metadata

        model_path = os.path.join(checkpoint_path, "model.pt")
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.pt")

        if not os.path.exists(model_path):
            available = os.listdir(checkpoint_path)
            raise FileNotFoundError(
                f"❌ model.pt not found in {checkpoint_path}\n"
                f"   Available files: {available}\n"
                f"   Expected: model.pt (state dict of the model)"
            )

        print(f"  Loading state dict from {model_path}...")
        try:
            state_dict = torch.load(model_path, map_location="cpu")
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load checkpoint file (PyTorch error): {e}\n"
                f"   File may be corrupted or from incompatible PyTorch version"
            ) from e
        except pickle.UnpicklingError as e:
            raise RuntimeError(
                f"Failed to unpickle checkpoint file: {e}\n"
                f"   File appears to be corrupted"
            ) from e
        except EOFError as e:
            raise RuntimeError(
                f"Checkpoint file is incomplete (truncated): {e}\n"
                f"   File may have been interrupted during save"
            ) from e

        # Charger métadonnées du trainer state si disponible
        if os.path.exists(trainer_state_path):
            try:
                trainer_state = torch.load(trainer_state_path, map_location="cpu")
                metadata["step"] = trainer_state.get("step", metadata["step"])
                metadata["loss"] = trainer_state.get("loss", None)
                metadata["timestamp"] = datetime.fromtimestamp(
                    os.path.getmtime(trainer_state_path)
                ).isoformat()
            except RuntimeError as e:
                print(f"⚠ Warning: PyTorch error loading trainer state: {e}")
                print(f"   File may be corrupted or from incompatible PyTorch version")
                # Continue without trainer metadata
            except pickle.UnpicklingError as e:
                print(f"⚠ Warning: Corrupted pickle data in trainer state: {e}")
                print(f"   File: {trainer_state_path}")
                # Continue without trainer metadata
            except EOFError as e:
                print(f"⚠ Warning: Incomplete trainer state file (truncated): {e}")
                print(f"   File may have been interrupted during save")
                # Continue without trainer metadata
            except OSError as e:
                print(f"⚠ Warning: OS error reading trainer state: {e}")
                print(f"   Check file permissions and disk health")
                # Continue without trainer metadata

    else:
        # Format direct: model.pt
        metadata["checkpoint_type"] = "file"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ Checkpoint file not found: {checkpoint_path}")

        print(f"  Loading state dict from file...")
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load checkpoint file (PyTorch error): {e}\n"
                f"   File may be corrupted or from incompatible PyTorch version"
            ) from e
        except pickle.UnpicklingError as e:
            raise RuntimeError(
                f"Failed to unpickle checkpoint file: {e}\n"
                f"   File appears to be corrupted"
            ) from e
        except EOFError as e:
            raise RuntimeError(
                f"Checkpoint file is incomplete (truncated): {e}\n"
                f"   File may have been interrupted during save"
            ) from e

        try:
            metadata["timestamp"] = datetime.fromtimestamp(
                os.path.getmtime(checkpoint_path)
            ).isoformat()
        except OSError as e:
            print(f"⚠ Warning: Could not get file modification time: {e}")
            metadata["timestamp"] = None

    # Charger les poids
    try:
        model.load_state_dict(state_dict)
        print("✓ Checkpoint loaded successfully")
        print(f"  Loaded {len(state_dict)} parameter tensors")

        # Vérifier que les poids ne sont pas random
        first_param = next(iter(state_dict.values()))
        param_mean = first_param.float().mean().item()
        print(
            "  Sanity check - first param mean: "
            f"{_format_metric(param_mean, decimal_places=6)}"
        )

        metadata["num_parameters"] = len(state_dict)
        metadata["first_param_mean"] = param_mean

        if metadata["step"] is not None:
            print(f"  Step: {metadata['step']}")
        if metadata["loss"] is not None:
            print(f"  Loss: {_format_metric(metadata['loss'])}")

    except (RuntimeError, TypeError, KeyError) as e:
        print(f"❌ Error loading state dict into model: {e}")
        print(f"   This usually means checkpoint architecture mismatch.")
        print(f"   Available keys in state dict: {list(state_dict.keys())[:5]}...")
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e

    return model, metadata



def validate_generation_params(args):
    """Valide les paramètres de génération avant utilisation.

    Args:
        args: Parsed command-line arguments

    Raises:
        SystemExit: If any parameter validation fails
    """
    errors = []
    warnings = []

    # Validation temperature
    if args.temperature < 0:
        errors.append(f"Temperature must be >= 0, got {args.temperature}")

    # Validation top_k
    if args.top_k is not None and args.top_k < 0:
        errors.append(f"top_k must be ≥ 0, got {args.top_k}")

    # Validation top_p (must be strictly positive, matching generate_text validation)
    if args.top_p is not None and not (0 < args.top_p <= 1):
        errors.append(f"top_p must be in (0, 1] (set ≥1.0 to disable), got {args.top_p}")

    # Validation max_tokens
    if args.max_tokens <= 0:
        errors.append(f"max_tokens must be > 0, got {args.max_tokens}")

    if args.repetition_penalty is not None and args.repetition_penalty < 1.0:
        errors.append(
            f"repetition_penalty must be ≥ 1.0 (1.0 disables it), got {args.repetition_penalty}"
        )

    if args.no_repeat_ngram_size is not None and args.no_repeat_ngram_size < 0:
        errors.append(
            f"no_repeat_ngram_size must be ≥ 0 (0 disables it), got {args.no_repeat_ngram_size}"
        )

    # ⚠️ NOUVEAU: Avertissement pour combinaison top_k + top_p
    # Seulement si l'utilisateur a EXPLICITEMENT fourni les deux
    # (pour éviter warning avec valeurs par défaut)
    # ✅ FIX: Détecter arguments qui commencent par --top-k ou --top-p
    #         (gère --top-k=40, --top-k 40, --top_k=40, etc.)
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

    # Si des erreurs, afficher et quitter
    if errors:
        print("=" * 80)
        print("❌ PARAMETER VALIDATION ERRORS")
        print("=" * 80)
        for err in errors:
            print(f"  • {err}")
        print("=" * 80)
        print("\nPlease fix the parameters and try again.")
        sys.exit(1)

    # Afficher les avertissements (non-bloquants)
    if warnings:
        print("=" * 80)
        print("⚠️  PARAMETER WARNINGS")
        print("=" * 80)
        for warn in warnings:
            print(f"  {warn}")
        print("=" * 80)
        print()


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
        help="Nucleus sampling (set ≥1.0 to disable)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.15,
        help="Penalty applied to previously generated tokens (1.0 disables)",
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
        help="EOS token ID for stopping generation (defaults to tokenizer.eos_token_id)",
    )

    # ✅ FIX: Utiliser un groupe mutuellement exclusif pour --stop-on-eos / --no-stop-on-eos
    eos_group = parser.add_mutually_exclusive_group()
    eos_group.add_argument(
        "--stop-on-eos",
        dest="stop_on_eos",
        action="store_true",
        default=True,
        help="Stop generation when EOS token is encountered (default)",
    )
    eos_group.add_argument(
        "--no-stop-on-eos",
        dest="stop_on_eos",
        action="store_false",
        help="Continue generation even after EOS token",
    )

    args = parser.parse_args()

    # Validate generation parameters before proceeding
    validate_generation_params(args)

    # Load config
    try:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Config file not found: {args.config}")
        print(f"   Please provide a valid config file with --config")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing config file: {e}")
        sys.exit(1)
    
    print("=" * 80)
    print("=== SLGA Text Generation (FIXED VERSION) ===")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print()
    
    # Tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded")
        print()
    except (OSError, ValueError, KeyError) as e:
        print(f"❌ Error loading tokenizer '{cfg.get('tokenizer', 'N/A')}': {e}")
        print(f"   Check that tokenizer name is correct in config file")
        sys.exit(1)

    # ✅ FIX Bug #31: Gérer vocab_size: null dans config
    # Si vocab_size est null, utiliser la taille du tokenizer
    if cfg["model"].get("vocab_size") is None:
        if hasattr(tokenizer, 'vocab_size') and tokenizer.vocab_size is not None:
            cfg["model"]["vocab_size"] = tokenizer.vocab_size
            print(f"ℹ️  vocab_size was null, using tokenizer vocab_size: {tokenizer.vocab_size}")
        else:
            print(f"⚠️  Warning: vocab_size is null and tokenizer.vocab_size unavailable")
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
        print(f"❌ Error creating model from config: {e}")
        print(f"   Check model configuration in {args.config}")
        print(f"   Available config keys: {list(cfg.get('model', {}).keys())}")
        sys.exit(1)
    
    # Load checkpoint (CORRECTED!)
    try:
        model, checkpoint_metadata = load_checkpoint(args.checkpoint, model)
        print()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)

    try:
        model = model.to(args.device)
        model.eval()
    except RuntimeError as e:
        print(f"❌ Error moving model to device '{args.device}': {e}")
        print(f"   Available CUDA devices: {torch.cuda.device_count()}")
        if args.device == "cuda" and not torch.cuda.is_available():
            print(f"   CUDA is not available. Try --device cpu")
        sys.exit(1)

    # Generation settings
    print("Generation settings:")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-K: {args.top_k if args.top_k > 0 else 'disabled'}")
    print(
        f"  Top-P: {args.top_p if args.top_p is not None and args.top_p < 1.0 else 'disabled'}"
    )
    print(
        "  Repetition penalty: "
        f"{args.repetition_penalty if args.repetition_penalty and args.repetition_penalty > 1.0 else 'disabled'}"
    )
    print(
        "  No-repeat n-gram: "
        f"{args.no_repeat_ngram_size if args.no_repeat_ngram_size and args.no_repeat_ngram_size > 0 else 'disabled'}"
    )
    print()

    # Déterminer l'EOS token ID
    eos_token_id = args.eos_token_id if args.eos_token_id is not None else tokenizer.eos_token_id

    # Log EOS configuration
    print(f"EOS token configuration:")
    print(f"  Token ID: {eos_token_id}")
    print(f"  Stop on EOS: {args.stop_on_eos}")
    if eos_token_id is not None:
        try:
            eos_token_str = tokenizer.decode([eos_token_id])
            print(f"  Token string: '{eos_token_str}'")
        except Exception:
            print(f"  Token string: [unable to decode]")
    print()

    # ⚠️ Log explicite si top_k + top_p sont utilisés ensemble
    # (seulement si explicitement fournis par l'utilisateur)
    # ✅ FIX: Détecter arguments qui commencent par --top-k ou --top-p
    user_set_top_k = any(arg.startswith('--top-k') or arg.startswith('--top_k')
                         for arg in sys.argv)
    user_set_top_p = any(arg.startswith('--top-p') or arg.startswith('--top_p')
                         for arg in sys.argv)

    if (user_set_top_k and user_set_top_p and
        args.top_k is not None and args.top_k > 0 and
        args.top_p is not None and args.top_p < 1.0):
        print("⚠️  NOTE: Using BOTH top_k and top_p filtering")
        print(f"   Sampling will apply: top_k={args.top_k} THEN top_p={args.top_p}")
        print(f"   This may result in very restrictive sampling.")
        print()

    # Generate
    print("=" * 80)
    print(f"PROMPT: {args.prompt}")
    print("=" * 80)
    print()

    generation_start = datetime.now()
    try:
        output = generate_text(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p if args.top_p is not None and args.top_p < 1.0 else None,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=(
                None
                if args.no_repeat_ngram_size in (None, 0)
                else args.no_repeat_ngram_size
            ),
            eos_token_id=eos_token_id,
            stop_on_eos=args.stop_on_eos,
            device=args.device,
        )
        generation_time = (datetime.now() - generation_start).total_seconds()
    except RuntimeError as e:
        print(f"\n❌ Error during text generation: {e}")
        if "out of memory" in str(e).lower():
            print(f"   GPU out of memory. Try:")
            print(f"   - Reducing --max-tokens")
            print(f"   - Using --device cpu")
            print(f"   - Reducing model size")
        sys.exit(1)
    except (ValueError, IndexError) as e:
        print(f"\n❌ Error during generation: {e}")
        print(f"   Check generation parameters (temperature, top_k, top_p)")
        sys.exit(1)

    print("=" * 80)
    print("GENERATED TEXT:")
    print("=" * 80)
    print(output)
    print("=" * 80)

    # Préparer métadonnées complètes
    generation_metadata = {
        "timestamp": datetime.now().isoformat(),
        "prompt": args.prompt,
        "generated_text": output,
        "generation_params": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k if args.top_k > 0 else None,
            "top_p": args.top_p if args.top_p is not None and args.top_p < 1.0 else None,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": (
                None
                if args.no_repeat_ngram_size in (None, 0)
                else args.no_repeat_ngram_size
            ),
        },
        "model_config": cfg["model"],
        "checkpoint": checkpoint_metadata,
        "device": args.device,
        "generation_time_seconds": generation_time,
        "model_params_millions": model.get_num_params() / 1e6,
    }

    # Save avec métadonnées complètes
    try:
        output_dir = cfg["save"]["out_dir"]
    except KeyError as e:
        print(f"\n⚠ Warning: 'save.out_dir' not found in config: {e}")
        output_dir = "."
        print(f"   Using current directory: {output_dir}")

    try:
        os.makedirs(output_dir, exist_ok=True)
    except PermissionError as e:
        print(f"\n❌ Error: No permission to create directory '{output_dir}': {e}")
        output_dir = "."
        print(f"   Falling back to current directory")
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"❌ Fatal: Cannot create any output directory: {e}")
            sys.exit(1)
    except OSError as e:
        print(f"\n❌ Error: OS error creating directory '{output_dir}': {e}")
        print(f"   Check disk space and path validity")
        output_dir = "."
        print(f"   Falling back to current directory")

    # 1. Fichier unique avec timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    step_str = (
        f"_step{checkpoint_metadata['step']}"
        if checkpoint_metadata['step'] is not None
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
            if checkpoint_metadata['step'] is not None:
                f.write(f"Step: {checkpoint_metadata['step']}\n")
            if checkpoint_metadata['loss'] is not None:
                f.write(f"Loss: {_format_metric(checkpoint_metadata['loss'])}\n")
            if checkpoint_metadata['timestamp']:
                f.write(f"Checkpoint saved: {checkpoint_metadata['timestamp']}\n")
            f.write(f"Parameters: {checkpoint_metadata['num_parameters']} tensors\n")
            f.write(
                "First param mean: "
                f"{_format_metric(checkpoint_metadata['first_param_mean'], decimal_places=6)}\n\n"
            )

            f.write("--- MODEL CONFIG ---\n")
            for key, value in cfg["model"].items():
                f.write(f"{key}: {value}\n")
            f.write(f"\nTotal parameters: {model.get_num_params() / 1e6:.2f}M\n\n")

            f.write("--- GENERATION PARAMS ---\n")
            f.write(f"Temperature: {args.temperature}\n")
            f.write(f"Top-K: {args.top_k if args.top_k > 0 else 'disabled'}\n")
            f.write(
                f"Top-P: {args.top_p if args.top_p is not None and args.top_p < 1.0 else 'disabled'}\n"
            )
            f.write(
                "Repetition penalty: "
                f"{args.repetition_penalty if args.repetition_penalty and args.repetition_penalty > 1.0 else 'disabled'}\n"
            )
            f.write(
                "No-repeat n-gram: "
                f"{args.no_repeat_ngram_size if args.no_repeat_ngram_size and args.no_repeat_ngram_size > 0 else 'disabled'}\n"
            )
            f.write(f"Max tokens: {args.max_tokens}\n")
            f.write(f"Device: {args.device}\n\n")

            f.write("=" * 80 + "\n")
            f.write(f"PROMPT:\n{args.prompt}\n")
            f.write("=" * 80 + "\n\n")

            f.write("=" * 80 + "\n")
            f.write(f"GENERATED TEXT:\n{output}\n")
            f.write("=" * 80 + "\n")
    except PermissionError as e:
        print(f"\n❌ Error: No permission to write file '{unique_output_path}': {e}")
        sys.exit(1)
    except OSError as e:
        print(f"\n❌ Error: OS error writing file '{unique_output_path}': {e}")
        print(f"   Check disk space and path validity")
        sys.exit(1)
    except UnicodeEncodeError as e:
        print(f"\n❌ Error: Encoding error writing file: {e}")
        print(f"   Generated text contains characters that cannot be encoded")
        sys.exit(1)

    # 2. Log centralisé (append mode)
    log_path = os.path.join(output_dir, "generation_history.jsonl")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(generation_metadata, ensure_ascii=False) + "\n")
    except PermissionError as e:
        print(f"\n⚠ Warning: No permission to write history log '{log_path}': {e}")
        print(f"   Generation completed but history not logged")
    except OSError as e:
        print(f"\n⚠ Warning: OS error writing history log '{log_path}': {e}")
        print(f"   Generation completed but history not logged")
    except (TypeError, ValueError) as e:
        print(f"\n⚠ Warning: Error serializing metadata to JSON: {e}")
        print(f"   Generation completed but history not logged")

    # 3. Fichier legacy pour compatibilité
    legacy_path = os.path.join(output_dir, "generated_sample.txt")
    try:
        with open(legacy_path, "w", encoding="utf-8") as f:
            f.write(output)
    except PermissionError as e:
        print(f"\n⚠ Warning: No permission to write legacy file '{legacy_path}': {e}")
    except OSError as e:
        print(f"\n⚠ Warning: OS error writing legacy file '{legacy_path}': {e}")
    except UnicodeEncodeError as e:
        print(f"\n⚠ Warning: Encoding error writing legacy file: {e}")

    print(f"\n✓ Output saved to:")
    print(f"  - Full log: {unique_output_path}")
    if os.path.exists(log_path):
        print(f"  - History: {log_path}")
    if os.path.exists(legacy_path):
        print(f"  - Quick view: {legacy_path}")


if __name__ == "__main__":
    main()