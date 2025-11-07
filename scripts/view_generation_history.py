#!/usr/bin/env python3
"""
Script pour consulter l'historique des générations.

Usage:
    python scripts/view_generation_history.py [output_dir]
    python scripts/view_generation_history.py out_slga_fineweb
    python scripts/view_generation_history.py --last 5
    python scripts/view_generation_history.py --step 4000
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict


def load_history(output_dir: str = "out_slga_fineweb") -> List[Dict]:
    """Charge l'historique des générations."""
    history_file = Path(output_dir) / "generation_history.jsonl"

    if not history_file.exists():
        print(f"❌ Aucun historique trouvé: {history_file}")
        return []

    history = []
    with open(history_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                history.append(entry)
            except json.JSONDecodeError:
                continue

    return history


def format_timestamp(iso_timestamp: str) -> str:
    """Formate un timestamp ISO en format lisible."""
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return iso_timestamp


def print_summary(history: List[Dict]):
    """Affiche un résumé de l'historique."""
    print(f"\n{'='*80}")
    print(f"HISTORIQUE DES GÉNÉRATIONS ({len(history)} entrées)")
    print(f"{'='*80}\n")

    for i, entry in enumerate(history, 1):
        checkpoint = entry.get("checkpoint", {})
        gen_params = entry.get("generation_params", {})

        print(f"[{i}] {format_timestamp(entry['timestamp'])}")
        print(f"    Checkpoint: step {checkpoint.get('step', 'N/A')}")
        if checkpoint.get('loss'):
            print(f"    Loss: {checkpoint['loss']:.4f}")

        print(f"    Prompt: {entry['prompt'][:50]}{'...' if len(entry['prompt']) > 50 else ''}")
        print(f"    Generated: {entry['generated_text'][:70]}{'...' if len(entry['generated_text']) > 70 else ''}")

        print(f"    Params: temp={gen_params.get('temperature', 'N/A')}, "
              f"top_k={gen_params.get('top_k', 'N/A')}, "
              f"top_p={gen_params.get('top_p', 'N/A')}")
        print(f"    Time: {entry.get('generation_time_seconds', 0):.2f}s")
        print()


def print_detailed(entry: Dict):
    """Affiche les détails complets d'une entrée."""
    print(f"\n{'='*80}")
    print(f"DÉTAILS DE GÉNÉRATION")
    print(f"{'='*80}\n")

    print(f"Timestamp: {format_timestamp(entry['timestamp'])}")
    print(f"Generation time: {entry.get('generation_time_seconds', 0):.2f}s")
    print()

    checkpoint = entry.get("checkpoint", {})
    print("--- CHECKPOINT INFO ---")
    print(f"Path: {checkpoint.get('checkpoint_path', 'N/A')}")
    print(f"Step: {checkpoint.get('step', 'N/A')}")
    if checkpoint.get('loss'):
        print(f"Loss: {checkpoint['loss']:.4f}")
    print(f"Parameters: {checkpoint.get('num_parameters', 'N/A')} tensors")
    print()

    print("--- MODEL CONFIG ---")
    model_cfg = entry.get("model_config", {})
    for key, value in model_cfg.items():
        print(f"{key}: {value}")
    print(f"\nTotal parameters: {entry.get('model_params_millions', 0):.2f}M")
    print()

    gen_params = entry.get("generation_params", {})
    print("--- GENERATION PARAMS ---")
    print(f"Temperature: {gen_params.get('temperature', 'N/A')}")
    print(f"Top-K: {gen_params.get('top_k', 'disabled')}")
    print(f"Top-P: {gen_params.get('top_p', 'disabled')}")
    print(f"Max tokens: {gen_params.get('max_tokens', 'N/A')}")
    print(f"Device: {entry.get('device', 'N/A')}")
    print()

    print(f"{'='*80}")
    print(f"PROMPT:\n{entry['prompt']}")
    print(f"{'='*80}\n")

    print(f"{'='*80}")
    print(f"GENERATED TEXT:\n{entry['generated_text']}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Consulter l'historique des générations")
    parser.add_argument("output_dir", nargs="?", default="out_slga_fineweb",
                        help="Dossier contenant l'historique")
    parser.add_argument("--last", type=int, help="Afficher les N dernières générations")
    parser.add_argument("--step", type=int, help="Filtrer par step du checkpoint")
    parser.add_argument("--detail", type=int, help="Afficher les détails de l'entrée N")
    parser.add_argument("--search", type=str, help="Rechercher dans les prompts")

    args = parser.parse_args()

    # Charger historique
    history = load_history(args.output_dir)

    if not history:
        return

    # Filtrer par step
    if args.step:
        history = [e for e in history if e.get("checkpoint", {}).get("step") == args.step]
        print(f"Filtré par step={args.step}: {len(history)} entrées")

    # Filtrer par recherche
    if args.search:
        search_lower = args.search.lower()
        history = [e for e in history if search_lower in e["prompt"].lower() or
                   search_lower in e["generated_text"].lower()]
        print(f"Recherche '{args.search}': {len(history)} entrées")

    # Limiter aux dernières
    if args.last:
        history = history[-args.last:]

    # Afficher détails d'une entrée spécifique
    if args.detail is not None:
        if 0 < args.detail <= len(history):
            print_detailed(history[args.detail - 1])
        else:
            print(f"❌ Entrée {args.detail} invalide (1-{len(history)})")
        return

    # Afficher résumé
    print_summary(history)

    # Stats
    if history:
        total_time = sum(e.get("generation_time_seconds", 0) for e in history)
        avg_time = total_time / len(history)

        steps = [e.get("checkpoint", {}).get("step") for e in history if e.get("checkpoint", {}).get("step")]
        unique_steps = set(steps)

        print(f"{'='*80}")
        print(f"STATISTIQUES")
        print(f"{'='*80}")
        print(f"Total générations: {len(history)}")
        print(f"Checkpoints utilisés: {len(unique_steps)} ({', '.join(f'step {s}' for s in sorted(unique_steps))})")
        print(f"Temps total: {total_time:.1f}s")
        print(f"Temps moyen: {avg_time:.2f}s")
        print()


if __name__ == "__main__":
    main()
