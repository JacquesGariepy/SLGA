#!/usr/bin/env python3
"""
Grid search utility for SLGA text generation.
Evaluates multiple sampling combinations for a given config/checkpoint pair.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import torch
import yaml
from transformers import AutoTokenizer

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import Config, LLMTransformer  # noqa: E402
from scripts.generate import load_checkpoint  # noqa: E402  pylint: disable=wrong-import-position


# Fixed English prompt suite kept stable for reproducibility across runs.
DEFAULT_PROMPTS: List[str] = [
    "Summarize the theory of relativity for a high-school science class.",
    "Provide a step-by-step recipe for a quick vegetarian dinner.",
    "Draft a short science-fiction scene set on a Martian research base.",
    "Explain the historical significance of Rome as the capital of Italy.",
    "Write a clean Python function that reverses a singly linked list.",
    "Describe a comprehensive daily workout plan for improving endurance.",
    "Outline the key talking points for a cybersecurity awareness briefing.",
]


@dataclass(frozen=True)
class SamplingCombo:
    temperature: float
    top_k: Optional[int]
    top_p: Optional[float]
    repetition_penalty: float
    no_repeat_ngram_size: Optional[int]

    def as_dict(self) -> Dict[str, Optional[Union[float, int]]]:
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
        }


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test plusieurs combinaisons de génération pour un checkpoint SLGA",
    )
    parser.add_argument("--config", required=True, type=Path, help="Fichier de configuration YAML")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Chemin vers le checkpoint")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Fichier texte avec un prompt par ligne",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="Prompts passés directement via la ligne de commande",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Nombre de tokens générés par prompt",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="*",
        default=[0.6, 0.8, 1.0],
        help="Valeurs de température à tester",
    )
    parser.add_argument(
        "--top-ks",
        type=int,
        nargs="*",
        default=[0, 40, 80],
        help="Valeurs de top-k à tester (0 pour désactiver)",
    )
    parser.add_argument(
        "--top-ps",
        type=float,
        nargs="*",
        default=[1.0, 0.95, 0.9],
        help="Valeurs de top-p à tester (>= 1.0 pour désactiver)",
    )
    parser.add_argument(
        "--repetition-penalties",
        type=float,
        nargs="*",
        default=[1.0, 1.1],
        help="Facteurs de pénalisation des répétitions (1.0 = désactivé)",
    )
    parser.add_argument(
        "--no-repeat-ngram-sizes",
        type=int,
        nargs="*",
        default=[0, 4],
        help="Tailles d'n-grammes à bannir (0 = désactivé)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device explicite (cuda, cuda:1, cpu, ...)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Seed de base pour reproductibilité",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Répertoire de sortie pour le rapport JSON",
    )
    parser.add_argument(
        "--truncate-text",
        type=int,
        default=400,
        help="Nombre maximal de caractères conservés par génération dans le rapport",
    )
    args = parser.parse_args()

    if not args.temperatures:
        parser.error("--temperatures doit contenir au moins une valeur")
    if not args.top_ks:
        parser.error("--top-ks doit contenir au moins une valeur")
    if not args.top_ps:
        parser.error("--top-ps doit contenir au moins une valeur")
    if not args.repetition_penalties:
        parser.error("--repetition-penalties doit contenir au moins une valeur")
    if not args.no_repeat_ngram_sizes:
        parser.error("--no-repeat-ngram-sizes doit contenir au moins une valeur")

    return args


def load_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompts:
        return [p.strip() for p in args.prompts if p.strip()]

    if args.prompt_file:
        if not args.prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
        prompts = [line.strip() for line in args.prompt_file.read_text(encoding="utf-8").splitlines()]
        prompts = [p for p in prompts if p]
        if prompts:
            return prompts

    # Stable English prompt set ensures comparable results across runs.
    return list(DEFAULT_PROMPTS)


def load_model_and_tokenizer(
    config_path: Path,
    checkpoint_path: Path,
    device: str,
) -> tuple[LLMTransformer, AutoTokenizer, Dict]:
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    tokenizer_name = cfg.get("tokenizer")
    if not tokenizer_name:
        raise KeyError("Le fichier de config doit contenir la clé 'tokenizer'")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_cfg = Config(**cfg["model"])
    model = LLMTransformer(model_cfg)
    model, _ = load_checkpoint(str(checkpoint_path), model)
    model = model.to(device)
    model.eval()

    return model, tokenizer, cfg


def build_combinations(args: argparse.Namespace) -> List[SamplingCombo]:
    combos: List[SamplingCombo] = []
    for temp, top_k, top_p, rep_penalty, no_repeat in itertools.product(
        args.temperatures,
        args.top_ks,
        args.top_ps,
        args.repetition_penalties,
        args.no_repeat_ngram_sizes,
    ):
        resolved_top_k = None if top_k <= 0 else top_k
        resolved_top_p = None if top_p >= 1.0 else round(top_p, 4)
        resolved_no_repeat = None if no_repeat <= 0 else no_repeat
        combos.append(
            SamplingCombo(
                temperature=temp,
                top_k=resolved_top_k,
                top_p=resolved_top_p,
                repetition_penalty=rep_penalty,
                no_repeat_ngram_size=resolved_no_repeat,
            )
        )
    return combos


def generate_once(
    model: LLMTransformer,
    tokenizer: AutoTokenizer,
    prompt: str,
    combo: SamplingCombo,
    max_new_tokens: int,
    device: str,
    seed: int,
) -> tuple[str, List[int], float]:
    encoded = tokenizer.encode(prompt, return_tensors="pt").to(device)

    gen_start = torch.cuda.Event(enable_timing=True) if device.startswith("cuda") else None
    gen_end = torch.cuda.Event(enable_timing=True) if device.startswith("cuda") else None

    cpu_start = time.perf_counter() if gen_start is None else None

    if gen_start and gen_end:
        gen_start.record()

    with torch.no_grad():
        output_ids = model.generate(
            encoded,
            max_new_tokens=max_new_tokens,
            temperature=combo.temperature,
            top_k=combo.top_k,
            top_p=combo.top_p,
            repetition_penalty=combo.repetition_penalty,
            no_repeat_ngram_size=combo.no_repeat_ngram_size,
            seed=seed,
        )

    if gen_start and gen_end:
        gen_end.record()
        torch.cuda.synchronize()
        elapsed = gen_start.elapsed_time(gen_end) / 1000.0
    else:
        elapsed = time.perf_counter() - cpu_start if cpu_start is not None else 0.0

    tokens = output_ids[0].tolist()
    prompt_len = encoded.size(1)
    new_tokens = tokens[prompt_len:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generated_text, new_tokens, elapsed


def compute_metrics(generated_text: str, token_ids: List[int]) -> Dict[str, float]:
    length = len(token_ids)
    words = generated_text.split()

    metrics: Dict[str, float] = {
        "tokens": float(length),
        "unique_tokens": float(len(set(token_ids))) if token_ids else 0.0,
        "token_diversity": (len(set(token_ids)) / length) if length else 0.0,
        "word_diversity": (len(set(words)) / len(words)) if words else 0.0,
    }

    # Distinct-n metrics to spot repetitions
    for n in (2, 3):
        ngrams = [tuple(token_ids[i : i + n]) for i in range(max(0, length - n + 1))]
        unique = len(set(ngrams))
        metrics[f"distinct_{n}"] = (unique / len(ngrams)) if ngrams else 0.0

    if length:
        # Immediate repetition rate
        repeats = sum(1 for i in range(length - 1) if token_ids[i] == token_ids[i + 1])
        metrics["immediate_repeat_rate"] = repeats / (length - 1) if length > 1 else 0.0

        # Longest repeating n-gram count (n=4 window)
        fourgrams = [tuple(token_ids[i : i + 4]) for i in range(max(0, length - 3))]
        counts = Counter(fourgrams)
        metrics["max_fourgram_reps"] = float(max(counts.values())) if counts else 0.0
    else:
        metrics["immediate_repeat_rate"] = 0.0
        metrics["max_fourgram_reps"] = 0.0

    return metrics


def score_metrics(metrics: Dict[str, float]) -> float:
    # Favorise diversité et pénalise les répétitions agressives
    return (
        metrics.get("token_diversity", 0.0) * 0.4
        + metrics.get("distinct_2", 0.0) * 0.3
        + metrics.get("distinct_3", 0.0) * 0.2
        + metrics.get("word_diversity", 0.0) * 0.1
        - metrics.get("immediate_repeat_rate", 0.0) * 0.3
        - metrics.get("max_fourgram_reps", 0.0) * 0.05
    )


def evaluate_combinations(
    model: LLMTransformer,
    tokenizer: AutoTokenizer,
    prompts: Iterable[str],
    combos: List[SamplingCombo],
    max_new_tokens: int,
    device: str,
    base_seed: int,
    truncate: int,
    history_log: Path,
) -> List[Dict]:
    results: List[Dict] = []
    combos_total = len(combos)
    prompts_list = list(prompts)
    prompts_total = len(prompts_list)

    print(
        f"Running grid search with {combos_total} combinations across {prompts_total} prompts...",
        flush=True,
    )

    for combo_idx, combo in enumerate(combos):
        prompt_entries = []
        aggregate_metrics: Dict[str, float] = {}
        successful = 0

        print(
                f"  -> [{combo_idx + 1}/{combos_total}] temp={combo.temperature:.2f} "
                f"top_k={combo.top_k} top_p={combo.top_p} "
                f"penalty={combo.repetition_penalty:.2f} "
                f"no_repeat={combo.no_repeat_ngram_size}",
            flush=True,
        )

        for prompt_idx, prompt in enumerate(prompts_list):
            seed = base_seed + combo_idx * 100 + prompt_idx
            generated_text, token_ids, elapsed = generate_once(
                model,
                tokenizer,
                prompt,
                combo,
                max_new_tokens,
                device,
                seed,
            )

            metrics = compute_metrics(generated_text, token_ids)
            metrics["generation_time_s"] = elapsed

            for key, value in metrics.items():
                aggregate_metrics[key] = aggregate_metrics.get(key, 0.0) + value

            prompt_entries.append(
                {
                    "prompt": prompt,
                    "temperature": combo.temperature,
                    "top_k": combo.top_k,
                    "top_p": combo.top_p,
                    "repetition_penalty": combo.repetition_penalty,
                    "no_repeat_ngram_size": combo.no_repeat_ngram_size,
                    "seed": seed,
                    "generation_time_s": elapsed,
                    "generated_text": generated_text[:truncate],
                    "metrics": metrics,
                }
            )
            successful += 1

            print(
                f"     Prompt {prompt_idx + 1}/{prompts_total}: time={elapsed:.2f}s "
                f"tokens={len(token_ids)} seed={seed}",
                flush=True,
            )
            print(f"       ▶ Prompt: {prompt}", flush=True)
            print(
                "       ▶ Output: "
                f"{generated_text[:truncate] + ('…' if len(generated_text) > truncate else '')}",
                flush=True,
            )
            print(
                "       ▶ Metrics: "
                f"diversity={metrics.get('token_diversity', 0.0):.3f} "
                f"distinct2={metrics.get('distinct_2', 0.0):.3f} "
                f"distinct3={metrics.get('distinct_3', 0.0):.3f} "
                f"repeat={metrics.get('immediate_repeat_rate', 0.0):.3f}",
                flush=True,
            )

            # Append to history log after every prompt so progress is persisted.
            append_history_entry(
                history_log,
                {
                    "timestamp": datetime.now().isoformat(),
                    "combo_index": combo_idx,
                    "combo_total": combos_total,
                    "prompt_index": prompt_idx,
                    "prompt_total": prompts_total,
                    "prompt": prompt,
                    "max_new_tokens": max_new_tokens,
                    "combo": combo.as_dict(),
                    "seed": seed,
                    "metrics": metrics,
                    "generated_preview": generated_text[:truncate],
                },
            )

        if successful:
            averaged = {key: value / successful for key, value in aggregate_metrics.items()}
            combo_score = score_metrics(averaged)
        else:
            averaged = {}
            combo_score = float("nan")

        results.append(
            {
                "combo": combo.as_dict(),
                "score": combo_score,
                "metrics": averaged,
                "prompt_samples": prompt_entries,
            }
        )

    return results


def append_history_entry(path: Path, entry: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_cli()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    prompts = load_prompts(args)
    combos = build_combinations(args)

    model, tokenizer, cfg = load_model_and_tokenizer(args.config, args.checkpoint, device)

    output_dir = args.output_dir or Path(cfg.get("save", {}).get("out_dir", ".")) / "grid_search"
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "generation_grid_history.jsonl"

    results = evaluate_combinations(
        model,
        tokenizer,
        prompts,
        combos,
        args.max_new_tokens,
        device,
        args.seed,
        args.truncate_text,
        history_path,
    )

    results.sort(key=lambda entry: entry["score"], reverse=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": timestamp,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": device,
        "max_new_tokens": args.max_new_tokens,
        "prompts": prompts,
        "combinations": results,
    }

    output_path = output_dir / f"grid_search_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print("\n=== TOP 5 COMBINAISONS ===")
    for rank, entry in enumerate(results[:5], 1):
        combo = entry["combo"]
        metrics = entry["metrics"]
        print(
            f"#{rank} temp={combo['temperature']:.2f} top_k={combo['top_k']} top_p={combo['top_p']} | "
            f"score={entry['score']:.3f} diversite={metrics.get('token_diversity', 0.0):.3f} "
            f"distinct2={metrics.get('distinct_2', 0.0):.3f} repeats={metrics.get('immediate_repeat_rate', 0.0):.3f} "
            f"penalty={combo['repetition_penalty']:.2f} no_repeat={combo['no_repeat_ngram_size']}"
        )

    print(f"\nRapport complet sauvegardé dans: {output_path}")


if __name__ == "__main__":
    main()
