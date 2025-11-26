#!/usr/bin/env python3
"""
🧪 Évaluation du Reasoning Model sur Benchmarks

Évalue le modèle sur:
- GSM8K (mathématiques grade school)
- Accuracy, exact match, step analysis

Usage:
    python eval_reasoning.py --checkpoint checkpoints/reasoning/checkpoint_best.pt
    python eval_reasoning.py --checkpoint ckpt.pt --samples 100 --verbose
"""

import os
import sys
import argparse
import json
import re
import torch
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

# Import du modèle
from train_reasoning_simple import (
    ReasoningModel,
    TrainConfig,
    SimpleTokenizer,
    SPECIAL_TOKENS,
    load_gsm8k,
)


def extract_answer(text: str) -> str:
    """Extrait la réponse finale du texte généré."""
    # Chercher <answer>...</answer>
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Chercher après "=" ou "is"
    match = re.search(r'(?:=|is)\s*([\d,.\-]+)', text)
    if match:
        return match.group(1).replace(',', '')

    # Dernier nombre dans le texte
    numbers = re.findall(r'[\d,]+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')

    return text.strip()


def normalize_answer(answer: str) -> str:
    """Normalise une réponse pour comparaison."""
    # Enlever les virgules, espaces, $, etc.
    answer = answer.replace(',', '').replace('$', '').replace('%', '')
    answer = answer.strip().lower()

    # Extraire le nombre si possible
    match = re.search(r'([\d.\-]+)', answer)
    if match:
        try:
            # Convertir en float puis en string normalisé
            num = float(match.group(1))
            if num == int(num):
                return str(int(num))
            return f"{num:.2f}"
        except:
            pass

    return answer


def count_reasoning_steps(text: str) -> int:
    """Compte le nombre d'étapes de raisonnement."""
    return text.count('<step>')


def evaluate_model(
    model: ReasoningModel,
    tokenizer: SimpleTokenizer,
    data: List[Dict[str, Any]],
    device: str = "cuda",
    max_samples: int = -1,
    verbose: bool = False,
    use_tot: bool = False,
) -> Dict[str, Any]:
    """
    Évalue le modèle sur un dataset.

    Returns:
        dict avec métriques d'évaluation
    """
    model.eval()

    results = {
        "correct": 0,
        "total": 0,
        "exact_match": 0,
        "avg_steps": 0,
        "examples": [],
    }

    samples = data[:max_samples] if max_samples > 0 else data
    total_steps = 0

    for item in tqdm(samples, desc="Evaluating"):
        question = item["question"]
        gold_answer = normalize_answer(item.get("answer", ""))

        # Préparer le prompt
        prompt = f"Question: {question}\n<think>"
        input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)

        # Générer
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=256,
                temperature=0.0,  # Déterministe pour éval
            )

        generated_text = tokenizer.decode(output_ids[0].tolist())

        # Extraire la réponse
        pred_answer = normalize_answer(extract_answer(generated_text))

        # Comparer
        is_correct = pred_answer == gold_answer
        is_exact = item.get("answer", "").strip() == extract_answer(generated_text)

        # Stats
        num_steps = count_reasoning_steps(generated_text)
        total_steps += num_steps

        results["total"] += 1
        if is_correct:
            results["correct"] += 1
        if is_exact:
            results["exact_match"] += 1

        # Log exemple
        example = {
            "question": question,
            "gold": gold_answer,
            "pred": pred_answer,
            "correct": is_correct,
            "num_steps": num_steps,
        }

        if verbose:
            status = "✓" if is_correct else "✗"
            print(f"\n{status} Q: {question[:60]}...")
            print(f"  Gold: {gold_answer}")
            print(f"  Pred: {pred_answer}")
            print(f"  Steps: {num_steps}")

        results["examples"].append(example)

    # Calculer métriques
    results["accuracy"] = results["correct"] / max(1, results["total"])
    results["exact_match_rate"] = results["exact_match"] / max(1, results["total"])
    results["avg_steps"] = total_steps / max(1, results["total"])

    return results


def print_results(results: Dict[str, Any]):
    """Affiche les résultats."""
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS D'ÉVALUATION")
    print("=" * 60)
    print(f"Total samples:     {results['total']}")
    print(f"Correct:           {results['correct']}")
    print(f"Accuracy:          {results['accuracy']:.1%}")
    print(f"Exact match:       {results['exact_match_rate']:.1%}")
    print(f"Avg steps:         {results['avg_steps']:.1f}")
    print("=" * 60)

    # Analyse des erreurs
    errors = [e for e in results["examples"] if not e["correct"]]
    if errors:
        print(f"\n❌ Exemples d'erreurs ({len(errors)} total):")
        for e in errors[:5]:
            print(f"  Q: {e['question'][:50]}...")
            print(f"     Gold: {e['gold']} | Pred: {e['pred']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Reasoning Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--samples", type=int, default=-1, help="Max samples (-1 = all)")
    parser.add_argument("--verbose", action="store_true", help="Print each example")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Charger le modèle
    print(f"\n📂 Chargement du checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    config = ckpt.get("config", TrainConfig())
    if isinstance(config, dict):
        config = TrainConfig(**config)

    model = ReasoningModel(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Modèle chargé: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # Tokenizer
    tokenizer = SimpleTokenizer()

    # Charger les données
    print(f"\n📚 Chargement du dataset: {args.dataset}")
    if args.dataset == "gsm8k":
        try:
            from datasets import load_dataset
            ds = load_dataset("openai/gsm8k", "main", split="test")
            data = []
            for item in ds:
                parts = item["answer"].split("####")
                if len(parts) == 2:
                    data.append({
                        "question": item["question"],
                        "answer": parts[1].strip(),
                    })
            print(f"GSM8K test: {len(data)} exemples")
        except Exception as e:
            print(f"Erreur chargement GSM8K: {e}")
            return

    # Évaluer
    results = evaluate_model(
        model,
        tokenizer,
        data,
        device=device,
        max_samples=args.samples,
        verbose=args.verbose,
    )

    # Afficher
    print_results(results)

    # Sauvegarder
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Résultats sauvegardés: {args.output}")


if __name__ == "__main__":
    main()
