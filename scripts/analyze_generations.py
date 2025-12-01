#!/usr/bin/env python3
"""
Analyse complète de l'historique des générations SLGA.

Génère des rapports détaillés et visualisations pour évaluer:
- Évolution de la qualité au fil du temps
- Comparaison entre checkpoints
- Identification des problèmes récurrents
- Statistiques détaillées par catégorie

Usage:
    python scripts/analyze_generations.py out_slga/
    python scripts/analyze_generations.py out_slga/ --compare-checkpoints
    python scripts/analyze_generations.py out_slga/ --export-report reports/
    python scripts/analyze_generations.py out_slga/ --tensorboard logs/
"""

import os
import sys
import json
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Optional imports for visualizations
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


# ======================================================================
# Data Loading
# ======================================================================

def load_generation_history(output_dir: str) -> List[Dict[str, Any]]:
    """Charge l'historique des générations."""
    history_file = Path(output_dir) / "generation_history.jsonl"

    if not history_file.exists():
        print(f"No history found at {history_file}")
        return []

    entries = []
    with open(history_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")

    return entries


# ======================================================================
# Analysis Functions
# ======================================================================

def compute_statistics(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcule des statistiques détaillées sur les générations."""
    if not entries:
        return {}

    # Extraire les métriques
    scores = []
    ppls = []
    distinct_2 = []
    rep_2 = []
    coherence = []
    gibberish = []
    generation_times = []
    tokens_per_second = []

    for e in entries:
        eval_data = e.get("evaluation", {})

        if isinstance(eval_data.get("overall_score"), (int, float)):
            scores.append(eval_data["overall_score"])

        ppl_data = eval_data.get("perplexity", {})
        if isinstance(ppl_data.get("sequence_perplexity"), (int, float)):
            ppls.append(ppl_data["sequence_perplexity"])

        div_data = eval_data.get("diversity", {})
        if isinstance(div_data.get("distinct_2"), (int, float)):
            distinct_2.append(div_data["distinct_2"])

        rep_data = eval_data.get("repetition", {})
        if isinstance(rep_data.get("rep_2"), (int, float)):
            rep_2.append(rep_data["rep_2"])

        coh_data = eval_data.get("coherence", {})
        if isinstance(coh_data.get("coherence_score"), (int, float)):
            coherence.append(coh_data["coherence_score"])
        if isinstance(coh_data.get("gibberish_score"), (int, float)):
            gibberish.append(coh_data["gibberish_score"])

        if isinstance(e.get("generation_time_seconds"), (int, float)):
            generation_times.append(e["generation_time_seconds"])

        if isinstance(eval_data.get("tokens_per_second"), (int, float)):
            tokens_per_second.append(eval_data["tokens_per_second"])

    def safe_stats(values: List[float], name: str) -> Dict[str, float]:
        if not values:
            return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
        arr = np.array(values)
        return {
            "count": len(arr),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
        }

    # Distribution des notes
    grades = [e.get("evaluation", {}).get("quality_grade") for e in entries
              if e.get("evaluation", {}).get("quality_grade")]
    grade_dist = dict(Counter(grades))

    # Issues fréquentes
    all_issues = []
    for e in entries:
        issues = e.get("evaluation", {}).get("issues", [])
        all_issues.extend(issues)
    issue_counts = dict(Counter(all_issues).most_common(10))

    return {
        "num_generations": len(entries),
        "overall_score": safe_stats(scores, "score"),
        "perplexity": safe_stats(ppls, "ppl"),
        "distinct_2": safe_stats(distinct_2, "distinct_2"),
        "repetition_2": safe_stats(rep_2, "rep_2"),
        "coherence": safe_stats(coherence, "coherence"),
        "gibberish": safe_stats(gibberish, "gibberish"),
        "generation_time": safe_stats(generation_times, "gen_time"),
        "tokens_per_second": safe_stats(tokens_per_second, "tok/s"),
        "grade_distribution": grade_dist,
        "top_issues": issue_counts,
    }


def analyze_by_checkpoint(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Analyse par checkpoint."""
    by_checkpoint: Dict[str, List[Dict]] = defaultdict(list)

    for e in entries:
        step = e.get("checkpoint", {}).get("step")
        if step is not None:
            by_checkpoint[f"step_{step}"].append(e)
        else:
            by_checkpoint["unknown"].append(e)

    results = {}
    for checkpoint, ckpt_entries in by_checkpoint.items():
        results[checkpoint] = compute_statistics(ckpt_entries)

    return results


def analyze_evolution(entries: List[Dict[str, Any]]) -> Dict[str, List]:
    """Analyse l'évolution temporelle des métriques."""
    evolution = {
        "timestamps": [],
        "scores": [],
        "ppls": [],
        "distinct_2": [],
        "rep_2": [],
        "steps": [],
    }

    for e in entries:
        ts = e.get("timestamp")
        eval_data = e.get("evaluation", {})
        step = e.get("checkpoint", {}).get("step")

        if ts:
            try:
                evolution["timestamps"].append(datetime.fromisoformat(ts))
            except (ValueError, TypeError):
                evolution["timestamps"].append(None)
        else:
            evolution["timestamps"].append(None)

        evolution["steps"].append(step)

        score = eval_data.get("overall_score")
        evolution["scores"].append(score if isinstance(score, (int, float)) else None)

        ppl = eval_data.get("perplexity", {}).get("sequence_perplexity")
        evolution["ppls"].append(ppl if isinstance(ppl, (int, float)) else None)

        d2 = eval_data.get("diversity", {}).get("distinct_2")
        evolution["distinct_2"].append(d2 if isinstance(d2, (int, float)) else None)

        r2 = eval_data.get("repetition", {}).get("rep_2")
        evolution["rep_2"].append(r2 if isinstance(r2, (int, float)) else None)

    return evolution


def find_best_worst_generations(entries: List[Dict[str, Any]], n: int = 5) -> Dict[str, List]:
    """Trouve les meilleures et pires générations."""
    scored_entries = [
        (e, e.get("evaluation", {}).get("overall_score", 0))
        for e in entries
        if isinstance(e.get("evaluation", {}).get("overall_score"), (int, float))
    ]

    scored_entries.sort(key=lambda x: x[1], reverse=True)

    best = scored_entries[:n]
    worst = scored_entries[-n:][::-1]

    return {
        "best": [{"entry": e, "score": s} for e, s in best],
        "worst": [{"entry": e, "score": s} for e, s in worst],
    }


# ======================================================================
# Report Generation
# ======================================================================

def print_text_report(stats: Dict[str, Any], by_checkpoint: Dict[str, Dict], best_worst: Dict):
    """Affiche un rapport texte détaillé."""
    print("\n" + "=" * 80)
    print("RAPPORT D'ANALYSE DES GÉNÉRATIONS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print(f"\nNombre total de générations: {stats['num_generations']}")

    # Statistiques globales
    print("\n" + "-" * 40)
    print("STATISTIQUES GLOBALES")
    print("-" * 40)

    metrics = [
        ("Score Global", "overall_score", "/100"),
        ("Perplexité", "perplexity", ""),
        ("Distinct-2", "distinct_2", ""),
        ("Répétition-2", "repetition_2", ""),
        ("Cohérence", "coherence", ""),
        ("Charabia", "gibberish", ""),
        ("Temps génération", "generation_time", "s"),
        ("Vitesse", "tokens_per_second", "tok/s"),
    ]

    for name, key, unit in metrics:
        data = stats.get(key, {})
        if data.get("count", 0) > 0:
            print(f"\n{name}:")
            print(f"  Moyenne: {data['mean']:.2f}{unit}")
            print(f"  Écart-type: {data['std']:.2f}")
            print(f"  Min/Max: {data['min']:.2f} - {data['max']:.2f}")
            print(f"  Médiane: {data['median']:.2f}")

    # Distribution des notes
    print("\n" + "-" * 40)
    print("DISTRIBUTION DES NOTES")
    print("-" * 40)
    for grade in ["A", "B", "C", "D", "F"]:
        count = stats.get("grade_distribution", {}).get(grade, 0)
        bar = "" * count
        print(f"  {grade}: {count:3d} {bar}")

    # Issues fréquentes
    if stats.get("top_issues"):
        print("\n" + "-" * 40)
        print("PROBLÈMES FRÉQUENTS")
        print("-" * 40)
        for issue, count in stats["top_issues"].items():
            print(f"  [{count:2d}x] {issue[:60]}")

    # Par checkpoint
    if len(by_checkpoint) > 1:
        print("\n" + "-" * 40)
        print("PAR CHECKPOINT")
        print("-" * 40)
        print(f"\n{'Checkpoint':<20} {'N':<6} {'Score':<10} {'PPL':<12} {'Distinct-2':<10}")
        print("-" * 60)

        sorted_ckpts = sorted(
            by_checkpoint.items(),
            key=lambda x: int(x[0].split("_")[-1]) if x[0] != "unknown" else 999999
        )

        for ckpt, ckpt_stats in sorted_ckpts:
            n = ckpt_stats.get("num_generations", 0)
            score = ckpt_stats.get("overall_score", {}).get("mean", 0)
            ppl = ckpt_stats.get("perplexity", {}).get("mean", 0)
            d2 = ckpt_stats.get("distinct_2", {}).get("mean", 0)
            print(f"{ckpt:<20} {n:<6} {score:<10.1f} {ppl:<12.2f} {d2:<10.3f}")

    # Meilleures/Pires générations
    print("\n" + "-" * 40)
    print("MEILLEURES GÉNÉRATIONS")
    print("-" * 40)
    for i, item in enumerate(best_worst.get("best", []), 1):
        e = item["entry"]
        score = item["score"]
        prompt = e.get("prompt", "")[:40]
        gen = e.get("generated_text", "")[:50]
        print(f"\n{i}. Score: {score:.1f}/100")
        print(f"   Prompt: {prompt}...")
        print(f"   Output: {gen}...")

    print("\n" + "-" * 40)
    print("PIRES GÉNÉRATIONS")
    print("-" * 40)
    for i, item in enumerate(best_worst.get("worst", []), 1):
        e = item["entry"]
        score = item["score"]
        prompt = e.get("prompt", "")[:40]
        gen = e.get("generated_text", "")[:50]
        issues = e.get("evaluation", {}).get("issues", [])
        print(f"\n{i}. Score: {score:.1f}/100")
        print(f"   Prompt: {prompt}...")
        print(f"   Output: {gen}...")
        if issues:
            print(f"   Issues: {issues[0]}")


def export_json_report(
    stats: Dict[str, Any],
    by_checkpoint: Dict[str, Dict],
    evolution: Dict[str, List],
    best_worst: Dict,
    output_path: Path,
):
    """Exporte le rapport au format JSON."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "statistics": stats,
        "by_checkpoint": by_checkpoint,
        "best_worst": {
            "best": [
                {"score": item["score"], "prompt": item["entry"].get("prompt", "")[:100]}
                for item in best_worst.get("best", [])
            ],
            "worst": [
                {"score": item["score"], "prompt": item["entry"].get("prompt", "")[:100]}
                for item in best_worst.get("worst", [])
            ],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nJSON report saved to {output_path}")


def generate_visualizations(
    evolution: Dict[str, List],
    by_checkpoint: Dict[str, Dict],
    output_dir: Path,
):
    """Génère des visualisations matplotlib."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not installed. Skipping visualizations.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Évolution du score dans le temps
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Score evolution
    ax = axes[0, 0]
    valid_scores = [(t, s) for t, s in zip(evolution["timestamps"], evolution["scores"])
                    if t is not None and s is not None]
    if valid_scores:
        times, scores = zip(*valid_scores)
        ax.plot(times, scores, 'b-', marker='o', markersize=3, alpha=0.7)
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Seuil C/D')
        ax.axhline(y=70, color='g', linestyle='--', alpha=0.5, label='Seuil B/C')
        ax.set_xlabel('Date')
        ax.set_ylabel('Score')
        ax.set_title('Évolution du Score Global')
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Perplexity evolution
    ax = axes[0, 1]
    valid_ppls = [(t, p) for t, p in zip(evolution["timestamps"], evolution["ppls"])
                  if t is not None and p is not None]
    if valid_ppls:
        times, ppls = zip(*valid_ppls)
        ax.semilogy(times, ppls, 'r-', marker='o', markersize=3, alpha=0.7)
        ax.axhline(y=100, color='orange', linestyle='--', alpha=0.5, label='Seuil bon')
        ax.set_xlabel('Date')
        ax.set_ylabel('Perplexité (log)')
        ax.set_title('Évolution de la Perplexité')
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Diversity evolution
    ax = axes[1, 0]
    valid_d2 = [(t, d) for t, d in zip(evolution["timestamps"], evolution["distinct_2"])
                if t is not None and d is not None]
    if valid_d2:
        times, d2s = zip(*valid_d2)
        ax.plot(times, d2s, 'g-', marker='o', markersize=3, alpha=0.7)
        ax.set_xlabel('Date')
        ax.set_ylabel('Distinct-2')
        ax.set_title('Évolution de la Diversité')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Repetition evolution
    ax = axes[1, 1]
    valid_r2 = [(t, r) for t, r in zip(evolution["timestamps"], evolution["rep_2"])
                if t is not None and r is not None]
    if valid_r2:
        times, r2s = zip(*valid_r2)
        ax.plot(times, r2s, 'm-', marker='o', markersize=3, alpha=0.7)
        ax.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='Seuil répétition')
        ax.set_xlabel('Date')
        ax.set_ylabel('Répétition-2')
        ax.set_title('Évolution de la Répétition')
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    fig.savefig(output_dir / "evolution.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'evolution.png'}")

    # 2. Comparaison par checkpoint
    if len(by_checkpoint) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        checkpoints = []
        scores = []
        ppls = []

        sorted_ckpts = sorted(
            by_checkpoint.items(),
            key=lambda x: int(x[0].split("_")[-1]) if x[0] != "unknown" and "_" in x[0] else 999999
        )

        for ckpt, stats in sorted_ckpts:
            if stats.get("overall_score", {}).get("count", 0) > 0:
                checkpoints.append(ckpt.replace("step_", ""))
                scores.append(stats["overall_score"]["mean"])
                ppl = stats.get("perplexity", {}).get("mean", 0)
                ppls.append(ppl if ppl > 0 else None)

        if checkpoints:
            ax = axes[0]
            bars = ax.bar(checkpoints, scores, color='steelblue', alpha=0.7)
            ax.set_xlabel('Checkpoint (Step)')
            ax.set_ylabel('Score Moyen')
            ax.set_title('Score par Checkpoint')
            ax.axhline(y=50, color='r', linestyle='--', alpha=0.5)
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{score:.0f}', ha='center', va='bottom', fontsize=8)

            ax = axes[1]
            valid_ppls = [(c, p) for c, p in zip(checkpoints, ppls) if p is not None]
            if valid_ppls:
                ckpts, ppl_vals = zip(*valid_ppls)
                bars = ax.bar(ckpts, ppl_vals, color='coral', alpha=0.7)
                ax.set_xlabel('Checkpoint (Step)')
                ax.set_ylabel('Perplexité Moyenne')
                ax.set_title('Perplexité par Checkpoint')
                ax.set_yscale('log')

            plt.tight_layout()
            fig.savefig(output_dir / "by_checkpoint.png", dpi=150)
            plt.close()
            print(f"Saved: {output_dir / 'by_checkpoint.png'}")


def export_to_tensorboard(
    entries: List[Dict[str, Any]],
    stats: Dict[str, Any],
    log_dir: Path,
):
    """Exporte les métriques vers TensorBoard."""
    if not HAS_TENSORBOARD:
        print("TensorBoard not installed. Skipping TensorBoard export.")
        return

    writer = SummaryWriter(log_dir=str(log_dir))

    # Log chaque génération comme un step
    for i, e in enumerate(entries):
        eval_data = e.get("evaluation", {})
        step = e.get("checkpoint", {}).get("step") or i

        if isinstance(eval_data.get("overall_score"), (int, float)):
            writer.add_scalar("generation/overall_score", eval_data["overall_score"], step)

        ppl_data = eval_data.get("perplexity", {})
        if isinstance(ppl_data.get("sequence_perplexity"), (int, float)):
            writer.add_scalar("generation/perplexity", ppl_data["sequence_perplexity"], step)

        div_data = eval_data.get("diversity", {})
        for key in ["distinct_1", "distinct_2", "distinct_3", "lexical_entropy"]:
            if isinstance(div_data.get(key), (int, float)):
                writer.add_scalar(f"generation/diversity/{key}", div_data[key], step)

        rep_data = eval_data.get("repetition", {})
        for key in ["rep_1", "rep_2", "rep_3"]:
            if isinstance(rep_data.get(key), (int, float)):
                writer.add_scalar(f"generation/repetition/{key}", rep_data[key], step)

        coh_data = eval_data.get("coherence", {})
        if isinstance(coh_data.get("coherence_score"), (int, float)):
            writer.add_scalar("generation/coherence", coh_data["coherence_score"], step)
        if isinstance(coh_data.get("gibberish_score"), (int, float)):
            writer.add_scalar("generation/gibberish", coh_data["gibberish_score"], step)

    # Histogrammes des distributions
    scores = [e.get("evaluation", {}).get("overall_score") for e in entries
              if isinstance(e.get("evaluation", {}).get("overall_score"), (int, float))]
    if scores:
        writer.add_histogram("generation/score_distribution", np.array(scores), 0)

    writer.close()
    print(f"\nTensorBoard logs saved to {log_dir}")
    print(f"Run: tensorboard --logdir={log_dir}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyse des générations SLGA")
    parser.add_argument("output_dir", type=str, help="Dossier contenant generation_history.jsonl")
    parser.add_argument("--export-report", type=str, help="Exporter le rapport JSON")
    parser.add_argument("--visualizations", type=str, help="Dossier pour les visualisations")
    parser.add_argument("--tensorboard", type=str, help="Dossier pour les logs TensorBoard")
    parser.add_argument("--compare-checkpoints", action="store_true", help="Afficher comparaison par checkpoint")
    parser.add_argument("--best-worst", type=int, default=5, help="Nombre de meilleures/pires à afficher")

    args = parser.parse_args()

    # Charger l'historique
    entries = load_generation_history(args.output_dir)

    if not entries:
        print("No entries found. Run some generations with --evaluate first.")
        return

    print(f"Loaded {len(entries)} generation entries")

    # Analyses
    stats = compute_statistics(entries)
    by_checkpoint = analyze_by_checkpoint(entries)
    evolution = analyze_evolution(entries)
    best_worst = find_best_worst_generations(entries, args.best_worst)

    # Rapport texte
    print_text_report(stats, by_checkpoint, best_worst)

    # Export JSON
    if args.export_report:
        export_json_report(
            stats, by_checkpoint, evolution, best_worst,
            Path(args.export_report)
        )

    # Visualisations
    if args.visualizations:
        generate_visualizations(evolution, by_checkpoint, Path(args.visualizations))

    # TensorBoard
    if args.tensorboard:
        export_to_tensorboard(entries, stats, Path(args.tensorboard))


if __name__ == "__main__":
    main()
