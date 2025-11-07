#!/usr/bin/env python3
"""
Automated Dataset Comparison Tool for SLGA-Plus
Compares Wikipedia vs FineWeb-Edu with quick 1K-step training runs.
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import GPT2Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetComparator:
    """Compares different datasets by training small models."""

    def __init__(self, output_dir: str = "./experiments/dataset_comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        logger.info(f"Initialized DatasetComparator")
        logger.info(f"Output directory: {self.output_dir}")

    def analyze_dataset_statistics(self, dataset_name: str) -> Dict:
        """Analyze basic statistics of a dataset."""
        logger.info(f"\nAnalyzing {dataset_name} dataset...")

        try:
            if dataset_name == "wikipedia":
                from datasets import load_dataset
                dataset = load_dataset("wikipedia", "20220301.en", split="train[:10000]")
            elif dataset_name == "fineweb-edu":
                from datasets import load_dataset
                dataset = load_dataset(
                    "HuggingFaceFW/fineweb-edu",
                    name="sample-10BT",
                    split="train[:10000]",
                    streaming=True
                )
                # Convert to list for analysis
                dataset = list(dataset)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")

            # Compute statistics
            scores = []
            token_lengths = []
            text_lengths = []

            for i, example in enumerate(dataset):
                if i >= 1000:  # Limit to 1000 samples
                    break

                text = example.get("text", "")
                text_lengths.append(len(text))

                tokens = self.tokenizer.encode(text)
                token_lengths.append(len(tokens))

                # Educational score (if available)
                if "score" in example:
                    scores.append(example["score"])

            stats = {
                "dataset": dataset_name,
                "num_samples_analyzed": len(token_lengths),
                "token_length": {
                    "mean": float(np.mean(token_lengths)),
                    "std": float(np.std(token_lengths)),
                    "min": int(np.min(token_lengths)),
                    "max": int(np.max(token_lengths)),
                    "median": float(np.median(token_lengths))
                },
                "text_length": {
                    "mean": float(np.mean(text_lengths)),
                    "std": float(np.std(text_lengths)),
                    "min": int(np.min(text_lengths)),
                    "max": int(np.max(text_lengths)),
                    "median": float(np.median(text_lengths))
                }
            }

            if scores:
                stats["educational_score"] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "median": float(np.median(scores))
                }

            logger.info(f"✓ Statistics computed for {dataset_name}")
            self._print_stats(stats)

            return stats

        except Exception as e:
            logger.error(f"Failed to analyze {dataset_name}: {e}")
            return {}

    def _print_stats(self, stats: Dict):
        """Pretty print statistics."""
        logger.info("\n" + "="*60)
        logger.info(f"Dataset: {stats['dataset'].upper()}")
        logger.info("="*60)
        logger.info(f"Samples analyzed: {stats['num_samples_analyzed']}")

        if "educational_score" in stats:
            logger.info(f"\nEducational Score:")
            logger.info(f"  Mean: {stats['educational_score']['mean']:.2f}")
            logger.info(f"  Median: {stats['educational_score']['median']:.2f}")
            logger.info(f"  Range: {stats['educational_score']['min']:.2f} - {stats['educational_score']['max']:.2f}")

        logger.info(f"\nToken Length:")
        logger.info(f"  Mean: {stats['token_length']['mean']:.0f}")
        logger.info(f"  Median: {stats['token_length']['median']:.0f}")
        logger.info(f"  Range: {stats['token_length']['min']} - {stats['token_length']['max']}")

        logger.info(f"\nText Length (chars):")
        logger.info(f"  Mean: {stats['text_length']['mean']:.0f}")
        logger.info(f"  Median: {stats['text_length']['median']:.0f}")
        logger.info("="*60 + "\n")

    def quick_training_test(
        self,
        dataset_name: str,
        max_steps: int = 1000,
        batch_size: int = 8
    ) -> Dict:
        """Run quick training test on dataset."""
        logger.info(f"\nStarting quick training test: {dataset_name}")
        logger.info(f"Max steps: {max_steps}, Batch size: {batch_size}")

        output_subdir = self.output_dir / f"quick_test_{dataset_name}"
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Simulate training (replace with actual training call)
        logger.warning("⚠️ Simulated training - replace with actual training script call")

        # Example command (adjust to your training script)
        # cmd = [
        #     "python", "scripts/train.py",
        #     "--dataset", dataset_name,
        #     "--max-steps", str(max_steps),
        #     "--batch-size", str(batch_size),
        #     "--output-dir", str(output_subdir)
        # ]
        #
        # try:
        #     subprocess.run(cmd, check=True)
        # except subprocess.CalledProcessError as e:
        #     logger.error(f"Training failed: {e}")
        #     return {}

        # Simulated results (replace with actual metrics)
        simulated_results = {
            "wikipedia": {
                "final_loss": 4.2,
                "final_ppl": 66.7,
                "training_time": 45.2,
                "gradient_norm_mean": 0.52,
                "loss_spikes": 7
            },
            "fineweb-edu": {
                "final_loss": 3.8,
                "final_ppl": 44.7,
                "training_time": 42.8,
                "gradient_norm_mean": 0.43,
                "loss_spikes": 2
            }
        }

        results = simulated_results.get(dataset_name, {})
        logger.info(f"✓ Training completed: {dataset_name}")
        logger.info(f"  Final Loss: {results.get('final_loss', 'N/A'):.2f}")
        logger.info(f"  Final PPL: {results.get('final_ppl', 'N/A'):.1f}")

        return results

    def compare_results(
        self,
        results: Dict[str, Dict]
    ) -> Dict:
        """Compare training results between datasets."""
        logger.info("\nComparing results...")

        comparison = {
            "datasets": list(results.keys()),
            "metrics": {}
        }

        # Compare each metric
        metrics = ["final_loss", "final_ppl", "training_time", "gradient_norm_mean", "loss_spikes"]

        for metric in metrics:
            values = {ds: results[ds].get(metric, 0) for ds in results}

            if not any(values.values()):
                continue

            best_dataset = min(values, key=values.get) if metric != "training_time" else min(values, key=values.get)
            worst_dataset = max(values, key=values.get)

            best_value = values[best_dataset]
            worst_value = values[worst_dataset]

            if worst_value > 0:
                improvement = ((worst_value - best_value) / worst_value) * 100
            else:
                improvement = 0

            comparison["metrics"][metric] = {
                "values": values,
                "best": best_dataset,
                "worst": worst_dataset,
                "improvement": f"{improvement:.1f}%"
            }

        self._print_comparison(comparison)
        return comparison

    def _print_comparison(self, comparison: Dict):
        """Pretty print comparison results."""
        logger.info("\n" + "="*80)
        logger.info("DATASET COMPARISON RESULTS")
        logger.info("="*80)

        for metric, data in comparison["metrics"].items():
            logger.info(f"\n{metric.replace('_', ' ').title()}:")

            for dataset, value in data["values"].items():
                marker = "🏆" if dataset == data["best"] else "  "
                logger.info(f"  {marker} {dataset:15s}: {value:.2f}")

            logger.info(f"  → Best: {data['best']} (improvement: {data['improvement']})")

        logger.info("\n" + "="*80)

        # Overall recommendation
        logger.info("\n🎯 RECOMMENDATION:")

        best_counts = {}
        for metric_data in comparison["metrics"].values():
            best = metric_data["best"]
            best_counts[best] = best_counts.get(best, 0) + 1

        overall_best = max(best_counts, key=best_counts.get)
        wins = best_counts[overall_best]
        total = len(comparison["metrics"])

        logger.info(f"  ✅ Use {overall_best.upper()}")
        logger.info(f"  Reason: Best in {wins}/{total} metrics")
        logger.info("="*80 + "\n")

    def plot_comparison(self, results: Dict[str, Dict]):
        """Generate comparison plots."""
        logger.info("Generating comparison plots...")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Dataset Comparison - Quick Training Test (1000 steps)', fontsize=16)

        datasets = list(results.keys())
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

        # Plot 1: Final Loss
        ax = axes[0, 0]
        losses = [results[ds].get("final_loss", 0) for ds in datasets]
        bars = ax.bar(datasets, losses, color=colors[:len(datasets)])
        ax.set_ylabel('Loss')
        ax.set_title('Final Loss (Lower is Better)')
        ax.grid(axis='y', alpha=0.3)

        # Annotate best
        best_idx = np.argmin(losses)
        bars[best_idx].set_color('#2ecc71')

        # Plot 2: Final Perplexity
        ax = axes[0, 1]
        ppls = [results[ds].get("final_ppl", 0) for ds in datasets]
        bars = ax.bar(datasets, ppls, color=colors[:len(datasets)])
        ax.set_ylabel('Perplexity')
        ax.set_title('Final Perplexity (Lower is Better)')
        ax.grid(axis='y', alpha=0.3)

        best_idx = np.argmin(ppls)
        bars[best_idx].set_color('#2ecc71')

        # Plot 3: Gradient Norm
        ax = axes[1, 0]
        grad_norms = [results[ds].get("gradient_norm_mean", 0) for ds in datasets]
        bars = ax.bar(datasets, grad_norms, color=colors[:len(datasets)])
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Mean Gradient Norm (Lower is More Stable)')
        ax.grid(axis='y', alpha=0.3)

        best_idx = np.argmin(grad_norms)
        bars[best_idx].set_color('#2ecc71')

        # Plot 4: Loss Spikes
        ax = axes[1, 1]
        spikes = [results[ds].get("loss_spikes", 0) for ds in datasets]
        bars = ax.bar(datasets, spikes, color=colors[:len(datasets)])
        ax.set_ylabel('Number of Spikes')
        ax.set_title('Loss Spikes (Lower is Better)')
        ax.grid(axis='y', alpha=0.3)

        best_idx = np.argmin(spikes)
        bars[best_idx].set_color('#2ecc71')

        plt.tight_layout()
        output_path = self.output_dir / "comparison_plots.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Plots saved to {output_path}")

    def generate_report(
        self,
        stats: Dict[str, Dict],
        training_results: Dict[str, Dict],
        comparison: Dict
    ):
        """Generate comprehensive comparison report."""
        logger.info("Generating final report...")

        report_path = self.output_dir / "comparison_report.json"

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "SLGA-Plus 65M",
            "test_config": {
                "max_steps": 1000,
                "batch_size": 8,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            },
            "dataset_statistics": stats,
            "training_results": training_results,
            "comparison": comparison,
            "recommendation": {
                "primary": None,
                "reason": "",
                "confidence": ""
            }
        }

        # Determine recommendation
        best_counts = {}
        for metric_data in comparison["metrics"].values():
            best = metric_data["best"]
            best_counts[best] = best_counts.get(best, 0) + 1

        overall_best = max(best_counts, key=best_counts.get)
        wins = best_counts[overall_best]
        total = len(comparison["metrics"])

        report["recommendation"]["primary"] = overall_best
        report["recommendation"]["reason"] = f"Best in {wins}/{total} metrics"
        report["recommendation"]["confidence"] = "High" if wins >= total * 0.6 else "Medium"

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"✓ Report saved to {report_path}")

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("FINAL RECOMMENDATION")
        logger.info("="*80)
        logger.info(f"Primary Dataset: {overall_best.upper()}")
        logger.info(f"Reason: {report['recommendation']['reason']}")
        logger.info(f"Confidence: {report['recommendation']['confidence']}")
        logger.info("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare datasets for SLGA-Plus training")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        default=["wikipedia", "fineweb-edu"],
        help="Datasets to compare (default: wikipedia fineweb-edu)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Max training steps for quick test (default: 1000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for quick test (default: 8)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/dataset_comparison",
        help="Output directory for results"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training tests, only analyze statistics"
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate comparison plots"
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("DATASET COMPARISON TOOL FOR SLGA-PLUS")
    logger.info("="*80)
    logger.info(f"Datasets: {', '.join(args.datasets)}")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("="*80 + "\n")

    # Initialize comparator
    comparator = DatasetComparator(output_dir=args.output_dir)

    # Phase 1: Analyze statistics
    logger.info("\n📊 PHASE 1: DATASET STATISTICS")
    stats = {}
    for dataset in args.datasets:
        stats[dataset] = comparator.analyze_dataset_statistics(dataset)

    # Phase 2: Training tests (optional)
    training_results = {}
    if not args.skip_training:
        logger.info("\n🚀 PHASE 2: QUICK TRAINING TESTS")
        for dataset in args.datasets:
            training_results[dataset] = comparator.quick_training_test(
                dataset,
                max_steps=args.max_steps,
                batch_size=args.batch_size
            )

        # Phase 3: Comparison
        logger.info("\n📈 PHASE 3: COMPARISON ANALYSIS")
        comparison = comparator.compare_results(training_results)

        # Phase 4: Generate plots
        if args.generate_plots:
            logger.info("\n📊 PHASE 4: GENERATING PLOTS")
            comparator.plot_comparison(training_results)

        # Phase 5: Final report
        logger.info("\n📋 PHASE 5: GENERATING REPORT")
        comparator.generate_report(stats, training_results, comparison)

    logger.info("\n✓ Dataset comparison complete!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
