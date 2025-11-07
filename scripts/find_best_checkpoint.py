#!/usr/bin/env python3
"""
Script pour identifier le meilleur checkpoint selon validation loss.

Usage:
    python scripts/find_best_checkpoint.py --out-dir out_slga_fineweb
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import re

def parse_tensorboard_logs(out_dir: Path) -> Dict[int, Dict[str, float]]:
    """Parse TensorBoard events pour extraire validation metrics."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        tb_dir = out_dir / "tensorboard"
        if not tb_dir.exists():
            print(f"⚠️  TensorBoard directory not found: {tb_dir}")
            return {}

        # Trouver fichier events
        event_files = list(tb_dir.glob("events.out.tfevents.*"))
        if not event_files:
            print(f"⚠️  No TensorBoard event files found in {tb_dir}")
            return {}

        print(f"📊 Loading TensorBoard events from {event_files[0].name}...")

        ea = EventAccumulator(str(event_files[0]))
        ea.Reload()

        metrics = {}

        # Extraire val/loss
        if "val/loss" in ea.Tags()["scalars"]:
            for event in ea.Scalars("val/loss"):
                step = event.step
                val_loss = event.value

                if step not in metrics:
                    metrics[step] = {}
                metrics[step]["val_loss"] = val_loss

        # Extraire val/perplexity
        if "val/perplexity" in ea.Tags()["scalars"]:
            for event in ea.Scalars("val/perplexity"):
                step = event.step
                val_ppl = event.value

                if step not in metrics:
                    metrics[step] = {}
                metrics[step]["val_ppl"] = val_ppl

        # Extraire train/loss pour calculer gap
        if "train/loss" in ea.Tags()["scalars"]:
            train_losses = {e.step: e.value for e in ea.Scalars("train/loss")}

            for step in metrics:
                # Trouver train loss le plus proche
                closest_step = min(train_losses.keys(), key=lambda s: abs(s - step))
                if abs(closest_step - step) < 100:  # Tolérance 100 steps
                    metrics[step]["train_loss"] = train_losses[closest_step]
                    metrics[step]["gap"] = metrics[step]["val_loss"] - train_losses[closest_step]

        return metrics

    except ImportError:
        print("⚠️  tensorboard package not installed. Install with: pip install tensorboard")
        return {}
    except Exception as e:
        print(f"⚠️  Error loading TensorBoard logs: {e}")
        return {}


def find_available_checkpoints(out_dir: Path) -> List[int]:
    """Liste tous les checkpoints disponibles."""
    ckpt_dirs = list(out_dir.glob("ckpt_*"))

    steps = []
    for ckpt_dir in ckpt_dirs:
        # Extraire numéro de step du nom
        match = re.search(r"ckpt_(\d+)", ckpt_dir.name)
        if match:
            step = int(match.group(1))

            # Vérifier que model.pt existe
            if (ckpt_dir / "model.pt").exists():
                steps.append(step)

    return sorted(steps)


def analyze_checkpoints(out_dir: Path) -> None:
    """Analyse tous les checkpoints et trouve le meilleur."""

    print("=" * 80)
    print("🔍 ANALYSE DES CHECKPOINTS")
    print("=" * 80)
    print()

    # 1. Trouver checkpoints disponibles
    available_steps = find_available_checkpoints(out_dir)

    if not available_steps:
        print(f"❌ Aucun checkpoint trouvé dans {out_dir}")
        print(f"   Vérifiez que le training a créé des checkpoints dans ce dossier.")
        return

    print(f"✅ {len(available_steps)} checkpoints trouvés")
    print(f"   Steps: {available_steps[:5]}{'...' if len(available_steps) > 5 else ''}")
    print()

    # 2. Charger validation metrics depuis TensorBoard
    metrics = parse_tensorboard_logs(out_dir)

    if not metrics:
        print("⚠️  Pas de métriques de validation dans TensorBoard")
        print("   Solution: Attendez qu'au moins une validation ait été effectuée")
        print(f"   (se produit tous les eval_every steps, voir config)")
        print()
        print("📋 Checkpoints disponibles:")
        for step in available_steps:
            ckpt_dir = out_dir / f"ckpt_{step}"
            size = sum(f.stat().st_size for f in ckpt_dir.glob("**/*") if f.is_file()) / (1024**2)
            print(f"   • Step {step:>6,} : {ckpt_dir} ({size:.1f} MB)")
        return

    print(f"✅ {len(metrics)} validations trouvées dans TensorBoard")
    print()

    # 3. Filtrer métriques pour steps avec checkpoint
    checkpoint_metrics = {
        step: m for step, m in metrics.items() if step in available_steps
    }

    if not checkpoint_metrics:
        print("⚠️  Aucun checkpoint n'a de métriques de validation")
        print("   Solution: Attendez que le training atteigne un step de validation")
        return

    # 4. Trier par validation loss (meilleur = plus bas)
    ranked = sorted(
        checkpoint_metrics.items(),
        key=lambda x: x[1].get("val_loss", float("inf"))
    )

    # 5. Afficher résultats
    print("=" * 80)
    print("🏆 CLASSEMENT DES MODÈLES (par Validation Loss)")
    print("=" * 80)
    print()
    print(f"{'Rang':<6} {'Step':<10} {'Val Loss':<12} {'Val PPL':<12} {'Gap':<10} {'Status'}")
    print("-" * 80)

    for rank, (step, m) in enumerate(ranked[:10], 1):  # Top 10
        val_loss = m.get("val_loss", 0)
        val_ppl = m.get("val_ppl", 0)
        gap = m.get("gap", 0)

        # Indicateurs de qualité
        status = []
        if rank == 1:
            status.append("⭐ MEILLEUR")
        if gap < 0.5:
            status.append("✅ Excellent gap")
        elif gap > 1.5:
            status.append("⚠️  Overfitting")

        print(
            f"#{rank:<5} {step:<10,} {val_loss:<12.4f} {val_ppl:<12.1f} "
            f"{gap:+.4f}    {' '.join(status)}"
        )

    # 6. Résumé du meilleur
    best_step, best_metrics = ranked[0]

    print()
    print("=" * 80)
    print("✨ RECOMMANDATION FINALE")
    print("=" * 80)
    print()
    print(f"🏆 Meilleur modèle : checkpoint au step {best_step:,}")
    print(f"   📂 Chemin      : {out_dir / f'ckpt_{best_step}'}")
    print(f"   📉 Val Loss    : {best_metrics['val_loss']:.4f}")
    print(f"   📊 Val PPL     : {best_metrics.get('val_ppl', 0):.2f}")
    if "gap" in best_metrics:
        print(f"   📏 Train/Val Gap : {best_metrics['gap']:+.4f}")
    print()
    print("💡 Utiliser ce checkpoint pour inference/generation:")
    print(f"   python scripts/generate.py \\")
    print(f"     --checkpoint {out_dir / f'ckpt_{best_step}'} \\")
    print(f"     --config config/config_fineweb_edu.yaml")
    print()

    # 7. Statistiques globales
    if len(ranked) > 1:
        worst_step, worst_metrics = ranked[-1]
        improvement = (worst_metrics["val_loss"] - best_metrics["val_loss"]) / worst_metrics["val_loss"] * 100

        print("📈 Progrès total:")
        print(f"   • Premier checkpoint (step {worst_step:,}): Val Loss {worst_metrics['val_loss']:.4f}")
        print(f"   • Meilleur checkpoint (step {best_step:,}): Val Loss {best_metrics['val_loss']:.4f}")
        print(f"   • Amélioration: {improvement:.1f}%")
        print()


def main():
    parser = argparse.ArgumentParser(description="Find best checkpoint by validation loss")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="out_slga_fineweb",
        help="Output directory containing checkpoints",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if not out_dir.exists():
        print(f"❌ Directory not found: {out_dir}")
        print(f"   Vérifiez le chemin ou lancez d'abord le training.")
        return

    analyze_checkpoints(out_dir)


if __name__ == "__main__":
    main()
