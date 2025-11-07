#!/usr/bin/env python3
"""
Script de monitoring en temps réel pour l'entraînement SLGA

Affiche les métriques d'entraînement en temps réel en lisant les logs TensorBoard.

Usage:
    python scripts/monitor.py
    python scripts/monitor.py --logdir out_slga/tensorboard
"""

import os
import sys
import time
import argparse
from pathlib import Path

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("❌ TensorBoard n'est pas installé")
    print("   Installez avec: pip install tensorboard")
    sys.exit(1)


def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_number(num, decimals=2):
    """Format number with K/M suffix"""
    if num >= 1e6:
        return f"{num/1e6:.{decimals}f}M"
    elif num >= 1e3:
        return f"{num/1e3:.{decimals}f}K"
    else:
        return f"{num:.{decimals}f}"


def get_latest_value(ea, tag):
    """Get latest value for a tag"""
    try:
        events = ea.Scalars(tag)
        if events:
            return events[-1].value
        return None
    except:
        return None


def get_latest_step(ea):
    """Get latest training step"""
    try:
        events = ea.Scalars("train/loss")
        if events:
            return events[-1].step
        return 0
    except:
        return 0


def monitor_training(logdir, refresh_rate=5):
    """Monitor training in real-time"""

    print("="*80)
    print("SLGA Training Monitor")
    print("="*80)
    print(f"Logdir: {logdir}")
    print(f"Refresh rate: {refresh_rate}s")
    print(f"Press Ctrl+C to exit")
    print("="*80 + "\n")

    logdir_path = Path(logdir)

    while True:
        try:
            # Find event files
            event_files = list(logdir_path.glob("events.out.tfevents.*"))

            if not event_files:
                print(f"⏳ Waiting for TensorBoard logs in {logdir}...")
                time.sleep(refresh_rate)
                continue

            # Use latest event file
            latest_file = max(event_files, key=lambda p: p.stat().st_mtime)

            # Load events
            ea = event_accumulator.EventAccumulator(str(latest_file))
            ea.Reload()

            # Get latest step
            step = get_latest_step(ea)

            if step == 0:
                print("⏳ Waiting for training to start...")
                time.sleep(refresh_rate)
                continue

            # Clear screen and display metrics
            clear_screen()

            print("="*80)
            print(f"SLGA Training Monitor - Step {step}")
            print("="*80)
            print()

            # Training metrics
            loss = get_latest_value(ea, "train/loss")
            ppl = get_latest_value(ea, "train/perplexity")
            lr = get_latest_value(ea, "train/learning_rate")
            grad_norm = get_latest_value(ea, "train/grad_norm")
            seq_len = get_latest_value(ea, "train/seq_len")
            global_weight = get_latest_value(ea, "train/global_weight")

            print("📊 Training Metrics:")
            print(f"  Loss:         {loss:.4f}" if loss else "  Loss:         N/A")
            print(f"  Perplexity:   {ppl:.2f}" if ppl else "  Perplexity:   N/A")
            print(f"  Learning Rate: {lr:.2e}" if lr else "  Learning Rate: N/A")
            print(f"  Grad Norm:    {grad_norm:.3f}" if grad_norm else "  Grad Norm:    N/A")
            print()

            # Model configuration
            print("⚙️  Model Configuration:")
            print(f"  Sequence Length: {int(seq_len)}" if seq_len else "  Sequence Length: N/A")
            print(f"  Global Weight:   {global_weight:.3f}" if global_weight else "  Global Weight:   N/A")
            print()

            # Loss components
            div_loss = get_latest_value(ea, "train/loss_diversity")
            spar_loss = get_latest_value(ea, "train/loss_sparsity")

            if div_loss or spar_loss:
                print("📉 Loss Components:")
                if div_loss:
                    print(f"  Diversity Loss: {div_loss:.6f}")
                if spar_loss:
                    print(f"  Sparsity Loss:  {spar_loss:.6f}")
                print()

            # Landmarks
            num_landmarks = get_latest_value(ea, "landmarks/num_selected")
            if num_landmarks:
                print("🎯 Landmarks:")
                print(f"  Selected: {int(num_landmarks)}")
                print()

            # Performance
            steps_per_sec = get_latest_value(ea, "perf/steps_per_sec")
            tokens_per_sec = get_latest_value(ea, "perf/tokens_per_sec")
            mem_alloc = get_latest_value(ea, "perf/gpu_memory_allocated_gb")
            mem_reserved = get_latest_value(ea, "perf/gpu_memory_reserved_gb")

            print("⚡ Performance:")
            print(f"  Steps/sec:   {steps_per_sec:.2f}" if steps_per_sec else "  Steps/sec:   N/A")
            print(f"  Tokens/sec:  {format_number(tokens_per_sec, 0)}" if tokens_per_sec else "  Tokens/sec:  N/A")
            if mem_alloc:
                print(f"  GPU Memory:  {mem_alloc:.1f}GB allocated")
            if mem_reserved:
                print(f"               {mem_reserved:.1f}GB reserved")
            print()

            # Validation metrics (if available)
            val_loss = get_latest_value(ea, "val/loss")
            val_ppl = get_latest_value(ea, "val/perplexity")

            if val_loss or val_ppl:
                print("✅ Validation:")
                if val_loss:
                    print(f"  Val Loss:       {val_loss:.4f}")
                if val_ppl:
                    print(f"  Val Perplexity: {val_ppl:.2f}")
                print()

            # Progress estimation
            max_steps = 100000  # From config
            progress_pct = (step / max_steps) * 100

            print("📈 Progress:")
            print(f"  Step: {step:,} / {max_steps:,} ({progress_pct:.1f}%)")

            # ETA estimation (very rough)
            if steps_per_sec and steps_per_sec > 0:
                remaining_steps = max_steps - step
                eta_seconds = remaining_steps / steps_per_sec
                eta_hours = eta_seconds / 3600
                print(f"  ETA: ~{eta_hours:.1f} hours")

            print()
            print("="*80)
            print(f"Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Refreshing in {refresh_rate}s... (Ctrl+C to exit)")

            time.sleep(refresh_rate)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(refresh_rate)


def main():
    parser = argparse.ArgumentParser(description='Monitor SLGA training in real-time')
    parser.add_argument('--logdir', type=str, default='out_slga/tensorboard',
                       help='TensorBoard log directory')
    parser.add_argument('--refresh', type=int, default=5,
                       help='Refresh rate in seconds (default: 5)')

    args = parser.parse_args()

    monitor_training(args.logdir, args.refresh)


if __name__ == "__main__":
    main()
