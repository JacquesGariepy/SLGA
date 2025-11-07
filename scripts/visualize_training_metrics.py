#!/usr/bin/env python3
"""
Visualize training metrics from training.log

Extracts and displays key metrics progression:
- Loss curve
- Learning rate schedule
- Spacing/sparsity losses
- Validation gap

Usage:
    python scripts/visualize_training_metrics.py [--log-file training.log]
"""

import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys


def parse_training_log(log_file: Path) -> Dict[str, List]:
    """Parse training log and extract metrics."""

    metrics = {
        'steps': [],
        'loss': [],
        'ppl': [],
        'lr': [],
        'spacing': [],
        'sparsity': [],
        'grad_norm': [],
        'val_loss': [],
        'val_steps': []
    }

    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Strip ANSI color codes
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    content_clean = ansi_escape.sub('', content)

    # Extract step metrics (every 10 steps) - they span 2 lines
    step_pattern = r'📊 Step ([\d,]+) │ Loss: (\d+\.\d+).*?│ PPL: (\d+\.\d+)'
    detail_pattern = r'LR: ([\d.e+-]+) │ Grad: ([\d.]+)'

    lines = content_clean.split('\n')
    for i, line in enumerate(lines):
        step_match = re.search(step_pattern, line)
        if step_match and i + 1 < len(lines):
            # Get details from next line
            detail_match = re.search(detail_pattern, lines[i + 1])
            if detail_match:
                step = int(step_match.group(1).replace(',', ''))
                loss = float(step_match.group(2))
                ppl = float(step_match.group(3))
                lr = float(detail_match.group(1))
                grad = float(detail_match.group(2))

                metrics['steps'].append(step)
                metrics['loss'].append(loss)
                metrics['ppl'].append(ppl)
                metrics['lr'].append(lr)
                metrics['grad_norm'].append(grad)

    # Extract spacing and sparsity
    spacing_pattern = r'Spacing: .*?(\d+\.\d+)'
    for i, match in enumerate(re.finditer(spacing_pattern, content_clean)):
        if i < len(metrics['steps']):
            metrics['spacing'].append(float(match.group(1)))

    sparsity_pattern = r'Sparsity: .*?(\d+\.\d+)'
    for i, match in enumerate(re.finditer(sparsity_pattern, content_clean)):
        if i < len(metrics['steps']):
            metrics['sparsity'].append(float(match.group(1)))

    # Extract validation metrics
    val_pattern = r'VALIDATION @ Step ([\d,]+)[^\n]*?Val Loss: ([\d.]+)'
    for match in re.finditer(val_pattern, content_clean):
        step = int(match.group(1).replace(',', ''))
        val_loss = float(match.group(2))
        metrics['val_steps'].append(step)
        metrics['val_loss'].append(val_loss)

    return metrics


def print_ascii_chart(data: List[float], width: int = 60, height: int = 10,
                      title: str = "", y_label: str = "") -> None:
    """Print an ASCII chart of the data."""

    if not data:
        print(f"  {title}: No data")
        return

    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val if max_val != min_val else 1

    print(f"\n{title}")
    print("─" * (width + 10))

    # Print chart from top to bottom
    for i in range(height, -1, -1):
        threshold = min_val + (range_val * i / height)

        # Y-axis label
        print(f"{threshold:7.2f} ┤", end="")

        # Plot line
        for j in range(0, len(data), max(1, len(data) // width)):
            if abs(data[j] - threshold) < range_val / height:
                print("●", end="")
            elif data[j] > threshold:
                print("│", end="")
            else:
                print(" ", end="")
        print()

    # X-axis
    print("        └" + "─" * width)
    print(f"        0{' ' * (width//2-5)}steps{' ' * (width//2-5)}{len(data)*10}")


def print_statistics(metrics: Dict[str, List]) -> None:
    """Print summary statistics."""

    print("\n" + "=" * 80)
    print("📊 TRAINING STATISTICS SUMMARY")
    print("=" * 80)

    if metrics['steps']:
        print(f"\n📈 Training Progress:")
        print(f"   Total steps: {metrics['steps'][-1]}")
        print(f"   Initial loss: {metrics['loss'][0]:.4f}")
        print(f"   Final loss: {metrics['loss'][-1]:.4f}")
        print(f"   Best loss: {min(metrics['loss']):.4f} @ step {metrics['steps'][metrics['loss'].index(min(metrics['loss']))]}")
        print(f"   Improvement: {(metrics['loss'][0] - metrics['loss'][-1]) / metrics['loss'][0] * 100:.1f}%")

    if metrics['lr']:
        print(f"\n🎓 Learning Rate:")
        print(f"   Initial LR: {metrics['lr'][0]:.2e}")
        print(f"   Peak LR: {max(metrics['lr']):.2e}")
        print(f"   Final LR: {metrics['lr'][-1]:.2e}")

    if metrics['spacing']:
        print(f"\n📏 Spacing Loss:")
        print(f"   Min: {min(metrics['spacing']):.4f}")
        print(f"   Max: {max(metrics['spacing']):.4f}")
        print(f"   Mean: {sum(metrics['spacing']) / len(metrics['spacing']):.4f}")
        print(f"   ⚠️  Target range: 0.5-1.5")

    if metrics['sparsity']:
        unique_sparsity = len(set(metrics['sparsity']))
        print(f"\n🔍 Sparsity Loss:")
        print(f"   Value: {metrics['sparsity'][0]:.4f} (constant: {unique_sparsity == 1})")
        if all(s == 0.0 for s in metrics['sparsity']):
            print(f"   ✅ Correctly disabled")

    if metrics['val_loss']:
        print(f"\n🎯 Validation:")
        for i, (step, val_loss) in enumerate(zip(metrics['val_steps'], metrics['val_loss'])):
            train_loss_idx = next((j for j, s in enumerate(metrics['steps']) if s == step), None)
            if train_loss_idx is not None:
                train_loss = metrics['loss'][train_loss_idx]
                gap = val_loss - train_loss
                status = "✅" if abs(gap) < 0.1 else "⚠️"
                print(f"   Step {step:4d}: Val={val_loss:.4f}, Train={train_loss:.4f}, Gap={gap:+.4f} {status}")


def print_recommendations(metrics: Dict[str, List]) -> None:
    """Print recommendations based on metrics."""

    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)

    recommendations = []

    # Check spacing loss
    if metrics['spacing']:
        avg_spacing = sum(metrics['spacing']) / len(metrics['spacing'])
        if avg_spacing < 0.5:
            recommendations.append({
                'priority': '🔴 CRITICAL',
                'issue': f'Spacing loss too small ({avg_spacing:.4f} < 0.5)',
                'action': 'Increase lambda_spacing from 50 to 1000 in config',
                'command': "sed -i 's/lambda_spacing: 50.0/lambda_spacing: 1000.0/' config/config.yaml"
            })

    # Check validation gap
    if len(metrics['val_loss']) >= 2:
        last_gap = metrics['val_loss'][-1] - metrics['loss'][metrics['steps'].index(metrics['val_steps'][-1])]
        if last_gap > 0.2:
            recommendations.append({
                'priority': '🟡 IMPORTANT',
                'issue': f'Validation gap increasing ({last_gap:.4f} > 0.2)',
                'action': 'Add early stopping to prevent overfitting',
                'command': 'Enable early_stopping in config with patience=100'
            })

    # Check loss plateau
    if len(metrics['loss']) > 50:
        recent_improvement = metrics['loss'][-50] - metrics['loss'][-1]
        if recent_improvement < 0.1:
            recommendations.append({
                'priority': '🟢 OPTIONAL',
                'issue': f'Loss plateau in last 500 steps (improvement: {recent_improvement:.4f})',
                'action': 'Consider increasing training steps or learning rate',
                'command': 'Set num_steps: 2000 or increase learning_rate in config'
            })

    if not recommendations:
        print("\n✅ No critical issues found. Training looks good!")
    else:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['priority']}")
            print(f"   Issue: {rec['issue']}")
            print(f"   Action: {rec['action']}")
            print(f"   Command: {rec['command']}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training metrics')
    parser.add_argument('--log-file', type=Path, default=Path('training.log'),
                       help='Path to training log file')
    parser.add_argument('--no-charts', action='store_true',
                       help='Skip ASCII charts (only show statistics)')
    args = parser.parse_args()

    if not args.log_file.exists():
        print(f"❌ Error: Log file not found: {args.log_file}")
        sys.exit(1)

    print(f"📖 Parsing training log: {args.log_file}")
    metrics = parse_training_log(args.log_file)

    if not metrics['steps']:
        print("❌ Error: No metrics found in log file")
        sys.exit(1)

    print(f"✅ Found {len(metrics['steps'])} training steps")

    # Print statistics
    print_statistics(metrics)

    # Print ASCII charts
    if not args.no_charts:
        print("\n" + "=" * 80)
        print("📊 METRICS VISUALIZATION")
        print("=" * 80)

        print_ascii_chart(metrics['loss'], title="Loss Progression", y_label="Loss")
        print_ascii_chart(metrics['lr'], title="Learning Rate Schedule", y_label="LR")

        if metrics['spacing']:
            print_ascii_chart(metrics['spacing'], title="Spacing Loss", y_label="Spacing")

    # Print recommendations
    print_recommendations(metrics)

    print("\n" + "=" * 80)
    print("✅ Analysis complete")
    print("=" * 80)
    print(f"\nFor detailed analysis, see:")
    print(f"  - docs/TRAINING_ANALYSIS_SPARSITY_DISABLED.md")
    print(f"  - docs/TRAINING_QUICK_VERDICT.md")


if __name__ == '__main__':
    main()
