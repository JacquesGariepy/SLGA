#!/usr/bin/env python3
"""
Diagnostic Tool: Analyze Landmark Selection Patterns

Checks for:
1. Landmark degeneration (clustering, repetition)
2. Gate value distribution (local vs global bias)
3. Temperature schedule effectiveness
4. Coverage metrics (gaps, distribution)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import yaml

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.legacy.model import LLMTransformer, Config
from transformers import GPT2Tokenizer


def load_model_and_config(checkpoint_path: str) -> Tuple[LLMTransformer, Config, Dict]:
    """Load model checkpoint and extract configuration"""
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # Extract config
    config_dict = ckpt.get('config', {})
    cfg = Config(**config_dict)

    # Load model
    model = LLMTransformer(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    return model, cfg, ckpt


def analyze_landmark_distribution(
    landmark_indices: torch.Tensor,
    seq_len: int,
    num_samples: int = 5
) -> Dict[str, float]:
    """
    Analyze spatial distribution of landmarks

    Args:
        landmark_indices: (B, G) landmark positions
        seq_len: Sequence length L
        num_samples: Number of samples to analyze

    Returns:
        metrics: Dict with distribution statistics
    """
    B, G = landmark_indices.shape
    num_samples = min(num_samples, B)

    metrics = {}

    for i in range(num_samples):
        idx = landmark_indices[i].cpu().numpy()
        sorted_idx = np.sort(idx)

        # Compute gaps
        gaps = np.diff(sorted_idx)

        # Ideal uniform spacing
        ideal_gap = seq_len / G

        # Metrics
        metrics[f'sample_{i}'] = {
            'min_gap': float(gaps.min()),
            'max_gap': float(gaps.max()),
            'mean_gap': float(gaps.mean()),
            'std_gap': float(gaps.std()),
            'ideal_gap': ideal_gap,
            'deviation': float(np.abs(gaps - ideal_gap).mean()),
            'coverage': float(G / seq_len * 100),  # Percentage
            'unique_landmarks': len(np.unique(idx)),
            'duplicates': G - len(np.unique(idx)),
        }

    # Aggregate statistics
    metrics['aggregate'] = {
        'mean_deviation': np.mean([m['deviation'] for m in metrics.values() if isinstance(m, dict)]),
        'mean_unique': np.mean([m['unique_landmarks'] for m in metrics.values() if isinstance(m, dict)]),
        'mean_duplicates': np.mean([m['duplicates'] for m in metrics.values() if isinstance(m, dict)]),
    }

    return metrics


def check_landmark_degeneration(
    model: LLMTransformer,
    tokenizer: GPT2Tokenizer,
    num_tests: int = 10,
    seq_len: int = 512
) -> Dict[str, any]:
    """
    Test if landmark selector has degenerated (always selects same positions)

    Args:
        model: SLGA model
        tokenizer: Tokenizer
        num_tests: Number of different inputs to test
        seq_len: Sequence length

    Returns:
        results: Degeneration metrics
    """
    model.eval()

    all_landmarks = []

    # Test with different random inputs
    for _ in range(num_tests):
        input_ids = torch.randint(0, model.cfg.vocab_size, (1, seq_len))

        with torch.no_grad():
            _, aux = model(input_ids, return_aux=True)

        if aux['landmark_indices'] is not None:
            all_landmarks.append(aux['landmark_indices'][0].cpu().numpy())

    if not all_landmarks:
        return {'error': 'No landmarks generated (learned_landmarks=False?)'}

    # Convert to array: (num_tests, G)
    landmarks_array = np.array(all_landmarks)

    # Check variance across different inputs
    position_variance = landmarks_array.var(axis=0)  # Variance per landmark position
    mean_variance = position_variance.mean()

    # Check if landmarks are always the same
    unique_patterns = len(set(tuple(lm) for lm in all_landmarks))

    # Check correlation between runs
    if num_tests > 1:
        correlation = np.corrcoef(landmarks_array)
        mean_correlation = (correlation.sum() - num_tests) / (num_tests * (num_tests - 1))
    else:
        mean_correlation = 1.0

    results = {
        'mean_variance': float(mean_variance),
        'unique_patterns': unique_patterns,
        'total_tests': num_tests,
        'pattern_diversity': unique_patterns / num_tests,
        'mean_correlation': float(mean_correlation),
        'degeneration_score': 1.0 - (unique_patterns / num_tests),  # 1.0 = fully degenerate
        'status': 'DEGENERATE' if unique_patterns < num_tests * 0.3 else 'HEALTHY',
    }

    return results


def analyze_temperature_schedule(checkpoint_path: str) -> Dict:
    """Analyze temperature decay from checkpoint"""
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # Check if landmark selector state exists
    if 'model_state_dict' not in ckpt:
        return {'error': 'No model state dict found'}

    state_dict = ckpt['model_state_dict']

    # Find step count (if saved)
    step_count = None
    for key in state_dict.keys():
        if 'landmark_selector.step_count' in key:
            step_count = state_dict[key].item()
            break

    # Get config
    cfg = ckpt.get('config', {})
    temp_init = cfg.get('temperature', 1.0)
    temp_decay = cfg.get('temperature_decay', 0.999)
    temp_min = cfg.get('min_temperature', 0.3)

    training_step = ckpt.get('step', 0)

    # Calculate current temperature
    if step_count is not None:
        current_temp = max(temp_init * (temp_decay ** step_count), temp_min)
    else:
        # Estimate from training step
        current_temp = max(temp_init * (temp_decay ** training_step), temp_min)
        step_count = training_step

    # Calculate when min temperature reached
    if temp_decay < 1.0:
        steps_to_min = np.log(temp_min / temp_init) / np.log(temp_decay)
    else:
        steps_to_min = float('inf')

    return {
        'initial_temp': temp_init,
        'decay_rate': temp_decay,
        'min_temp': temp_min,
        'current_temp': float(current_temp),
        'step_count': int(step_count) if step_count is not None else None,
        'training_step': training_step,
        'steps_to_min': int(steps_to_min),
        'reached_min': current_temp <= temp_min + 1e-6,
        'status': 'AT_MIN' if current_temp <= temp_min + 1e-6 else 'DECAYING',
    }


def visualize_landmark_coverage(
    landmark_indices: torch.Tensor,
    seq_len: int,
    save_path: str = None
):
    """
    Visualize landmark coverage across sequence

    Args:
        landmark_indices: (B, G) landmark positions
        seq_len: Sequence length
        save_path: Path to save plot (optional)
    """
    B, G = landmark_indices.shape

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Histogram of landmark positions (aggregated)
    all_positions = landmark_indices.flatten().cpu().numpy()
    axes[0, 0].hist(all_positions, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axhline(y=len(all_positions)/50, color='r', linestyle='--', label='Uniform')
    axes[0, 0].set_title('Landmark Position Distribution (All Samples)')
    axes[0, 0].set_xlabel('Position in Sequence')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()

    # Plot 2: Coverage visualization (first 5 samples)
    num_samples = min(5, B)
    for i in range(num_samples):
        positions = landmark_indices[i].cpu().numpy()
        y_offset = i * 0.2
        axes[0, 1].scatter(positions, [y_offset] * len(positions), alpha=0.6, s=50)
    axes[0, 1].set_title('Landmark Positions (First 5 Samples)')
    axes[0, 1].set_xlabel('Position in Sequence')
    axes[0, 1].set_yticks([])
    axes[0, 1].set_xlim(0, seq_len)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Gap distribution
    gaps_list = []
    for i in range(min(10, B)):
        sorted_idx = np.sort(landmark_indices[i].cpu().numpy())
        gaps = np.diff(sorted_idx)
        gaps_list.extend(gaps)

    ideal_gap = seq_len / G
    axes[1, 0].hist(gaps_list, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=ideal_gap, color='r', linestyle='--', linewidth=2, label=f'Ideal Gap ({ideal_gap:.1f})')
    axes[1, 0].set_title('Gap Distribution Between Landmarks')
    axes[1, 0].set_xlabel('Gap Size (tokens)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()

    # Plot 4: Coverage metrics
    coverage_pct = (G / seq_len) * 100
    metrics_text = f"""
    Coverage Metrics:

    Total Landmarks: {G}
    Sequence Length: {seq_len}
    Coverage: {coverage_pct:.2f}%

    Ideal Gap: {ideal_gap:.1f} tokens
    Mean Gap: {np.mean(gaps_list):.1f} tokens
    Std Gap: {np.std(gaps_list):.1f} tokens

    Max Gap: {np.max(gaps_list):.0f} tokens
    Min Gap: {np.min(gaps_list):.0f} tokens
    """

    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                     verticalalignment='center', transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Coverage Statistics')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Diagnose SLGA Landmark Selection')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint to analyze')
    parser.add_argument('--output', type=str, default='landmark_analysis.png',
                       help='Output path for visualization')
    parser.add_argument('--num-tests', type=int, default=10,
                       help='Number of degeneration tests')
    parser.add_argument('--seq-len', type=int, default=512,
                       help='Sequence length for testing')
    args = parser.parse_args()

    print("=" * 80)
    print("SLGA Landmark Selection Diagnostic Tool")
    print("=" * 80)
    print(f"\nAnalyzing checkpoint: {args.checkpoint}\n")

    # Load model
    print("Loading model...")
    model, cfg, ckpt = load_model_and_config(args.checkpoint)
    print(f"✓ Model loaded: {model.get_num_params() / 1e6:.2f}M parameters")
    print(f"  - Learned landmarks: {cfg.learned_landmarks}")
    print(f"  - Global K: {cfg.global_k}")
    print(f"  - Layers: {cfg.n_layers}")
    print(f"  - Embed dim: {cfg.embed_dim}")

    # Check temperature schedule
    print("\n" + "-" * 80)
    print("1. Temperature Schedule Analysis")
    print("-" * 80)
    temp_metrics = analyze_temperature_schedule(args.checkpoint)

    print(f"Initial temperature: {temp_metrics['initial_temp']}")
    print(f"Decay rate: {temp_metrics['decay_rate']}")
    print(f"Min temperature: {temp_metrics['min_temp']}")
    print(f"Current temperature: {temp_metrics['current_temp']:.4f}")
    print(f"Training step: {temp_metrics['training_step']}")
    print(f"Steps to min: {temp_metrics['steps_to_min']}")
    print(f"Status: {temp_metrics['status']}")

    if temp_metrics['reached_min']:
        print("\n⚠️  WARNING: Temperature has reached minimum!")
        print("   Selection is now deterministic (no exploration)")
        print("   Consider increasing temperature_decay or min_temperature")

    # Check degeneration
    if cfg.learned_landmarks:
        print("\n" + "-" * 80)
        print("2. Landmark Degeneration Test")
        print("-" * 80)

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        degen_metrics = check_landmark_degeneration(
            model, tokenizer,
            num_tests=args.num_tests,
            seq_len=args.seq_len
        )

        print(f"Tests run: {degen_metrics['total_tests']}")
        print(f"Unique patterns: {degen_metrics['unique_patterns']}")
        print(f"Pattern diversity: {degen_metrics['pattern_diversity']:.2%}")
        print(f"Mean variance: {degen_metrics['mean_variance']:.2f}")
        print(f"Mean correlation: {degen_metrics['mean_correlation']:.4f}")
        print(f"Degeneration score: {degen_metrics['degeneration_score']:.2%}")
        print(f"\nStatus: {degen_metrics['status']}")

        if degen_metrics['status'] == 'DEGENERATE':
            print("\n🔴 CRITICAL: Landmark selector has degenerated!")
            print("   Recommendations:")
            print("   - Increase temperature_decay (e.g., 0.9999)")
            print("   - Increase min_temperature (e.g., 0.5)")
            print("   - Check if diversity losses are active")
        else:
            print("\n✓ Landmark selector appears healthy")

        # Analyze distribution
        print("\n" + "-" * 80)
        print("3. Landmark Distribution Analysis")
        print("-" * 80)

        # Generate sample
        input_ids = torch.randint(0, cfg.vocab_size, (5, args.seq_len))
        with torch.no_grad():
            _, aux = model(input_ids, return_aux=True)

        if aux['landmark_indices'] is not None:
            dist_metrics = analyze_landmark_distribution(
                aux['landmark_indices'],
                args.seq_len,
                num_samples=5
            )

            print(f"Coverage: {dist_metrics['sample_0']['coverage']:.2f}%")
            print(f"Ideal gap: {dist_metrics['sample_0']['ideal_gap']:.1f} tokens")
            print(f"Mean gap: {dist_metrics['sample_0']['mean_gap']:.1f} tokens")
            print(f"Std gap: {dist_metrics['sample_0']['std_gap']:.1f} tokens")
            print(f"Mean deviation from ideal: {dist_metrics['aggregate']['mean_deviation']:.1f}")
            print(f"Mean unique landmarks: {dist_metrics['aggregate']['mean_unique']:.1f}")
            print(f"Mean duplicates: {dist_metrics['aggregate']['mean_duplicates']:.1f}")

            # Visualize
            print(f"\nGenerating visualization: {args.output}")
            visualize_landmark_coverage(
                aux['landmark_indices'],
                args.seq_len,
                save_path=args.output
            )
        else:
            print("No landmarks to analyze (learned_landmarks=False)")
    else:
        print("\n⚠️  Learned landmarks disabled, skipping degeneration tests")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    issues = []

    if temp_metrics['reached_min']:
        issues.append("Temperature at minimum (deterministic selection)")

    if cfg.learned_landmarks and degen_metrics.get('status') == 'DEGENERATE':
        issues.append("Landmark selector degenerated")

    if cfg.global_k < 32:
        issues.append(f"Low global_k ({cfg.global_k} < 32 recommended)")

    coverage = (cfg.global_k * 2) / cfg.max_seq_len * 100  # Approximate
    if coverage < 5.0:
        issues.append(f"Low landmark coverage ({coverage:.1f}% < 5% recommended)")

    if issues:
        print("\n🔴 Issues Found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\nRecommendations:")
        print("  - Review docs/ARCHITECTURE_REVIEW_GENERATION_QUALITY.md")
        print("  - Consider increasing global_k to 64+")
        print("  - Adjust temperature_decay to 0.9999")
        print("  - Check training loss composition")
    else:
        print("\n✓ No major issues detected")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
