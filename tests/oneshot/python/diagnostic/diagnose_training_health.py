#!/usr/bin/env python3
"""
Training Health Diagnostic Tool for SLGA Model

Diagnoses critical training issues:
1. GPU utilization bottlenecks (4-5% usage is CRITICALLY LOW)
2. Data loading pipeline efficiency
3. Gradient flow and backpropagation health
4. Loss-quality disconnect analysis
5. Landmark selection patterns
6. Batch collation and memory efficiency

Usage:
    python scripts/diagnose_training_health.py --config config/config.yaml --checkpoint checkpoints/ckpt_33800
"""

import os
import sys
import yaml
import torch
import torch.nn.functional as F
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.legacy.model import Config, LLMTransformer
from src.legacy.data import get_tokenizer, load_text_dataset, CollatorLocal
from scripts.utils import load_checkpoint
from torch.utils.data import DataLoader


class TrainingHealthDiagnostic:
    """Comprehensive training health diagnostics"""

    def __init__(self, cfg: dict, checkpoint_path: str = None):
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        print("=" * 80)
        print("SLGA TRAINING HEALTH DIAGNOSTIC")
        print("=" * 80)

        self.model_cfg = Config(**cfg["model"])
        self.model = LLMTransformer(self.model_cfg).to(self.device)

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\n📁 Loading checkpoint: {checkpoint_path}")
            from torch.optim import AdamW
            from transformers import get_cosine_schedule_with_warmup

            optimizer = AdamW(self.model.parameters(), lr=cfg["train"]["lr"])
            scheduler = get_cosine_schedule_with_warmup(optimizer, 100, 1000)

            step = load_checkpoint(checkpoint_path, self.model, optimizer, scheduler, device=str(self.device))
            print(f"✅ Loaded checkpoint at step {step}")

        # Load tokenizer and data
        self.tokenizer = get_tokenizer(cfg["tokenizer"])

        print("\n" + "=" * 80)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 80)

    def diagnose_gpu_utilization(self) -> Dict[str, Any]:
        """
        🚨 CRITICAL: Diagnose GPU utilization bottleneck

        Expected: 80-95% GPU utilization during training
        Observed: 4-5% GPU utilization (CRITICALLY LOW!)

        Possible causes:
        1. Data loading bottleneck (CPU-bound)
        2. num_workers=0 (synchronous data loading)
        3. Small batch size
        4. Gradient accumulation too high
        5. CPU preprocessing dominating
        """
        print("\n" + "=" * 80)
        print("1. GPU UTILIZATION DIAGNOSTIC")
        print("=" * 80)

        results = {}

        # Check training configuration
        batch_size = self.cfg["train"]["batch_size"]
        accum_steps = self.cfg["train"]["accum_steps"]
        num_workers = self.cfg["data"].get("num_workers", 2)

        effective_batch = batch_size * accum_steps

        print(f"\n📊 Training Configuration:")
        print(f"   Batch size: {batch_size}")
        print(f"   Gradient accumulation: {accum_steps}")
        print(f"   Effective batch: {effective_batch}")
        print(f"   Num workers: {num_workers}")

        results['batch_size'] = batch_size
        results['accum_steps'] = accum_steps
        results['num_workers'] = num_workers

        # DIAGNOSIS: Check if data loading is bottleneck
        print(f"\n🔍 DIAGNOSIS:")

        issues = []

        if num_workers == 0:
            issues.append("❌ CRITICAL: num_workers=0 (synchronous data loading)")
            print(f"   ❌ CRITICAL: num_workers=0 causes CPU bottleneck!")
            print(f"      → GPU waits for CPU to load each batch")
            print(f"      → FIX: Set num_workers=4 or higher")
        elif num_workers < 4:
            issues.append(f"⚠️  WARNING: num_workers={num_workers} is low")
            print(f"   ⚠️  num_workers={num_workers} may be insufficient")
            print(f"      → Recommend num_workers=4-8 for optimal pipeline")
        else:
            print(f"   ✅ num_workers={num_workers} is adequate")

        if batch_size < 8:
            issues.append(f"⚠️  WARNING: Small batch_size={batch_size}")
            print(f"   ⚠️  Batch size {batch_size} is small")
            print(f"      → GPU underutilized with small batches")
            print(f"      → Recommend batch_size=16-32")

        if accum_steps > 8:
            issues.append(f"⚠️  WARNING: High accum_steps={accum_steps}")
            print(f"   ⚠️  Gradient accumulation {accum_steps} is high")
            print(f"      → More frequent sync barriers reduce GPU efficiency")
            print(f"      → Prefer larger batch_size over high accum_steps")

        # Measure actual data loading speed
        print(f"\n⏱️  Measuring Data Loading Speed...")

        ds_train = load_text_dataset(
            self.cfg["data"]["dataset"],
            self.cfg["data"].get("subset"),
            self.cfg["data"]["split_train"],
        )

        max_train = self.cfg["data"].get("max_train_samples")
        if max_train and len(ds_train) > max_train:
            ds_train = ds_train.select(range(max_train))

        collator = CollatorLocal(self.tokenizer, self.cfg["train"].get("seq_len_start", 512))

        # Test with different num_workers
        for nw in [0, 2, 4]:
            loader = DataLoader(
                ds_train,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=nw,
                pin_memory=True,
            )

            start = time.time()
            batches_loaded = 0
            for batch in loader:
                batches_loaded += 1
                if batches_loaded >= 10:  # Test 10 batches
                    break
            elapsed = time.time() - start

            batches_per_sec = batches_loaded / elapsed if elapsed > 0 else 0
            samples_per_sec = batches_per_sec * batch_size

            print(f"   num_workers={nw}: {batches_per_sec:.2f} batches/sec, {samples_per_sec:.1f} samples/sec")

        results['issues'] = issues

        return results

    def diagnose_gradient_flow(self) -> Dict[str, Any]:
        """
        Check if gradients are flowing properly through the network

        Issues to detect:
        1. Vanishing gradients (gradients too small)
        2. Exploding gradients (gradients too large)
        3. Dead neurons (no gradient flow)
        4. Landmark scorer gradients
        """
        print("\n" + "=" * 80)
        print("2. GRADIENT FLOW DIAGNOSTIC")
        print("=" * 80)

        self.model.train()

        # Create dummy batch
        B, L = 4, 512
        input_ids = torch.randint(0, self.model_cfg.vocab_size, (B, L), device=self.device)
        labels = torch.randint(0, self.model_cfg.vocab_size, (B, L), device=self.device)

        # Forward pass
        logits, aux = self.model(input_ids, return_aux=True)

        # Compute loss
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            labels[:, :-1].contiguous().view(-1),
            ignore_index=-100,
        )

        # Backward
        loss.backward()

        # Analyze gradients
        print(f"\n📊 Gradient Statistics:")

        results = {}
        layer_grads = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()

                layer_grads[name] = {
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std,
                }

        # Show top layers by gradient norm
        sorted_layers = sorted(layer_grads.items(), key=lambda x: x[1]['norm'], reverse=True)

        print(f"\n   Top 10 layers by gradient norm:")
        for i, (name, stats) in enumerate(sorted_layers[:10]):
            print(f"   {i+1}. {name:50s} | norm: {stats['norm']:8.4f} | mean: {stats['mean']:8.4f}")

        # Check for potential issues
        print(f"\n🔍 GRADIENT HEALTH CHECK:")

        issues = []

        # Check landmark selector gradients
        scorer_grads = [v for k, v in layer_grads.items() if 'landmark_selector.scorer' in k]
        if scorer_grads:
            avg_scorer_grad = np.mean([g['norm'] for g in scorer_grads])
            print(f"   Landmark scorer avg gradient norm: {avg_scorer_grad:.6f}")

            if avg_scorer_grad < 1e-5:
                issues.append("⚠️  Landmark scorer gradients very small (possible vanishing)")
                print(f"   ⚠️  Scorer gradients are very small ({avg_scorer_grad:.2e})")
                print(f"      → May need higher learning rate for scorer")
            else:
                print(f"   ✅ Scorer gradients healthy")
        else:
            issues.append("❌ No landmark selector found in model")
            print(f"   ❌ No landmark_selector gradients found!")

        # Check for vanishing gradients in early layers
        early_layers = [v for k, v in layer_grads.items() if 'blocks.0' in k or 'blocks.1' in k]
        if early_layers:
            avg_early_grad = np.mean([g['norm'] for g in early_layers])
            late_layers = [v for k, v in layer_grads.items() if 'blocks.10' in k or 'blocks.11' in k]
            avg_late_grad = np.mean([g['norm'] for g in late_layers]) if late_layers else avg_early_grad

            ratio = avg_early_grad / (avg_late_grad + 1e-10)
            print(f"   Early/Late layer gradient ratio: {ratio:.4f}")

            if ratio < 0.1:
                issues.append("⚠️  Possible vanishing gradients in early layers")
                print(f"   ⚠️  Early layer gradients much smaller than late layers")
                print(f"      → May indicate vanishing gradient problem")

        results['layer_grads'] = layer_grads
        results['issues'] = issues

        return results

    def diagnose_landmark_selection(self) -> Dict[str, Any]:
        """
        Analyze landmark selection patterns

        Issues to detect:
        1. Landmarks clustering (not uniformly distributed)
        2. Landmarks not being selected (scores too low)
        3. Invariant selection (always same positions)
        """
        print("\n" + "=" * 80)
        print("3. LANDMARK SELECTION DIAGNOSTIC")
        print("=" * 80)

        if self.model.landmark_selector is None:
            print("\n❌ Model does not have learned landmarks enabled")
            return {'error': 'No landmark selector'}

        self.model.eval()

        # Create test batch
        B, L = 8, 2048
        input_ids = torch.randint(0, self.model_cfg.vocab_size, (B, L), device=self.device)

        with torch.no_grad():
            logits, aux = self.model(input_ids, return_aux=True)

        landmark_indices = aux['landmark_indices']  # (B, G)
        landmark_scores = aux['landmark_scores']    # (B, L)

        G = landmark_indices.size(1)

        print(f"\n📊 Landmark Selection Statistics:")
        print(f"   Sequence length: {L}")
        print(f"   Num landmarks selected: {G}")
        print(f"   Expected spacing: {L / G:.1f} tokens")

        # Analyze spacing
        sorted_idx = torch.sort(landmark_indices, dim=-1)[0]
        gaps = sorted_idx[:, 1:] - sorted_idx[:, :-1]

        mean_gap = gaps.float().mean().item()
        std_gap = gaps.float().std().item()
        min_gap = gaps.min().item()
        max_gap = gaps.max().item()

        print(f"\n   Actual spacing statistics:")
        print(f"   Mean gap: {mean_gap:.1f} tokens")
        print(f"   Std gap: {std_gap:.1f} tokens")
        print(f"   Min gap: {min_gap} tokens")
        print(f"   Max gap: {max_gap} tokens")

        # Check for clustering
        ideal_gap = L / G
        clustering_ratio = std_gap / ideal_gap

        print(f"\n🔍 CLUSTERING ANALYSIS:")
        print(f"   Clustering ratio (std/ideal): {clustering_ratio:.3f}")

        issues = []

        if clustering_ratio > 0.5:
            issues.append(f"⚠️  High clustering (ratio={clustering_ratio:.3f})")
            print(f"   ⚠️  Landmarks are clustering (high std deviation)")
            print(f"      → spacing_loss may need higher weight")
        elif clustering_ratio < 0.2:
            print(f"   ✅ Landmarks well-distributed (low clustering)")
        else:
            print(f"   ℹ️  Moderate clustering (acceptable)")

        # Check score distribution
        scores_topk = torch.topk(landmark_scores, k=G, dim=-1)[0]
        mass_in_topk = scores_topk.sum() / landmark_scores.sum()

        print(f"\n📊 Score Distribution:")
        print(f"   Mass in top-{G}: {mass_in_topk.item():.3f}")
        print(f"   Score mean: {landmark_scores.mean().item():.6f}")
        print(f"   Score std: {landmark_scores.std().item():.6f}")

        if mass_in_topk < 0.6:
            issues.append(f"⚠️  Low mass concentration ({mass_in_topk:.3f})")
            print(f"   ⚠️  Scores are too dispersed (mass_in_topk < 0.6)")
            print(f"      → sparsity_loss may need higher weight")
        else:
            print(f"   ✅ Good score concentration")

        return {
            'mean_gap': mean_gap,
            'std_gap': std_gap,
            'clustering_ratio': clustering_ratio,
            'mass_in_topk': mass_in_topk.item(),
            'issues': issues,
        }

    def diagnose_loss_quality_disconnect(self) -> Dict[str, Any]:
        """
        Analyze disconnect between low validation loss and poor generation quality

        Possible causes:
        1. Overfitting to common n-grams
        2. Loss dominated by frequent tokens
        3. Model not learning long-range dependencies
        4. Landmark attention not working properly
        """
        print("\n" + "=" * 80)
        print("4. LOSS-QUALITY DISCONNECT DIAGNOSTIC")
        print("=" * 80)

        self.model.eval()

        # Generate samples
        prompts = [
            "The capital of France is",
            "In the year 2024, artificial intelligence",
            "Once upon a time, there was a",
        ]

        print(f"\n📝 Generation Quality Test:")
        print(f"   Temperature: 0.8")
        print(f"   Max tokens: 50")

        results = {}

        for i, prompt in enumerate(prompts):
            print(f"\n   Prompt {i+1}: \"{prompt}\"")

            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=50,
                    temperature=0.8,
                    top_k=40,
                    stop_on_eos=True,
                )

            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            print(f"   Generated: \"{generated_text}\"")

            # Analyze repetition
            tokens = generated_text.split()
            unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0

            # Check for common repetition patterns
            has_repetition = False
            for j in range(len(tokens) - 2):
                if tokens[j] == tokens[j+1] == tokens[j+2]:
                    has_repetition = True
                    break

            print(f"   Unique token ratio: {unique_ratio:.3f}")
            print(f"   Has 3+ token repetition: {has_repetition}")

            results[f'prompt_{i}'] = {
                'prompt': prompt,
                'generated': generated_text,
                'unique_ratio': unique_ratio,
                'has_repetition': has_repetition,
            }

        # DIAGNOSIS
        print(f"\n🔍 QUALITY DIAGNOSIS:")

        avg_unique_ratio = np.mean([v['unique_ratio'] for v in results.values()])
        repetition_count = sum([v['has_repetition'] for v in results.values()])

        issues = []

        if avg_unique_ratio < 0.6:
            issues.append(f"❌ High repetition (unique_ratio={avg_unique_ratio:.3f})")
            print(f"   ❌ High repetition detected (unique ratio: {avg_unique_ratio:.3f})")
            print(f"      → Model may be overfitting to frequent patterns")

        if repetition_count > 0:
            issues.append(f"❌ {repetition_count}/3 samples have immediate repetition")
            print(f"   ❌ {repetition_count}/3 samples have 3+ token repetition")
            print(f"      → Possible mode collapse or attention issues")

        if avg_unique_ratio >= 0.7 and repetition_count == 0:
            print(f"   ✅ Generation quality acceptable")

        results['avg_unique_ratio'] = avg_unique_ratio
        results['repetition_count'] = repetition_count
        results['issues'] = issues

        return results

    def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run all diagnostics and generate comprehensive report"""

        all_results = {}

        # 1. GPU Utilization
        all_results['gpu_utilization'] = self.diagnose_gpu_utilization()

        # 2. Gradient Flow
        all_results['gradient_flow'] = self.diagnose_gradient_flow()

        # 3. Landmark Selection
        all_results['landmark_selection'] = self.diagnose_landmark_selection()

        # 4. Loss-Quality Disconnect
        all_results['loss_quality'] = self.diagnose_loss_quality_disconnect()

        # Generate summary
        print("\n" + "=" * 80)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 80)

        all_issues = []

        for category, result in all_results.items():
            if 'issues' in result and result['issues']:
                all_issues.extend([(category, issue) for issue in result['issues']])

        if all_issues:
            print(f"\n🚨 FOUND {len(all_issues)} ISSUES:\n")
            for i, (category, issue) in enumerate(all_issues, 1):
                print(f"{i}. [{category.upper()}] {issue}")
        else:
            print(f"\n✅ No critical issues detected!")

        # RECOMMENDATIONS
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        print(f"\n🔧 IMMEDIATE FIXES:")

        # GPU utilization fixes
        if all_results['gpu_utilization']['num_workers'] < 4:
            print(f"\n1. FIX DATA LOADING BOTTLENECK:")
            print(f"   In config.yaml:")
            print(f"   data:")
            print(f"     num_workers: 4  # Currently {all_results['gpu_utilization']['num_workers']}")
            print(f"   This will enable async data loading and increase GPU utilization from 4-5% to 60-80%+")

        # Gradient flow fixes
        if 'gradient_flow' in all_results and all_results['gradient_flow'].get('issues'):
            print(f"\n2. FIX GRADIENT FLOW:")
            print(f"   - Consider using higher learning rate for landmark scorer")
            print(f"   - Check for gradient clipping value (currently set to {self.cfg['train']['grad_clip']})")

        # Landmark selection fixes
        if 'landmark_selection' in all_results and all_results['landmark_selection'].get('issues'):
            print(f"\n3. IMPROVE LANDMARK SELECTION:")
            print(f"   In config.yaml:")
            print(f"   train:")
            print(f"     lambda_spacing: 0.05  # Increase to reduce clustering")
            print(f"     lambda_sparsity: 0.005  # Increase to improve concentration")

        # Generation quality fixes
        if 'loss_quality' in all_results and all_results['loss_quality'].get('issues'):
            print(f"\n4. IMPROVE GENERATION QUALITY:")
            print(f"   - Model may be overfitting - consider:")
            print(f"     • Increase dropout (currently {self.cfg['model']['dropout_rate']})")
            print(f"     • Add more diverse training data")
            print(f"     • Use nucleus sampling (top_p=0.9) instead of top_k")

        return all_results


def main():
    parser = argparse.ArgumentParser(description="Diagnose SLGA training health")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint directory")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Find latest checkpoint if directory provided
    checkpoint_path = None
    if args.checkpoint:
        if os.path.isdir(args.checkpoint):
            checkpoints = [d for d in os.listdir(args.checkpoint) if d.startswith("ckpt_")]
            if checkpoints:
                latest = max(checkpoints, key=lambda x: int(x.split("_")[1]))
                checkpoint_path = os.path.join(args.checkpoint, latest)
        else:
            checkpoint_path = args.checkpoint

    # Run diagnostic
    diagnostic = TrainingHealthDiagnostic(cfg, checkpoint_path)
    results = diagnostic.run_full_diagnostic()

    print("\n" + "=" * 80)
    print("✅ DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
