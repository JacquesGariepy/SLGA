#!/usr/bin/env python3
"""
Test script to verify SLGA inference bugs.

This script demonstrates the 6 inference-specific bugs identified in the analysis:
1. Stale landmarks during generation
2. Missing cache_global_ids
3. Eval mode strategy mismatch
4. Disabled diversity
5. Missing position information
6. No KV-cache (performance)

Run this to see the bugs in action before applying fixes.
"""

import os
import sys
import torch
import yaml
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import Config, LLMTransformer


def test_bug1_stale_landmarks(model: LLMTransformer, device: str = "cuda"):
    """
    Bug #1: Landmarks never update during generation.

    Expected: Landmarks should change as sequence grows
    Actual: Landmarks remain static (if using cache_global_ids) or recomputed incorrectly
    """
    print("\n" + "="*80)
    print("BUG #1: Stale Landmarks During Generation")
    print("="*80)

    model.eval()
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 10)).to(device)

    print(f"Starting with prompt length: {prompt.size(1)}")
    print("\nGenerating 20 tokens and tracking landmark indices...\n")

    landmark_history = []

    with torch.no_grad():
        for step in range(20):
            # Forward pass with aux info
            logits, aux = model(prompt, return_aux=True)

            if aux['landmark_indices'] is not None:
                landmarks = aux['landmark_indices'][0].cpu().tolist()
                landmark_history.append(landmarks)

                print(f"Step {step:2d} | L={prompt.size(1):3d} | Landmarks: {landmarks[:5]}...")
            else:
                print(f"Step {step:2d} | L={prompt.size(1):3d} | Landmarks: None (using cache_global_ids)")

            # Sample next token (greedy for determinism)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            prompt = torch.cat([prompt, next_token], dim=1)

    # Analyze landmark changes
    if landmark_history:
        print("\n" + "-"*80)
        print("ANALYSIS:")

        # Check if landmarks are changing
        unique_landmark_sets = len(set(tuple(l) for l in landmark_history))
        print(f"  Unique landmark sets: {unique_landmark_sets} / {len(landmark_history)}")

        if unique_landmark_sets < len(landmark_history) * 0.5:
            print("  ⚠️  BUG CONFIRMED: Landmarks are NOT updating frequently enough!")
            print("  Expected: Landmarks should adapt as context grows")
        else:
            print("  ✓ Landmarks are updating (bug may be fixed or using learned landmarks)")

        # Check if early landmarks persist
        if len(landmark_history) > 10:
            early = set(landmark_history[0])
            late = set(landmark_history[-1])
            overlap = early.intersection(late)
            print(f"  Overlap between first and last: {len(overlap)} / {len(early)} positions")
            if len(overlap) > len(early) * 0.5:
                print("  ⚠️  BUG CONFIRMED: Early landmarks persisting too long!")


def test_bug2_missing_cache_global_ids(model: LLMTransformer, device: str = "cuda"):
    """
    Bug #2: cache_global_ids never computed in generate() method.

    Expected: Should compute heuristic landmarks or use learned selector
    Actual: Always passes None, disabling/misconfiguring global attention
    """
    print("\n" + "="*80)
    print("BUG #2: Missing cache_global_ids Computation")
    print("="*80)

    print(f"\nModel config:")
    print(f"  learned_landmarks: {model.cfg.learned_landmarks}")
    print(f"  global_k: {model.cfg.global_k}")

    # Check generate() signature and implementation
    import inspect
    source = inspect.getsource(model.generate)

    print("\nChecking generate() method implementation...")

    if "cache_global_ids = " in source or "_compute_heuristic_landmarks" in source:
        print("  ✓ generate() appears to compute cache_global_ids")
    else:
        print("  ⚠️  BUG CONFIRMED: generate() never computes cache_global_ids!")
        print("  The parameter exists but is always None unless passed by caller.")

    # Test actual behavior
    model.eval()
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 10)).to(device)

    print("\nTesting with learned_landmarks=True:")
    with torch.no_grad():
        logits, aux = model(prompt, return_aux=True)
        if aux['landmark_indices'] is not None:
            print(f"  Landmark selector produced: {aux['landmark_indices'].shape}")
        else:
            print("  ⚠️  No landmarks selected!")


def test_bug3_eval_strategy_mismatch(model: LLMTransformer, device: str = "cuda"):
    """
    Bug #3: Training uses soft selection, inference uses hard selection.

    Expected: Same selection strategy in both modes
    Actual: Different code paths lead to distribution mismatch
    """
    print("\n" + "="*80)
    print("BUG #3: Training vs Eval Selection Strategy Mismatch")
    print("="*80)

    if not model.cfg.learned_landmarks or model.landmark_selector is None:
        print("  Skipping: Model doesn't use learned landmarks")
        return

    x = torch.randn(1, 100, model.cfg.embed_dim).to(device)

    print("\nComparing landmark selection in train vs eval mode...\n")

    # Training mode
    model.train()
    with torch.no_grad():  # No gradients, just comparing forward pass
        indices_train, states_train, scores_train = model.landmark_selector(x)

    # Eval mode
    model.eval()
    with torch.no_grad():
        indices_eval, states_eval, scores_eval = model.landmark_selector(x)

    print(f"Training mode:")
    print(f"  Indices: {indices_train[0].cpu().tolist()[:10]}...")
    print(f"  Scores mean: {scores_train[0].mean().item():.4f}, std: {scores_train[0].std().item():.4f}")

    print(f"\nEval mode:")
    print(f"  Indices: {indices_eval[0].cpu().tolist()[:10]}...")
    if scores_eval is not None:
        print(f"  Scores mean: {scores_eval[0].mean().item():.4f}, std: {scores_eval[0].std().item():.4f}")
    else:
        print(f"  Scores: None (not returned in eval mode)")

    # Check if indices differ
    indices_match = torch.equal(indices_train, indices_eval)

    print(f"\n" + "-"*80)
    print("ANALYSIS:")
    print(f"  Indices identical: {indices_match}")

    if not indices_match:
        print("  ⚠️  BUG CONFIRMED: Different selection strategies in train vs eval!")
        print("  This causes train/test distribution mismatch.")
    else:
        print("  ✓ Selection appears consistent (bug may be fixed)")


def test_bug4_disabled_diversity(model: LLMTransformer, device: str = "cuda"):
    """
    Bug #4: Diversity mechanism disabled during inference.

    Expected: Different heads select different landmarks
    Actual: All heads may select same landmarks in eval mode
    """
    print("\n" + "="*80)
    print("BUG #4: Disabled Multi-Head Diversity")
    print("="*80)

    # We need to hook into the SLGA module to see per-head selections
    # This is more complex, so we'll check the code instead

    import inspect
    from src.slga import SLGAModule

    source = inspect.getsource(SLGAModule._diverse_topk)

    print("\nChecking _diverse_topk() implementation...")

    if "not self.training" in source:
        print("  ⚠️  BUG CONFIRMED: _diverse_topk() disabled during inference!")
        print("  Line contains: 'if not self.diverse_topk or not self.training'")
        print("  This means diversity is OFF in eval mode.")
    else:
        print("  ✓ Diversity mechanism appears active in eval mode")

    print(f"\nModel config:")
    print(f"  diverse_topk: {model.cfg.diverse_topk}")
    print(f"  num_heads: {model.cfg.num_heads}")

    if model.cfg.diverse_topk:
        print("\n  Impact: During training, 8 heads select different landmarks.")
        print("          During inference, all 8 heads may select SAME landmarks.")
        print("          → Multi-head attention degenerates to single-head!")


def test_bug5_missing_position_info(model: LLMTransformer, device: str = "cuda"):
    """
    Bug #5: cache_positions never passed during generation.

    Expected: Position info for causal masking of global attention
    Actual: Positions missing, causal masking incomplete
    """
    print("\n" + "="*80)
    print("BUG #5: Missing Position Information")
    print("="*80)

    # Check if forward() supports cache_positions
    import inspect
    forward_source = inspect.getsource(model.forward)
    generate_source = inspect.getsource(model.generate)

    print("\nChecking method signatures...")

    supports_positions = "cache_positions" in forward_source
    print(f"  forward() supports cache_positions: {supports_positions}")

    passes_positions = "cache_positions=" in generate_source
    print(f"  generate() passes cache_positions: {passes_positions}")

    if supports_positions and not passes_positions:
        print("\n  ⚠️  BUG CONFIRMED: forward() accepts positions but generate() never passes them!")
        print("  Impact: Global attention missing causal masking information.")


def test_bug6_no_kv_cache(model: LLMTransformer, device: str = "cuda"):
    """
    Bug #6: No KV-cache implementation.

    Expected: Cache past key/value states, O(1) per token
    Actual: Full recomputation, O(L²) per token
    """
    print("\n" + "="*80)
    print("BUG #6: Missing KV-Cache (Performance Bug)")
    print("="*80)

    import time

    model.eval()
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 10)).to(device)

    print("\nTiming generation with different sequence lengths...\n")

    timings = []

    for n_tokens in [5, 10, 20]:
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.time()

        with torch.no_grad():
            output = model.generate(prompt, max_new_tokens=n_tokens, temperature=0.8)

        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.time() - start

        timings.append((n_tokens, elapsed))
        print(f"  {n_tokens:2d} tokens: {elapsed:.2f}s ({elapsed/n_tokens:.3f}s per token)")

    print(f"\n" + "-"*80)
    print("ANALYSIS:")

    # Check if time per token increases (indicates no caching)
    time_per_token = [elapsed / n for n, elapsed in timings]

    if time_per_token[-1] > time_per_token[0] * 1.5:
        print("  ⚠️  BUG CONFIRMED: Time per token INCREASES with sequence length!")
        print("  This indicates no KV-cache: full recomputation every step.")
        print(f"  Complexity: O(L²) per token instead of O(1)")
    else:
        print("  ✓ Time per token relatively constant (may have caching)")

    # Check implementation
    import inspect
    source = inspect.getsource(model.generate)

    if "kv_cache" in source or "past_key_values" in source:
        print("  ✓ generate() appears to use KV-cache")
    else:
        print("  ⚠️  generate() does NOT implement KV-cache!")


def main():
    """Run all bug tests."""

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*80)
    print("SLGA INFERENCE BUGS TEST SUITE")
    print("="*80)
    print(f"\nDevice: {device}")
    print(f"Config: {config_path}")

    # Create model (don't need trained weights for bug testing)
    print("\nCreating model...")
    model_cfg = Config(**cfg["model"])
    model = LLMTransformer(model_cfg).to(device)
    model.eval()

    print(f"Model: {model.get_num_params() / 1e6:.2f}M parameters")
    print(f"  learned_landmarks: {model.cfg.learned_landmarks}")
    print(f"  diverse_topk: {model.cfg.diverse_topk}")

    # Run tests
    try:
        test_bug1_stale_landmarks(model, device)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

    try:
        test_bug2_missing_cache_global_ids(model, device)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

    try:
        test_bug3_eval_strategy_mismatch(model, device)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

    try:
        test_bug4_disabled_diversity(model, device)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

    try:
        test_bug5_missing_position_info(model, device)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

    try:
        test_bug6_no_kv_cache(model, device)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

    # Summary
    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80)
    print("\nSee detailed analysis in:")
    print("  - docs/SLGA_INFERENCE_BUGS_ANALYSIS.md")
    print("  - docs/INFERENCE_BUGS_SUMMARY.md")
    print("  - docs/INFERENCE_BUG_DIAGRAM.md")
    print("\nNext steps:")
    print("  1. Implement Fix #1 and #2 (critical)")
    print("  2. Implement Fix #3 and #4 (important)")
    print("  3. Re-run this test suite to verify fixes")


if __name__ == "__main__":
    main()
