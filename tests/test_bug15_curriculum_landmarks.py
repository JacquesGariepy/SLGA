#!/usr/bin/env python3
"""
Test Bug #15: Heuristic landmarks collapse during curriculum learning

When curriculum truncates sequences to shorter lengths, heuristic landmark
positions (computed for full sequence) exceed the new length and ALL get
clamped to the same position (L_cur-1), collapsing global attention.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.legacy.model import Config, LLMTransformer


def test_landmark_collapse_during_curriculum():
    """
    Simulate Bug #15: Curriculum truncation causes landmark clustering
    """
    print("=" * 80)
    print("BUG #15 TEST: Landmark Collapse During Curriculum")
    print("=" * 80)
    print()

    # Setup
    cfg = Config(
        vocab_size=1000,
        max_seq_len=512,
        embed_dim=64,
        num_heads=2,
        n_layers=2,
        learned_landmarks=False,  # Heuristic mode
        global_k=8,
    )

    model = LLMTransformer(cfg)
    model.eval()

    # Simuler le scénario du bug
    print("Scenario: Curriculum Learning")
    print("-" * 80)

    full_seq_len = 512
    truncated_seq_len = 128  # Curriculum truncates to this

    # Collator génère landmarks pour séquence complète
    landmark_positions_full = torch.linspace(0, full_seq_len-1, cfg.global_k).long()
    print(f"\n1. Collator generates landmarks for full sequence (L={full_seq_len}):")
    print(f"   Positions: {landmark_positions_full.tolist()}")
    print(f"   Range: [{landmark_positions_full.min()}, {landmark_positions_full.max()}]")

    # Training tronque à current_seq_len
    print(f"\n2. Training truncates to curriculum length (L={truncated_seq_len}):")
    print(f"   input_ids: (B, {full_seq_len}) → (B, {truncated_seq_len})")

    # model.forward() reçoit positions originales
    cache_global_ids = landmark_positions_full.unsqueeze(0)  # (1, G)
    print(f"\n3. Model receives original landmark positions:")
    print(f"   cache_global_ids: {cache_global_ids[0].tolist()}")

    # ❌ BUG: model.forward() clamp à L_cur-1
    L_cur = truncated_seq_len
    landmark_indices_safe = torch.clamp(cache_global_ids, 0, L_cur - 1)
    print(f"\n4. ❌ BUG #15: Model clamps to [0, {L_cur-1}]:")
    print(f"   Result: {landmark_indices_safe[0].tolist()}")
    print(f"   Range: [{landmark_indices_safe.min()}, {landmark_indices_safe.max()}]")

    # Check pour collapse
    unique_positions = torch.unique(landmark_indices_safe)
    print(f"\n5. Unique landmark positions after clamp: {len(unique_positions)}")
    print(f"   Values: {unique_positions.tolist()}")

    if len(unique_positions) < cfg.global_k // 2:
        print(f"\n   ❌ DISASTER: Only {len(unique_positions)}/{cfg.global_k} unique positions!")
        print(f"   ❌ Landmarks collapsed! Most/all at position {L_cur-1}")
        print(f"   ❌ Global attention becomes nearly useless")
        bug_present = True
    else:
        print(f"\n   ✓ Reasonable diversity: {len(unique_positions)}/{cfg.global_k} unique")
        bug_present = False

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Calculer combien de landmarks sont > L_cur-1
    out_of_bounds = (landmark_positions_full >= L_cur).sum().item()
    print(f"\nLandmarks out of bounds: {out_of_bounds}/{cfg.global_k}")

    if out_of_bounds > cfg.global_k // 2:
        print(f"  ❌ More than half landmarks clamped to same position!")
        print(f"  ❌ Global attention severely degraded")

    # ✅ Solution proposée
    print("\n" + "=" * 80)
    print("PROPOSED FIX")
    print("=" * 80)
    print()

    print("Option 1: Regenerate landmarks after truncation")
    regenerated = torch.linspace(0, L_cur-1, cfg.global_k).long()
    print(f"  Regenerated: {regenerated.tolist()}")
    print(f"  Unique: {len(torch.unique(regenerated))}/{cfg.global_k}")
    print(f"  ✅ Full diversity restored!")

    print("\nOption 2: Filter out-of-bounds landmarks")
    valid_mask = landmark_positions_full < L_cur
    filtered = landmark_positions_full[valid_mask]
    print(f"  Filtered: {filtered.tolist()}")
    print(f"  Count: {len(filtered)} (may be < {cfg.global_k})")
    print(f"  ⚠️  Fewer landmarks, need padding")

    print("\n" + "=" * 80)
    if bug_present:
        print("❌ BUG #15 is PRESENT")
        print("\nRecommendation: Regenerate landmarks in train.py after truncation")
        print("  (Option 1 is cleanest)")
        return 1
    else:
        print("✓ BUG #15 is FIXED")
        return 0


if __name__ == "__main__":
    sys.exit(test_landmark_collapse_during_curriculum())
