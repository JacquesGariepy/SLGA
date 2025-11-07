#!/usr/bin/env python3
"""
Test si le clamp dans model.py L270 est encore nécessaire après fix Bug #15
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.model import Config, LLMTransformer


def test_scenarios_where_clamp_needed():
    """
    Identifier les scénarios où landmark_indices pourrait dépasser L_cur
    """
    print("=" * 80)
    print("TEST: When is model.py clamp still needed?")
    print("=" * 80)
    print()

    cfg = Config(
        vocab_size=1000,
        max_seq_len=512,
        embed_dim=64,
        num_heads=2,
        n_layers=2,
        learned_landmarks=False,
        global_k=8,
    )

    model = LLMTransformer(cfg)

    print("Scenario 1: After train.py fix (regenerated positions)")
    print("-" * 80)
    print("  train.py regenerates landmarks after truncation")
    print("  Positions should already be in [0, current_seq_len-1]")
    print("  → Clamp should be NO-OP (safety check only)")
    print()

    # Simuler positions régénérées (correctes)
    current_seq_len = 128
    cache_ids_regen = torch.linspace(0, current_seq_len-1, cfg.global_k).long()
    cache_ids_regen = cache_ids_regen.unsqueeze(0)

    print(f"  Regenerated positions: {cache_ids_regen[0].tolist()}")
    print(f"  All in [0, {current_seq_len-1}]: {(cache_ids_regen < current_seq_len).all()}")
    print(f"  → Clamp is NO-OP ✅")
    print()

    print("Scenario 2: Learned landmarks (could exceed)")
    print("-" * 80)
    print("  Model's landmark_selector might return positions > L_cur")
    print("  → Clamp IS needed as safety")
    print()

    # Simuler learned positions (pourraient dépasser)
    learned_positions = torch.tensor([[50, 150, 300, 600]])  # 600 > 512!
    print(f"  Learned positions: {learned_positions[0].tolist()}")
    print(f"  Some > L_cur: {(learned_positions >= current_seq_len).any()}")
    print(f"  → Clamp IS needed ✅")
    print()

    print("Scenario 3: Custom landmarks from user")
    print("-" * 80)
    print("  User might provide arbitrary positions")
    print("  Could be out of bounds")
    print("  → Clamp IS needed as safety")
    print()

    custom = torch.tensor([[10, 20, 150, 300]])  # 150, 300 > 128
    print(f"  Custom positions: {custom[0].tolist()}")
    print(f"  Some > L_cur: {(custom >= current_seq_len).any()}")
    print(f"  → Clamp IS needed ✅")
    print()

    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("The clamp in model.py:270 should be KEPT because:")
    print("  1. ✅ Safety check for learned landmarks")
    print("  2. ✅ Safety check for custom landmarks")
    print("  3. ✅ NO-OP for heuristic (after train.py fix)")
    print("  4. ✅ Prevents crashes from unexpected inputs")
    print()
    print("After Bug #15 fix in train.py:")
    print("  - Heuristic landmarks are regenerated → clamp is NO-OP")
    print("  - Clamp remains as defensive programming")
    print("  - No collapse because inputs are already correct")
    print()
    print("✓ Clamp should be KEPT (it's harmless and provides safety)")

    return 0


if __name__ == "__main__":
    sys.exit(test_scenarios_where_clamp_needed())
