#!/usr/bin/env python3
"""
Test Bugs #18 et #19 dans slga.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from src.slga import SLGAModule


def test_bug18_nan_global_attention():
    """
    Bug #18: Global attention NaN quand toutes valeurs sont -inf après masque
    """
    print("=" * 80)
    print("BUG #18 TEST: NaN in Global Attention (Fully Masked)")
    print("=" * 80)
    print()

    # Setup SLGA module
    slga = SLGAModule(
        embed_dim=64,
        num_heads=4,
        local_window=16,
        global_k=8,
        causal=True,
    )
    slga.eval()

    # Test avec séquence très courte où premiers tokens n'ont pas de landmarks visibles
    batch_size = 2
    seq_len = 10  # Très court
    embed_dim = 64

    x = torch.randn(batch_size, seq_len, embed_dim)

    # Landmarks globaux (très peu)
    cache_global = torch.randn(batch_size, 2, embed_dim)  # Seulement 2 landmarks

    print(f"Input shape: {x.shape}")
    print(f"Global cache: {cache_global.shape}")
    print(f"Scenario: Very short sequence with causal masking")
    print(f"  → Some early tokens may have ALL global scores masked to -inf")
    print()

    try:
        with torch.no_grad():
            output = slga(x, cache_global=cache_global)

        print(f"✓ Forward pass succeeded")
        print(f"  Output shape: {output.shape}")

        # Vérifier pas de NaN
        if torch.isnan(output).any():
            print(f"  ❌ FAIL: Output contains NaN")
            print(f"  Bug #18 is PRESENT")
            return 1
        else:
            print(f"  ✓ PASS: No NaN in output")
            print(f"  Bug #18 is FIXED")

    except Exception as e:
        print(f"❌ FAIL: Exception during forward: {e}")
        return 1

    return 0


if __name__ == "__main__":
    result1 = test_bug18_nan_global_attention()

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    if result1 == 0:
        print("✓ ALL TESTS PASSED")
        print()
        print("Bugs fixed:")
        print("  ✓ Bug #18: NaN in global attention (fully masked)")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)
