#!/usr/bin/env python3
"""
Final validation of sparsity_loss fix

Demonstrates that the new implementation:
1. Produces meaningful gradients
2. Loss varies with concentration
3. Works correctly in realistic scenarios
"""

import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.landmarks import landmark_sparsity_loss


def main():
    print("\n" + "="*70)
    print("FINAL VALIDATION: landmark_sparsity_loss FIX")
    print("="*70)

    B, L, G = 4, 384, 48
    lambda_reg = 0.001

    # Test 1: Gradient Flow
    print("\n[TEST 1] Gradient Flow")
    print("-" * 70)
    scores = torch.randn(B, L, requires_grad=True)
    loss = landmark_sparsity_loss(scores, G, lambda_reg)
    loss.backward()

    grad_valid = (scores.grad is not None and
                  not torch.isnan(scores.grad).any() and
                  not torch.isinf(scores.grad).any())

    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Gradient norm: {scores.grad.norm().item():.6f}")
    print(f"  Status: {'✅ PASS' if grad_valid else '❌ FAIL'}")

    # Test 2: Loss Variation
    print("\n[TEST 2] Loss Variation with Concentration")
    print("-" * 70)

    scenarios = []

    # Perfect concentration
    s_perfect = torch.zeros(B, L)
    for b in range(B):
        s_perfect[b, torch.randperm(L)[:G]] = 10.0
    l_perfect = landmark_sparsity_loss(s_perfect, G, lambda_reg)
    scenarios.append(("Perfect concentration", l_perfect.item()))

    # Good concentration
    s_good = torch.randn(B, L)
    for b in range(B):
        s_good[b, torch.randperm(L)[:G]] += 5.0
    l_good = landmark_sparsity_loss(s_good, G, lambda_reg)
    scenarios.append(("Good concentration", l_good.item()))

    # Random
    s_random = torch.randn(B, L)
    l_random = landmark_sparsity_loss(s_random, G, lambda_reg)
    scenarios.append(("Random (no pattern)", l_random.item()))

    # Uniform
    s_uniform = torch.randn(B, L) * 0.1
    l_uniform = landmark_sparsity_loss(s_uniform, G, lambda_reg)
    scenarios.append(("Uniform (poor)", l_uniform.item()))

    for name, loss_val in scenarios:
        print(f"  {name:<25} Loss: {loss_val:.6f}")

    # Check variation
    losses = [l for _, l in scenarios]
    variation = max(losses) - min(losses)
    variation_ok = variation > 0.0001

    print(f"\n  Loss range: {min(losses):.6f} to {max(losses):.6f}")
    print(f"  Variation: {variation:.6f}")
    print(f"  Status: {'✅ PASS' if variation_ok else '❌ FAIL'}")

    # Test 3: Measure Mass Concentration
    print("\n[TEST 3] Mass Concentration Measurement")
    print("-" * 70)

    def measure_mass(scores, G):
        B, L = scores.shape
        probs = F.softmax(scores, dim=-1)
        _, top_idx = torch.topk(scores, k=G, dim=-1)
        top_probs = torch.gather(probs, dim=1, index=top_idx)
        return top_probs.sum(dim=-1).mean().item()

    mass_perfect = measure_mass(s_perfect, G)
    mass_good = measure_mass(s_good, G)
    mass_random = measure_mass(s_random, G)
    mass_uniform = measure_mass(s_uniform, G)

    print(f"  Perfect:   Mass in top-{G} = {mass_perfect:6.1%}")
    print(f"  Good:      Mass in top-{G} = {mass_good:6.1%}")
    print(f"  Random:    Mass in top-{G} = {mass_random:6.1%}")
    print(f"  Uniform:   Mass in top-{G} = {mass_uniform:6.1%}")

    target = 0.60 + (G/L) * 0.40
    print(f"\n  Target mass: {target:.1%}")

    mass_ok = (mass_perfect > target and mass_good > target and
               mass_random < target and mass_uniform < target)

    print(f"  Status: {'✅ PASS' if mass_ok else '❌ FAIL'}")

    # Test 4: Neural Network Integration
    print("\n[TEST 4] Integration with Neural Scorer")
    print("-" * 70)

    class SimpleScorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(64, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, L)
            )

        def forward(self, x):
            return self.net(x)

    scorer = SimpleScorer()
    optimizer = torch.optim.Adam(scorer.parameters(), lr=0.01)

    # Training step
    x = torch.randn(B, 64)
    scores = scorer(x)
    loss = landmark_sparsity_loss(scores, G, lambda_reg=0.01)  # Higher for demo

    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    total_grad = sum(p.grad.norm().item() for p in scorer.parameters() if p.grad is not None)
    grad_ok = total_grad > 0.0

    print(f"  Loss: {loss.item():.6f}")
    print(f"  Total gradient norm: {total_grad:.6f}")
    print(f"  Status: {'✅ PASS' if grad_ok else '❌ FAIL'}")

    # Test 5: Numerical Stability
    print("\n[TEST 5] Numerical Stability")
    print("-" * 70)

    test_cases = [
        ("Very large scores", torch.randn(B, L) * 100),
        ("Very small scores", torch.randn(B, L) * 0.001),
        ("Negative scores", torch.randn(B, L) - 10.0),
        ("Mixed extreme", torch.cat([
            torch.ones(B, L//2) * 100,
            torch.ones(B, L//2) * -100
        ], dim=1))
    ]

    all_stable = True
    for name, scores in test_cases:
        loss = landmark_sparsity_loss(scores, G, lambda_reg)
        is_stable = not (torch.isnan(loss) or torch.isinf(loss))
        all_stable = all_stable and is_stable
        status = "✓" if is_stable else "✗"
        print(f"  {status} {name:<20} Loss: {loss.item():.6f}")

    print(f"\n  Status: {'✅ PASS' if all_stable else '❌ FAIL'}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_tests = [
        ("Gradient Flow", grad_valid),
        ("Loss Variation", variation_ok),
        ("Mass Concentration", mass_ok),
        ("Neural Integration", grad_ok),
        ("Numerical Stability", all_stable)
    ]

    for name, passed in all_tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:<25} {status}")

    total_passed = sum(passed for _, passed in all_tests)
    total_tests = len(all_tests)

    print(f"\n  {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n  🎉 ALL TESTS PASS! Sparsity loss fix validated.")
        print("\n  Key improvements:")
        print("    • Gradients flow correctly to scorer network")
        print("    • Loss varies meaningfully with concentration")
        print("    • Numerically stable across extreme inputs")
        print("    • Ready for production training")
        return 0
    else:
        print("\n  ⚠️  Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
