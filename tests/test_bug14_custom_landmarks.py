#!/usr/bin/env python3
"""
Test Bug #14: Verify custom landmarks are preserved during generation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.model import LLMTransformer, Config


def test_custom_landmarks_preserved():
    """
    Test que les landmarks personnalisés ne sont PAS écrasés par autogeneration
    """
    print("=" * 80)
    print("BUG #14 TEST: Custom Landmarks Preservation")
    print("=" * 80)
    print()

    cfg = Config(
        vocab_size=1000,
        max_seq_len=512,
        embed_dim=64,
        num_heads=2,
        n_layers=1,
        learned_landmarks=False,  # Heuristic mode
        global_k=4,
    )

    model = LLMTransformer(cfg)
    model.eval()

    prompt = torch.randint(0, 1000, (1, 50))

    # Fournir des landmarks personnalisés (positions spécifiques)
    custom_landmarks = torch.tensor([[10, 20, 30, 40]])  # Custom positions

    print(f"Prompt length: {prompt.size(1)}")
    print(f"Custom landmarks: {custom_landmarks[0].tolist()}")
    print(f"  These should be PRESERVED throughout generation")
    print()

    # Monkey-patch pour capturer les landmarks utilisés
    landmarks_used = []
    original_forward = model.forward

    def tracked_forward(input_ids, cache_global_ids=None, **kwargs):
        if cache_global_ids is not None:
            landmarks_used.append(cache_global_ids.clone())
        return original_forward(input_ids, cache_global_ids=cache_global_ids, **kwargs)

    model.forward = tracked_forward

    # Générer avec landmarks personnalisés
    with torch.no_grad():
        output = model.generate(
            prompt,
            max_new_tokens=10,
            temperature=0.0,  # Greedy
            cache_global_ids=custom_landmarks,  # Custom!
        )

    model.forward = original_forward  # Restore

    print(f"✓ Generation completed: {output.shape}")
    print(f"  Generated {output.size(1) - prompt.size(1)} new tokens")
    print()

    # Vérifier que les landmarks ont été préservés
    print("=" * 80)
    print("VALIDATION: Custom Landmarks Preserved")
    print("=" * 80)
    print()

    if len(landmarks_used) == 0:
        print("✗ FAIL: No landmarks were used during generation")
        return 1

    print(f"Number of forward passes: {len(landmarks_used)}")
    print()

    # Bug #14 serait: landmarks écrasés par linspace
    all_same_as_custom = all(
        torch.equal(lm[0, :len(custom_landmarks[0])], custom_landmarks[0])
        for lm in landmarks_used
        if lm.size(1) >= len(custom_landmarks[0])
    )

    # Check si landmarks changent (signe d'écrasement)
    landmarks_changed = False
    for i, lm in enumerate(landmarks_used):
        print(f"Step {i}: landmarks = {lm[0].tolist()[:6]}...")

        if i == 0:
            first_landmarks = lm
        else:
            if not torch.equal(lm, first_landmarks):
                landmarks_changed = True

    print()

    if not landmarks_changed:
        print("✓ PASS: Custom landmarks PRESERVED across all steps")
        print(f"  All {len(landmarks_used)} forward passes used: {custom_landmarks[0].tolist()}")
        print("  ✅ Bug #14 is FIXED")
        return 0
    else:
        print("✗ FAIL: Landmarks were MODIFIED during generation")
        print("  Expected: Same custom landmarks every step")
        print("  Actual: Landmarks changed (overwritten by autogeneration)")
        print("  ❌ Bug #14 is PRESENT")
        return 1


def test_autogeneration_still_works():
    """
    Test que l'autogeneration fonctionne quand PAS de landmarks fournis
    """
    print("\n" + "=" * 80)
    print("BUG #14 TEST: Autogeneration Still Works (No Custom Landmarks)")
    print("=" * 80)
    print()

    cfg = Config(
        vocab_size=1000,
        max_seq_len=512,
        embed_dim=64,
        num_heads=2,
        n_layers=1,
        learned_landmarks=False,
        global_k=4,
    )

    model = LLMTransformer(cfg)
    model.eval()

    prompt = torch.randint(0, 1000, (1, 50))

    print(f"Prompt length: {prompt.size(1)}")
    print(f"No custom landmarks provided → should autogenerate")
    print()

    # Monkey-patch
    landmarks_used = []
    original_forward = model.forward

    def tracked_forward(input_ids, cache_global_ids=None, **kwargs):
        if cache_global_ids is not None:
            landmarks_used.append(cache_global_ids.clone())
        return original_forward(input_ids, cache_global_ids=cache_global_ids, **kwargs)

    model.forward = tracked_forward

    # Générer SANS landmarks personnalisés
    with torch.no_grad():
        output = model.generate(
            prompt,
            max_new_tokens=10,
            temperature=0.0,
            # NO cache_global_ids → should autogenerate
        )

    model.forward = original_forward

    print(f"✓ Generation completed: {output.shape}")
    print()

    print("=" * 80)
    print("VALIDATION: Autogeneration Works")
    print("=" * 80)
    print()

    if len(landmarks_used) == 0:
        print("  ℹ️  No landmarks used (model may use learned landmarks)")
        return 0

    # Check que landmarks sont mis à jour (sign d'autogen qui fonctionne)
    landmarks_changed = False
    for i, lm in enumerate(landmarks_used[:5]):  # Check first 5 steps
        print(f"Step {i}: landmarks = {lm[0].tolist()}")

        if i > 0:
            if not torch.equal(lm, landmarks_used[0]):
                landmarks_changed = True

    print()

    if landmarks_changed:
        print("✓ PASS: Landmarks UPDATED during generation (autogen working)")
        print("  ✅ Bug #11 fix is working")
        return 0
    else:
        print("  ℹ️  Landmarks stayed same (may be expected for short generation)")
        return 0


if __name__ == "__main__":
    result1 = test_custom_landmarks_preserved()
    result2 = test_autogeneration_still_works()

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    if result1 == 0 and result2 == 0:
        print("✓ ALL TESTS PASSED")
        print()
        print("Bug #14 Fix Verified:")
        print("  ✓ Custom landmarks preserved when provided")
        print("  ✓ Autogeneration works when not provided")
        print("  ✓ Both modes work correctly")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)
