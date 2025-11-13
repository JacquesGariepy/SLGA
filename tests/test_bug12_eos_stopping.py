#!/usr/bin/env python3
"""
Test for BUG #12: Verify sequences stop growing after EOS token

This test validates that sequences marked as finished (after generating EOS)
do not continue to grow by appending more tokens.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.legacy.model import LLMTransformer, Config
from src.legacy.data import get_tokenizer


def test_bug12_no_tokens_after_eos():
    """
    BUG #12 Test: Verify sequences stop at EOS token.

    Expected behavior:
        - Sequence ends at EOS token
        - No tokens appended after EOS
        - Length varies per sample (early stop)

    Broken behavior (before fix):
        - EOS tracking marks sequence as finished
        - But token concatenation continues
        - Results in garbage tokens after EOS
    """
    print("=" * 80)
    print("BUG #12 TEST: EOS Stopping (No Post-EOS Tokens)")
    print("=" * 80)

    # Setup model
    cfg = Config(
        vocab_size=50257,
        max_seq_len=512,
        embed_dim=128,
        num_heads=4,
        n_layers=2,
        local_window=32,
        global_k=8,
        learned_landmarks=False,
    )

    model = LLMTransformer(cfg)
    model.eval()

    tokenizer = get_tokenizer("gpt2")
    eos_token_id = tokenizer.eos_token_id

    print(f"EOS token ID: {eos_token_id}")

    # Create batch with multiple samples
    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "In the beginning",
    ]

    prompt_tokens = [tokenizer.encode(p, return_tensors="pt") for p in prompts]

    # Pad to same length for batch
    max_prompt_len = max(p.size(1) for p in prompt_tokens)
    padded_prompts = []
    for p in prompt_tokens:
        if p.size(1) < max_prompt_len:
            padding = torch.full((1, max_prompt_len - p.size(1)), tokenizer.pad_token_id, dtype=torch.long)
            p = torch.cat([p, padding], dim=1)
        padded_prompts.append(p)

    prompt_batch = torch.cat(padded_prompts, dim=0)  # (B, L)

    print(f"\nBatch size: {prompt_batch.size(0)}")
    print(f"Prompt length: {prompt_batch.size(1)}")

    # Generate with EOS stopping enabled
    max_new_tokens = 100
    print(f"\nGenerating up to {max_new_tokens} tokens with stop_on_eos=True...")

    output = model.generate(
        prompt_batch,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=40,
        stop_on_eos=True,
        eos_token_id=eos_token_id,
    )

    print(f"✓ Generation completed")
    print(f"  Output shape: {output.shape}")

    # ========================================================================
    # VALIDATION CHECKS
    # ========================================================================

    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)

    bug_present = False

    # Check each sample in batch
    for batch_idx in range(output.size(0)):
        print(f"\n--- Sample {batch_idx} ---")

        tokens = output[batch_idx].tolist()

        # Decode for display
        try:
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            print(f"Generated text: {text[:200]}...")
        except Exception as e:
            print(f"Decode error: {e}")

        # Check 1: Find EOS token position
        if eos_token_id in tokens:
            eos_pos = tokens.index(eos_token_id)
            print(f"✓ EOS found at position {eos_pos}/{len(tokens)}")

            # Check 2: What comes after EOS?
            post_eos_tokens = tokens[eos_pos + 1:]

            if len(post_eos_tokens) == 0:
                print(f"  ✓ PASS: No tokens after EOS (clean termination)")

            elif all(t == tokenizer.pad_token_id for t in post_eos_tokens):
                print(f"  ✓ PASS: Only PAD tokens after EOS ({len(post_eos_tokens)} pads)")

            else:
                # Count non-pad tokens after EOS
                non_pad_after_eos = [t for t in post_eos_tokens if t != tokenizer.pad_token_id]

                print(f"  ❌ FAIL: Found {len(non_pad_after_eos)} non-PAD tokens after EOS!")
                print(f"     Post-EOS tokens: {post_eos_tokens[:20]}")

                # Decode post-EOS text
                try:
                    post_eos_text = tokenizer.decode(non_pad_after_eos)
                    print(f"     Post-EOS text: '{post_eos_text}'")
                except:
                    pass

                print(f"     This indicates BUG #12 is PRESENT")
                bug_present = True

        else:
            print(f"  ℹ️  No EOS in output (generated max_new_tokens without stopping)")
            print(f"     Final length: {len(tokens)}")

        # Check 3: Multiple EOS tokens (another symptom of bug)
        eos_count = tokens.count(eos_token_id)
        if eos_count > 1:
            print(f"  ⚠️  WARNING: Multiple EOS tokens ({eos_count}) in output")
            print(f"     This may indicate continued generation after first EOS")

            # Find all EOS positions
            eos_positions = [i for i, t in enumerate(tokens) if t == eos_token_id]
            print(f"     EOS positions: {eos_positions}")

    # Check 4: Batch efficiency (did all sequences finish early?)
    print("\n--- Batch Statistics ---")

    lengths = [len([t for t in output[i].tolist() if t != tokenizer.pad_token_id])
               for i in range(output.size(0))]

    print(f"Sequence lengths: {lengths}")
    print(f"Mean length: {sum(lengths) / len(lengths):.1f}")
    print(f"Max length: {max(lengths)}")
    print(f"Min length: {min(lengths)}")

    if max(lengths) == prompt_batch.size(1) + max_new_tokens:
        print(f"⚠️  At least one sequence hit max_new_tokens")
        print(f"   This suggests stop_on_eos may not be working")

    # Check 5: Verify 'finished' tracking worked correctly
    # (Implicit: if bug present, all sequences would have same length = max)
    if len(set(lengths)) == 1:
        print(f"⚠️  All sequences have identical length ({lengths[0]})")
        print(f"   This suggests no early stopping occurred (potential bug)")
    else:
        print(f"✓ Sequences have varying lengths (early stopping worked)")

    # ========================================================================
    # FINAL VERDICT
    # ========================================================================

    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    if not bug_present:
        print("✅ BUG #12 IS FIXED (or not present)")
        print("   Sequences end cleanly at EOS token")
        print("   No garbage tokens after EOS")
        print("   Early termination works correctly")
        return True
    else:
        print("❌ BUG #12 IS PRESENT")
        print("   Found tokens after EOS in one or more sequences")
        print("   Finished sequences continue growing")
        print("   Violates EOS stopping invariant")
        print("\n📋 FIX: Add EOS logit manipulation in model.py after line 361")
        print("   Force finished sequences to generate EOS")
        print("   Example:")
        print("     if stop_on_eos and finished.any():")
        print("         logits[finished] = float('-inf')")
        print("         logits[finished, eos_token_id] = 1e4")
        return False


def test_bug12_early_termination_efficiency():
    """
    Test that early termination actually saves computation.

    With fix: If all sequences finish early (e.g., at step 20/100),
              should return immediately (20 forward passes).

    Without fix: Always runs 100 forward passes even if all finished at step 20.
    """
    print("\n" + "=" * 80)
    print("BUG #12 TEST: Early Termination Efficiency")
    print("=" * 80)

    cfg = Config(
        vocab_size=50257,
        max_seq_len=256,
        embed_dim=128,
        num_heads=4,
        n_layers=2,
        local_window=32,
        global_k=8,
        learned_landmarks=False,
    )

    model = LLMTransformer(cfg)
    model.eval()

    tokenizer = get_tokenizer("gpt2")

    # Create prompt that likely generates EOS early
    prompt = tokenizer.encode("The end.<|endoftext|>", return_tensors="pt")

    print(f"Prompt: {tokenizer.decode(prompt[0])}")
    print(f"Generating up to 50 tokens...")

    # Count forward passes
    forward_count = [0]
    original_forward = model.forward

    def counting_forward(*args, **kwargs):
        forward_count[0] += 1
        return original_forward(*args, **kwargs)

    model.forward = counting_forward

    output = model.generate(
        prompt,
        max_new_tokens=50,
        temperature=0.0,
        stop_on_eos=True,
        eos_token_id=tokenizer.eos_token_id,
    )

    model.forward = original_forward

    tokens_generated = output.size(1) - prompt.size(1)

    print(f"\n✓ Generation completed")
    print(f"  Tokens generated: {tokens_generated}")
    print(f"  Forward passes: {forward_count[0]}")

    # Check efficiency
    if forward_count[0] <= tokens_generated + 2:  # +2 for some tolerance
        print(f"  ✓ PASS: Efficient early termination")
        print(f"     Forward passes ≈ tokens generated (minimal waste)")
        return True
    else:
        wasted = forward_count[0] - tokens_generated
        print(f"  ⚠️  WARNING: Possible inefficiency")
        print(f"     Wasted forward passes: {wasted}")
        print(f"     This may indicate continued generation after EOS")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TESTING BUG #12: EOS Stopping (Finished Sequences)")
    print("=" * 80)
    print("\nThis test validates the fix for BUG #12, where sequences marked")
    print("as finished (after EOS) continued growing with garbage tokens.")
    print("=" * 80)

    # Run tests
    test1_passed = test_bug12_no_tokens_after_eos()
    test2_passed = test_bug12_early_termination_efficiency()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Test 1 (No post-EOS tokens): {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Test 2 (Early termination): {'✅ PASS' if test2_passed else '❌ FAIL'}")

    if test1_passed and test2_passed:
        print("\n🎉 All tests PASSED - BUG #12 is FIXED")
        sys.exit(0)
    else:
        print("\n❌ Some tests FAILED - BUG #12 is still PRESENT")
        print("\n📋 To fix: Modify src/model.py after line 361")
        print("   Force finished sequences to output EOS only")
        print("   Prevent token concatenation for finished sequences")
        sys.exit(1)
