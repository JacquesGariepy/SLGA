"""
Script de diagnostic pour identifier les problèmes du modèle SLGA
"""
import torch
import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import LLMTransformer, Config
from transformers import GPT2Tokenizer
import yaml

def load_checkpoint_if_provided(model, checkpoint_path):
    """Load checkpoint if path is provided"""
    if checkpoint_path and os.path.exists(checkpoint_path):
        model_path = os.path.join(checkpoint_path, "model.pt")
        if os.path.exists(model_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            state_dict = torch.load(model_path, map_location="cuda")
            model.load_state_dict(state_dict)
            print(f"✓ Checkpoint loaded")
            return True
        else:
            print(f"⚠️  Model file not found at {model_path}")
            return False
    return False

def test_forward_pass(checkpoint_path=None):
    """Test basic forward pass for anomalies"""
    print("="*80)
    print("DIAGNOSTIC: Forward Pass Test")
    print("="*80)

    # Load config
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    model_cfg = Config(**cfg["model"])
    model = LLMTransformer(model_cfg).cuda()

    print(f"✓ Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    # Load checkpoint if provided
    is_trained = load_checkpoint_if_provided(model, checkpoint_path)

    # Test input
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    text = "The quick brown fox jumps over the lazy dog"
    input_ids = tokenizer.encode(text, return_tensors="pt").cuda()

    print(f"✓ Input shape: {input_ids.shape}")

    # Forward pass
    with torch.no_grad():
        try:
            logits, aux = model(input_ids, return_aux=True)
            print(f"✓ Forward pass succeeded")
            print(f"  - Logits shape: {logits.shape}")
            print(f"  - Logits mean: {logits.mean().item():.4f}")
            print(f"  - Logits std: {logits.std().item():.4f}")
            print(f"  - Has NaN: {torch.isnan(logits).any().item()}")
            print(f"  - Has Inf: {torch.isinf(logits).any().item()}")

            # Check aux info
            if 'landmark_gates' in aux and aux['landmark_gates'] is not None:
                gates = aux['landmark_gates']
                print(f"\n  Landmark gates:")
                print(f"  - Shape: {gates.shape}")
                print(f"  - Mean: {gates.mean().item():.4f}")
                print(f"  - Min: {gates.min().item():.4f}")
                print(f"  - Max: {gates.max().item():.4f}")
                print(f"  - Num selected: {(gates > 0.5).sum().item()}")

            # Compute loss
            logits_shifted = logits[:, :-1, :].contiguous()
            labels_shifted = input_ids[:, 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(
                logits_shifted.view(-1, logits_shifted.size(-1)),
                labels_shifted.view(-1),
            )

            ppl = torch.exp(loss).item()
            print(f"\n  Loss: {loss.item():.4f}")
            print(f"  Perplexity: {ppl:.2f}")

            if is_trained:
                # Évaluation pour modèle entraîné
                if ppl > 1000:
                    print("  ❌ CRITICAL: Model hasn't learned (checkpoint may be corrupted)")
                elif ppl > 100:
                    print("  ⚠️  WARNING: Poor performance for trained model")
                elif ppl > 50:
                    print("  🟡 MODERATE: Reasonable but could be better")
                else:
                    print("  ✅ GOOD: Solid performance")
            else:
                # Évaluation pour modèle non-entraîné
                if ppl > 10000:
                    print("  ℹ️  INFO: Normal untrained behavior")
                else:
                    print("  ⚠️  Unexpected for untrained model")

        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    return True

def test_attention_mechanism():
    """Test if attention mechanism works correctly"""
    print("\n" + "="*80)
    print("DIAGNOSTIC: Attention Mechanism Test")
    print("="*80)

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    from src.slga import SLGAModule

    # Small attention module
    attn = SLGAModule(
        embed_dim=512,
        num_heads=8,
        local_window=32,
        global_k=8,
        gated_fusion=True
    ).cuda()

    # Test input
    B, L, D = 2, 64, 512
    x = torch.randn(B, L, D).cuda()

    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        try:
            out = attn(x)
            print(f"✓ Attention forward succeeded")
            print(f"  - Output shape: {out.shape}")
            print(f"  - Output mean: {out.mean().item():.4f}")
            print(f"  - Output std: {out.std().item():.4f}")
            print(f"  - Has NaN: {torch.isnan(out).any().item()}")
            print(f"  - Has Inf: {torch.isinf(out).any().item()}")

            if torch.isnan(out).any() or torch.isinf(out).any():
                print("❌ CRITICAL: Attention produces NaN/Inf")
                return False
            else:
                print("✅ Attention mechanism OK")
                return True

        except Exception as e:
            print(f"❌ Attention failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_learned_landmarks():
    """Test learned landmark selector"""
    print("\n" + "="*80)
    print("DIAGNOSTIC: Learned Landmarks Test")
    print("="*80)

    from src.landmarks import LearnableLandmarkSelector

    selector = LearnableLandmarkSelector(
        embed_dim=512,
        num_landmarks=24
    ).cuda()

    B, L, D = 2, 128, 512
    x = torch.randn(B, L, D).cuda()

    with torch.no_grad():
        try:
            # Le selector retourne: (indices, states, scores)
            indices, states, scores = selector(x)
            print(f"✓ Landmark selector succeeded")
            print(f"  - Indices shape: {indices.shape}")
            print(f"  - States shape: {states.shape}")
            print(f"  - Scores shape: {scores.shape}")
            print(f"  - Num landmarks selected: {indices.size(1)} per batch")
            print(f"  - Scores mean: {scores.mean().item():.4f}")
            print(f"  - Scores max: {scores.max().item():.4f}")
            print(f"  - Scores min: {scores.min().item():.4f}")

            # Vérifier que les indices sont valides
            if indices.max() >= L or indices.min() < 0:
                print(f"❌ CRITICAL: Invalid indices! Range: [{indices.min()}, {indices.max()}], Expected: [0, {L-1}]")
                return False
            else:
                print("✅ Landmark selection OK")
                return True

        except Exception as e:
            print(f"❌ Landmark selector failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diagnostic tool for SLGA model')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                       help='Path to checkpoint to test (optional)')
    args = parser.parse_args()

    print("SLGA Model Diagnostic Tool")
    print("="*80)
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    else:
        print("Mode: Testing untrained model")
    print("="*80 + "\n")

    results = []

    # Test 1: Forward pass
    results.append(("Forward Pass", test_forward_pass(args.checkpoint)))

    # Test 2: Attention mechanism
    results.append(("Attention Mechanism", test_attention_mechanism()))

    # Test 3: Learned landmarks
    results.append(("Learned Landmarks", test_learned_landmarks()))

    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:<30} {status}")

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n✅ All tests passed - architecture seems OK")
        if args.checkpoint:
            print("   Checkpoint loaded successfully and functions correctly")
        else:
            print("   Untrained model architecture is correct")
    else:
        print("\n❌ Some tests failed - architecture has issues")
