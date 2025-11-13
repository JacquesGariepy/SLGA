# scripts/eval_perplexity.py
"""
Comprehensive perplexity test for SLGA model
Tests model understanding on clean English text
"""

import torch
import yaml
import sys
import argparse
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.legacy.model import LLMTransformer, Config
from transformers import GPT2Tokenizer


def load_checkpoint(checkpoint_dir, model=None, device="cuda"):
    """Load checkpoint from directory - custom implementation"""
    model_path = os.path.join(checkpoint_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    
    if model is not None:
        model.load_state_dict(state_dict)
        model.to(device)
    
    # Get step number
    trainer_path = os.path.join(checkpoint_dir, "trainer_state.pt")
    step = 0
    if os.path.exists(trainer_path):
        trainer_state = torch.load(trainer_path, map_location=device)
        step = trainer_state.get('step', 0)
    
    return step


def test_checkpoint(checkpoint_path, verbose=True):
    """Test a single checkpoint on clean text samples"""
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing checkpoint: {checkpoint_path}")
        print(f"{'='*80}\n")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Load config
    try:
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ config.yaml not found")
        return None
    
    # Create model from YOUR custom architecture
    model_cfg = Config(**cfg["model"])
    model = LLMTransformer(model_cfg)
    
    # Load checkpoint using custom load_checkpoint function
    try:
        step = load_checkpoint(checkpoint_path, model, device="cuda")
        if verbose:
            print(f"✓ Checkpoint loaded successfully (step {step})")
            param_count = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"✓ Model parameters: {param_count:.2f}M\n")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    model = model.cuda().eval()
    
    # Test texts - diverse and progressively complex
    test_categories = {
        "Simple Sentences": [
            "The cat sat on the mat.",
            "Hello world!",
            "I love you.",
        ],
        "Common Phrases": [
            "The quick brown fox jumps over the lazy dog.",
            "Once upon a time there was a princess.",
            "In the beginning God created the heavens and the earth.",
        ],
        "Encyclopedic": [
            "Anarchism is a political philosophy and movement.",
            "The history of artificial intelligence began in the 1950s.",
            "In the year 2024 many people use computers daily.",
        ],
        "Narrative": [
            "Once upon a time in a faraway land there lived a wise old wizard.",
            "The sun was setting over the mountains as the travelers approached.",
        ]
    }
    
    all_results = []
    category_results = {}
    
    if verbose:
        print(f"{'Category':<20} {'Text':<60} {'Loss':<8} {'PPL':<10} Status")
        print("-" * 110)
    
    for category, texts in test_categories.items():
        category_losses = []
        category_ppls = []
        
        for text in texts:
            input_ids = tokenizer.encode(text, return_tensors="pt").cuda()

            with torch.no_grad():
                # Get logits from model
                logits = model(input_ids)  # (B, L, V)

                # Calculate loss manually (shift for causal LM)
                logits_shifted = logits[:, :-1, :].contiguous()  # (B, L-1, V)
                labels_shifted = input_ids[:, 1:].contiguous()  # (B, L-1)

                loss_tensor = torch.nn.functional.cross_entropy(
                    logits_shifted.view(-1, logits_shifted.size(-1)),
                    labels_shifted.view(-1),
                )

                loss = loss_tensor.item()
                ppl = torch.exp(loss_tensor).item()
            
            category_losses.append(loss)
            category_ppls.append(ppl)
            all_results.append((text, loss, ppl))
            
            # Status indicator
            if ppl > 1000:
                status = "❌ CRITICAL"
            elif ppl > 500:
                status = "⚠️  WARNING"
            elif ppl > 100:
                status = "🟡 MODERATE"
            else:
                status = "✅ OK"
            
            # Truncate text for display
            display_text = text[:55] + "..." if len(text) > 55 else text
            
            if verbose:
                print(f"{category:<20} {display_text:<60} {loss:>7.4f} {ppl:>9.2f} {status}")
        
        category_results[category] = {
            'avg_loss': sum(category_losses) / len(category_losses),
            'avg_ppl': sum(category_ppls) / len(category_ppls)
        }
    
    # Overall statistics
    all_losses = [r[1] for r in all_results]
    all_ppls = [r[2] for r in all_results]
    
    avg_loss = sum(all_losses) / len(all_losses)
    avg_ppl = sum(all_ppls) / len(all_ppls)
    median_ppl = sorted(all_ppls)[len(all_ppls) // 2]
    max_ppl = max(all_ppls)
    min_ppl = min(all_ppls)
    
    if verbose:
        print("\n" + "="*80)
        print("STATISTICS")
        print("="*80)
        print(f"Average Loss:   {avg_loss:.4f}")
        print(f"Average PPL:    {avg_ppl:.2f}")
        print(f"Median PPL:     {median_ppl:.2f}")
        print(f"Min PPL:        {min_ppl:.2f}")
        print(f"Max PPL:        {max_ppl:.2f}")
        
        print("\nPer Category:")
        for cat, results in category_results.items():
            print(f"  {cat:<20} Loss: {results['avg_loss']:.4f}  PPL: {results['avg_ppl']:.2f}")
        
        # Verdict
        print("\n" + "="*80)
        print("VERDICT")
        print("="*80)
        
        if avg_ppl > 1000:
            print("🚨 CRITICAL: Model hasn't learned basic language patterns")
            print("   → Training is fundamentally broken")
            print("   → Loss may be decreasing but model isn't learning language")
            print("   → RECOMMENDATION: Restart training with verified setup")
        elif avg_ppl > 500:
            print("⚠️  WARNING: Model has very weak understanding")
            print("   → Significant issues in training process")
            print("   → Dataset quality or architecture may be problematic")
            print("   → RECOMMENDATION: Investigate training data and architecture")
        elif avg_ppl > 200:
            print("🟡 MODERATE: Model has some understanding but limited")
            print("   → Training is working but suboptimal")
            print("   → May improve with more steps")
            print("   → RECOMMENDATION: Continue but monitor closely")
        elif avg_ppl > 100:
            print("🟢 FAIR: Model shows reasonable understanding")
            print("   → Training is progressing normally")
            print("   → Generation quality should improve with more training")
            print("   → RECOMMENDATION: Continue training")
        else:
            print("✅ GOOD: Model has solid understanding")
            print("   → Training is successful")
            print("   → Poor generation is likely a sampling/decoding issue")
            print("   → RECOMMENDATION: Debug generation script, not training")
    
    return {
        'checkpoint': checkpoint_path,
        'step': step,
        'avg_loss': avg_loss,
        'avg_ppl': avg_ppl,
        'median_ppl': median_ppl,
        'all_ppls': all_ppls,
        'category_results': category_results
    }


def compare_checkpoints(checkpoint_paths, verbose=True):
    """Compare multiple checkpoints"""
    
    if verbose:
        print("="*80)
        print("COMPARATIVE PERPLEXITY ANALYSIS")
        print("="*80)
    
    results = {}
    
    for ckpt_path in checkpoint_paths:
        result = test_checkpoint(ckpt_path, verbose=verbose)
        if result:
            results[ckpt_path] = result
    
    if len(results) < 2:
        return results
    
    # Comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Checkpoint':<30} {'Step':<10} {'Avg Loss':<12} {'Avg PPL':<12} {'Median PPL':<12}")
    print("-" * 80)
    
    # Sort by step number
    sorted_ckpts = sorted(results.items(), key=lambda x: x[1]['step'])
    
    for ckpt_path, res in sorted_ckpts:
        ckpt_name = ckpt_path.split('/')[-1]
        print(f"{ckpt_name:<30} {res['step']:<10} {res['avg_loss']:>11.4f} {res['avg_ppl']:>11.2f} {res['median_ppl']:>11.2f}")
    
    # Calculate improvement
    if len(sorted_ckpts) >= 2:
        first_step = sorted_ckpts[0][1]['step']
        last_step = sorted_ckpts[-1][1]['step']
        
        first_ppl = sorted_ckpts[0][1]['avg_ppl']
        last_ppl = sorted_ckpts[-1][1]['avg_ppl']
        
        improvement = (first_ppl - last_ppl) / first_ppl * 100
        
        print(f"\n{'='*80}")
        print(f"Step {first_step} → Step {last_step}")
        print(f"PPL: {first_ppl:.2f} → {last_ppl:.2f}")
        
        if improvement > 0:
            print(f"Improvement: {improvement:.1f}% ✅")
        else:
            print(f"Degradation: {abs(improvement):.1f}% ❌")
        print(f"{'='*80}")
    
    return results


def main():
    """Main test function"""
    
    parser = argparse.ArgumentParser(description='Evaluate perplexity on SLGA checkpoints')
    parser.add_argument('--checkpoint', '-c', type=str, nargs='+', 
                       help='Checkpoint path(s) to test')
    parser.add_argument('--auto', '-a', action='store_true',
                       help='Auto-detect and test all checkpoints in out_slga/')
    
    args = parser.parse_args()
    
    # Determine which checkpoints to test
    checkpoints = []
    
    if args.auto:
        # Auto-detect checkpoints
        ckpt_dir = "out_slga"
        if os.path.exists(ckpt_dir):
            all_dirs = [os.path.join(ckpt_dir, d) for d in os.listdir(ckpt_dir) 
                       if d.startswith('ckpt_') and os.path.isdir(os.path.join(ckpt_dir, d))]
            checkpoints = sorted(all_dirs, key=lambda x: int(x.split('_')[-1]))
        else:
            print(f"❌ Directory {ckpt_dir} not found")
            return
    elif args.checkpoint:
        checkpoints = args.checkpoint
    else:
        # Default: test latest available
        default_ckpts = ["out_slga/ckpt_10000", "out_slga/ckpt_22000", "out_slga/ckpt_30000"]
        checkpoints = [c for c in default_ckpts if os.path.exists(c)]
    
    # Filter to only existing checkpoints
    existing_ckpts = [c for c in checkpoints if os.path.exists(c)]
    
    if not existing_ckpts:
        print("❌ No valid checkpoints found!")
        print(f"Searched for: {checkpoints}")
        print("\nUsage examples:")
        print("  python scripts/eval_perplexity.py --checkpoint out_slga/ckpt_30000")
        print("  python scripts/eval_perplexity.py --checkpoint out_slga/ckpt_10000 out_slga/ckpt_30000")
        print("  python scripts/eval_perplexity.py --auto  # Test all checkpoints")
        return
    
    print(f"Found {len(existing_ckpts)} checkpoint(s) to test")
    
    if len(existing_ckpts) == 1:
        test_checkpoint(existing_ckpts[0], verbose=True)
    else:
        compare_checkpoints(existing_ckpts, verbose=True)


if __name__ == "__main__":
    main()