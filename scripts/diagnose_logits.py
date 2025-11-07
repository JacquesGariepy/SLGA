#!/usr/bin/env python3
"""
Diagnostic script to analyze model logits and sampling behavior.
Verifies that the model produces reasonable predictions and that
sampling correctly selects tokens.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import LLMTransformer, Config
from transformers import AutoTokenizer
import yaml


def load_checkpoint(ckpt_path: str, device: str = "cuda", config_path: str = "config.yaml"):
    """Load model checkpoint and tokenizer."""
    print(f"Loading checkpoint from: {ckpt_path}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"Checkpoint keys: {list(ckpt.keys())}")

    # Load config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    model_cfg = config_dict['model']
    print(f"Model config: {model_cfg}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Create Config object from yaml config
    cfg = Config(
        vocab_size=len(tokenizer),
        max_seq_len=model_cfg['max_seq_len'],
        embed_dim=model_cfg['embed_dim'],
        num_heads=model_cfg['num_heads'],
        ff_hidden_multiplier=model_cfg.get('ff_hidden_multiplier', 4),
        n_layers=model_cfg['n_layers'],
        dropout_rate=model_cfg.get('dropout_rate', 0.1),
        local_window=model_cfg.get('local_window', 128),
        global_k=model_cfg.get('global_k', 24),
        gated_fusion=model_cfg.get('gated_fusion', True),
        learned_landmarks=model_cfg.get('learned_landmarks', True),
        dilated_windows=model_cfg.get('dilated_windows', True),
        diverse_topk=model_cfg.get('diverse_topk', True),
        grad_checkpointing=False,  # Disable for inference
    )

    # Initialize model
    model = LLMTransformer(cfg).to(device)

    # Load state dict
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
        print(f"Loaded model from checkpoint (step {ckpt.get('step', 'unknown')})")
    else:
        model.load_state_dict(ckpt)
        print("Loaded model from checkpoint (legacy format)")

    model.eval()
    return model, tokenizer


def analyze_logits(logits: torch.Tensor, tokenizer, top_k: int = 20):
    """Analyze logits and return detailed statistics."""
    # Get the last token's logits
    last_logits = logits[0, -1, :]  # Shape: [vocab_size]

    # Compute probabilities
    probs = F.softmax(last_logits, dim=-1)

    # Get top-k tokens
    top_probs, top_indices = torch.topk(probs, top_k)

    # Get argmax (what temperature=0.0 would select)
    argmax_idx = torch.argmax(last_logits)
    argmax_prob = probs[argmax_idx]

    # Check where "Paris" ranks
    paris_token_id = 6342  # " Paris" in GPT-2
    paris_logit = last_logits[paris_token_id].item()
    paris_prob = probs[paris_token_id].item()

    # Find rank of Paris
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    paris_rank = (sorted_indices == paris_token_id).nonzero(as_tuple=True)[0].item() + 1

    results = {
        'logits_stats': {
            'min': last_logits.min().item(),
            'max': last_logits.max().item(),
            'mean': last_logits.mean().item(),
            'std': last_logits.std().item(),
        },
        'top_tokens': [
            {
                'rank': i + 1,
                'token': tokenizer.decode([idx]),
                'token_id': idx.item(),
                'logit': last_logits[idx].item(),
                'probability': prob.item(),
            }
            for i, (idx, prob) in enumerate(zip(top_indices, top_probs))
        ],
        'argmax': {
            'token': tokenizer.decode([argmax_idx]),
            'token_id': argmax_idx.item(),
            'logit': last_logits[argmax_idx].item(),
            'probability': argmax_prob.item(),
        },
        'paris_info': {
            'token': ' Paris',
            'token_id': paris_token_id,
            'rank': paris_rank,
            'logit': paris_logit,
            'probability': paris_prob,
        }
    }

    return results


def apply_temperature_sampling(logits: torch.Tensor, temperature: float):
    """Apply temperature and return selected token."""
    last_logits = logits[0, -1, :]

    if temperature == 0.0:
        # Greedy sampling
        return torch.argmax(last_logits).item()
    else:
        # Temperature sampling
        scaled_logits = last_logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()


def print_results(results: dict, tokenizer):
    """Print analysis results in a readable format."""
    print("\n" + "="*80)
    print("LOGITS ANALYSIS")
    print("="*80)

    # Logits statistics
    stats = results['logits_stats']
    print("\n📊 Logits Statistics:")
    print(f"  Min:  {stats['min']:>10.4f}")
    print(f"  Max:  {stats['max']:>10.4f}")
    print(f"  Mean: {stats['mean']:>10.4f}")
    print(f"  Std:  {stats['std']:>10.4f}")

    # Top tokens
    print("\n🔝 Top-20 Tokens:")
    print(f"{'Rank':<6} {'Token':<20} {'Token ID':<10} {'Logit':<12} {'Probability':<12}")
    print("-" * 80)
    for token_info in results['top_tokens']:
        token_repr = repr(token_info['token'])
        print(f"{token_info['rank']:<6} {token_repr:<20} {token_info['token_id']:<10} "
              f"{token_info['logit']:<12.4f} {token_info['probability']:<12.6f}")

    # Argmax selection
    print("\n🎯 Argmax Selection (Temperature = 0.0):")
    argmax = results['argmax']
    print(f"  Token:       {repr(argmax['token'])}")
    print(f"  Token ID:    {argmax['token_id']}")
    print(f"  Logit:       {argmax['logit']:.4f}")
    print(f"  Probability: {argmax['probability']:.6f}")

    # Paris analysis
    print("\n🗼 'Paris' Analysis (Expected Answer):")
    paris = results['paris_info']
    print(f"  Token:       {repr(paris['token'])}")
    print(f"  Token ID:    {paris['token_id']}")
    print(f"  Rank:        #{paris['rank']}")
    print(f"  Logit:       {paris['logit']:.4f}")
    print(f"  Probability: {paris['probability']:.6f}")

    # Check if argmax is reasonable
    print("\n✅ Sanity Checks:")
    if argmax['probability'] > 0.01:
        print(f"  ✓ Argmax probability is reasonable: {argmax['probability']:.2%}")
    else:
        print(f"  ⚠ Argmax probability is very low: {argmax['probability']:.2%}")

    if argmax['token_id'] == results['top_tokens'][0]['token_id']:
        print(f"  ✓ Argmax matches top-1 prediction")
    else:
        print(f"  ✗ Argmax does NOT match top-1 (this should never happen!)")

    if paris['rank'] > 100:
        print(f"  ⚠ 'Paris' ranks very low (#{paris['rank']}) - model lacks factual knowledge")
    elif paris['rank'] > 20:
        print(f"  ⚠ 'Paris' not in top-20 (rank #{paris['rank']}) - needs more training")
    else:
        print(f"  ✓ 'Paris' is in top-20 (rank #{paris['rank']})")


def main():
    print("="*80)
    print("SLGA Logits Diagnostic Tool")
    print("="*80)

    # Configuration
    checkpoint_path = "/mnt/d/ai/SLGA/out_slga/ckpt_11000/model.pt"
    test_prompt = "The capital of France is"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n⚙️  Configuration:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Prompt:     '{test_prompt}'")
    print(f"  Device:     {device}")

    # Load model and tokenizer
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    model, tokenizer = load_checkpoint(checkpoint_path, device)

    # Tokenize input
    print("\n" + "="*80)
    print("TOKENIZING INPUT")
    print("="*80)
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
    print(f"Input tokens: {input_ids[0].tolist()}")
    print(f"Decoded: {[repr(tokenizer.decode([t])) for t in input_ids[0]]}")

    # Run forward pass
    print("\n" + "="*80)
    print("FORWARD PASS")
    print("="*80)
    with torch.no_grad():
        logits = model(input_ids)
        print(f"Output logits shape: {logits.shape}")

    # Analyze logits
    print("\n" + "="*80)
    print("ANALYZING LOGITS")
    print("="*80)
    results = analyze_logits(logits, tokenizer, top_k=20)
    print_results(results, tokenizer)

    # Test temperature sampling
    print("\n" + "="*80)
    print("TEMPERATURE SAMPLING TESTS")
    print("="*80)
    temperatures = [0.0, 0.5, 1.0]
    for temp in temperatures:
        token_id = apply_temperature_sampling(logits, temp)
        token = tokenizer.decode([token_id])
        print(f"  Temperature {temp:.1f}: {repr(token)} (token_id={token_id})")

    # Store results in memory
    print("\n" + "="*80)
    print("STORING RESULTS")
    print("="*80)

    memory_summary = {
        'checkpoint': checkpoint_path,
        'prompt': test_prompt,
        'argmax_token': results['argmax']['token'],
        'argmax_probability': results['argmax']['probability'],
        'top_5_tokens': [
            f"{t['token']} ({t['probability']:.4f})"
            for t in results['top_tokens'][:5]
        ],
        'logits_stats': results['logits_stats']
    }

    print(f"Memory key: diagnostics/logits-analysis")
    print(f"Summary: {memory_summary}")

    print("\n" + "="*80)
    print("✅ DIAGNOSTIC COMPLETE")
    print("="*80)

    return results, memory_summary


if __name__ == "__main__":
    results, memory_summary = main()
