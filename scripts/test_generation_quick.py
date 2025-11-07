#!/usr/bin/env python3
"""
Quick Generation Parameter Testing - Fast version with fewer configs
"""

import torch
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import Config, LLMTransformer
from transformers import GPT2Tokenizer

def calculate_metrics(text: str, tokens: List[int]) -> Dict:
    """Calculate repetition metrics."""
    words = text.split()

    metrics = {
        'total_tokens': len(tokens),
        'unique_tokens': len(set(tokens)),
        'token_diversity': len(set(tokens)) / len(tokens) if tokens else 0,
        'unique_words': len(set(words)),
        'word_diversity': len(set(words)) / len(words) if words else 0,
    }

    # Bigrams
    bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
    bigram_counts = Counter(bigrams)
    metrics['bigram_diversity'] = len(bigram_counts) / len(bigrams) if bigrams else 0
    metrics['max_bigram_rep'] = max(bigram_counts.values()) if bigram_counts else 0

    # Trigrams
    trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
    trigram_counts = Counter(trigrams)
    metrics['trigram_diversity'] = len(trigram_counts) / len(trigrams) if trigrams else 0
    metrics['max_trigram_rep'] = max(trigram_counts.values()) if trigram_counts else 0

    # Immediate repetitions
    immediate_reps = sum(1 for i in range(len(tokens)-1) if tokens[i] == tokens[i+1])
    metrics['immediate_rep_rate'] = immediate_reps / len(tokens) if tokens else 0

    return metrics

def generate(model, tokenizer, prompt, max_len, temp=1.0, top_k=0, top_p=1.0, rep_penalty=1.0, device='cuda'):
    """Generate text with parameters."""
    model.eval()
    start = time.time()

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids[0].tolist()

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(input_ids)
            next_logits = logits[0, -1, :]

            # Repetition penalty
            if rep_penalty != 1.0:
                for tid in set(generated):
                    next_logits[tid] /= rep_penalty

            # Temperature
            if temp != 1.0:
                next_logits = next_logits / temp

            # Top-k
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = -float('Inf')

            # Top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = -float('Inf')

            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    gen_time = time.time() - start
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, generated, gen_time

def main():
    print("="*80)
    print("QUICK GENERATION PARAMETER TEST")
    print("="*80)

    # Load model
    checkpoint_path = "out_slga_fineweb/ckpt_2000/model.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n📦 Loading from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Extract state dict
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        config_dict = ckpt.get('config', {})
    else:
        state_dict = ckpt
        config_dict = {
            'vocab_size': state_dict['token_emb.weight'].shape[0],
            'embed_dim': state_dict['token_emb.weight'].shape[1],
            'max_seq_len': state_dict['pos_emb.weight'].shape[0],
            'n_layers': len([k for k in state_dict if k.startswith('blocks.') and k.endswith('.norm1.weight')]),
            'num_heads': 8,
        }

    config = Config(**config_dict)
    model = LLMTransformer(config).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"✅ Model loaded: {config.n_layers} layers, {config.embed_dim}d, {config.num_heads} heads")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Test prompts
    prompts = [
        "The future of artificial intelligence",
        "In a small village",
        "The scientist discovered"
    ]

    # Key configurations to test
    configs = [
        {'name': 'baseline', 'temp': 1.0, 'top_k': 0, 'top_p': 1.0, 'rep': 1.0},
        {'name': 'low_temp', 'temp': 0.5, 'top_k': 0, 'top_p': 1.0, 'rep': 1.0},
        {'name': 'high_temp', 'temp': 1.2, 'top_k': 0, 'top_p': 1.0, 'rep': 1.0},
        {'name': 'top_k_40', 'temp': 1.0, 'top_k': 40, 'top_p': 1.0, 'rep': 1.0},
        {'name': 'top_p_0.9', 'temp': 1.0, 'top_k': 0, 'top_p': 0.9, 'rep': 1.0},
        {'name': 'combined', 'temp': 0.8, 'top_k': 0, 'top_p': 0.95, 'rep': 1.0},
        {'name': 'aggressive', 'temp': 0.8, 'top_k': 50, 'top_p': 0.95, 'rep': 1.5},
    ]

    max_len = 100
    results = []

    print(f"\n{'='*80}")
    print(f"TESTING {len(configs)} CONFIGURATIONS x {len(prompts)} PROMPTS x {max_len} TOKENS")
    print(f"{'='*80}\n")

    for cfg in configs:
        print(f"\n🧪 {cfg['name']}: temp={cfg['temp']}, top_k={cfg['top_k']}, top_p={cfg['top_p']}, rep={cfg['rep']}")

        cfg_results = {'config': cfg, 'tests': []}

        for pidx, prompt in enumerate(prompts):
            try:
                text, tokens, gen_time = generate(
                    model, tokenizer, prompt, max_len,
                    cfg['temp'], cfg['top_k'], cfg['top_p'], cfg['rep'], device
                )

                metrics = calculate_metrics(text, tokens)
                metrics['gen_time'] = gen_time

                cfg_results['tests'].append({
                    'prompt': prompt,
                    'text': text,
                    'metrics': metrics
                })

                print(f"   P{pidx+1}: {text[:60]}...")
                print(f"       tok_div={metrics['token_diversity']:.3f}, "
                      f"word_div={metrics['word_diversity']:.3f}, "
                      f"bigram_div={metrics['bigram_diversity']:.3f}, "
                      f"time={gen_time:.2f}s")

            except Exception as e:
                print(f"   P{pidx+1}: ❌ Error: {e}")
                cfg_results['tests'].append({'prompt': prompt, 'error': str(e)})

        # Aggregate
        valid = [t for t in cfg_results['tests'] if 'metrics' in t]
        if valid:
            avg_metrics = {
                'avg_token_div': np.mean([t['metrics']['token_diversity'] for t in valid]),
                'avg_word_div': np.mean([t['metrics']['word_diversity'] for t in valid]),
                'avg_bigram_div': np.mean([t['metrics']['bigram_diversity'] for t in valid]),
                'avg_trigram_div': np.mean([t['metrics']['trigram_diversity'] for t in valid]),
                'avg_time': np.mean([t['metrics']['gen_time'] for t in valid])
            }
            cfg_results['aggregate'] = avg_metrics
            print(f"   📊 Avg: tok={avg_metrics['avg_token_div']:.3f}, "
                  f"word={avg_metrics['avg_word_div']:.3f}, "
                  f"bigram={avg_metrics['avg_bigram_div']:.3f}")

        results.append(cfg_results)

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS & RECOMMENDATIONS")
    print("="*80)

    scored = []
    for r in results:
        if 'aggregate' in r:
            # Composite score (higher diversity = better)
            score = (r['aggregate']['avg_token_div'] * 0.35 +
                    r['aggregate']['avg_word_div'] * 0.35 +
                    r['aggregate']['avg_bigram_div'] * 0.3)
            scored.append((score, r))

    scored.sort(reverse=True)

    print("\n🏆 TOP 3 CONFIGURATIONS:\n")
    for rank, (score, r) in enumerate(scored[:3], 1):
        cfg = r['config']
        agg = r['aggregate']
        print(f"{rank}. {cfg['name']} (score: {score:.4f})")
        print(f"   Params: temp={cfg['temp']}, top_k={cfg['top_k']}, top_p={cfg['top_p']}, rep={cfg['rep']}")
        print(f"   Token div: {agg['avg_token_div']:.3f}")
        print(f"   Word div: {agg['avg_word_div']:.3f}")
        print(f"   Bigram div: {agg['avg_bigram_div']:.3f}")
        print(f"   Time: {agg['avg_time']:.2f}s")
        print()

    # Compare best to baseline
    if len(scored) > 0:
        best = scored[0]
        baseline = [r for r in results if r['config']['name'] == 'baseline']

        if baseline and 'aggregate' in baseline[0]:
            best_div = best[1]['aggregate']['avg_token_div']
            base_div = baseline[0]['aggregate']['avg_token_div']
            improvement = ((best_div - base_div) / base_div) * 100

            print("🔍 CONCLUSION:")
            print(f"   Baseline diversity: {base_div:.3f}")
            print(f"   Best config diversity: {best_div:.3f}")
            print(f"   Improvement: {improvement:+.1f}%")
            print()

            if improvement > 15:
                print("   ✅ Sampling strategies provide SIGNIFICANT improvement")
                print("      Issue can be largely mitigated through better generation params")
            elif improvement > 5:
                print("   ⚠️  Sampling strategies provide MODERATE improvement")
                print("      Both training and sampling contribute to the issue")
            else:
                print("   ❌ Issue appears FUNDAMENTAL (training-related)")
                print("      Sampling strategies provide minimal improvement")
                print("      Model may need retraining")

    # Save results
    output_path = "tests/generation_quick_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': results,
            'top_configs': [{'score': s, 'config': r['config'], 'metrics': r.get('aggregate')}
                           for s, r in scored[:3]]
        }, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")
    print("\n✨ Test complete!")

if __name__ == "__main__":
    main()
