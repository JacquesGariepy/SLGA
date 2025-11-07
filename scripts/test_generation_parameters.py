#!/usr/bin/env python3
"""
Comprehensive Generation Parameter Testing Script
Tests various sampling strategies to mitigate repetitive output
"""

import torch
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import Config, LLMTransformer
from transformers import GPT2Tokenizer

def calculate_repetition_metrics(text: str, tokens: List[int]) -> Dict:
    """Calculate various repetition metrics for generated text."""

    # N-gram repetition analysis
    words = text.split()

    metrics = {
        'total_tokens': len(tokens),
        'unique_tokens': len(set(tokens)),
        'token_diversity': len(set(tokens)) / len(tokens) if tokens else 0,
        'text_length': len(text),
        'word_count': len(words),
        'unique_words': len(set(words)),
        'word_diversity': len(set(words)) / len(words) if words else 0,
    }

    # Bigram repetition
    bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
    bigram_counts = Counter(bigrams)
    metrics['unique_bigrams'] = len(bigram_counts)
    metrics['bigram_diversity'] = len(bigram_counts) / len(bigrams) if bigrams else 0
    metrics['max_bigram_repetition'] = max(bigram_counts.values()) if bigram_counts else 0

    # Trigram repetition
    trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
    trigram_counts = Counter(trigrams)
    metrics['unique_trigrams'] = len(trigram_counts)
    metrics['trigram_diversity'] = len(trigram_counts) / len(trigrams) if trigrams else 0
    metrics['max_trigram_repetition'] = max(trigram_counts.values()) if trigram_counts else 0

    # 4-gram repetition
    fourgrams = [tuple(tokens[i:i+4]) for i in range(len(tokens)-3)]
    fourgram_counts = Counter(fourgrams)
    metrics['unique_fourgrams'] = len(fourgram_counts)
    metrics['fourgram_diversity'] = len(fourgram_counts) / len(fourgrams) if fourgrams else 0
    metrics['max_fourgram_repetition'] = max(fourgram_counts.values()) if fourgram_counts else 0

    # Detect immediate repetitions (same token repeated consecutively)
    immediate_reps = sum(1 for i in range(len(tokens)-1) if tokens[i] == tokens[i+1])
    metrics['immediate_repetitions'] = immediate_reps
    metrics['immediate_repetition_rate'] = immediate_reps / len(tokens) if tokens else 0

    # Detect phrase repetitions (sliding window)
    phrase_length = 5
    phrases = [tuple(tokens[i:i+phrase_length]) for i in range(len(tokens)-phrase_length+1)]
    phrase_counts = Counter(phrases)
    metrics['repeated_phrases'] = sum(1 for count in phrase_counts.values() if count > 1)
    metrics['max_phrase_repetition'] = max(phrase_counts.values()) if phrase_counts else 0

    return metrics

def generate_with_params(
    model: LLMTransformer,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_length: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    device: str = 'cuda'
) -> Tuple[str, List[int], float]:
    """Generate text with specific parameters and measure time."""

    model.eval()
    start_time = time.time()

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    generated_tokens = input_ids[0].tolist()

    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_tokens):
                    next_token_logits[token_id] /= repetition_penalty

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    0, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float('Inf')

            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token.item())

            # Update input for next iteration
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    generation_time = time.time() - start_time
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text, generated_tokens, generation_time

def run_experiments(checkpoint_path: str, device: str = 'cuda'):
    """Run comprehensive generation experiments."""

    print("=" * 80)
    print("GENERATION PARAMETER TESTING")
    print("=" * 80)

    # Load model
    print(f"\n📦 Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check if checkpoint is wrapped or direct state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config_dict = checkpoint.get('config', checkpoint.get('model_config', {}))
    elif 'config' in checkpoint:
        state_dict = checkpoint
        config_dict = checkpoint.get('config', checkpoint.get('model_config', {}))
    else:
        # Direct state dict, need to load config from separate file or use defaults
        state_dict = checkpoint
        # Try to infer config from state dict
        config_dict = {
            'vocab_size': state_dict['token_emb.weight'].shape[0],
            'embed_dim': state_dict['token_emb.weight'].shape[1],
            'max_seq_len': state_dict['pos_emb.weight'].shape[0],
        }
        # Count layers
        n_layers = len([k for k in state_dict.keys() if k.startswith('blocks.') and k.endswith('.norm1.weight')])
        config_dict['n_layers'] = n_layers
        # Infer num_heads from attention weights
        attn_weight = state_dict['blocks.0.attn.qkv_proj.weight']
        # qkv_proj outputs 3 * embed_dim (for q, k, v)
        embed_dim = config_dict['embed_dim']
        config_dict['num_heads'] = 8  # Default assumption

    config = Config(**config_dict)

    model = LLMTransformer(config).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"✅ Model loaded successfully")
    print(f"   Vocab size: {config.vocab_size}")
    print(f"   Hidden dim: {config.embed_dim}")
    print(f"   Num layers: {config.n_layers}")
    print(f"   Num heads: {config.num_heads}")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Define test prompts (diverse set)
    test_prompts = [
        "The future of artificial intelligence",
        "In a small village nestled in the mountains",
        "The scientific breakthrough was announced",
        "Once upon a time in a distant galaxy",
        "The chef carefully prepared"
    ]

    # Define parameter configurations to test
    configurations = [
        # Baseline
        {'name': 'baseline', 'temp': 1.0, 'top_k': 0, 'top_p': 1.0, 'rep_penalty': 1.0},

        # Temperature variations
        {'name': 'temp_0.1', 'temp': 0.1, 'top_k': 0, 'top_p': 1.0, 'rep_penalty': 1.0},
        {'name': 'temp_0.5', 'temp': 0.5, 'top_k': 0, 'top_p': 1.0, 'rep_penalty': 1.0},
        {'name': 'temp_0.8', 'temp': 0.8, 'top_k': 0, 'top_p': 1.0, 'rep_penalty': 1.0},
        {'name': 'temp_1.2', 'temp': 1.2, 'top_k': 0, 'top_p': 1.0, 'rep_penalty': 1.0},

        # Top-k variations
        {'name': 'top_k_10', 'temp': 1.0, 'top_k': 10, 'top_p': 1.0, 'rep_penalty': 1.0},
        {'name': 'top_k_20', 'temp': 1.0, 'top_k': 20, 'top_p': 1.0, 'rep_penalty': 1.0},
        {'name': 'top_k_40', 'temp': 1.0, 'top_k': 40, 'top_p': 1.0, 'rep_penalty': 1.0},
        {'name': 'top_k_80', 'temp': 1.0, 'top_k': 80, 'top_p': 1.0, 'rep_penalty': 1.0},

        # Top-p (nucleus) variations
        {'name': 'top_p_0.9', 'temp': 1.0, 'top_k': 0, 'top_p': 0.9, 'rep_penalty': 1.0},
        {'name': 'top_p_0.95', 'temp': 1.0, 'top_k': 0, 'top_p': 0.95, 'rep_penalty': 1.0},

        # Combined strategies
        {'name': 'temp_0.8_top_p_0.95', 'temp': 0.8, 'top_k': 0, 'top_p': 0.95, 'rep_penalty': 1.0},
        {'name': 'temp_0.7_top_k_40', 'temp': 0.7, 'top_k': 40, 'top_p': 1.0, 'rep_penalty': 1.0},
        {'name': 'temp_0.9_top_p_0.9_rep_1.2', 'temp': 0.9, 'top_k': 0, 'top_p': 0.9, 'rep_penalty': 1.2},

        # Aggressive anti-repetition
        {'name': 'aggressive_diverse', 'temp': 0.8, 'top_k': 50, 'top_p': 0.95, 'rep_penalty': 1.5},
    ]

    # Test with different generation lengths
    test_lengths = [50, 100, 200]

    all_results = []

    for length in test_lengths:
        print(f"\n{'='*80}")
        print(f"TESTING WITH MAX LENGTH: {length}")
        print(f"{'='*80}")

        for config_params in configurations:
            config_name = config_params['name']
            print(f"\n🧪 Testing configuration: {config_name}")
            print(f"   Temperature: {config_params['temp']}")
            print(f"   Top-k: {config_params['top_k']}")
            print(f"   Top-p: {config_params['top_p']}")
            print(f"   Repetition penalty: {config_params['rep_penalty']}")

            config_results = {
                'config_name': config_name,
                'max_length': length,
                'parameters': config_params,
                'prompt_results': []
            }

            for prompt_idx, prompt in enumerate(test_prompts):
                try:
                    generated_text, tokens, gen_time = generate_with_params(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_length=length,
                        temperature=config_params['temp'],
                        top_k=config_params['top_k'],
                        top_p=config_params['top_p'],
                        repetition_penalty=config_params['rep_penalty'],
                        device=device
                    )

                    # Calculate metrics
                    metrics = calculate_repetition_metrics(generated_text, tokens)
                    metrics['generation_time'] = gen_time

                    prompt_result = {
                        'prompt': prompt,
                        'prompt_idx': prompt_idx,
                        'generated_text': generated_text,
                        'metrics': metrics
                    }

                    config_results['prompt_results'].append(prompt_result)

                    # Print summary for this prompt
                    print(f"\n   Prompt {prompt_idx + 1}: '{prompt[:40]}...'")
                    print(f"   Generated: '{generated_text[:80]}...'")
                    print(f"   Token diversity: {metrics['token_diversity']:.3f}")
                    print(f"   Word diversity: {metrics['word_diversity']:.3f}")
                    print(f"   Bigram diversity: {metrics['bigram_diversity']:.3f}")
                    print(f"   Max phrase repetition: {metrics['max_phrase_repetition']}")
                    print(f"   Time: {gen_time:.2f}s")

                except Exception as e:
                    print(f"   ❌ Error with prompt {prompt_idx + 1}: {e}")
                    config_results['prompt_results'].append({
                        'prompt': prompt,
                        'prompt_idx': prompt_idx,
                        'error': str(e)
                    })

            # Calculate aggregate metrics for this configuration
            valid_results = [r for r in config_results['prompt_results'] if 'metrics' in r]
            if valid_results:
                avg_metrics = {
                    'avg_token_diversity': np.mean([r['metrics']['token_diversity'] for r in valid_results]),
                    'avg_word_diversity': np.mean([r['metrics']['word_diversity'] for r in valid_results]),
                    'avg_bigram_diversity': np.mean([r['metrics']['bigram_diversity'] for r in valid_results]),
                    'avg_trigram_diversity': np.mean([r['metrics']['trigram_diversity'] for r in valid_results]),
                    'avg_max_phrase_rep': np.mean([r['metrics']['max_phrase_repetition'] for r in valid_results]),
                    'avg_gen_time': np.mean([r['metrics']['generation_time'] for r in valid_results])
                }
                config_results['aggregate_metrics'] = avg_metrics

                print(f"\n   📊 Aggregate metrics:")
                print(f"      Token diversity: {avg_metrics['avg_token_diversity']:.3f}")
                print(f"      Word diversity: {avg_metrics['avg_word_diversity']:.3f}")
                print(f"      Bigram diversity: {avg_metrics['avg_bigram_diversity']:.3f}")
                print(f"      Avg max phrase repetition: {avg_metrics['avg_max_phrase_rep']:.1f}")

            all_results.append(config_results)

    return all_results, configurations, test_prompts

def analyze_and_report(results: List[Dict], configurations: List[Dict], prompts: List[str]):
    """Analyze results and generate comprehensive report."""

    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS")
    print("="*80)

    # Group by max_length
    results_by_length = {}
    for result in results:
        length = result['max_length']
        if length not in results_by_length:
            results_by_length[length] = []
        results_by_length[length].append(result)

    best_configs = []

    for length, length_results in sorted(results_by_length.items()):
        print(f"\n📏 MAX LENGTH: {length}")
        print("-" * 80)

        # Sort by aggregate metrics
        scored_configs = []
        for result in length_results:
            if 'aggregate_metrics' in result:
                # Composite score: higher diversity is better, lower repetition is better
                score = (
                    result['aggregate_metrics']['avg_token_diversity'] * 0.3 +
                    result['aggregate_metrics']['avg_word_diversity'] * 0.3 +
                    result['aggregate_metrics']['avg_bigram_diversity'] * 0.2 +
                    result['aggregate_metrics']['avg_trigram_diversity'] * 0.2 -
                    result['aggregate_metrics']['avg_max_phrase_rep'] * 0.05
                )
                scored_configs.append((score, result))

        scored_configs.sort(reverse=True, key=lambda x: x[0])

        # Print top 5 configurations
        print("\n🏆 TOP 5 CONFIGURATIONS:")
        for rank, (score, result) in enumerate(scored_configs[:5], 1):
            config_name = result['config_name']
            metrics = result['aggregate_metrics']

            print(f"\n{rank}. {config_name} (Score: {score:.4f})")
            print(f"   Token diversity: {metrics['avg_token_diversity']:.3f}")
            print(f"   Word diversity: {metrics['avg_word_diversity']:.3f}")
            print(f"   Bigram diversity: {metrics['avg_bigram_diversity']:.3f}")
            print(f"   Trigram diversity: {metrics['avg_trigram_diversity']:.3f}")
            print(f"   Avg max phrase rep: {metrics['avg_max_phrase_rep']:.1f}")
            print(f"   Avg gen time: {metrics['avg_gen_time']:.2f}s")

            # Store best config for this length
            if rank == 1:
                best_configs.append({
                    'max_length': length,
                    'config_name': config_name,
                    'score': score,
                    'metrics': metrics,
                    'parameters': result['parameters']
                })

    # Overall recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    print("\n💡 Best configurations by generation length:")
    for best in best_configs:
        print(f"\n   Length {best['max_length']}: {best['config_name']}")
        print(f"      Parameters: temp={best['parameters']['temp']}, "
              f"top_k={best['parameters']['top_k']}, "
              f"top_p={best['parameters']['top_p']}, "
              f"rep_penalty={best['parameters']['rep_penalty']}")
        print(f"      Score: {best['score']:.4f}")

    # Determine if issue is fundamental or can be mitigated
    print("\n🔍 Issue Analysis:")

    # Compare best performance to baseline
    best_overall = max(best_configs, key=lambda x: x['score'])
    baseline_results = [r for r in results if r['config_name'] == 'baseline']

    if baseline_results and 'aggregate_metrics' in baseline_results[0]:
        baseline_diversity = baseline_results[0]['aggregate_metrics']['avg_token_diversity']
        best_diversity = best_overall['metrics']['avg_token_diversity']
        improvement = ((best_diversity - baseline_diversity) / baseline_diversity) * 100

        print(f"\n   Baseline token diversity: {baseline_diversity:.3f}")
        print(f"   Best config diversity: {best_diversity:.3f}")
        print(f"   Improvement: {improvement:+.1f}%")

        if improvement > 20:
            print("\n   ✅ CONCLUSION: Sampling strategies CAN significantly mitigate repetition")
            print("      The issue is partially addressable through better generation parameters")
        elif improvement > 5:
            print("\n   ⚠️  CONCLUSION: Sampling strategies provide MODERATE improvement")
            print("      Both training and sampling contribute to the issue")
        else:
            print("\n   ❌ CONCLUSION: Issue appears FUNDAMENTAL (training-related)")
            print("      Sampling strategies provide minimal improvement")
            print("      Model may need retraining or architectural changes")

    return best_configs

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test generation parameters")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output', type=str, default='generation_experiments_results.json',
                       help='Output JSON file')

    args = parser.parse_args()

    # Run experiments
    results, configurations, prompts = run_experiments(args.checkpoint, args.device)

    # Analyze and report
    best_configs = analyze_and_report(results, configurations, prompts)

    # Save results
    output_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'checkpoint': args.checkpoint,
        'results': results,
        'best_configs': best_configs,
        'configurations_tested': configurations,
        'test_prompts': prompts
    }

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")
    print("\n✨ Experiment complete!")
