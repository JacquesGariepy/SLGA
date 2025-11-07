"""
Script de comparaison exhaustive des paramètres de sampling.

Usage:
    python scripts/compare_sampling.py --checkpoint out_slga/ckpt_11000 --config config.yaml
    
Génère un rapport HTML comparatif avec toutes les configurations de sampling.
"""

from __future__ import annotations
import os
import sys
import yaml
import torch
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer
from src.model import Config, LLMTransformer


# ============================================================================
# CONFIGURATIONS DE TEST
# ============================================================================

TEST_PROMPTS = [
    # Factual - Tests basiques
    ("factual_simple", "The capital of France is", 10),
    ("factual_who", "Who is Albert Einstein?", 40),
    ("factual_what", "What is photosynthesis?", 30),
    
    # Continuation - Tests de cohérence
    ("continuation_sentence", "The quick brown fox jumps over", 20),
    ("continuation_story", "Once upon a time, there was a", 50),
    ("continuation_tech", "Machine learning is a field of", 40),
    
    # Creative - Tests de génération
    ("creative_fiction", "In a distant galaxy,", 60),
    ("creative_future", "In the year 2050,", 50),
    ("creative_describe", "The ancient castle stood on", 40),
    
    # Structured - Tests de format
    ("structured_list", "The five largest cities in the world are:", 50),
    ("structured_explain", "To bake a cake, first you need to", 60),
]

SAMPLING_CONFIGS = [
    # (name, description, temperature, top_k, top_p)
    
    # Greedy & Deterministic
    ("greedy", "Greedy (temp=0, déterministe)", 0.0, 1, None),
    ("near_greedy", "Near-Greedy (temp=0.3)", 0.3, 10, None),
    
    # Conservative - Pour PPL élevé
    ("conservative", "Conservative (PPL ~25)", 0.5, 15, 0.75),
    ("conservative_plus", "Conservative+ (PPL ~20)", 0.6, 20, 0.8),
    
    # Balanced - Standard
    ("balanced", "Balanced (GPT-3 style)", 0.8, 40, 0.9),
    ("balanced_topk", "Balanced (top-k only)", 0.9, 40, None),
    ("balanced_topp", "Balanced (top-p only)", 0.9, 0, 0.9),
    
    # Creative
    ("creative", "Creative (temp=1.0)", 1.0, 50, 0.92),
    ("creative_high", "Creative High (temp=1.2)", 1.2, 80, 0.95),
    
    # Experimental
    ("experimental", "Experimental (temp=1.5)", 1.5, 100, 0.95),
    ("chaotic", "Chaotic (temp=2.0)", 2.0, 200, 0.98),
    
    # Special configurations
    ("nucleus_only", "Pure Nucleus (p=0.9)", 0.8, 0, 0.9),
    ("nucleus_strict", "Nucleus Strict (p=0.7)", 0.8, 0, 0.7),
    ("topk_only", "Pure Top-K (k=50)", 0.9, 50, None),
    ("topk_low", "Top-K Low (k=10)", 0.8, 10, None),
]


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def load_model(checkpoint_path: str, config_path: str, device: str):
    """Charge le modèle et la config."""
    print(f"Loading config from {config_path}...")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Creating model...")
    model_cfg = Config(**cfg["model"])
    model = LLMTransformer(model_cfg)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    if os.path.isdir(checkpoint_path):
        model_path = os.path.join(checkpoint_path, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model.pt not found in {checkpoint_path}")
        state_dict = torch.load(model_path, map_location="cpu")
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded: {model.get_num_params() / 1e6:.2f}M parameters")
    
    return model, tokenizer, cfg


def generate_text(
    model: LLMTransformer,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: str,
) -> Tuple[str, float]:
    """Génère du texte et retourne (texte, temps_ms)."""
    import time
    
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p,
        )
    elapsed_ms = (time.time() - start) * 1000
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text, elapsed_ms


def evaluate_quality(text: str, prompt: str) -> Dict[str, any]:
    """Évalue la qualité de la génération (heuristiques basiques)."""
    generated_part = text[len(prompt):].strip()
    
    # Métriques basiques
    num_words = len(generated_part.split())
    num_chars = len(generated_part)
    
    # Répétitions (bigrams)
    words = generated_part.lower().split()
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    unique_bigrams = len(set(bigrams))
    repetition_ratio = unique_bigrams / max(len(bigrams), 1)
    
    # Tokens spéciaux/étranges
    has_special = any(c in generated_part for c in ['�', '###', '@@@'])
    
    # Diversité lexicale
    unique_words = len(set(words))
    lexical_diversity = unique_words / max(num_words, 1)
    
    return {
        "num_words": num_words,
        "num_chars": num_chars,
        "repetition_ratio": repetition_ratio,
        "lexical_diversity": lexical_diversity,
        "has_special": has_special,
    }


# ============================================================================
# GÉNÉRATION DU RAPPORT
# ============================================================================

def generate_html_report(results: List[Dict], output_path: str, checkpoint_name: str):
    """Génère un rapport HTML comparatif."""
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SLGA Sampling Comparison - {checkpoint_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        .info-box {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .config-section {{
            margin-bottom: 40px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .config-header {{
            background: #3498db;
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            margin-bottom: 15px;
        }}
        .config-name {{
            font-size: 1.2em;
            font-weight: bold;
        }}
        .config-params {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-top: 5px;
        }}
        .prompt-result {{
            margin-bottom: 20px;
            border-left: 3px solid #e0e0e0;
            padding-left: 15px;
        }}
        .prompt-label {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .prompt-text {{
            color: #7f8c8d;
            font-style: italic;
            margin-bottom: 8px;
        }}
        .generated-text {{
            background: #ecf0f1;
            padding: 12px;
            border-radius: 5px;
            margin: 8px 0;
            line-height: 1.6;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 8px;
            border-radius: 4px;
            font-size: 0.85em;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .metric-value {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .summary-table th {{
            background: #34495e;
            color: white;
            padding: 10px;
            text-align: left;
        }}
        .summary-table td {{
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }}
        .summary-table tr:hover {{
            background: #f8f9fa;
        }}
        .quality-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .quality-good {{ background: #2ecc71; color: white; }}
        .quality-medium {{ background: #f39c12; color: white; }}
        .quality-poor {{ background: #e74c3c; color: white; }}
        .timestamp {{
            color: #95a5a6;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>🎯 SLGA Sampling Comparison Report</h1>
    
    <div class="info-box">
        <strong>Checkpoint:</strong> {checkpoint_name}<br>
        <strong>Generated:</strong> <span class="timestamp">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span><br>
        <strong>Total Configurations:</strong> {len(SAMPLING_CONFIGS)}<br>
        <strong>Total Prompts:</strong> {len(TEST_PROMPTS)}
    </div>
"""
    
    # Tableau récapitulatif
    html += """
    <h2>📊 Summary Table</h2>
    <table class="summary-table">
        <thead>
            <tr>
                <th>Configuration</th>
                <th>Temperature</th>
                <th>Top-K</th>
                <th>Top-P</th>
                <th>Avg Words</th>
                <th>Avg Diversity</th>
                <th>Avg Time (ms)</th>
            </tr>
        </thead>
        <tbody>
"""
    
    # Calculer moyennes par config
    config_stats = {}
    for result in results:
        config_name = result["config_name"]
        if config_name not in config_stats:
            config_stats[config_name] = {
                "description": result["config_description"],
                "temperature": result["temperature"],
                "top_k": result["top_k"],
                "top_p": result["top_p"],
                "words": [],
                "diversity": [],
                "time": [],
            }
        config_stats[config_name]["words"].append(result["metrics"]["num_words"])
        config_stats[config_name]["diversity"].append(result["metrics"]["lexical_diversity"])
        config_stats[config_name]["time"].append(result["time_ms"])
    
    for config_name, stats in config_stats.items():
        avg_words = sum(stats["words"]) / len(stats["words"])
        avg_diversity = sum(stats["diversity"]) / len(stats["diversity"])
        avg_time = sum(stats["time"]) / len(stats["time"])
        
        html += f"""
            <tr>
                <td><strong>{stats['description']}</strong></td>
                <td>{stats['temperature']}</td>
                <td>{stats['top_k'] if stats['top_k'] else 'None'}</td>
                <td>{stats['top_p'] if stats['top_p'] else 'None'}</td>
                <td>{avg_words:.1f}</td>
                <td>{avg_diversity:.2f}</td>
                <td>{avg_time:.1f}</td>
            </tr>
"""
    
    html += """
        </tbody>
    </table>
"""
    
    # Résultats détaillés par configuration
    html += "<h2>📝 Detailed Results</h2>"
    
    current_config = None
    for result in results:
        if result["config_name"] != current_config:
            if current_config is not None:
                html += "</div>"  # Close previous config section
            
            current_config = result["config_name"]
            html += f"""
    <div class="config-section">
        <div class="config-header">
            <div class="config-name">{result['config_description']}</div>
            <div class="config-params">
                Temperature: {result['temperature']} | 
                Top-K: {result['top_k'] if result['top_k'] else 'None'} | 
                Top-P: {result['top_p'] if result['top_p'] else 'None'}
            </div>
        </div>
"""
        
        # Prompt result
        quality_class = "quality-good" if result["metrics"]["lexical_diversity"] > 0.7 else \
                       "quality-medium" if result["metrics"]["lexical_diversity"] > 0.5 else "quality-poor"
        
        html += f"""
        <div class="prompt-result">
            <div class="prompt-label">📌 {result['prompt_name']}</div>
            <div class="prompt-text">"{result['prompt']}"</div>
            <div class="generated-text">{result['generated_text']}</div>
            <div class="metrics">
                <div class="metric">
                    <span class="metric-label">Words:</span>
                    <span class="metric-value">{result['metrics']['num_words']}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Diversity:</span>
                    <span class="metric-value">{result['metrics']['lexical_diversity']:.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Repetition:</span>
                    <span class="metric-value">{result['metrics']['repetition_ratio']:.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Time:</span>
                    <span class="metric-value">{result['time_ms']:.1f}ms</span>
                </div>
            </div>
        </div>
"""
    
    html += "</div>"  # Close last config section
    
    html += """
</body>
</html>
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"✓ HTML report saved to: {output_path}")


def generate_text_report(results: List[Dict], output_path: str, checkpoint_name: str):
    """Génère un rapport texte simple."""
    
    report = f"""
{'='*80}
SLGA SAMPLING COMPARISON REPORT
{'='*80}

Checkpoint: {checkpoint_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Configurations: {len(SAMPLING_CONFIGS)}
Total Prompts: {len(TEST_PROMPTS)}

{'='*80}

"""
    
    current_config = None
    for result in results:
        if result["config_name"] != current_config:
            current_config = result["config_name"]
            report += f"\n{'='*80}\n"
            report += f"CONFIG: {result['config_description']}\n"
            report += f"  Temperature: {result['temperature']}\n"
            report += f"  Top-K: {result['top_k'] if result['top_k'] else 'None'}\n"
            report += f"  Top-P: {result['top_p'] if result['top_p'] else 'None'}\n"
            report += f"{'='*80}\n\n"
        
        report += f"--- {result['prompt_name']} ---\n"
        report += f"Prompt: \"{result['prompt']}\"\n\n"
        report += f"Generated:\n{result['generated_text']}\n\n"
        report += f"Metrics:\n"
        report += f"  Words: {result['metrics']['num_words']}\n"
        report += f"  Diversity: {result['metrics']['lexical_diversity']:.2f}\n"
        report += f"  Repetition: {result['metrics']['repetition_ratio']:.2f}\n"
        report += f"  Time: {result['time_ms']:.1f}ms\n"
        report += f"\n{'-'*80}\n\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"✓ Text report saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare sampling configurations")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="comparison_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="Quick test (fewer configs)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract checkpoint name
    checkpoint_name = Path(args.checkpoint).name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("="*80)
    print("SLGA SAMPLING COMPARISON")
    print("="*80)
    print()
    
    # Load model
    model, tokenizer, cfg = load_model(args.checkpoint, args.config, args.device)
    print()
    
    # Select configs to test
    if args.quick:
        configs_to_test = SAMPLING_CONFIGS[:5]  # Only first 5
        prompts_to_test = TEST_PROMPTS[:3]       # Only first 3
        print("⚡ Quick mode: Testing 5 configs on 3 prompts")
    else:
        configs_to_test = SAMPLING_CONFIGS
        prompts_to_test = TEST_PROMPTS
        print(f"🔬 Full mode: Testing {len(configs_to_test)} configs on {len(prompts_to_test)} prompts")
    
    print()
    print("="*80)
    print()
    
    # Run tests
    results = []
    total_tests = len(configs_to_test) * len(prompts_to_test)
    current_test = 0
    
    for config_name, config_desc, temperature, top_k, top_p in configs_to_test:
        print(f"\n{'─'*80}")
        print(f"Testing: {config_desc}")
        print(f"  temp={temperature}, top_k={top_k}, top_p={top_p}")
        print(f"{'─'*80}\n")
        
        for prompt_name, prompt, max_tokens in prompts_to_test:
            current_test += 1
            progress = (current_test / total_tests) * 100
            
            print(f"[{current_test}/{total_tests}] ({progress:.1f}%) {prompt_name}...", end=" ", flush=True)
            
            try:
                generated_text, time_ms = generate_text(
                    model, tokenizer, prompt, max_tokens,
                    temperature, top_k, top_p, args.device
                )
                
                metrics = evaluate_quality(generated_text, prompt)
                
                results.append({
                    "config_name": config_name,
                    "config_description": config_desc,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "prompt_name": prompt_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "generated_text": generated_text,
                    "time_ms": time_ms,
                    "metrics": metrics,
                })
                
                print(f"✓ ({time_ms:.0f}ms, {metrics['num_words']} words)")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
    
    print()
    print("="*80)
    print("GENERATING REPORTS...")
    print("="*80)
    print()
    
    # Generate reports
    html_path = os.path.join(args.output_dir, f"comparison_{checkpoint_name}_{timestamp}.html")
    text_path = os.path.join(args.output_dir, f"comparison_{checkpoint_name}_{timestamp}.txt")
    
    generate_html_report(results, html_path, checkpoint_name)
    generate_text_report(results, text_path, checkpoint_name)
    
    print()
    print("="*80)
    print("✓ COMPARISON COMPLETE!")
    print("="*80)
    print(f"\n📊 View results:")
    print(f"   HTML: {html_path}")
    print(f"   Text: {text_path}")
    print()


if __name__ == "__main__":
    main()