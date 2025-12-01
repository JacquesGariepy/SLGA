#!/usr/bin/env python3
"""
SLGA Text Generation Script with Quality Evaluation
Usage:
    python generate.py --checkpoint out_slga/best_overall --prompt "Einstein was"
    python generate.py --config config.yaml --interactive
    python generate.py --checkpoint out_slga/ckpt_1000 --prompt "Hello" --evaluate
    python generate.py --checkpoint out_slga/ckpt_1000 --benchmark  # Run benchmark suite
    python generate.py --history out_slga/  # View generation history
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import Config, LLMTransformer
from src.data import get_tokenizer
from src.evaluation import GenerationEvaluator, GenerationQualityMetrics


# ======================================================================
# Model Loading
# ======================================================================

def load_model(checkpoint_dir: str, config_path: str = None, device: str = "cuda"):
    """Load model from checkpoint."""

    # Load config
    if config_path:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        # Try to find config in checkpoint
        config_path = os.path.join(checkpoint_dir, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        else:
            # Use default
            cfg = {"model": {}, "tokenizer": {"name": "Qwen/Qwen2-7B"}}

    # Tokenizer
    tokenizer = get_tokenizer(cfg["tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Update vocab size
    cfg["model"]["vocab_size"] = len(tokenizer)

    # Model
    model_cfg = Config(**cfg["model"])
    model = LLMTransformer(model_cfg)

    # Load weights
    model_path = os.path.join(checkpoint_dir, "model.pt")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {model_path}")
    else:
        print(f"No weights found at {model_path}")

    model = model.to(device)
    model.eval()

    return model, tokenizer, cfg


def get_checkpoint_info(checkpoint_dir: str) -> Dict[str, Any]:
    """Get checkpoint metadata."""
    info = {
        "checkpoint_path": checkpoint_dir,
        "checkpoint_type": "directory",
        "step": None,
        "loss": None,
        "timestamp": datetime.now().isoformat(),
    }

    # Try to get step from trainer state
    trainer_path = os.path.join(checkpoint_dir, "trainer_state.pt")
    if os.path.exists(trainer_path):
        try:
            state = torch.load(trainer_path, map_location="cpu")
            info["step"] = state.get("step", None)
            info["loss"] = state.get("best_loss", None)
        except Exception:
            pass

    # Try to get step from directory name
    if info["step"] is None:
        dir_name = os.path.basename(checkpoint_dir)
        if "ckpt_" in dir_name:
            try:
                info["step"] = int(dir_name.split("_")[-1])
            except ValueError:
                pass

    return info


# ======================================================================
# Generation with Evaluation
# ======================================================================

def generate_text(
    model: LLMTransformer,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3,
    device: str = "cuda",
) -> tuple:
    """
    Generate text from prompt with timing.

    Returns:
        (output_text, generation_time_seconds)
    """
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate with timing
    start_time = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=tokenizer.eos_token_id,
            stop_on_eos=True,
        )

    generation_time = time.time() - start_time

    # Decode
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text, generation_time


def generate_and_evaluate(
    model: LLMTransformer,
    tokenizer,
    evaluator: GenerationEvaluator,
    prompt: str,
    generation_params: Dict[str, Any],
    device: str = "cuda",
    verbose: bool = True,
) -> tuple:
    """
    Generate text and evaluate quality.

    Returns:
        (output_text, metrics, generation_time)
    """
    # Generate
    output_text, gen_time = generate_text(
        model, tokenizer, prompt,
        max_new_tokens=generation_params.get("max_new_tokens", 100),
        temperature=generation_params.get("temperature", 0.7),
        top_k=generation_params.get("top_k", 50),
        top_p=generation_params.get("top_p", 0.9),
        repetition_penalty=generation_params.get("repetition_penalty", 1.2),
        no_repeat_ngram_size=generation_params.get("no_repeat_ngram_size", 3),
        device=device,
    )

    # Evaluate
    metrics = evaluator.evaluate(
        prompt=prompt,
        generated_text=output_text,
        generation_time=gen_time,
        compute_perplexity=True,
    )

    if verbose:
        print_evaluation_summary(metrics)

    return output_text, metrics, gen_time


def print_evaluation_summary(metrics: GenerationQualityMetrics):
    """Print a summary of evaluation metrics."""
    print("\n" + "=" * 60)
    print(f"QUALITY EVALUATION: {metrics.overall_score:.1f}/100 ({metrics.quality_grade})")
    print("=" * 60)

    print(f"\nPerplexity:")
    print(f"  Sequence PPL: {metrics.perplexity.sequence_perplexity:.2f}")
    print(f"  Token PPL (avg): {metrics.perplexity.token_perplexity:.2f}")
    print(f"  Confident tokens: {metrics.perplexity.confident_tokens_ratio*100:.1f}%")

    print(f"\nDiversity:")
    print(f"  Distinct-1: {metrics.diversity.distinct_1:.3f}")
    print(f"  Distinct-2: {metrics.diversity.distinct_2:.3f}")
    print(f"  Distinct-3: {metrics.diversity.distinct_3:.3f}")
    print(f"  Lexical entropy: {metrics.diversity.lexical_entropy:.2f}")
    print(f"  Type-Token Ratio: {metrics.diversity.type_token_ratio:.3f}")

    print(f"\nRepetition:")
    print(f"  Rep-1: {metrics.repetition.rep_1:.3f}")
    print(f"  Rep-2: {metrics.repetition.rep_2:.3f}")
    print(f"  Rep-3: {metrics.repetition.rep_3:.3f}")
    if metrics.repetition.loop_detected:
        print(f"  LOOP DETECTED: '{metrics.repetition.loop_pattern}'")

    print(f"\nCoherence:")
    print(f"  Coherence score: {metrics.coherence.coherence_score:.3f}")
    print(f"  Gibberish score: {metrics.coherence.gibberish_score:.3f}")
    if metrics.coherence.has_broken_words:
        print(f"  Broken words detected")
    if metrics.coherence.has_special_char_spam:
        print(f"  Special char spam detected")

    print(f"\nPerformance:")
    print(f"  Tokens: {metrics.num_tokens}")
    print(f"  Time: {metrics.generation_time:.2f}s")
    print(f"  Speed: {metrics.tokens_per_second:.1f} tok/s")

    if metrics.issues:
        print(f"\nIssues:")
        for issue in metrics.issues:
            print(f"  - {issue}")

    if metrics.recommendations:
        print(f"\nRecommendations:")
        for rec in metrics.recommendations:
            print(f"  - {rec}")

    print("=" * 60)


# ======================================================================
# History Logging
# ======================================================================

def log_generation(
    output_dir: str,
    prompt: str,
    generated_text: str,
    generation_params: Dict[str, Any],
    model_config: Dict[str, Any],
    checkpoint_info: Dict[str, Any],
    metrics: Optional[GenerationQualityMetrics] = None,
    generation_time: float = 0.0,
    device: str = "cuda",
):
    """
    Log generation to JSONL history file with full metrics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    history_file = output_path / "generation_history.jsonl"

    # Build log entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "generated_text": generated_text,
        "generation_params": generation_params,
        "model_config": model_config,
        "checkpoint": checkpoint_info,
        "device": device,
        "generation_time_seconds": generation_time,
    }

    # Add evaluation metrics if available
    if metrics:
        entry["evaluation"] = {
            "overall_score": metrics.overall_score,
            "quality_grade": metrics.quality_grade,
            "perplexity": metrics.perplexity.to_dict(),
            "diversity": metrics.diversity.to_dict(),
            "repetition": metrics.repetition.to_dict(),
            "coherence": metrics.coherence.to_dict(),
            "issues": metrics.issues,
            "recommendations": metrics.recommendations,
            "num_tokens": metrics.num_tokens,
            "tokens_per_second": metrics.tokens_per_second,
        }

    # Append to history
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return str(history_file)


# ======================================================================
# Benchmark Suite
# ======================================================================

BENCHMARK_PROMPTS = [
    # Simple completions
    ("simple", "The cat sat on the"),
    ("simple", "Hello, how are you"),
    ("simple", "The weather today is"),

    # Knowledge queries
    ("knowledge", "Albert Einstein was a German-born theoretical physicist who"),
    ("knowledge", "The capital of France is"),
    ("knowledge", "In 1969, humans first landed on the"),

    # Narrative continuation
    ("narrative", "Once upon a time, in a land far away, there lived a"),
    ("narrative", "The detective examined the crime scene carefully and noticed"),
    ("narrative", "As the sun set over the mountains, she realized that"),

    # Technical text
    ("technical", "Machine learning is a subset of artificial intelligence that"),
    ("technical", "The function takes two parameters and returns"),
    ("technical", "To install the package, run the following command:"),

    # Creative writing
    ("creative", "The old lighthouse keeper had a secret that"),
    ("creative", "In the year 3000, humanity discovered"),
    ("creative", "The melody drifted through the empty streets like"),
]


def run_benchmark(
    model: LLMTransformer,
    tokenizer,
    evaluator: GenerationEvaluator,
    output_dir: str,
    generation_params: Dict[str, Any],
    device: str = "cuda",
):
    """
    Run benchmark suite with multiple prompts.
    """
    print("\n" + "=" * 80)
    print("BENCHMARK SUITE")
    print("=" * 80 + "\n")

    results_by_category: Dict[str, List[GenerationQualityMetrics]] = {}

    for category, prompt in BENCHMARK_PROMPTS:
        print(f"\n[{category.upper()}] {prompt[:50]}...")

        output_text, metrics, gen_time = generate_and_evaluate(
            model, tokenizer, evaluator,
            prompt, generation_params, device,
            verbose=False,
        )

        if category not in results_by_category:
            results_by_category[category] = []
        results_by_category[category].append(metrics)

        # Brief output
        print(f"  Score: {metrics.overall_score:.1f} ({metrics.quality_grade})")
        print(f"  PPL: {metrics.perplexity.sequence_perplexity:.2f}")
        print(f"  Output: {metrics.generated_only[:60]}...")

    # Summary by category
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print(f"\n{'Category':<15} {'Avg Score':<12} {'Avg PPL':<12} {'Distinct-2':<12} {'Rep-2':<10}")
    print("-" * 60)

    all_metrics = []
    for category, metrics_list in results_by_category.items():
        all_metrics.extend(metrics_list)

        avg_score = sum(m.overall_score for m in metrics_list) / len(metrics_list)
        ppls = [m.perplexity.sequence_perplexity for m in metrics_list if m.perplexity.sequence_perplexity > 0]
        avg_ppl = sum(ppls) / len(ppls) if ppls else 0
        avg_distinct = sum(m.diversity.distinct_2 for m in metrics_list) / len(metrics_list)
        avg_rep = sum(m.repetition.rep_2 for m in metrics_list) / len(metrics_list)

        print(f"{category:<15} {avg_score:<12.1f} {avg_ppl:<12.2f} {avg_distinct:<12.3f} {avg_rep:<10.3f}")

    # Overall
    print("-" * 60)
    overall_score = sum(m.overall_score for m in all_metrics) / len(all_metrics)
    ppls = [m.perplexity.sequence_perplexity for m in all_metrics if m.perplexity.sequence_perplexity > 0]
    overall_ppl = sum(ppls) / len(ppls) if ppls else 0
    overall_distinct = sum(m.diversity.distinct_2 for m in all_metrics) / len(all_metrics)
    overall_rep = sum(m.repetition.rep_2 for m in all_metrics) / len(all_metrics)

    print(f"{'OVERALL':<15} {overall_score:<12.1f} {overall_ppl:<12.2f} {overall_distinct:<12.3f} {overall_rep:<10.3f}")

    # Grade distribution
    grades = [m.quality_grade for m in all_metrics]
    from collections import Counter
    grade_dist = Counter(grades)
    print(f"\nGrade distribution: {dict(sorted(grade_dist.items()))}")

    # Generate report
    report_path = Path(output_dir) / "benchmark_report"
    report_files = evaluator.generate_report(report_path, format="all")
    print(f"\nReports saved to:")
    for fmt, path in report_files.items():
        print(f"  {fmt}: {path}")

    return all_metrics


# ======================================================================
# History Viewer
# ======================================================================

def view_history(output_dir: str, last_n: int = 10, filter_grade: str = None):
    """View generation history."""
    history_file = Path(output_dir) / "generation_history.jsonl"

    if not history_file.exists():
        print(f"No history found at {history_file}")
        return

    entries = []
    with open(history_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                entries.append(entry)
            except json.JSONDecodeError:
                continue

    if filter_grade:
        entries = [
            e for e in entries
            if e.get("evaluation", {}).get("quality_grade") == filter_grade
        ]

    entries = entries[-last_n:]

    print("\n" + "=" * 80)
    print(f"GENERATION HISTORY ({len(entries)} entries)")
    print("=" * 80 + "\n")

    for i, entry in enumerate(entries, 1):
        eval_data = entry.get("evaluation", {})
        score = eval_data.get("overall_score", "N/A")
        grade = eval_data.get("quality_grade", "N/A")
        ppl = eval_data.get("perplexity", {}).get("sequence_perplexity", "N/A")

        print(f"[{i}] {entry['timestamp'][:19]}")
        print(f"    Prompt: {entry['prompt'][:50]}...")
        print(f"    Output: {entry['generated_text'][:60]}...")

        if isinstance(score, (int, float)):
            print(f"    Score: {score:.1f}/100 ({grade})")
        if isinstance(ppl, (int, float)):
            print(f"    PPL: {ppl:.2f}")

        checkpoint = entry.get("checkpoint", {})
        if checkpoint.get("step"):
            print(f"    Checkpoint: step {checkpoint['step']}")

        issues = eval_data.get("issues", [])
        if issues:
            print(f"    Issues: {issues[0]}")

        print()

    # Summary stats
    scores = [
        e.get("evaluation", {}).get("overall_score")
        for e in entries
        if isinstance(e.get("evaluation", {}).get("overall_score"), (int, float))
    ]

    if scores:
        print("=" * 80)
        print("STATISTICS")
        print("=" * 80)
        print(f"Average score: {sum(scores)/len(scores):.1f}")
        print(f"Min/Max: {min(scores):.1f} / {max(scores):.1f}")

        grades = [e.get("evaluation", {}).get("quality_grade") for e in entries if e.get("evaluation", {}).get("quality_grade")]
        grade_counts = Counter(grades)
        print(f"Grades: {dict(sorted(grade_counts.items()))}")


# ======================================================================
# Interactive Mode
# ======================================================================

def interactive_mode(
    model: LLMTransformer,
    tokenizer,
    evaluator: GenerationEvaluator,
    cfg: Dict[str, Any],
    checkpoint_info: Dict[str, Any],
    output_dir: str,
    device: str,
):
    """Interactive generation loop with evaluation."""
    print("\n" + "=" * 60)
    print("SLGA Interactive Generation with Quality Evaluation")
    print("=" * 60)
    print("Commands: /quit, /config, /eval, /report, /help")
    print("=" * 60 + "\n")

    # Get inference params
    inf_cfg = cfg.get("inference", {})
    gen_params = {
        "max_new_tokens": inf_cfg.get("max_new_tokens", 100),
        "temperature": inf_cfg.get("temperature", 0.7),
        "top_k": inf_cfg.get("top_k", 50),
        "top_p": inf_cfg.get("top_p", 0.9),
        "repetition_penalty": inf_cfg.get("repetition_penalty", 1.2),
        "no_repeat_ngram_size": inf_cfg.get("no_repeat_ngram_size", 3),
    }

    evaluate_all = False  # Toggle evaluation for each generation

    while True:
        try:
            prompt = input("\nPrompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue

        if prompt.lower() == "/quit":
            print("Goodbye!")
            break

        if prompt.lower() == "/config":
            print(f"\nCurrent settings:")
            for key, value in gen_params.items():
                print(f"  {key}: {value}")
            print(f"  evaluate_all: {evaluate_all}")
            continue

        if prompt.lower() == "/eval":
            evaluate_all = not evaluate_all
            print(f"Evaluation mode: {'ON' if evaluate_all else 'OFF'}")
            continue

        if prompt.lower() == "/report":
            report_path = Path(output_dir) / "interactive_report"
            files = evaluator.generate_report(report_path, format="all")
            print("Reports generated:")
            for fmt, path in files.items():
                print(f"  {fmt}: {path}")
            continue

        if prompt.lower() == "/help":
            print("\nCommands:")
            print("  /quit   - Exit")
            print("  /config - Show settings")
            print("  /eval   - Toggle evaluation mode")
            print("  /report - Generate evaluation report")
            print("  /help   - This message")
            continue

        print("\nGenerating...", end="", flush=True)

        try:
            if evaluate_all:
                output, metrics, gen_time = generate_and_evaluate(
                    model, tokenizer, evaluator,
                    prompt, gen_params, device, verbose=True
                )
            else:
                output, gen_time = generate_text(
                    model, tokenizer, prompt,
                    device=device,
                    **gen_params,
                )
                metrics = None
                print("\r" + " " * 20 + "\r", end="")

            print(f"\nOutput:\n{output}")

            # Log
            log_generation(
                output_dir=output_dir,
                prompt=prompt,
                generated_text=output,
                generation_params=gen_params,
                model_config=cfg.get("model", {}),
                checkpoint_info=checkpoint_info,
                metrics=metrics,
                generation_time=gen_time,
                device=device,
            )

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="SLGA Text Generation with Quality Evaluation")

    # Model arguments
    parser.add_argument("--checkpoint", type=str, help="Checkpoint directory")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    # Generation mode
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark suite")
    parser.add_argument("--history", type=str, help="View history from output dir")

    # Generation parameters
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.2, help="Repetition penalty")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3, help="No repeat n-gram")

    # Evaluation options
    parser.add_argument("--evaluate", action="store_true", help="Evaluate generation quality")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for logs")
    parser.add_argument("--no-log", action="store_true", help="Disable logging")

    # History options
    parser.add_argument("--last", type=int, default=10, help="Show last N history entries")
    parser.add_argument("--filter-grade", type=str, help="Filter history by grade (A/B/C/D/F)")

    args = parser.parse_args()

    # View history mode
    if args.history:
        view_history(args.history, args.last, args.filter_grade)
        return

    # Need checkpoint for generation
    if not args.checkpoint:
        parser.error("--checkpoint is required for generation (or use --history)")

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, tokenizer, cfg = load_model(args.checkpoint, args.config, args.device)
    print(f"Model loaded ({model.get_num_params()/1e6:.1f}M params)")

    if args.device == "cuda":
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory: {mem:.2f} GB")

    # Get checkpoint info
    checkpoint_info = get_checkpoint_info(args.checkpoint)

    # Output directory
    output_dir = args.output_dir or os.path.dirname(args.checkpoint) or "out_slga"

    # Generation parameters
    gen_params = {
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }

    # Create evaluator
    evaluator = GenerationEvaluator(model, tokenizer, args.device)

    # Run mode
    if args.benchmark:
        run_benchmark(
            model, tokenizer, evaluator,
            output_dir, gen_params, args.device
        )

    elif args.interactive:
        interactive_mode(
            model, tokenizer, evaluator, cfg,
            checkpoint_info, output_dir, args.device
        )

    elif args.prompt:
        print(f"\nPrompt: {args.prompt}")
        print("Generating...")

        if args.evaluate:
            output, metrics, gen_time = generate_and_evaluate(
                model, tokenizer, evaluator,
                args.prompt, gen_params, args.device,
                verbose=True,
            )
        else:
            output, gen_time = generate_text(
                model, tokenizer, args.prompt,
                device=args.device,
                **gen_params,
            )
            metrics = None

        print(f"\nOutput:\n{output}")

        # Log
        if not args.no_log:
            log_file = log_generation(
                output_dir=output_dir,
                prompt=args.prompt,
                generated_text=output,
                generation_params=gen_params,
                model_config=cfg.get("model", {}),
                checkpoint_info=checkpoint_info,
                metrics=metrics,
                generation_time=gen_time,
                device=args.device,
            )
            print(f"\nLogged to {log_file}")

    else:
        print("Please provide --prompt, --interactive, --benchmark, or --history")
        parser.print_help()


if __name__ == "__main__":
    main()
