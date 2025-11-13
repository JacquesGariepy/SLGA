#!/usr/bin/env python3
"""
Quick diagnostic script for step 1000 checkpoint quality issues
"""

import torch
import re
import sys
from pathlib import Path
from transformers import GPT2Tokenizer
from collections import Counter

def analyze_generation_file(filepath):
    """Analyze a generation output file"""
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract generated text
    match = re.search(r'GENERATED TEXT:\n(.*?)={5,}', content, re.DOTALL)
    if not match:
        return None

    generated = match.group(1).strip()

    # Analyze
    lines = generated.split('\n')
    words = []
    newlines = 0

    for line in lines:
        line = line.strip()
        if not line:
            newlines += 1
        else:
            words.extend(line.split())

    total_tokens = len(words)
    unique_tokens = len(set(words))
    diversity = unique_tokens / total_tokens if total_tokens > 0 else 0

    # Token frequency
    word_counts = Counter(words)
    most_common = word_counts.most_common(5)

    # Calculate newline ratio
    total_lines = len(lines)
    newline_ratio = newlines / total_lines if total_lines > 0 else 0

    return {
        'filepath': filepath,
        'total_words': total_tokens,
        'unique_words': unique_tokens,
        'diversity': diversity,
        'newlines': newlines,
        'newline_ratio': newline_ratio,
        'most_common': most_common,
        'text_preview': ' '.join(words[:30])
    }

def analyze_training_log(logpath):
    """Extract key metrics from training log"""
    with open(logpath, 'r') as f:
        content = f.read()

    # Find step 1000
    pattern = r'Step\s+1[,.]?000.*?Loss.*?(\d+\.\d+).*?PPL.*?(\d+\.\d+)'
    match = re.search(pattern, content, re.IGNORECASE)

    if not match:
        return None

    train_loss = float(match.group(1))
    train_ppl = float(match.group(2))

    # Find validation
    val_pattern = r'VALIDATION.*?Step.*?1[,.]?000.*?Loss.*?(\d+\.\d+).*?PPL.*?(\d+\.\d+)'
    val_match = re.search(val_pattern, content, re.IGNORECASE | re.DOTALL)

    val_loss = float(val_match.group(1)) if val_match else None
    val_ppl = float(val_match.group(2)) if val_match else None

    # Find best loss
    best_pattern = r'best:.*?(\d+\.\d+)'
    best_match = re.search(best_pattern, content)
    best_loss = float(best_match.group(1)) if best_match else None

    return {
        'train_loss': train_loss,
        'train_ppl': train_ppl,
        'val_loss': val_loss,
        'val_ppl': val_ppl,
        'best_loss': best_loss,
        'gap': val_loss - train_loss if val_loss else None
    }

def check_dataset_quality(data_dir):
    """Check if dataset has quality issues"""
    # This is a placeholder - would need actual dataset access
    print("⚠️  Dataset quality check requires manual inspection")
    print("    Run: python scripts/inspect_training_batch.py --checkpoint out_slga/ckpt_1000")
    return None

def print_report(gen_analysis, log_analysis):
    """Print diagnostic report"""
    print("="*80)
    print("🔍 DIAGNOSTIC REPORT - Step 1000 Checkpoint")
    print("="*80)
    print()

    # Training metrics
    if log_analysis:
        print("📊 TRAINING METRICS")
        print("-" * 80)
        print(f"Training Loss:    {log_analysis['train_loss']:.4f}")
        print(f"Training PPL:     {log_analysis['train_ppl']:.1f}")
        if log_analysis['val_loss']:
            print(f"Validation Loss:  {log_analysis['val_loss']:.4f}")
            print(f"Validation PPL:   {log_analysis['val_ppl']:.1f}")
            print(f"Train/Val Gap:    {log_analysis['gap']:+.4f}")
        if log_analysis['best_loss']:
            print(f"Best Loss (ever): {log_analysis['best_loss']:.4f}")
            print(f"Regression:       {log_analysis['train_loss'] - log_analysis['best_loss']:+.4f}")

        print()
        print("🔴 STATUS: ", end="")
        if log_analysis['train_loss'] > 6.0:
            print("CRITICAL - Loss too high (expected 3.5-4.5)")
        elif log_analysis['train_loss'] > 5.0:
            print("WARNING - Loss higher than expected")
        else:
            print("OK")

        if log_analysis['train_ppl'] > 500:
            print("🔴 STATUS: CRITICAL - Perplexity catastrophic (expected 30-100)")
        elif log_analysis['train_ppl'] > 200:
            print("🟡 STATUS: WARNING - Perplexity high")
        print()

    # Generation quality
    if gen_analysis:
        print("📝 GENERATION QUALITY")
        print("-" * 80)
        for i, analysis in enumerate(gen_analysis, 1):
            print(f"\n--- Generation {i} ---")
            print(f"File: {Path(analysis['filepath']).name}")
            print(f"Total words:      {analysis['total_words']}")
            print(f"Unique words:     {analysis['unique_words']}")
            print(f"Diversity:        {analysis['diversity']:.2%}")
            print(f"Empty lines:      {analysis['newlines']}")
            print(f"Newline ratio:    {analysis['newline_ratio']:.2%}")
            print(f"\nMost common tokens:")
            for word, count in analysis['most_common']:
                print(f"  '{word}': {count}×")
            print(f"\nPreview: {analysis['text_preview']}")

            # Status
            print("\n🔴 ISSUES DETECTED:")
            issues = []
            if analysis['newline_ratio'] > 0.5:
                issues.append(f"  - Excessive newlines ({analysis['newline_ratio']:.0%})")
            if analysis['diversity'] < 0.5:
                issues.append(f"  - Low diversity ({analysis['diversity']:.0%})")
            if analysis['total_words'] < 30:
                issues.append(f"  - Too few words ({analysis['total_words']})")

            # Check for "the" overuse
            the_count = sum(count for word, count in analysis['most_common'] if word.lower() == 'the')
            the_ratio = the_count / analysis['total_words'] if analysis['total_words'] > 0 else 0
            if the_ratio > 0.15:
                issues.append(f"  - 'the' overuse ({the_ratio:.0%}, expected ~7%)")

            if issues:
                for issue in issues:
                    print(issue)
            else:
                print("  None (relative to extremely low baseline)")

    print()
    print("="*80)
    print("🎯 RECOMMENDATIONS")
    print("="*80)
    print()

    if log_analysis and log_analysis['train_loss'] > 6.0:
        print("1. 🔥 CRITICAL: Increase training to 10,000 steps minimum")
        print("   Current: 1,000 steps (insufficient for convergence)")
        print("   → Edit config: max_steps = 10000")
        print()

    if gen_analysis and any(a['newline_ratio'] > 0.4 for a in gen_analysis):
        print("2. 🔥 IMPORTANT: Clean dataset to reduce newlines")
        print("   → Run: python scripts/clean_wikipedia_dataset.py")
        print()

    if log_analysis and log_analysis.get('gap') and log_analysis.get('gap') > 0.5:
        print("3. 🟡 WARNING: Add regularization (dropout, weight decay)")
        print()

    print("4. 📋 Next diagnostic steps:")
    print("   → python scripts/inspect_training_batch.py --checkpoint out_slga/ckpt_1000")
    print("   → python scripts/check_training_losses.py")
    print("   → python scripts/compare_sampling.py --checkpoint out_slga/ckpt_1000")
    print()

    print("5. 🎯 Expected improvements after fixes:")
    print("   Loss:       6.99 → 3.5-4.5 (35-55% reduction)")
    print("   Perplexity: 1091 → 30-80 (90-95% reduction)")
    print("   Quality:    2/10 → 6-7/10 (3-4× better)")
    print()

def main():
    print("🔍 Diagnosing Step 1000 Checkpoint...")
    print()

    base_dir = Path(__file__).parent.parent

    # Find generation files
    gen_dir = base_dir / "out_slga"
    gen_files = sorted(gen_dir.glob("generation_*_step1000.txt"))

    gen_analysis = []
    for gen_file in gen_files:
        analysis = analyze_generation_file(gen_file)
        if analysis:
            gen_analysis.append(analysis)

    # Analyze training log
    log_file = base_dir / "training.log"
    log_analysis = None
    if log_file.exists():
        log_analysis = analyze_training_log(log_file)

    # Print report
    print_report(gen_analysis, log_analysis)

    # Return exit code based on severity
    if log_analysis and log_analysis['train_loss'] > 6.0:
        return 2  # Critical
    elif gen_analysis and any(a['newline_ratio'] > 0.5 for a in gen_analysis):
        return 1  # Warning
    else:
        return 0  # OK

if __name__ == "__main__":
    sys.exit(main())
