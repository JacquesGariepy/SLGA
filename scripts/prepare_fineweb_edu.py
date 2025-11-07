#!/usr/bin/env python3
"""
FineWeb-Edu Dataset Preparation Script

Downloads and validates the FineWeb-Edu dataset for SLGA-Plus training.

Usage:
    python scripts/prepare_fineweb_edu.py --subset sample-10BT
    python scripts/prepare_fineweb_edu.py --subset sample-10BT --validate
    python scripts/prepare_fineweb_edu.py --subset sample-10BT --cache-dir /mnt/ssd/cache
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def download_dataset(subset: str = "sample-10BT", cache_dir: str = None, retry: int = 3):
    """Download FineWeb-Edu dataset."""
    print("=" * 80)
    print("=== FineWeb-Edu Dataset Preparation ===")
    print("=" * 80)
    print(f"Subset: {subset}")
    print(f"Cache dir: {cache_dir or 'default (~/.cache/huggingface)'}")
    print(f"Retries: {retry}")
    print()

    subset_info = {
        "sample-10BT": {
            "size_gb": 30,
            "tokens": "10B",
            "description": "10 billion tokens (recommended for experimentation)"
        },
        "sample-100BT": {
            "size_gb": 300,
            "tokens": "100B",
            "description": "100 billion tokens (production training)"
        },
    }

    if subset in subset_info:
        info = subset_info[subset]
        print(f"📊 Dataset Info:")
        print(f"  Size: ~{info['size_gb']}GB")
        print(f"  Tokens: {info['tokens']}")
        print(f"  Description: {info['description']}")
        print()

    print("🔄 Starting download...")
    print("⚠️  First download will take 1-2 hours depending on network speed")
    print()

    for attempt in range(1, retry + 1):
        try:
            print(f"Attempt {attempt}/{retry}...")

            dataset = load_dataset(
                "HuggingFaceFW/fineweb-edu",
                name=subset,
                split="train",
                cache_dir=cache_dir,
                trust_remote_code=True,
            )

            print()
            print("✅ Download completed successfully!")
            print(f"📈 Total samples: {len(dataset):,}")
            print()

            return dataset

        except Exception as e:
            print(f"❌ Attempt {attempt} failed: {e}")
            if attempt < retry:
                print(f"⏳ Retrying in 5 seconds...")
                import time
                time.sleep(5)
            else:
                print("❌ All download attempts failed")
                raise


def validate_dataset(dataset, num_samples: int = 100):
    """Validate dataset integrity."""
    print("=" * 80)
    print("=== Dataset Validation ===")
    print("=" * 80)

    print(f"📊 Total samples: {len(dataset):,}")
    print()

    print(f"🔍 Inspecting first {num_samples} samples...")

    text_lengths = []

    for i in tqdm(range(min(num_samples, len(dataset))), desc="Validating"):
        sample = dataset[i]
        assert "text" in sample, f"Sample {i} missing 'text' field"
        text_lengths.append(len(sample["text"]))

    print()
    print("📈 Text Length Statistics:")
    print(f"  Mean: {sum(text_lengths) / len(text_lengths):.0f} chars")
    print(f"  Min: {min(text_lengths)} chars")
    print(f"  Max: {max(text_lengths)} chars")

    print()
    print("📄 Sample Preview:")
    print("-" * 80)
    print(dataset[0]["text"][:500])
    print("...")
    print("-" * 80)
    print()
    print("✅ Validation passed!")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare FineWeb-Edu dataset")
    parser.add_argument("--subset", type=str, default="sample-10BT", help="Dataset subset")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--retry", type=int, default=3, help="Download retries")
    parser.add_argument("--validate", action="store_true", help="Validate after download")
    args = parser.parse_args()

    try:
        dataset = download_dataset(subset=args.subset, cache_dir=args.cache_dir, retry=args.retry)
        
        if args.validate:
            validate_dataset(dataset)

        print("=" * 80)
        print("✅ Dataset preparation complete!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
