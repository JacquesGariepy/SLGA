#!/usr/bin/env python3
"""
Script pour télécharger et préparer les datasets avant l'entraînement.

Usage:
    python download_dataset.py --dataset wikimedia/wikipedia --subset 20231101.en
"""

import argparse
import os
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour importer src
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from src.data import get_tokenizer


def download_and_prepare(
    dataset_name: str,
    subset: str = None,
    split: str = "train",
    cache_dir: str = None,
):
    """
    Télécharge et prépare un dataset HuggingFace.
    
    Args:
        dataset_name: Nom du dataset (ex: "wikimedia/wikipedia")
        subset: Subset/config (ex: "20231101.en")
        split: Split à télécharger
        cache_dir: Répertoire de cache personnalisé
    """
    print("=" * 80)
    print("SLGA-Plus Dataset Downloader")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    if subset:
        print(f"Subset: {subset}")
    print(f"Split: {split}")
    if cache_dir:
        print(f"Cache dir: {cache_dir}")
    print()
    
    # Télécharger dataset
    print("Downloading dataset...")
    try:
        if subset:
            ds = load_dataset(
                dataset_name,
                subset,
                split=split,
                cache_dir=cache_dir,
            )
        else:
            ds = load_dataset(
                dataset_name,
                split=split,
                cache_dir=cache_dir,
            )
        
        print(f"✓ Downloaded successfully!")
        print(f"  Number of examples: {len(ds):,}")
        
        # Afficher exemple
        if len(ds) > 0:
            print(f"\nFirst example keys: {list(ds[0].keys())}")
            print(f"Example text preview:")
            text_key = "text" if "text" in ds[0] else list(ds[0].keys())[0]
            print(f"  {ds[0][text_key][:200]}...")
        
        # Info stockage
        cache_location = os.path.expanduser("~/.cache/huggingface/datasets/")
        print(f"\nDataset cached at: {cache_location}")
        
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        return False
    
    return True


def list_popular_datasets():
    """Affiche une liste de datasets populaires pour LLM."""
    print("\n" + "=" * 80)
    print("Popular Datasets for LLM Training")
    print("=" * 80)
    
    datasets = [
        {
            "name": "wikimedia/wikipedia",
            "subset": "20231101.en",
            "size": "~20 GB",
            "desc": "Wikipedia EN (Nov 2023)",
        },
        {
            "name": "wikitext",
            "subset": "wikitext-103-v1",
            "size": "~500 MB",
            "desc": "WikiText-103 (smaller, good for testing)",
        },
        {
            "name": "wikitext",
            "subset": "wikitext-2-v1",
            "size": "~12 MB",
            "desc": "WikiText-2 (tiny, for debugging)",
        },
        {
            "name": "bookcorpus",
            "subset": None,
            "size": "~5 GB",
            "desc": "BookCorpus (books)",
        },
        {
            "name": "c4",
            "subset": "en",
            "size": "~750 GB",
            "desc": "C4 (Colossal Clean Crawled Corpus)",
        },
        {
            "name": "openwebtext",
            "subset": None,
            "size": "~38 GB",
            "desc": "OpenWebText (Reddit links)",
        },
    ]
    
    print(f"{'Dataset':<30} {'Subset':<20} {'Size':<10} {'Description'}")
    print("-" * 80)
    for ds in datasets:
        subset_str = ds['subset'] or "None"
        print(f"{ds['name']:<30} {subset_str:<20} {ds['size']:<10} {ds['desc']}")
    
    print("\nUsage example:")
    print("  python download_dataset.py --dataset wikitext --subset wikitext-2-v1")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for SLGA training"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikimedia/wikipedia",
        help="Dataset name (default: wikimedia/wikipedia)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="20231101.en",
        help="Dataset subset/config (default: 20231101.en)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to download (default: train)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List popular datasets",
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_popular_datasets()
        return
    
    # Télécharger
    success = download_and_prepare(
        args.dataset,
        args.subset,
        args.split,
        args.cache_dir,
    )
    
    if success:
        print("\n" + "=" * 80)
        print("✓ Dataset ready for training!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Run: python scripts/train.py")
        print("  2. Or edit configs/config_default.yaml to use different dataset")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()