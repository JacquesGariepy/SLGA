#!/usr/bin/env python3
"""
Script d'inspection des batches d'entraînement

Vérifie que les données sont correctes avant de lancer l'entraînement.

Usage:
    python scripts/inspect_training_batch.py
    python scripts/inspect_training_batch.py --config config_3090.yaml
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import yaml
from transformers import GPT2Tokenizer
from src.data import get_tokenizer, load_text_dataset, CollatorLocal, CollatorLocalGlobal
from torch.utils.data import DataLoader


def build_loaders(cfg):
    """Build dataloaders (copié de train.py)"""
    tokenizer = get_tokenizer(cfg["tokenizer"])

    # Datasets
    try:
        ds_train = load_text_dataset(
            cfg["data"]["dataset"],
            cfg["data"].get("subset"),
            cfg["data"]["split_train"],
        )
        print(f"✓ Dataset loaded: {len(ds_train)} training samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None, None

    # Limiter taille pour inspection rapide
    max_samples = 1000
    if len(ds_train) > max_samples:
        ds_train = ds_train.select(range(max_samples))
        print(f"  (Limited to {max_samples} samples for inspection)")

    # Collator
    use_learned = cfg["model"].get("learned_landmarks", True)
    seq_len = cfg["train"].get("seq_len_start", 512)

    if use_learned:
        collate_fn = CollatorLocal(tokenizer, seq_len)
    else:
        collate_fn = CollatorLocalGlobal(
            tokenizer,
            seq_len,
            cfg["train"]["global_every"],
            cfg["train"]["max_global"],
        )

    # Dataloader
    train_loader = DataLoader(
        ds_train,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,  # Don't shuffle for inspection
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=0,  # Single thread for debugging
    )

    return tokenizer, train_loader


def inspect_batch(batch, tokenizer, cfg, batch_idx=0):
    """Inspect a single batch"""
    print("="*80)
    print(f"BATCH #{batch_idx} INSPECTION")
    print("="*80)

    # Extract data
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    cache_ids = batch.get("cache_global_ids")

    B, L = input_ids.shape

    print(f"\n📊 Shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  labels: {labels.shape}")
    if cache_ids is not None:
        print(f"  cache_global_ids: {cache_ids.shape}")

    print(f"\n📈 Statistics:")
    print(f"  Batch size: {B}")
    print(f"  Sequence length: {L}")
    print(f"  Min token ID: {input_ids.min().item()}")
    print(f"  Max token ID: {input_ids.max().item()}")
    print(f"  Vocab size: {cfg['model']['vocab_size']}")

    # Check for invalid tokens
    vocab_size = cfg['model']['vocab_size']
    invalid_mask = (input_ids >= vocab_size) | (input_ids < 0)
    num_invalid = invalid_mask.sum().item()

    if num_invalid > 0:
        print(f"\n❌ PROBLEM: {num_invalid} invalid tokens found!")
        print(f"  Tokens >= vocab_size or < 0")
        invalid_positions = torch.where(invalid_mask)
        print(f"  First few: {invalid_positions[0][:5]}, {invalid_positions[1][:5]}")
    else:
        print(f"\n✅ Token IDs valid: All in range [0, {vocab_size})")

    # Check padding
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
    num_padding = (input_ids == pad_id).sum().item()
    padding_pct = (num_padding / (B * L)) * 100
    print(f"\n📝 Padding:")
    print(f"  Pad token ID: {pad_id}")
    print(f"  Padding tokens: {num_padding} / {B * L} ({padding_pct:.1f}%)")

    # Check labels
    print(f"\n🎯 Labels:")
    print(f"  Min label: {labels.min().item()}")
    print(f"  Max label: {labels.max().item()}")
    print(f"  Ignore index (-100): {(labels == -100).sum().item()} tokens")

    # Decode first example
    print(f"\n📖 First Example (input_ids[0]):")
    first_input = input_ids[0]

    # Remove padding for cleaner display
    non_pad_mask = first_input != pad_id if pad_id >= 0 else torch.ones_like(first_input, dtype=torch.bool)
    first_input_clean = first_input[non_pad_mask]

    decoded = tokenizer.decode(first_input_clean, skip_special_tokens=False)
    print(f"  Length: {len(first_input_clean)} tokens")
    print(f"  Text (first 500 chars):")
    print(f"  {decoded[:500]}")
    if len(decoded) > 500:
        print(f"  ... (truncated, total {len(decoded)} chars)")

    # Check cache_global_ids if present
    if cache_ids is not None:
        print(f"\n🎯 Cache Global IDs:")
        print(f"  Shape: {cache_ids.shape}")
        print(f"  Min: {cache_ids.min().item()}")
        print(f"  Max: {cache_ids.max().item()}")
        print(f"  Should be < seq_len ({L}): {(cache_ids < L).all().item()}")

        if (cache_ids >= L).any():
            print(f"  ❌ PROBLEM: Some cache IDs >= sequence length!")
        else:
            print(f"  ✅ All cache IDs valid")

        # Show first example's cache IDs
        print(f"  First example cache IDs: {cache_ids[0].tolist()}")

    # Token distribution
    print(f"\n📊 Token Distribution:")
    unique_tokens = torch.unique(first_input_clean)
    print(f"  Unique tokens in first example: {len(unique_tokens)}")

    # Most common tokens
    token_counts = {}
    for token_id in first_input_clean.tolist():
        token_counts[token_id] = token_counts.get(token_id, 0) + 1

    top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Most common tokens:")
    for token_id, count in top_tokens:
        token_str = tokenizer.decode([token_id])
        print(f"    ID {token_id:5d}: '{token_str}' ({count} times)")


def main():
    parser = argparse.ArgumentParser(description='Inspect training batches')
    parser.add_argument('--config', type=str, default='config_3090.yaml',
                       help='Config file to use')
    parser.add_argument('--num-batches', type=int, default=3,
                       help='Number of batches to inspect')

    args = parser.parse_args()

    print("="*80)
    print("TRAINING BATCH INSPECTION")
    print("="*80)
    print(f"Config: {args.config}")
    print()

    # Load config
    try:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        print("✓ Config loaded")
    except FileNotFoundError:
        print(f"❌ Config file not found: {args.config}")
        return

    # Build loaders
    print("\nBuilding dataloaders...")
    result = build_loaders(cfg)
    if result is None:
        return

    tokenizer, train_loader = result
    print(f"✓ Dataloaders built")
    print(f"  Total batches: {len(train_loader)}")
    print()

    # Inspect batches
    for i, batch in enumerate(train_loader):
        if i >= args.num_batches:
            break

        inspect_batch(batch, tokenizer, cfg, batch_idx=i)
        print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Inspected {min(args.num_batches, len(train_loader))} batches")
    print(f"✓ Config: {args.config}")
    print(f"✓ Batch size: {cfg['train']['batch_size']}")
    print(f"✓ Sequence length: {cfg['train']['seq_len_start']}")
    print(f"✓ Vocab size: {cfg['model']['vocab_size']}")
    print()
    print("If all checks passed (✅), data is ready for training!")
    print()


if __name__ == "__main__":
    main()
