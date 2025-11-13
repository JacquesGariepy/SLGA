#!/usr/bin/env python3
"""
Debug script pour inspecter exactement ce qui est envoyé au modèle.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch
from src.legacy.data import get_tokenizer, load_text_dataset, CollatorLocal
from torch.utils.data import DataLoader

def debug_real_batch():
    """Inspecter un vrai batch du dataloader de training"""

    print("="*80)
    print("DEBUGGING REAL TRAINING BATCH")
    print("="*80)

    # Charger config
    with open("config/config_fineweb_edu_3090_optimized.yaml") as f:
        cfg = yaml.safe_load(f)

    # Setup
    tokenizer = get_tokenizer(cfg["tokenizer"])
    pad_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    print(f"\nTokenizer info:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Pad token ID: {pad_id}")

    # Charger dataset (juste quelques exemples)
    print(f"\nLoading dataset...")
    ds_train = load_text_dataset(
        cfg["data"]["dataset"],
        cfg["data"].get("subset"),
        cfg["data"]["split_train"],
    )

    # Limiter pour test rapide
    ds_train = ds_train.select(range(100))
    print(f"  Loaded {len(ds_train)} samples")

    # Créer collator
    seq_len = cfg["train"].get("seq_len_start", 384)
    collator = CollatorLocal(tokenizer, seq_len)

    # Créer dataloader
    dataloader = DataLoader(
        ds_train,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
    )

    # Prendre premier batch
    print(f"\nGetting first batch...")
    batch = next(iter(dataloader))

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    print(f"\nBatch info:")
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  labels shape: {labels.shape}")

    # Analyser labels
    print(f"\nLabel analysis:")
    unique_labels = torch.unique(labels)
    print(f"  Unique values ({len(unique_labels)}): {unique_labels.tolist()[:30]}...")

    # Chercher valeurs invalides
    invalid_mask = (labels != -100) & ((labels < 0) | (labels >= vocab_size))
    num_invalid = invalid_mask.sum().item()

    if num_invalid > 0:
        print(f"\n  ❌ FOUND {num_invalid} INVALID LABELS!")

        # Trouver les valeurs invalides
        invalid_labels = labels[invalid_mask]
        unique_invalid = torch.unique(invalid_labels)
        print(f"     Invalid values: {unique_invalid.tolist()}")

        # Trouver où elles sont
        for i in range(labels.size(0)):  # Pour chaque exemple
            invalid_in_example = invalid_mask[i].sum().item()
            if invalid_in_example > 0:
                print(f"\n     Example {i}: {invalid_in_example} invalid labels")
                invalid_positions = torch.where(invalid_mask[i])[0][:10]
                print(f"       Positions: {invalid_positions.tolist()}")
                print(f"       Values: {labels[i][invalid_positions].tolist()}")
                print(f"       Corresponding input_ids: {input_ids[i][invalid_positions].tolist()}")

        return False
    else:
        print(f"  ✅ All labels valid!")

    # Vérifier pad masking
    num_masked = (labels == -100).sum().item()
    num_pads_in_input = (input_ids == pad_id).sum().item()

    print(f"\n  Masked labels (-100): {num_masked}")
    print(f"  Pads in input_ids: {num_pads_in_input}")

    # Tester sur GPU
    if torch.cuda.is_available():
        print(f"\n  Testing CUDA loss computation...")

        try:
            input_ids_cuda = input_ids.cuda()
            labels_cuda = labels.cuda()

            # Créer logits aléatoires
            logits = torch.randn(
                input_ids_cuda.size(0),
                input_ids_cuda.size(1),
                vocab_size,
                device='cuda'
            )

            # Compute loss (là où ça crash)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab_size),
                labels_cuda.view(-1),
                ignore_index=-100
            )

            print(f"  ✅ Loss computed successfully: {loss.item():.4f}")
            print(f"  ✅ NO CUDA ASSERTION!")

        except RuntimeError as e:
            print(f"  ❌ CUDA error: {e}")
            return False

    print(f"\n{'='*80}")
    print(f"✅ DEBUGGING COMPLETE - No issues found!")
    print(f"{'='*80}")
    return True

if __name__ == "__main__":
    success = debug_real_batch()
    sys.exit(0 if success else 1)
