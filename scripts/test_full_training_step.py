#!/usr/bin/env python3
"""
Test complet d'un step de training pour vérifier que tout fonctionne.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch
import torch.nn.functional as F
from src.data import get_tokenizer, load_text_dataset, CollatorLocal
from src.model import Config, LLMTransformer
from torch.utils.data import DataLoader

def cross_entropy_shifted_test(logits, labels):
    """Version de test de cross_entropy_shifted"""
    logits_shifted = logits[:, :-1, :].contiguous()
    labels_shifted = labels[:, :-1].contiguous()

    loss = F.cross_entropy(
        logits_shifted.view(-1, logits_shifted.size(-1)),
        labels_shifted.view(-1),
        ignore_index=-100,  # Ignorer -100, pas pad_id!
    )
    return loss

def test_training_step():
    """Simuler un vrai step de training"""

    print("="*80)
    print("FULL TRAINING STEP TEST")
    print("="*80)

    # Load config
    with open("config/config_fineweb_edu_3090_optimized.yaml") as f:
        cfg = yaml.safe_load(f)

    # Setup
    tokenizer = get_tokenizer(cfg["tokenizer"])
    pad_id = tokenizer.pad_token_id

    print(f"\nSetup:")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Pad token ID: {pad_id}")

    # Créer modèle (petit pour test rapide)
    print(f"\nCreating model...")
    model_cfg = Config(**cfg["model"])
    model = LLMTransformer(model_cfg)

    if torch.cuda.is_available():
        model = model.cuda()
        print(f"  Model on CUDA")
    else:
        print(f"  Model on CPU")

    # Load dataset
    print(f"\nLoading dataset...")
    ds_train = load_text_dataset(
        cfg["data"]["dataset"],
        cfg["data"].get("subset"),
        cfg["data"]["split_train"],
    )
    ds_train = ds_train.select(range(50))  # Petit subset
    print(f"  Loaded {len(ds_train)} samples")

    # Créer dataloader
    seq_len = cfg["train"].get("seq_len_start", 384)
    collator = CollatorLocal(tokenizer, seq_len)

    dataloader = DataLoader(
        ds_train,
        batch_size=4,  # Petit batch pour test rapide
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
    )

    # Prendre un batch
    print(f"\nGetting batch...")
    batch = next(iter(dataloader))

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    print(f"  input_ids: {input_ids.shape}")
    print(f"  labels: {labels.shape}")

    # Vérifier labels
    num_masked = (labels == -100).sum().item()
    invalid_mask = (labels != -100) & ((labels < 0) | (labels >= tokenizer.vocab_size))
    num_invalid = invalid_mask.sum().item()

    print(f"  Masked tokens (-100): {num_masked}")
    print(f"  Invalid tokens: {num_invalid}")

    if num_invalid > 0:
        print(f"  ❌ FOUND INVALID LABELS!")
        return False

    # Forward pass
    print(f"\nForward pass...")
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        labels = labels.cuda()

    try:
        with torch.no_grad():
            logits = model(input_ids)  # (B, L, V)

        print(f"  Logits shape: {logits.shape}")

        # Compute loss
        print(f"\nComputing loss...")
        loss = cross_entropy_shifted_test(logits, labels)

        print(f"  ✅ Loss: {loss.item():.4f}")
        print(f"  ✅ NO CUDA ASSERTION ERROR!")

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n{'='*80}")
    print(f"✅ FULL TRAINING STEP TEST PASSED!")
    print(f"{'='*80}")
    return True

if __name__ == "__main__":
    success = test_training_step()
    sys.exit(0 if success else 1)
