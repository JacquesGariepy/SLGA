#!/usr/bin/env python3
"""
Test du flow de validation EXACT qui crashe.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch
import torch.nn.functional as F
from src.data import get_tokenizer, load_text_dataset
from src.model import Config, LLMTransformer
from torch.utils.data import DataLoader

def cross_entropy_shifted(logits, labels, pad_id):
    """Version EXACTE de train.py ligne 85-113"""
    logits_shifted = logits[:, :-1, :].contiguous()
    labels_shifted = labels[:, :-1].contiguous()

    loss = F.cross_entropy(
        logits_shifted.view(-1, logits_shifted.size(-1)),
        labels_shifted.view(-1),
        ignore_index=-100,  # Doit être -100, pas pad_id!
    )
    return loss

def validate(model, val_loader, pad_id, device):
    """Version EXACTE de validate() dans train.py"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 3:  # Just 3 batches for test
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            print(f"\n  Batch {i}:")
            print(f"    input_ids shape: {input_ids.shape}")
            print(f"    labels shape: {labels.shape}")

            # Vérifier labels AVANT forward
            invalid_mask = (labels != -100) & ((labels < 0) | (labels >= 50257))
            if invalid_mask.any():
                print(f"    ❌ FOUND INVALID LABELS!")
                invalid_count = invalid_mask.sum().item()
                print(f"       Count: {invalid_count}")
                invalid_values = labels[invalid_mask][:10]
                print(f"       Values: {invalid_values.tolist()}")
                return False

            # Forward
            try:
                logits = model(input_ids)
                loss = cross_entropy_shifted(logits, labels, pad_id)
                print(f"    ✅ Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"    ❌ ERROR: {e}")
                return False

            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    print(f"\n  ✅ VALIDATION PASSED!")
    return True

def test_validation():
    """Test complet du flow de validation"""

    print("="*80)
    print("VALIDATION FLOW TEST")
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

    # Créer modèle
    print(f"\nCreating model...")
    model_cfg = Config(**cfg["model"])
    model = LLMTransformer(model_cfg)

    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device("cuda")
        print(f"  Model on CUDA")
    else:
        device = torch.device("cpu")
        print(f"  Model on CPU")

    # Load VALIDATION dataset
    print(f"\nLoading VALIDATION dataset...")
    ds_val = load_text_dataset(
        cfg["data"]["dataset"],
        cfg["data"].get("subset"),
        cfg["data"]["split_val"],  # Important: split VAL!
    )

    # Limit for speed
    if len(ds_val) > 100:
        ds_val = ds_val.select(range(100))
    print(f"  Loaded {len(ds_val)} validation samples")

    # Utiliser LE MÊME collator que dans train.py (collate_val_reduced)
    max_len_val = 512

    def collate_val_reduced(examples):
        """EXACTEMENT le même collator que dans train.py ligne 199-290"""
        import torch

        def _stack_input_ids(ids_list):
            tens = []
            for ids in ids_list:
                if not torch.is_tensor(ids):
                    ids = torch.as_tensor(ids, dtype=torch.long)
                ids = ids.view(-1)
                if ids.numel() >= max_len_val + 1:
                    ids = ids[: max_len_val + 1]
                else:
                    pad = torch.full((max_len_val + 1 - ids.numel(),), pad_id, dtype=torch.long)
                    ids = torch.cat([ids, pad], dim=0)
                tens.append(ids)
            input_ids = torch.stack(tens, dim=0)  # (B, L+1)

            # Créer input_ids et labels AVANT masking
            input_ids_final = input_ids[:, :-1]  # (B, L)
            labels = input_ids[:, 1:].clone()  # (B, L)

            # Masquer pads
            pad_mask = (labels == pad_id)
            labels[pad_mask] = -100

            return {
                "input_ids": input_ids_final,
                "labels": labels,
                "cache_global_ids": None,
            }

        # Détection format
        ex0 = examples[0]
        if isinstance(ex0, dict) and "input_ids" in ex0:
            return _stack_input_ids([ex["input_ids"] for ex in examples])

        # Texte brut
        text_key = None
        if isinstance(ex0, dict):
            for k in ("text", "content", "document"):
                if k in ex0 and isinstance(ex0[k], str):
                    text_key = k
                    break

        if text_key is None and isinstance(ex0, str):
            texts = list(examples)
        elif text_key is not None:
            texts = [ex[text_key] for ex in examples]
        else:
            raise KeyError(f"Format non supporté: {type(ex0)}")

        # Tokenize
        encoded = tokenizer(
            texts,
            max_length=max_len_val + 1,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]

        # Créer labels
        input_ids_final = input_ids[:, :-1]
        labels = input_ids[:, 1:].clone()

        # Masquer pads
        pad_mask = (labels == pad_id)
        labels[pad_mask] = -100

        return {
            "input_ids": input_ids_final,
            "labels": labels,
            "cache_global_ids": None,
        }

    # Créer dataloader
    val_batch_size = 7  # Même taille que dans train.py
    val_loader = DataLoader(
        ds_val,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate_val_reduced,
        drop_last=False,
        num_workers=0,
    )

    # Test validation
    print(f"\nRunning validation...")
    success = validate(model, val_loader, pad_id, device)

    if success:
        print(f"\n{'='*80}")
        print(f"✅ VALIDATION FLOW TEST PASSED!")
        print(f"{'='*80}")
        return True
    else:
        print(f"\n{'='*80}")
        print(f"❌ VALIDATION FLOW TEST FAILED!")
        print(f"{'='*80}")
        return False

if __name__ == "__main__":
    success = test_validation()
    sys.exit(0 if success else 1)
