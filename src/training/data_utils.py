"""Data loading utilities for training."""

from __future__ import annotations
import os
import sys
import subprocess
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader

from src.legacy.data import get_tokenizer, load_text_dataset, CollatorLocal, CollatorLocalGlobal


def build_loaders(cfg: Dict[str, Any], tokenizer=None) -> Tuple:
    """
    Build tokenizer and data loaders for training and validation.

    Handles dataset loading, automatic downloading if needed, and creates
    appropriate collators based on whether learned landmarks are used.

    Args:
        cfg: Configuration dictionary
        tokenizer: Optional pre-initialized tokenizer

    Returns:
        Tuple of (tokenizer, train_loader, val_loader)
    """
    tokenizer = tokenizer or get_tokenizer(cfg["tokenizer"])

    # Load datasets
    try:
        ds_train = load_text_dataset(
            cfg["data"]["dataset"],
            cfg["data"].get("subset"),
            cfg["data"]["split_train"],
        )
        ds_val = load_text_dataset(
            cfg["data"]["dataset"],
            cfg["data"].get("subset"),
            cfg["data"]["split_val"],
        )
    except Exception as e:
        print(f"Warning: Could not load dataset: {e}")
        print("Attempting to download dataset automatically...")

        # Auto-download dataset
        download_script = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "download_dataset.py")
        dataset_name = cfg["data"]["dataset"]
        subset = cfg["data"].get("subset")
        split_train = cfg["data"]["split_train"]

        # Download training split
        cmd = [sys.executable, download_script, "--dataset", dataset_name, "--split", split_train]
        if subset:
            cmd.extend(["--subset", subset])

        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Download output:")
            print(result.stdout)
            if result.stderr:
                print("Download stderr:")
                print(result.stderr)
        except subprocess.CalledProcessError as download_error:
            print(f"❌ Failed to download dataset: {download_error}")
            print("Please download the dataset manually or check your configuration.")
            raise download_error

        # Retry loading after download
        try:
            ds_train = load_text_dataset(
                cfg["data"]["dataset"],
                cfg["data"].get("subset"),
                cfg["data"]["split_train"],
            )
            ds_val = load_text_dataset(
                cfg["data"]["dataset"],
                cfg["data"].get("subset"),
                cfg["data"]["split_val"],
            )
        except Exception as retry_error:
            print(f"Warning: Could not load validation split after download: {retry_error}")
            print("Using subset of training data for validation")
            ds_all = load_text_dataset(
                cfg["data"]["dataset"],
                cfg["data"].get("subset"),
                cfg["data"]["split_train"],
            )
            # Manual split
            split_idx = int(len(ds_all) * 0.95)
            ds_train = ds_all.select(range(split_idx))
            ds_val = ds_all.select(range(split_idx, len(ds_all)))

    # Limit dataset size if specified
    max_train = cfg["data"].get("max_train_samples")
    max_val = cfg["data"].get("max_val_samples")

    if max_train and len(ds_train) > max_train:
        ds_train = ds_train.select(range(max_train))
    if max_val and len(ds_val) > max_val:
        ds_val = ds_val.select(range(max_val))

    print(f"Train samples: {len(ds_train)}")
    print(f"Val samples: {len(ds_val)}")

    use_learned = cfg["model"].get("learned_landmarks", True)

    seq_len_start = cfg["train"].get("seq_len_start", 512)
    seq_len_final = cfg["train"].get("seq_len_final", seq_len_start)

    def build_collator(max_length: int):
        if use_learned:
            return CollatorLocal(tokenizer, max_length)
        return CollatorLocalGlobal(
            tokenizer,
            max_length,
            cfg["train"]["global_every"],
            cfg["train"]["max_global"],
        )

    collate_train = build_collator(seq_len_final)
    collate_val = build_reduced_collator(tokenizer)

    # DataLoaders
    train_loader = DataLoader(
        ds_train,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=collate_train,
        num_workers=cfg["data"].get("num_workers", 2),
        pin_memory=True,
    )

    # Validation with reduced batch_size to avoid OOM
    val_batch_size = max(1, cfg["train"]["batch_size"] // 2)
    print(f"Validation config: batch_size={val_batch_size} (train: {cfg['train']['batch_size']}), seq_len=512 (train: up to {seq_len_final})")

    val_loader = DataLoader(
        ds_val,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_val,
        num_workers=0,  # Simpler for validation
        pin_memory=False,
    )

    return tokenizer, train_loader, val_loader


def build_reduced_collator(tokenizer):
    """
    Create a robust validation collator that handles both raw text and pre-tokenized datasets.

    Args:
        tokenizer: Tokenizer instance

    Returns:
        Collator function
    """
    import torch

    def collate_val_reduced(examples):
        """
        Robust validation collator.
        - Truncates/pads to 512 (+1 for label shifting).
        - Auto-detects format: looks for 'input_ids' or text fields.
        """
        max_len_val = 512
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        def _stack_input_ids(ids_list):
            tens = []
            for ids in ids_list:
                if not torch.is_tensor(ids):
                    ids = torch.as_tensor(ids, dtype=torch.long)
                ids = ids.view(-1)
                # truncate/pad to max_len_val+1
                if ids.numel() >= max_len_val + 1:
                    ids = ids[: max_len_val + 1]
                else:
                    pad = torch.full((max_len_val + 1 - ids.numel(),), pad_id, dtype=torch.long)
                    ids = torch.cat([ids, pad], dim=0)
                tens.append(ids)
            input_ids = torch.stack(tens, dim=0)  # (B, L+1)

            # Create input_ids and labels BEFORE masking
            input_ids_final = input_ids[:, :-1]  # (B, L)
            labels = input_ids[:, 1:].clone()  # (B, L)

            # Mask padding tokens with -100
            pad_mask = (labels == pad_id)
            labels[pad_mask] = -100

            return {
                "input_ids": input_ids_final,
                "labels": labels,
                "cache_global_ids": None,
            }

        # 1) Pre-tokenized case
        ex0 = examples[0]
        if isinstance(ex0, dict) and "input_ids" in ex0:
            return _stack_input_ids([ex["input_ids"] for ex in examples])

        # 2) Raw text case
        text_key = None
        if isinstance(ex0, dict):
            for k in ("text", "content", "document", "raw", "prompt"):
                if k in ex0 and isinstance(ex0[k], str):
                    text_key = k
                    break

        if text_key is None and isinstance(ex0, str):
            texts = list(examples)
        elif text_key is not None:
            texts = [ex[text_key] for ex in examples]
        else:
            if isinstance(ex0, dict):
                candidates = [k for k, v in ex0.items() if isinstance(v, str)]
                if candidates:
                    texts = [ex[candidates[0]] for ex in examples]
                else:
                    raise KeyError(f"Unable to find text field. Keys: {list(ex0.keys())}")
            else:
                raise KeyError(f"Unsupported example format: {type(ex0).__name__}")

        # Tokenize
        encoded = tokenizer(
            texts,
            max_length=max_len_val + 1,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]

        # Create input_ids and labels
        input_ids_final = input_ids[:, :-1]
        labels = input_ids[:, 1:].clone()
        pad_mask = (labels == pad_id)
        labels[pad_mask] = -100

        return {
            "input_ids": input_ids_final,
            "labels": labels,
            "cache_global_ids": None,
        }

    return collate_val_reduced
