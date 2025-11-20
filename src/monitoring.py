"""Monitoring utilities for SLGA training.

Provides synthetic evaluation routines and analytical helpers to assess
landmark selection quality and long-range dependency handling.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import torch


def _generate_long_dependency_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    dependency_gap: int,
    pairs_per_sample: int,
    device: torch.device,
) -> tuple[torch.Tensor, List[torch.Tensor]]:
    """Create synthetic sequences with long-range copy dependencies.

    Each sample contains random tokens plus ``pairs_per_sample`` key/target pairs
    separated by ``dependency_gap`` tokens. The key and target share the same
    token id, forcing the model to rely on global context to link them.
    """

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
    key_positions: List[torch.Tensor] = []

    max_start = max(1, seq_len - dependency_gap - 1)
    for b in range(batch_size):
        keys: List[int] = []
        for _ in range(pairs_per_sample):
            start_pos = torch.randint(0, max_start, (1,), device=device).item()
            key_pos = start_pos
            target_pos = min(seq_len - 1, key_pos + dependency_gap)

            token_id = torch.randint(0, vocab_size, (1,), device=device).item()
            tokens[b, key_pos] = token_id
            tokens[b, target_pos] = token_id
            keys.append(key_pos)

        if keys:
            key_positions.append(torch.tensor(keys, dtype=torch.long, device=device))
        else:
            key_positions.append(torch.empty(0, dtype=torch.long, device=device))

    return tokens, key_positions


def compute_long_dependency_recall(
    model: torch.nn.Module,
    device: torch.device,
    cfg: Dict,
    *,
    k_values: Sequence[int] = (8, 16, 24),
    batch_size: int = 4,
    num_batches: int = 2,
    pairs_per_sample: int | None = None,
    dependency_gap: int | None = None,
    seq_len: int | None = None,
    global_weight: float = 1.0,
) -> Dict[str, float]:
    """Estimate Recall@k of landmark selection on synthetic long dependencies.

    Args:
        model: SLGA model (already on ``device``).
        device: Target device for synthetic batches.
        cfg: Full config dictionary (expects ``cfg['model']`` keys).
        k_values: Sequence of k values to evaluate.
        batch_size: Synthetic batch size per evaluation pass.
        num_batches: Number of batches to average.
        pairs_per_sample: Number of long-range pairs per sample (defaults to ``global_k``).
        dependency_gap: Distance between key and target (defaults to ``2 * local_window``).
        seq_len: Synthetic sequence length (defaults to ``max_seq_len``).

    Returns:
        Dictionary mapping ``recall_at_{k}`` to averaged recall percentages.
    """

    if not hasattr(model, "forward"):
        raise ValueError("Model must implement forward for evaluation")

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    vocab_size = model_cfg.get("vocab_size", 50257)
    max_seq_len = seq_len or model_cfg.get("max_seq_len", 2048)
    local_window = model_cfg.get("local_window", 128)
    global_k = model_cfg.get("global_k", 24)

    # Memory optimization: Cap sequence length for monitoring to prevent OOM
    # Use config setting if available, otherwise default to 2K (conservative for large vocab)
    config_max_len = train_cfg.get("monitor_max_seq_len", 2048)
    max_monitoring_len = min(max_seq_len, config_max_len)
    if max_seq_len > max_monitoring_len:
        print(f"⚠️  Capping monitoring seq_len from {max_seq_len} to {max_monitoring_len} to prevent OOM")

    gap = dependency_gap or max(local_window * 2, local_window + 32)
    pairs = pairs_per_sample or max(1, global_k)

    k_list = sorted(set(int(max(1, k)) for k in k_values))
    recall_hits = {k: 0 for k in k_list}
    total_pairs = 0

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for _ in range(num_batches):
            batch_tokens, key_positions = _generate_long_dependency_batch(
                batch_size=batch_size,
                seq_len=max_monitoring_len,  # Use capped length
                vocab_size=vocab_size,
                dependency_gap=gap,
                pairs_per_sample=pairs,
                device=device,
            )

            # Additional memory optimization: clear cache before forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logits, aux = model(batch_tokens, return_aux=True, global_weight=global_weight)

            # Immediately move to CPU and clear GPU memory
            del logits
            landmark_scores = aux.get("landmark_scores")

            if landmark_scores is None:
                # Clear auxiliary data to free memory
                del aux
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            landmark_scores_cpu = landmark_scores.detach().cpu()
            key_positions_cpu = [kp.detach().cpu() for kp in key_positions]

            # Free GPU memory immediately after moving to CPU
            del landmark_scores, aux, batch_tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for b in range(landmark_scores_cpu.size(0)):
                keys = key_positions_cpu[b]
                if keys.numel() == 0:
                    continue

                total_pairs += keys.numel()
                scores_b = landmark_scores_cpu[b]

                for k in k_list:
                    topk_indices = torch.topk(scores_b, k=min(k, scores_b.numel()), dim=-1).indices
                    hits = sum(1 for key in keys.tolist() if (topk_indices == key).any())
                    recall_hits[k] += hits

            # Clear CPU tensors after processing
            del landmark_scores_cpu, key_positions_cpu

    if was_training:
        model.train()

    recalls = {}
    for k in k_list:
        if total_pairs == 0:
            recalls[f"recall_at_{k}"] = 0.0
        else:
            recalls[f"recall_at_{k}"] = recall_hits[k] / total_pairs

    return recalls
