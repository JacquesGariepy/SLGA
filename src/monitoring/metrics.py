"""Monitoring utilities for SLGA training.

Provides synthetic evaluation routines and analytical helpers to assess
landmark selection quality and long-range dependency handling.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor


def _generate_long_dependency_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    dependency_gap: int,
    pairs_per_sample: int,
    device: torch.device,
) -> Tuple[Tensor, List[Tensor]]:
    """Create synthetic sequences with long-range copy dependencies.

    Each sample contains random tokens plus ``pairs_per_sample`` key/target pairs
    separated by ``dependency_gap`` tokens. The key and target share the same
    token id, forcing the model to rely on global context to link them.
    """

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
    key_positions: List[torch.Tensor] = []

    max_start: int = max(1, seq_len - dependency_gap - 1)
    for b in range(batch_size):
        keys: List[int] = []
        for _ in range(pairs_per_sample):
            start_pos: int = int(torch.randint(0, max_start, (1,), device=device).item())
            key_pos: int = start_pos
            target_pos: int = min(seq_len - 1, key_pos + dependency_gap)

            token_id: int = int(torch.randint(0, vocab_size, (1,), device=device).item())
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
    cfg: Dict[str, Dict[str, int]],
    *,
    k_values: Sequence[int] = (8, 16, 24),
    batch_size: int = 4,
    num_batches: int = 2,
    pairs_per_sample: Optional[int] = None,
    dependency_gap: Optional[int] = None,
    seq_len: Optional[int] = None,
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

    model_cfg: Dict[str, int] = cfg.get("model", {})
    vocab_size: int = model_cfg.get("vocab_size", 50257)
    max_seq_len: int = seq_len or model_cfg.get("max_seq_len", 2048)
    local_window: int = model_cfg.get("local_window", 128)
    global_k: int = model_cfg.get("global_k", 24)

    gap: int = dependency_gap or max(local_window * 2, local_window + 32)
    pairs: int = pairs_per_sample or max(1, global_k)

    k_list: List[int] = sorted(set(int(max(1, k)) for k in k_values))
    recall_hits: Dict[int, int] = dict.fromkeys(k_list, 0)
    total_pairs: int = 0

    was_training: bool = model.training
    model.eval()

    with torch.no_grad():
        for _ in range(num_batches):
            batch_tokens: Tensor
            key_positions: List[Tensor]
            batch_tokens, key_positions = _generate_long_dependency_batch(
                batch_size=batch_size,
                seq_len=max_seq_len,
                vocab_size=vocab_size,
                dependency_gap=gap,
                pairs_per_sample=pairs,
                device=device,
            )

            logits: Tensor
            aux: Dict[str, Tensor]
            logits, aux = model(batch_tokens, return_aux=True, global_weight=global_weight)
            landmark_scores: Optional[Tensor] = aux.get("landmark_scores")

            if landmark_scores is None:
                continue

            landmark_scores_cpu: Tensor = landmark_scores.detach().cpu()
            key_positions_cpu: List[Tensor] = [kp.detach().cpu() for kp in key_positions]

            for b in range(landmark_scores_cpu.size(0)):
                keys: Tensor = key_positions_cpu[b]
                if keys.numel() == 0:
                    continue

                total_pairs += keys.numel()
                scores_b: Tensor = landmark_scores_cpu[b]

                for k in k_list:
                    topk_indices: Tensor = torch.topk(
                        scores_b, k=min(k, scores_b.numel()), dim=-1
                    ).indices
                    hits: int = sum(1 for key in keys.tolist() if (topk_indices == key).any())
                    recall_hits[k] += hits

    if was_training:
        model.train()

    recalls: Dict[str, float] = {}
    for k in k_list:
        if total_pairs == 0:
            recalls[f"recall_at_{k}"] = 0.0
        else:
            recalls[f"recall_at_{k}"] = recall_hits[k] / total_pairs

    return recalls
