"""
Minimal test to debug KV cache correctness issue.
"""

import torch
from src.model import Config, LLMTransformer


def test_single_token_generation():
    """Test that a single token generation matches with/without cache."""
    cfg = Config(
        vocab_size=100,
        max_seq_len=64,
        embed_dim=64,
        num_heads=4,
        ff_hidden_multiplier=2,
        n_layers=2,
        dropout_rate=0.0,  # NO dropout
        local_window=8,
        global_k=4,
        learned_landmarks=False,
    )

    model = LLMTransformer(cfg)
    model.eval()

    # Fix seed
    torch.manual_seed(42)
    prompt = torch.tensor([[10, 20, 30, 40, 50]])  # 5 tokens

    print("Prompt:", prompt)

    # === Method 1: Generate without cache (baseline) ===
    with torch.no_grad():
        torch.manual_seed(100)
        logits_baseline = model(prompt)  # (1, 5, 100)
        next_token_baseline = logits_baseline[:, -1, :].argmax(dim=-1)
        print(f"Baseline next token: {next_token_baseline.item()}")

    # === Method 2: Generate with cache ===
    from src.kv_cache import KVCache

    with torch.no_grad():
        # First, process prompt to populate cache
        torch.manual_seed(100)
        kv_cache = KVCache.create(
            batch_size=1,
            num_layers=2,
            num_heads=4,
            head_dim=16,
            max_seq_len=64,
            device=prompt.device,
            dtype=torch.float32,
        )

        # Forward pass to populate cache
        logits_cached = model._forward_with_cache(
            prompt, cache_global_ids=None, kv_cache=kv_cache
        )
        next_token_cached = logits_cached[:, -1, :].argmax(dim=-1)
        print(f"Cached next token: {next_token_cached.item()}")

    # Compare
    print(f"\nLogits match: {torch.allclose(logits_baseline, logits_cached, atol=1e-5)}")
    print(f"Tokens match: {next_token_baseline.item() == next_token_cached.item()}")

    if not torch.allclose(logits_baseline, logits_cached, atol=1e-5):
        diff = (logits_baseline - logits_cached).abs()
        print(f"Max diff: {diff.max().item():.6f}")
        print(f"Mean diff: {diff.mean().item():.6f}")


if __name__ == "__main__":
    test_single_token_generation()
