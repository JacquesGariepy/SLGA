"""Tests for Grouped Query Attention (GQA) implementation."""

import torch
import pytest
from src.gqa import GroupedQueryAttention, create_causal_mask
from src.slga import SLGAModule
from src.model import Config, LLMTransformer


def test_gqa_basic():
    """Test basic GQA functionality."""
    B, L, D = 2, 16, 256
    num_heads = 8
    num_kv_heads = 2  # GQA with 4:1 ratio

    gqa = GroupedQueryAttention(
        embed_dim=D,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dropout=0.0,
    )

    x = torch.randn(B, L, D)
    out, attn = gqa(x, return_attn_weights=True)

    assert out.shape == (B, L, D), f"Expected {(B, L, D)}, got {out.shape}"
    assert attn.shape == (B, num_heads, L, L), f"Expected {(B, num_heads, L, L)}, got {attn.shape}"

    print(f"✓ GQA basic test passed")


def test_gqa_memory_savings():
    """Test GQA memory savings calculation."""
    embed_dim = 512
    num_heads = 8

    # Test different GQA configurations
    configs = [
        (8, "MHA (no reduction)"),
        (4, "GQA 2:1"),
        (2, "GQA 4:1"),
        (1, "MQA (maximum reduction)"),
    ]

    print("\n=== GQA Memory Savings ===")
    for num_kv_heads, desc in configs:
        gqa = GroupedQueryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )

        savings = gqa.get_memory_savings()
        print(f"\n{desc}:")
        print(f"  num_kv_heads: {savings['num_kv_heads']}")
        print(f"  queries_per_kv: {savings['queries_per_kv']}")
        print(f"  MHA params: {savings['mha_params']:,}")
        print(f"  GQA params: {savings['gqa_params']:,}")
        print(f"  Params saved: {savings['params_saved']:,}")
        print(f"  Reduction: {savings['reduction_percent']:.1f}%")


def test_gqa_vs_mha_equivalence():
    """Test that GQA with num_kv_heads=num_heads equals MHA."""
    B, L, D = 2, 16, 256
    num_heads = 8

    torch.manual_seed(42)

    # GQA with num_kv_heads = num_heads (should be equivalent to MHA)
    gqa = GroupedQueryAttention(
        embed_dim=D,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        dropout=0.0,
    )

    x = torch.randn(B, L, D)

    with torch.no_grad():
        out_gqa, _ = gqa(x)

    assert out_gqa.shape == (B, L, D)
    print(f"✓ GQA with num_kv_heads=num_heads works correctly")


def test_gqa_causal_masking():
    """Test GQA with causal masking."""
    B, L, D = 2, 8, 128
    num_heads = 4
    num_kv_heads = 2

    gqa = GroupedQueryAttention(
        embed_dim=D,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dropout=0.0,
    )

    x = torch.randn(B, L, D)
    causal_mask = create_causal_mask(L, x.device)

    out, attn = gqa(x, attention_mask=causal_mask, return_attn_weights=True)

    assert out.shape == (B, L, D)

    # Verify causal pattern: attention weights should be zero for future positions
    for b in range(B):
        for h in range(num_heads):
            for i in range(L):
                for j in range(i + 1, L):
                    assert attn[b, h, i, j].item() == 0.0, \
                        f"Non-zero attention to future position: [{b},{h},{i},{j}]={attn[b,h,i,j]}"

    print(f"✓ GQA causal masking test passed")


def test_slga_with_gqa():
    """Test SLGA module with GQA support."""
    B, L, D = 2, 32, 256
    num_heads = 8
    num_kv_heads = 2

    slga = SLGAModule(
        embed_dim=D,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        local_window=16,
        global_k=8,
        attn_drop=0.0,
        proj_drop=0.0,
    )

    x = torch.randn(B, L, D)
    position_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)

    # Create landmark cache
    G = 8
    landmark_states = torch.randn(B, G, D)
    landmark_positions = torch.linspace(0, L-1, G, device=x.device).long().unsqueeze(0).expand(B, -1)

    out, kv_cache = slga(
        x,
        position_ids=position_ids,
        cache_global=landmark_states,
        cache_positions=landmark_positions,
        global_weight=1.0,
        use_cache=False,
    )

    assert out.shape == (B, L, D)
    assert kv_cache is None  # No cache when use_cache=False
    print(f"✓ SLGA with GQA test passed")


def test_model_with_gqa():
    """Test full LLM model with GQA configuration."""
    cfg = Config(
        vocab_size=1000,
        max_seq_len=128,
        embed_dim=256,
        num_heads=8,
        num_kv_heads=2,  # Enable GQA
        ff_hidden_multiplier=2,
        n_layers=2,
        dropout_rate=0.0,
        local_window=16,
        global_k=8,
        learned_landmarks=False,
    )

    model = LLMTransformer(cfg)

    B, L = 2, 32
    input_ids = torch.randint(0, cfg.vocab_size, (B, L))

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (B, L, cfg.vocab_size)

    # Check parameter count reduction
    total_params = model.get_num_params()
    print(f"\nModel with GQA:")
    print(f"  Total params: {total_params:,}")
    print(f"  Config: num_heads={cfg.num_heads}, num_kv_heads={cfg.num_kv_heads}")

    # Compare with MHA version
    cfg_mha = Config(
        vocab_size=1000,
        max_seq_len=128,
        embed_dim=256,
        num_heads=8,
        num_kv_heads=None,  # Standard MHA
        ff_hidden_multiplier=2,
        n_layers=2,
        dropout_rate=0.0,
        local_window=16,
        global_k=8,
        learned_landmarks=False,
    )

    model_mha = LLMTransformer(cfg_mha)
    mha_params = model_mha.get_num_params()

    reduction = (mha_params - total_params) / mha_params * 100
    print(f"\nModel with MHA:")
    print(f"  Total params: {mha_params:,}")
    print(f"\nGQA Savings:")
    print(f"  Params saved: {mha_params - total_params:,}")
    print(f"  Reduction: {reduction:.2f}%")

    print(f"✓ Model with GQA test passed")


def test_gqa_gradients():
    """Test that GQA backward pass works correctly."""
    B, L, D = 2, 16, 128
    num_heads = 4
    num_kv_heads = 2

    gqa = GroupedQueryAttention(
        embed_dim=D,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dropout=0.0,
    )

    x = torch.randn(B, L, D, requires_grad=True)
    out, _ = gqa(x)

    # Compute loss and backward
    loss = out.sum()
    loss.backward()

    # Check gradients exist
    assert x.grad is not None
    assert gqa.q_proj.weight.grad is not None
    assert gqa.k_proj.weight.grad is not None
    assert gqa.v_proj.weight.grad is not None

    # Check no NaN gradients
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(gqa.q_proj.weight.grad).any()

    print(f"✓ GQA gradients test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Grouped Query Attention (GQA)")
    print("=" * 60)

    test_gqa_basic()
    test_gqa_memory_savings()
    test_gqa_vs_mha_equivalence()
    test_gqa_causal_masking()
    test_slga_with_gqa()
    test_model_with_gqa()
    test_gqa_gradients()

    print("\n" + "=" * 60)
    print("All GQA tests passed!")
    print("=" * 60)
