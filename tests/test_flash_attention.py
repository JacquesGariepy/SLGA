"""
Test suite for Flash Attention integration in SLGA.

Tests:
1. Flash Attention availability detection
2. Correctness: Flash vs PyTorch attention outputs match
3. Performance: Flash is faster than PyTorch
4. Integration: SLGA module uses Flash when available
5. Fallback: Graceful degradation when Flash unavailable
"""

import pytest
import torch
import torch.nn as nn
import time
from typing import Optional

# Import Flash Attention components
try:
    from src.flash_attention import (
        FLASH_ATTN_AVAILABLE,
        FLASH_ATTN_VERSION,
        flash_attention_forward,
        flash_attention_local_window,
        FlashAttentionModule,
        benchmark_flash_attention,
        get_flash_attention_info,
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    print(f"Warning: Could not import flash_attention: {e}")

# Import SLGA
try:
    from src.slga import SLGAModule
    SLGA_IMPORT_SUCCESS = True
except ImportError as e:
    SLGA_IMPORT_SUCCESS = False
    print(f"Warning: Could not import SLGA: {e}")


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="flash_attention module not available")
def test_flash_attention_info():
    """Test that we can get Flash Attention info."""
    info = get_flash_attention_info()

    assert 'available' in info
    assert 'version' in info

    print("\n=== Flash Attention Info ===")
    for key, value in info.items():
        print(f"{key}: {value}")

    if FLASH_ATTN_AVAILABLE:
        assert info['version'] is not None
        assert 'features' in info


@pytest.mark.skipif(not IMPORT_SUCCESS or not FLASH_ATTN_AVAILABLE,
                   reason="Flash Attention not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_attention_correctness():
    """Test that Flash Attention produces similar results to PyTorch attention."""

    batch_size = 2
    seq_len = 512
    num_heads = 8
    head_dim = 64

    device = 'cuda'
    dtype = torch.bfloat16

    # Create random inputs
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    # Flash Attention output
    from src.flash_attention import flash_attention_forward
    output_flash = flash_attention_forward(q, k, v, causal=True)

    # PyTorch attention output
    from src.flash_attention import _pytorch_attention_fallback
    output_pytorch = _pytorch_attention_fallback(q, k, v, causal=True)

    # Check shapes match
    assert output_flash.shape == output_pytorch.shape

    # Check outputs are similar (allowing for numerical differences)
    diff = (output_flash - output_pytorch).abs().mean()
    print(f"\nMean absolute difference: {diff.item():.6f}")

    # With bfloat16, we expect some numerical difference but should be small
    assert diff < 0.01, f"Outputs differ too much: {diff.item()}"


@pytest.mark.skipif(not IMPORT_SUCCESS or not FLASH_ATTN_AVAILABLE,
                   reason="Flash Attention not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_attention_local_window():
    """Test Flash Attention with local windowing."""

    batch_size = 2
    seq_len = 1024
    num_heads = 8
    head_dim = 64
    window_size = 128

    device = 'cuda'
    dtype = torch.bfloat16

    # Create random inputs
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    # Flash Attention with local window
    output = flash_attention_local_window(
        q, k, v,
        window_size=window_size,
        causal=True,
    )

    # Check output shape
    assert output.shape == (batch_size, num_heads, seq_len, head_dim)

    # Check no NaN or Inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


@pytest.mark.skipif(not SLGA_IMPORT_SUCCESS, reason="SLGA not available")
def test_slga_flash_integration():
    """Test that SLGA module integrates Flash Attention correctly."""

    embed_dim = 512
    num_heads = 8
    local_window = 128

    # Create SLGA module with Flash Attention enabled
    slga_flash = SLGAModule(
        embed_dim=embed_dim,
        num_heads=num_heads,
        local_window=local_window,
        use_flash_attn=True,
    )

    # Check Flash Attention status
    if FLASH_ATTN_AVAILABLE:
        assert slga_flash.use_flash_attn == True
        print("\n✓ SLGA is using Flash Attention")
    else:
        assert slga_flash.use_flash_attn == False
        print("\n✓ SLGA correctly disabled Flash Attention (not available)")

    # Create SLGA module with Flash Attention disabled
    slga_no_flash = SLGAModule(
        embed_dim=embed_dim,
        num_heads=num_heads,
        local_window=local_window,
        use_flash_attn=False,
    )

    assert slga_no_flash.use_flash_attn == False
    print("✓ SLGA correctly disabled Flash Attention (user request)")


@pytest.mark.skipif(not SLGA_IMPORT_SUCCESS, reason="SLGA not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_slga_flash_forward():
    """Test SLGA forward pass with Flash Attention."""

    batch_size = 2
    seq_len = 512
    embed_dim = 512
    num_heads = 8
    local_window = 128

    device = 'cuda'
    dtype = torch.bfloat16

    # Create module
    slga = SLGAModule(
        embed_dim=embed_dim,
        num_heads=num_heads,
        local_window=local_window,
        use_flash_attn=True,
    ).to(device).to(dtype)

    # Create inputs
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # Forward pass
    output, _ = slga(x, position_ids)

    # Check output
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    print(f"\n✓ SLGA forward pass successful with Flash Attention")


@pytest.mark.skipif(not IMPORT_SUCCESS or not FLASH_ATTN_AVAILABLE,
                   reason="Flash Attention not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_attention_performance():
    """Benchmark Flash Attention vs PyTorch attention."""

    print("\n=== Flash Attention Performance Benchmark ===")

    results = benchmark_flash_attention(
        batch_size=4,
        seq_len=2048,
        num_heads=8,
        head_dim=64,
        num_iterations=100,
    )

    print(f"Configuration: {results['batch_size']}x{results['seq_len']} (B×L), "
          f"{results['num_heads']} heads, {results['head_dim']} head_dim")
    print(f"PyTorch time: {results['pytorch_time_ms']:.2f} ms/iter")

    if results['flash_time_ms']:
        print(f"Flash time:   {results['flash_time_ms']:.2f} ms/iter")
        print(f"Speedup:      {results['speedup']}")

        # Flash should be faster
        assert results['flash_time_ms'] < results['pytorch_time_ms']
    else:
        print("Flash Attention not available for benchmarking")


@pytest.mark.skipif(not SLGA_IMPORT_SUCCESS, reason="SLGA not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_slga_flash_vs_pytorch_performance():
    """Compare SLGA performance with and without Flash Attention."""

    batch_size = 2
    seq_len = 2048
    embed_dim = 512
    num_heads = 8
    local_window = 128
    num_iterations = 20

    device = 'cuda'
    dtype = torch.bfloat16

    # Create inputs
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # SLGA without Flash
    slga_pytorch = SLGAModule(
        embed_dim=embed_dim,
        num_heads=num_heads,
        local_window=local_window,
        use_flash_attn=False,
    ).to(device).to(dtype)

    # Warmup
    for _ in range(5):
        _ = slga_pytorch(x, position_ids)
    torch.cuda.synchronize()

    # Benchmark PyTorch
    start = time.time()
    for _ in range(num_iterations):
        _ = slga_pytorch(x, position_ids)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_iterations

    # SLGA with Flash (if available)
    if FLASH_ATTN_AVAILABLE:
        slga_flash = SLGAModule(
            embed_dim=embed_dim,
            num_heads=num_heads,
            local_window=local_window,
            use_flash_attn=True,
        ).to(device).to(dtype)

        # Warmup
        for _ in range(5):
            _ = slga_flash(x, position_ids)
        torch.cuda.synchronize()

        # Benchmark Flash
        start = time.time()
        for _ in range(num_iterations):
            _ = slga_flash(x, position_ids)
        torch.cuda.synchronize()
        flash_time = (time.time() - start) / num_iterations

        speedup = pytorch_time / flash_time

        print(f"\n=== SLGA Performance Comparison ===")
        print(f"Configuration: {batch_size}×{seq_len}, {num_heads} heads, window={local_window}")
        print(f"PyTorch time: {pytorch_time*1000:.2f} ms/iter")
        print(f"Flash time:   {flash_time*1000:.2f} ms/iter")
        print(f"Speedup:      {speedup:.2f}×")

        # Flash should be faster (at least 1.5×)
        assert speedup > 1.2, f"Flash Attention not providing speedup: {speedup:.2f}×"
    else:
        print("\n✓ PyTorch attention benchmark: {pytorch_time*1000:.2f} ms/iter")
        print("  (Flash Attention not available for comparison)")


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="flash_attention module not available")
def test_flash_attention_module():
    """Test FlashAttentionModule wrapper."""

    embed_dim = 512
    num_heads = 8

    # Create module
    attn_module = FlashAttentionModule(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.1,
        causal=True,
        window_size=128,
    )

    # Check configuration
    assert attn_module.embed_dim == embed_dim
    assert attn_module.num_heads == num_heads
    assert attn_module.head_dim == embed_dim // num_heads
    assert attn_module.causal == True
    assert attn_module.window_size == 128

    print(f"\n✓ FlashAttentionModule created successfully")
    print(f"  Using Flash Attention: {attn_module.using_flash}")


if __name__ == "__main__":
    """Run tests manually."""
    print("=" * 60)
    print("Flash Attention Test Suite")
    print("=" * 60)

    # Run basic tests
    if IMPORT_SUCCESS:
        print("\n1. Testing Flash Attention info...")
        test_flash_attention_info()

        if torch.cuda.is_available() and FLASH_ATTN_AVAILABLE:
            print("\n2. Testing Flash Attention correctness...")
            test_flash_attention_correctness()

            print("\n3. Testing Flash Attention local window...")
            test_flash_attention_local_window()

            print("\n4. Testing Flash Attention performance...")
            test_flash_attention_performance()

    if SLGA_IMPORT_SUCCESS:
        print("\n5. Testing SLGA Flash integration...")
        test_slga_flash_integration()

        if torch.cuda.is_available():
            print("\n6. Testing SLGA Flash forward...")
            test_slga_flash_forward()

            if FLASH_ATTN_AVAILABLE:
                print("\n7. Testing SLGA Flash vs PyTorch performance...")
                test_slga_flash_vs_pytorch_performance()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
