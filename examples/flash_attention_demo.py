"""
Flash Attention Demo for SLGA

This script demonstrates:
1. Basic Flash Attention usage
2. SLGA integration
3. Performance comparison
4. Memory usage comparison
"""

import torch
import time
from typing import Optional

# Import Flash Attention components
try:
    from src.flash_attention import (
        FLASH_ATTN_AVAILABLE,
        get_flash_attention_info,
        benchmark_flash_attention,
        flash_attention_forward,
        flash_attention_local_window,
    )
    FLASH_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"❌ Could not import flash_attention: {e}")
    FLASH_IMPORT_SUCCESS = False

# Import SLGA
try:
    from src.slga import SLGAModule
    SLGA_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"❌ Could not import SLGA: {e}")
    SLGA_IMPORT_SUCCESS = False


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_flash_attention_info():
    """Demonstrate Flash Attention info retrieval."""
    print_section("Flash Attention Information")

    if not FLASH_IMPORT_SUCCESS:
        print("❌ Flash Attention module not available")
        return

    info = get_flash_attention_info()

    print(f"\n{'Available:':<20} {info['available']}")
    print(f"{'Version:':<20} {info.get('version', 'N/A')}")

    if info['available']:
        print(f"\n{'Feature Support:':}")
        features = info.get('features', {})
        for feature, supported in features.items():
            status = "✓" if supported else "✗"
            print(f"  {status} {feature}")

        if 'cuda_capability' in info:
            print(f"\n{'CUDA Capability:':<20} {info['cuda_capability']}")
            print(f"{'Optimal Hardware:':<20} {info.get('optimal_hardware', False)}")
    else:
        print("\n⚠️  Flash Attention not installed")
        print("   Install with: pip install flash-attn --no-build-isolation")


def demo_basic_flash_attention():
    """Demonstrate basic Flash Attention usage."""
    print_section("Basic Flash Attention Usage")

    if not FLASH_IMPORT_SUCCESS or not FLASH_ATTN_AVAILABLE:
        print("❌ Flash Attention not available")
        return

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    # Configuration
    batch_size = 2
    seq_len = 1024
    num_heads = 8
    head_dim = 64

    device = 'cuda'
    dtype = torch.bfloat16

    print(f"\nConfiguration:")
    print(f"  Batch size:  {batch_size}")
    print(f"  Seq length:  {seq_len}")
    print(f"  Num heads:   {num_heads}")
    print(f"  Head dim:    {head_dim}")
    print(f"  Device:      {device}")
    print(f"  Dtype:       {dtype}")

    # Create random tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    # Full attention
    print("\n1. Full causal attention...")
    output = flash_attention_forward(q, k, v, causal=True)
    print(f"   Output shape: {output.shape}")
    print(f"   ✓ Success")

    # Local windowed attention
    print("\n2. Local windowed attention (window=128)...")
    output = flash_attention_local_window(q, k, v, window_size=128, causal=True)
    print(f"   Output shape: {output.shape}")
    print(f"   ✓ Success")


def demo_slga_integration():
    """Demonstrate SLGA integration with Flash Attention."""
    print_section("SLGA Integration")

    if not SLGA_IMPORT_SUCCESS:
        print("❌ SLGA not available")
        return

    # Configuration
    embed_dim = 512
    num_heads = 8
    local_window = 128
    batch_size = 2
    seq_len = 1024

    # Create SLGA with Flash Attention
    print("\n1. Creating SLGA module with Flash Attention...")
    slga_flash = SLGAModule(
        embed_dim=embed_dim,
        num_heads=num_heads,
        local_window=local_window,
        use_flash_attn=True,
    )

    if FLASH_ATTN_AVAILABLE:
        status = "✓ Using Flash Attention"
    else:
        status = "⚠️  Using PyTorch fallback"
    print(f"   {status}")
    print(f"   Flash enabled: {slga_flash.use_flash_attn}")

    # Forward pass
    if torch.cuda.is_available():
        print("\n2. Running forward pass...")
        device = 'cuda'
        dtype = torch.bfloat16

        slga_flash = slga_flash.to(device).to(dtype)

        x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        output, _ = slga_flash(x, position_ids)

        print(f"   Input shape:  {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   ✓ Success")

        # Check output validity
        has_nan = torch.isnan(output).any()
        has_inf = torch.isinf(output).any()
        print(f"\n3. Output validation:")
        print(f"   Has NaN: {has_nan}")
        print(f"   Has Inf: {has_inf}")
        if not has_nan and not has_inf:
            print(f"   ✓ Output is valid")
    else:
        print("\n⚠️  CUDA not available, skipping forward pass")


def demo_performance_comparison():
    """Demonstrate performance comparison."""
    print_section("Performance Comparison")

    if not FLASH_IMPORT_SUCCESS or not FLASH_ATTN_AVAILABLE:
        print("❌ Flash Attention not available for benchmarking")
        return

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    print("\nRunning benchmark (this may take a minute)...")

    results = benchmark_flash_attention(
        batch_size=4,
        seq_len=2048,
        num_heads=8,
        head_dim=64,
        num_iterations=100,
    )

    print("\n" + "-" * 60)
    print("Benchmark Results")
    print("-" * 60)
    print(f"{'Configuration:':<20}")
    print(f"  {'Batch size:':<18} {results['batch_size']}")
    print(f"  {'Sequence length:':<18} {results['seq_len']}")
    print(f"  {'Num heads:':<18} {results['num_heads']}")
    print(f"  {'Head dim:':<18} {results['head_dim']}")

    print(f"\n{'Performance:':<20}")
    print(f"  {'PyTorch:':<18} {results['pytorch_time_ms']:.2f} ms/iter")

    if results['flash_time_ms']:
        print(f"  {'Flash Attention:':<18} {results['flash_time_ms']:.2f} ms/iter")
        print(f"\n  {'Speedup:':<18} {results['speedup']}")

        # Visual speedup bar
        speedup_val = float(results['speedup'].replace('x', ''))
        bar_length = int(speedup_val * 10)
        bar = "█" * bar_length
        print(f"\n  Speedup visualization: {bar} ({speedup_val:.2f}×)")
    else:
        print("  Flash Attention:   Not available")


def demo_slga_performance():
    """Demonstrate SLGA performance with and without Flash Attention."""
    print_section("SLGA Performance Comparison")

    if not SLGA_IMPORT_SUCCESS:
        print("❌ SLGA not available")
        return

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    # Configuration
    batch_size = 2
    seq_len = 2048
    embed_dim = 512
    num_heads = 8
    local_window = 128
    num_iterations = 20

    device = 'cuda'
    dtype = torch.bfloat16

    print(f"\nConfiguration:")
    print(f"  Batch size:    {batch_size}")
    print(f"  Seq length:    {seq_len}")
    print(f"  Embed dim:     {embed_dim}")
    print(f"  Num heads:     {num_heads}")
    print(f"  Local window:  {local_window}")
    print(f"  Iterations:    {num_iterations}")

    # Create inputs
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # SLGA without Flash
    print("\n1. Benchmarking PyTorch attention...")
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

    # Benchmark
    start = time.time()
    for _ in range(num_iterations):
        _ = slga_pytorch(x, position_ids)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_iterations

    print(f"   PyTorch time: {pytorch_time*1000:.2f} ms/iter")

    # SLGA with Flash (if available)
    if FLASH_ATTN_AVAILABLE:
        print("\n2. Benchmarking Flash Attention...")
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

        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            _ = slga_flash(x, position_ids)
        torch.cuda.synchronize()
        flash_time = (time.time() - start) / num_iterations

        print(f"   Flash time:   {flash_time*1000:.2f} ms/iter")

        # Calculate speedup
        speedup = pytorch_time / flash_time

        print(f"\n3. Results:")
        print(f"   Speedup: {speedup:.2f}×")

        if speedup > 1.5:
            print(f"   ✓ Flash Attention provides significant speedup!")
        elif speedup > 1.0:
            print(f"   ✓ Flash Attention is faster")
        else:
            print(f"   ⚠️  Flash Attention not providing expected speedup")
    else:
        print("\n⚠️  Flash Attention not available, skipping comparison")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  SLGA Flash Attention Demo")
    print("=" * 60)

    # Check imports
    if not FLASH_IMPORT_SUCCESS:
        print("\n❌ Flash Attention module not available")
        print("   Install the flash_attention module first")
        return

    # Run demos
    demo_flash_attention_info()
    demo_basic_flash_attention()
    demo_slga_integration()
    demo_performance_comparison()
    demo_slga_performance()

    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
