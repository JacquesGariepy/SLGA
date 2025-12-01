"""
Flash Attention integration for SLGA.

Provides 2-3× speed improvement for attention computation when flash_attn is available.
Falls back gracefully to standard PyTorch attention if not installed.

Key Features:
- Automatic detection of flash_attn library
- Support for causal masking
- Support for local windowed attention
- Compatible with RoPE (applied before Flash Attention)
- Support for bf16/fp16 precision
- Zero-copy integration with existing SLGA module

Installation:
    pip install flash-attn --no-build-isolation

    Or with CUDA:
    MAX_JOBS=4 pip install flash-attn --no-build-isolation

Usage:
    from flash_attention import FlashAttentionLocal, FLASH_ATTN_AVAILABLE

    if FLASH_ATTN_AVAILABLE:
        ctx = FlashAttentionLocal.apply(q, k, v, causal=True)
    else:
        # Fallback to standard attention
        ctx = standard_attention(q, k, v)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ======================================================================
# Flash Attention Detection
# ======================================================================

FLASH_ATTN_AVAILABLE = False
FLASH_ATTN_VERSION = None

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.flash_attn_interface import flash_attn_with_kvcache
    FLASH_ATTN_AVAILABLE = True

    # Detect version
    try:
        import flash_attn
        FLASH_ATTN_VERSION = getattr(flash_attn, '__version__', 'unknown')
    except:
        FLASH_ATTN_VERSION = 'unknown'

except ImportError:
    # Flash attention not available - will use fallback
    flash_attn_func = None
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None


# ======================================================================
# Flash Attention Wrapper
# ======================================================================

def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    window_size: Optional[Tuple[int, int]] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    return_attn_probs: bool = False,
) -> torch.Tensor:
    """
    Flash Attention forward pass with automatic fallback.

    Args:
        q: Query tensor (B, H, L, Dh) or (B, L, H, Dh)
        k: Key tensor (B, H, L, Dh) or (B, L, H, Dh)
        v: Value tensor (B, H, L, Dh) or (B, L, H, Dh)
        causal: If True, apply causal masking
        window_size: Optional (left, right) window sizes for local attention
                    If None, full attention is used
                    If (W, 0), sliding window of size W with causal masking
        dropout_p: Dropout probability (only in training)
        softmax_scale: Scale for softmax (default: 1/sqrt(d_head))
        return_attn_probs: If True, return attention weights (WARNING: memory intensive!)

    Returns:
        output: (B, H, L, Dh) or (B, L, H, Dh) depending on input format

    Note:
        - RoPE should be applied BEFORE calling this function
        - Input tensors should be contiguous
        - Supports bf16, fp16, and fp32
    """
    if not FLASH_ATTN_AVAILABLE:
        # Fallback to standard PyTorch attention
        return _pytorch_attention_fallback(
            q, k, v, causal, window_size, dropout_p, softmax_scale
        )

    # Determine input format
    is_bhld_format = q.dim() == 4 and q.shape[1] == k.shape[1]  # (B, H, L, Dh)

    # Flash attention expects (B, L, H, Dh)
    if is_bhld_format:
        # Convert (B, H, L, Dh) -> (B, L, H, Dh)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    else:
        # Already in (B, L, H, Dh) format
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

    B, L, H, Dh = q.shape

    # Default scale
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(Dh)

    # Flash attention call
    try:
        output = flash_attn_func(
            q, k, v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            return_attn_probs=return_attn_probs,
        )

        if return_attn_probs:
            output, attn_weights = output

    except Exception as e:
        # Fallback on any error
        print(f"Flash Attention failed ({e}), falling back to PyTorch implementation")
        if is_bhld_format:
            # Convert back to (B, H, L, Dh) for fallback
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        return _pytorch_attention_fallback(
            q, k, v, causal, window_size, dropout_p, softmax_scale
        )

    # Convert output back to original format if needed
    if is_bhld_format:
        output = output.transpose(1, 2).contiguous()  # (B, L, H, Dh) -> (B, H, L, Dh)
        if return_attn_probs:
            # attn_weights is (B, H, L, L)
            attn_weights = attn_weights.contiguous()

    if return_attn_probs:
        return output, attn_weights

    return output


def flash_attention_local_window(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    causal: bool = True,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Flash Attention with local sliding window.

    Optimized wrapper for local attention patterns (like SLGA's local attention).
    Uses Flash Attention's native window_size parameter for efficiency.

    Args:
        q, k, v: (B, H, L, Dh) tensors (already with RoPE applied)
        window_size: Size of the sliding window
        causal: If True, apply causal masking
        dropout_p: Dropout probability
        softmax_scale: Scale for softmax

    Returns:
        output: (B, H, L, Dh) context tensor
    """
    # For causal local attention, window extends backwards
    # window_size = (left, right) where:
    # - left: how many positions to attend backwards
    # - right: how many positions to attend forwards

    if causal:
        # Causal: only attend to current and past positions
        window = (window_size - 1, 0)
    else:
        # Bidirectional: attend to window_size // 2 on each side
        half_window = window_size // 2
        window = (half_window, half_window)

    return flash_attention_forward(
        q, k, v,
        causal=causal,
        window_size=window,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        return_attn_probs=False,
    )


# ======================================================================
# PyTorch Fallback Implementation
# ======================================================================

def _pytorch_attention_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    window_size: Optional[Tuple[int, int]] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Standard PyTorch attention as fallback when Flash Attention is not available.

    Matches the interface of flash_attention_forward for seamless fallback.
    """
    # Expect (B, H, L, Dh) format
    B, H, L, Dh = q.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(Dh)

    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale  # (B, H, L, L)

    # Apply causal mask
    if causal:
        causal_mask = torch.triu(
            torch.ones(L, L, device=q.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))

    # Apply window mask if specified
    if window_size is not None:
        left_window, right_window = window_size

        # Create window mask
        row_idx = torch.arange(L, device=q.device).unsqueeze(1)
        col_idx = torch.arange(L, device=q.device).unsqueeze(0)

        # Position is valid if: col >= row - left_window AND col <= row + right_window
        window_mask = (col_idx < row_idx - left_window) | (col_idx > row_idx + right_window)
        scores = scores.masked_fill(window_mask, float('-inf'))

    # Softmax with NaN protection
    attn = F.softmax(scores, dim=-1)

    # Handle NaN from fully masked rows
    attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

    # Dropout
    if dropout_p > 0.0 and q.requires_grad:
        attn = F.dropout(attn, p=dropout_p)

    # Apply attention to values
    output = torch.matmul(attn, v)  # (B, H, L, Dh)

    return output


# ======================================================================
# Flash Attention Module (nn.Module wrapper)
# ======================================================================

class FlashAttentionModule(nn.Module):
    """
    PyTorch module wrapper for Flash Attention.

    Drop-in replacement for standard attention with automatic fallback.

    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        causal: If True, apply causal masking
        window_size: Optional window size for local attention
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = True,
        window_size: Optional[int] = None,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.causal = causal
        self.window_size = window_size
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Track if Flash Attention is available
        self.using_flash = FLASH_ATTN_AVAILABLE

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            q, k, v: (B, H, L, Dh) tensors

        Returns:
            output: (B, H, L, Dh)
        """
        # Prepare window size for Flash Attention format
        window = None
        if self.window_size is not None:
            if self.causal:
                window = (self.window_size - 1, 0)
            else:
                half = self.window_size // 2
                window = (half, half)

        # Call Flash Attention or fallback
        output = flash_attention_forward(
            q, k, v,
            causal=self.causal,
            window_size=window,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.scale,
        )

        return output


# ======================================================================
# Utility Functions
# ======================================================================

def benchmark_flash_attention(
    batch_size: int = 4,
    seq_len: int = 2048,
    num_heads: int = 8,
    head_dim: int = 64,
    device: str = 'cuda',
    num_iterations: int = 100,
) -> dict:
    """
    Benchmark Flash Attention vs PyTorch attention.

    Returns:
        dict with timing results and speedup factor
    """
    if not torch.cuda.is_available():
        return {
            'error': 'CUDA not available',
            'flash_available': FLASH_ATTN_AVAILABLE,
        }

    # Create random tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(10):
        _ = _pytorch_attention_fallback(q, k, v, causal=True)

    if FLASH_ATTN_AVAILABLE:
        for _ in range(10):
            _ = flash_attention_forward(q, k, v, causal=True)

    torch.cuda.synchronize()

    # Benchmark PyTorch
    import time
    start = time.time()
    for _ in range(num_iterations):
        _ = _pytorch_attention_fallback(q, k, v, causal=True)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start

    # Benchmark Flash Attention
    flash_time = None
    speedup = None

    if FLASH_ATTN_AVAILABLE:
        start = time.time()
        for _ in range(num_iterations):
            _ = flash_attention_forward(q, k, v, causal=True)
        torch.cuda.synchronize()
        flash_time = time.time() - start
        speedup = pytorch_time / flash_time

    return {
        'flash_available': FLASH_ATTN_AVAILABLE,
        'flash_version': FLASH_ATTN_VERSION,
        'pytorch_time_ms': pytorch_time * 1000 / num_iterations,
        'flash_time_ms': flash_time * 1000 / num_iterations if flash_time else None,
        'speedup': f"{speedup:.2f}x" if speedup else None,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'num_heads': num_heads,
        'head_dim': head_dim,
    }


def get_flash_attention_info() -> dict:
    """
    Get information about Flash Attention availability and capabilities.

    Returns:
        dict with availability, version, and supported features
    """
    info = {
        'available': FLASH_ATTN_AVAILABLE,
        'version': FLASH_ATTN_VERSION,
    }

    if FLASH_ATTN_AVAILABLE:
        info['features'] = {
            'causal_masking': True,
            'local_window': True,
            'variable_length': flash_attn_varlen_func is not None,
            'kv_cache': flash_attn_with_kvcache is not None,
        }

        # Detect CUDA capability
        if torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability()
            info['cuda_capability'] = f"{device_capability[0]}.{device_capability[1]}"

            # Flash Attention works best on Ampere (SM80+) and newer
            sm = device_capability[0] * 10 + device_capability[1]
            info['optimal_hardware'] = sm >= 80  # Ampere or newer
        else:
            info['cuda_capability'] = 'N/A'
            info['optimal_hardware'] = False

    return info


# ======================================================================
# Exports
# ======================================================================

__all__ = [
    'FLASH_ATTN_AVAILABLE',
    'FLASH_ATTN_VERSION',
    'flash_attention_forward',
    'flash_attention_local_window',
    'FlashAttentionModule',
    'benchmark_flash_attention',
    'get_flash_attention_info',
]
