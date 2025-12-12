from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ============================================================
# Helpers
# ============================================================

def has_active_kv_cache(cfg, kv_cache: Optional[List[Optional[Dict[str, Any]]]]) -> bool:
    """
    Cache actif seulement si:
    - cfg.use_kv_cache True
    - kv_cache existe
    - toutes les couches ont lc != None
    - k et v existent ensemble et ne sont pas None
    """
    if not (cfg.use_kv_cache and kv_cache):
        return False
    for lc in kv_cache:
        if lc is None:
            return False
        k = lc.get("k", None)
        v = lc.get("v", None)
        if (k is None) != (v is None):
            return False
        if k is None:
            return False
    return True


def current_pos_offset(kv_cache: Optional[List[Optional[Dict[str, Any]]]], fallback: int) -> int:
    """
    Position absolue canonique: base_pos + longueur du cache (layer 0).
    Robuste aux structures [None] * n_layers.
    """
    if not kv_cache:
        return int(fallback)

    c0 = kv_cache[0]
    if c0 is None:
        return int(fallback)

    k0 = c0.get("k", None)
    if k0 is None:
        return int(fallback)

    base = int(c0.get("base_pos", 0))
    if base < 0:
        raise ValueError("KV cache base_pos invalid.")

    Lk = int(k0.size(2))
    if Lk < 0:
        raise ValueError("KV cache length invalid.")

    return base + Lk


# ============================================================
# RoPE (pairwise even/odd) + cache cos/sin
# ============================================================

def build_rope_cache_pairwise(
    seq_len: int,
    head_dim: int,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 2 == 0
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (L, half)
    cos = freqs.cos().to(dtype=dtype)
    sin = freqs.sin().to(dtype=dtype)
    return cos, sin


def apply_rope_pairwise_1dpos(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """
    x: (B,H,L,Hd), pos: (L,)
    cos/sin: (Lmax, Hd/2)
    """
    B, H, L, Hd = x.shape
    half = Hd // 2
    if pos.numel() != L:
        raise ValueError("RoPE pos must be length L.")
    if pos.min().item() < 0 or int(pos.max().item()) >= int(cos.size(0)):
        raise ValueError("RoPE index out of range (q/k).")

    c = cos.index_select(0, pos).view(1, 1, L, half)
    s = sin.index_select(0, pos).view(1, 1, L, half)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * c - x_odd * s
    out_odd = x_even * s + x_odd * c

    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


def apply_rope_pairwise_2dpos(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, pos_bg: torch.Tensor) -> torch.Tensor:
    """
    x: (B,H,G,Hd), pos_bg: (B,G)
    """
    if pos_bg.numel() == 0:
        return x
    if pos_bg.min().item() < 0 or int(pos_bg.max().item()) >= int(cos.size(0)):
        raise ValueError("RoPE index out of range (global).")

    B, H, G, Hd = x.shape
    half = Hd // 2
    c = cos[pos_bg].view(B, 1, G, half)
    s = sin[pos_bg].view(B, 1, G, half)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * c - x_odd * s
    out_odd = x_even * s + x_odd * c

    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


# ============================================================
# Norm / FFN
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(norm + self.eps)) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# ============================================================
# SLGA Attention (local window + optional global bank)
# - KV cache compatible decoding
# - global softmax/matmul skipped for bad rows
# - full KV invariants
# ============================================================

class SLGAAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int],
        local_window: int,
        attn_drop: float,
        proj_drop: float,
        causal: bool = True,
        gated_fusion: bool = True,
        require_odd_window_noncausal: bool = True,
        max_cache_len: Optional[int] = None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads) if num_kv_heads is not None else int(num_heads)
        assert self.num_heads % self.num_kv_heads == 0
        self.kv_repeat = self.num_heads // self.num_kv_heads

        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim % 2 == 0

        self.local_window = int(local_window)
        self.causal = bool(causal)
        self.gated_fusion = bool(gated_fusion)
        self.require_odd_window_noncausal = bool(require_odd_window_noncausal)
        self.max_cache_len = max_cache_len

        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)

        self.gk_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.gv_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.gated_fusion:
            self.gate_proj = nn.Linear(self.embed_dim, 1, bias=True)

        self.last_monitor: Dict[str, Any] = {}

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.kv_repeat == 1 else x.repeat_interleave(self.kv_repeat, dim=1)

    @staticmethod
    def _mask_min(dtype: torch.dtype) -> float:
        if dtype == torch.float16:
            return -65504.0
        return -1e9

    def _maybe_evict(self, kv_cache: Dict[str, Any]):
        if self.max_cache_len is None:
            return
        k = kv_cache.get("k", None)
        v = kv_cache.get("v", None)
        if k is None or v is None:
            return
        Lk = int(k.size(2))
        if Lk <= int(self.max_cache_len):
            return
        drop = Lk - int(self.max_cache_len)
        kv_cache["k"] = k[:, :, drop:, :].contiguous()
        kv_cache["v"] = v[:, :, drop:, :].contiguous()
        kv_cache["base_pos"] = int(kv_cache.get("base_pos", 0)) + int(drop)

    def _local_attend_queries_only(
        self,
        q: torch.Tensor,          # (B,H,Lq,Hd)
        k_full: torch.Tensor,     # (B,H,Lk,Hd)
        v_full: torch.Tensor,     # (B,H,Lk,Hd)
        q_pos_abs: torch.Tensor,  # (Lq,)
        cache_base_pos: int,      # absolute position of k_full[:, :, 0]
    ) -> torch.Tensor:
        B, H, Lq, Hd = q.shape
        Lk = int(k_full.size(2))
        W = max(1, int(self.local_window))
        scale = 1.0 / math.sqrt(Hd)

        if not self.causal and self.require_odd_window_noncausal and (W % 2 == 0):
            raise ValueError("Noncausal: local_window doit être impair.")

        if self.causal:
            rel = torch.arange(W - 1, -1, -1, device=q.device)
            idx_abs = q_pos_abs[:, None] - rel[None, :]
        else:
            left = (W - 1) // 2
            right = (W - 1) - left
            rel = torch.arange(-left, right + 1, device=q.device)
            idx_abs = q_pos_abs[:, None] + rel[None, :]

        idx = idx_abs - int(cache_base_pos)            # cache indices
        mask = (idx >= 0) & (idx < Lk)
        idx = idx.clamp(0, Lk - 1)

        idx_b = idx.view(1, 1, Lq, W).expand(B, H, -1, -1)
        k_win = torch.gather(k_full, 2, idx_b.unsqueeze(-1).expand(B, H, Lq, W, Hd))
        v_win = torch.gather(v_full, 2, idx_b.unsqueeze(-1).expand(B, H, Lq, W, Hd))

        scores = torch.einsum("bhld,bhlwd->bhlw", q, k_win) * scale
        scores = scores.masked_fill(~mask.view(1, 1, Lq, W), self._mask_min(scores.dtype))
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        return torch.einsum("bhlw,bhlwd->bhld", attn, v_win)

    def forward(
        self,
        x: torch.Tensor,                               # (B,Lq,D)
        cache_global: Optional[torch.Tensor],           # (B,G,D)
        global_indices: Optional[torch.Tensor],         # (B,G) abs positions
        global_valid_mask: Optional[torch.Tensor],      # (B,G) True=valid
        global_weight: float,
        kv_cache: Optional[Dict[str, Any]],             # {"k","v","base_pos"}
        pos_offset: int,                                # abs start pos for this chunk
        rope_cos: Optional[torch.Tensor],               # (Lmax, Hd/2)
        rope_sin: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, Lq, D = x.shape
        Hd = self.head_dim

        q = self.q_proj(x).view(B, Lq, self.num_heads, Hd).transpose(1, 2)          # (B,H,Lq,Hd)
        k = self.k_proj(x).view(B, Lq, self.num_kv_heads, Hd).transpose(1, 2)       # (B,Hkv,Lq,Hd)
        v = self.v_proj(x).view(B, Lq, self.num_kv_heads, Hd).transpose(1, 2)       # (B,Hkv,Lq,Hd)

        q_pos_abs = torch.arange(Lq, device=x.device, dtype=torch.long) + int(pos_offset)

        if rope_cos is not None and rope_sin is not None:
            q = apply_rope_pairwise_1dpos(q, rope_cos, rope_sin, q_pos_abs)
            k = apply_rope_pairwise_1dpos(k, rope_cos, rope_sin, q_pos_abs)

        cache_base_pos = 0
        if kv_cache is not None:
            cache_base_pos = int(kv_cache.get("base_pos", 0))
            k_prev = kv_cache.get("k", None)
            v_prev = kv_cache.get("v", None)

            if (k_prev is None) != (v_prev is None):
                raise ValueError("KV cache corrompu: k et v doivent être présents ensemble.")

            if k_prev is not None:
                if (
                    k_prev.shape[:2] != v_prev.shape[:2]
                    or k_prev.shape[3:] != v_prev.shape[3:]
                ):
                    raise ValueError("KV cache corrompu: shapes k/v incompatibles.")
                if k_prev.device != v_prev.device or k_prev.dtype != v_prev.dtype:
                    raise ValueError("KV cache corrompu: device/dtype k/v incompatibles.")
                if int(k_prev.size(2)) != int(v_prev.size(2)):
                    raise ValueError("KV cache corrompu: longueurs k/v incompatibles.")

                expected = cache_base_pos + int(k_prev.size(2))
                if int(pos_offset) != int(expected):
                    raise ValueError("KV invariant violé: pos_offset doit == base_pos + len(cache) pour append.")

                k = torch.cat([k_prev, k], dim=2)
                v = torch.cat([v_prev, v], dim=2)

            kv_cache["k"] = k
            kv_cache["v"] = v
            kv_cache["base_pos"] = cache_base_pos

            self._maybe_evict(kv_cache)

            cache_base_pos = int(kv_cache.get("base_pos", 0))
            k = kv_cache["k"]
            v = kv_cache["v"]

        k_full = self._repeat_kv(k)  # (B,H,Lk,Hd)
        v_full = self._repeat_kv(v)  # (B,H,Lk,Hd)

        local_out = self._local_attend_queries_only(q, k_full, v_full, q_pos_abs, cache_base_pos)

        global_out = None
        gate_scalar: Optional[float] = None

        if cache_global is not None and cache_global.numel() > 0 and float(global_weight) != 0.0:
            if global_indices is None:
                raise ValueError("global_indices requis si cache_global est fourni.")
            if cache_global.dim() != 3 or cache_global.size(0) != B or cache_global.size(2) != D:
                raise ValueError("cache_global doit être (B,G,D).")
            G = int(cache_global.size(1))

            if global_indices.shape != (B, G):
                raise ValueError("global_indices doit être (B,G).")
            global_indices = global_indices.to(device=x.device, dtype=torch.long)

            if global_valid_mask is not None:
                if global_valid_mask.shape != (B, G):
                    raise ValueError("global_valid_mask doit être (B,G).")
                global_valid_mask = global_valid_mask.to(device=x.device, dtype=torch.bool)

            gk = self.gk_proj(cache_global).view(B, G, self.num_kv_heads, Hd).transpose(1, 2)  # (B,Hkv,G,Hd)
            gv = self.gv_proj(cache_global).view(B, G, self.num_kv_heads, Hd).transpose(1, 2)
            gk = self._repeat_kv(gk)  # (B,H,G,Hd)
            gv = self._repeat_kv(gv)

            if rope_cos is not None and rope_sin is not None:
                gk = apply_rope_pairwise_2dpos(gk, rope_cos, rope_sin, global_indices)

            valid = torch.ones((B, Lq, G), device=x.device, dtype=torch.bool)
            if global_valid_mask is not None:
                valid &= global_valid_mask[:, None, :]
            if self.causal:
                valid &= (global_indices[:, None, :] <= q_pos_abs[None, :, None])

            any_ok = valid.any(dim=-1)  # (B,Lq)
            global_out = torch.zeros((B, self.num_heads, Lq, Hd), device=x.device, dtype=q.dtype)

            if any_ok.any():
                b_idx, t_idx = any_ok.nonzero(as_tuple=True)
                q_bt = q.permute(0, 2, 1, 3)[b_idx, t_idx]  # (N,H,Hd)
                gk_b = gk[b_idx]                            # (N,H,G,Hd)
                gv_b = gv[b_idx]                            # (N,H,G,Hd)
                vmask = valid[b_idx, t_idx]                 # (N,G)

                scores = torch.einsum("nhd,nhgd->nhg", q_bt, gk_b) / math.sqrt(Hd)
                scores = scores.masked_fill(~vmask[:, None, :], self._mask_min(scores.dtype))
                attn = self.attn_drop(F.softmax(scores, dim=-1))
                out_bt = torch.einsum("nhg,nhgd->nhd", attn, gv_b)
                global_out[b_idx, :, t_idx, :] = out_bt

            if self.gated_fusion:
                gate = torch.sigmoid(self.gate_proj(x)).view(B, 1, Lq, 1)
                global_out = global_out * gate
                gate_scalar = float(gate.detach().mean().item())

        out = local_out if global_out is None else (local_out + float(global_weight) * global_out)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.num_heads * Hd)
        out = self.proj_drop(self.o_proj(out))

        self.last_monitor = {"gate_scalar": gate_scalar}
        return out


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    vocab_size: int = 50257
    max_seq_len: int = 2048

    embed_dim: int = 512
    num_heads: int = 8
    num_kv_heads: Optional[int] = 2

    ff_hidden_multiplier: float = 4.0
    n_layers: int = 12
    dropout_rate: float = 0.1

    local_window: int = 128
    global_weight: float = 1.0
    gated_fusion: bool = True
    causal: bool = True

    norm_eps: float = 1e-6
    use_swiglu: bool = True

    use_rope: bool = True
    rope_base: float = 10000.0

    grad_checkpointing: bool = False
    resid_dropout: float = 0.0
    init_std: float = 0.02
    use_scaled_residual_init: bool = True

    use_kv_cache: bool = True
    max_cache_len: Optional[int] = None
    require_odd_window_noncausal: bool = True

    enforce_max_seq_len: bool = True


# ============================================================
# Transformer Block (checkpoint-safe)
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.norm1 = RMSNorm(cfg.embed_dim, eps=cfg.norm_eps)
        self.norm2 = RMSNorm(cfg.embed_dim, eps=cfg.norm_eps)

        self.attn = SLGAAttention(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            local_window=cfg.local_window,
            attn_drop=cfg.dropout_rate,
            proj_drop=cfg.dropout_rate,
            causal=cfg.causal,
            gated_fusion=cfg.gated_fusion,
            require_odd_window_noncausal=cfg.require_odd_window_noncausal,
            max_cache_len=cfg.max_cache_len,
        )

        hidden = int(cfg.embed_dim * cfg.ff_hidden_multiplier)
        self.ffn = SwiGLU(cfg.embed_dim, hidden, cfg.dropout_rate) if cfg.use_swiglu else nn.Sequential(
            nn.Linear(cfg.embed_dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, cfg.embed_dim, bias=False),
            nn.Dropout(cfg.dropout_rate),
        )

        self.resid_drop = nn.Dropout(cfg.resid_dropout) if cfg.resid_dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cache_global: Optional[torch.Tensor],
        global_indices: Optional[torch.Tensor],
        global_valid_mask: Optional[torch.Tensor],
        global_weight: float,
        kv_cache: Optional[Dict[str, Any]],
        pos_offset: int,
        rope_cos: Optional[torch.Tensor],
        rope_sin: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[float]]:
        if self.cfg.grad_checkpointing and self.training:
            cg = cache_global
            gi = global_indices
            gvm = global_valid_mask
            rc = rope_cos
            rs = rope_sin
            gw = float(global_weight)
            po = int(pos_offset)

            def _pure(x_in: torch.Tensor, gw_in: torch.Tensor, po_in: torch.Tensor) -> torch.Tensor:
                x1 = self.norm1(x_in)
                att = self.attn(
                    x1,
                    cache_global=cg,
                    global_indices=gi,
                    global_valid_mask=gvm,
                    global_weight=float(gw_in.item()),
                    kv_cache=None,
                    pos_offset=int(po_in.item()),
                    rope_cos=rc,
                    rope_sin=rs,
                )
                x2 = x_in + self.resid_drop(att)
                x3 = self.norm2(x2)
                ff = self.ffn(x3)
                return x2 + self.resid_drop(ff)

            x = checkpoint(
                _pure,
                x,
                torch.tensor(gw, device=x.device, dtype=torch.float32),
                torch.tensor(po, device=x.device, dtype=torch.long),
                use_reentrant=False,
            )
            return x, None

        x1 = self.norm1(x)
        att = self.attn(x1, cache_global, global_indices, global_valid_mask, global_weight, kv_cache, pos_offset, rope_cos, rope_sin)
        x = x + self.resid_drop(att)
        x = x + self.resid_drop(self.ffn(self.norm2(x)))
        return x, self.attn.last_monitor.get("gate_scalar", None)


# ============================================================
# Model (hard cap RoPE + canonical pos_offset under cache)
# ============================================================

class LLMTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # strict init hardening before blocks construction
        if not self.cfg.use_kv_cache:
            self.cfg.max_cache_len = None
        if self.cfg.enforce_max_seq_len and self.cfg.max_cache_len is not None:
            self.cfg.max_cache_len = min(int(self.cfg.max_cache_len), int(self.cfg.max_seq_len))

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.emb_drop = nn.Dropout(cfg.dropout_rate)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.embed_dim, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.register_buffer("_rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("_rope_sin", torch.empty(0), persistent=False)

        self.apply(self._init_weights)
        self._post_init_scale()

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, self.cfg.init_std)

    def _post_init_scale(self):
        if not self.cfg.use_scaled_residual_init:
            return
        scale = 1.0 / math.sqrt(2.0 * self.cfg.n_layers)
        with torch.no_grad():
            for blk in self.blocks:
                blk.attn.o_proj.weight.mul_(scale)
                if isinstance(blk.ffn, SwiGLU):
                    blk.ffn.w3.weight.mul_(scale)

    def _ensure_rope(self, need_len: int, device: torch.device, dtype: torch.dtype):
        if not self.cfg.use_rope:
            return
        head_dim = self.cfg.embed_dim // self.cfg.num_heads
        need = (
            self._rope_cos.numel() == 0
            or int(self._rope_cos.size(0)) < int(need_len)
            or self._rope_cos.device != device
            or self._rope_cos.dtype != dtype
        )
        if need:
            cos, sin = build_rope_cache_pairwise(int(need_len), head_dim, self.cfg.rope_base, device, dtype)
            self.register_buffer("_rope_cos", cos, persistent=False)
            self.register_buffer("_rope_sin", sin, persistent=False)

    @torch.no_grad()
    def init_kv_cache(self, batch_size: int, device: torch.device) -> List[Dict[str, Any]]:
        return [{"k": None, "v": None, "base_pos": 0} for _ in range(self.cfg.n_layers)]

    def _rope_need_len(
        self,
        pos_offset: int,
        Lq: int,
        kv_cache: Optional[List[Optional[Dict[str, Any]]]],
        global_indices: Optional[torch.Tensor],
    ) -> int:
        need = int(pos_offset) + int(Lq)

        if global_indices is not None and global_indices.numel() > 0:
            need = max(need, int(global_indices.max().item()) + 1)

        if kv_cache is not None and self.cfg.use_kv_cache:
            for lc in kv_cache:
                if lc is None:
                    continue
                k = lc.get("k", None)
                if k is None:
                    continue
                base = int(lc.get("base_pos", 0))
                need = max(need, base + int(k.size(2)))

        return need

    def forward(
        self,
        input_ids: torch.Tensor,
        pos_offset: int = 0,
        kv_cache: Optional[List[Optional[Dict[str, Any]]]] = None,
        cache_global: Optional[torch.Tensor] = None,
        global_indices: Optional[torch.Tensor] = None,
        global_valid_mask: Optional[torch.Tensor] = None,
        global_weight: Optional[float] = None,
        enforce_max_seq_len: Optional[bool] = None,
        return_aux: bool = False,
    ):
        B, Lq = input_ids.shape
        x = self.token_emb(input_ids)
        x = self.emb_drop(x)

        # normalize kv_cache structure
        if kv_cache is None:
            kv_cache = [None] * self.cfg.n_layers
        else:
            if len(kv_cache) != self.cfg.n_layers:
                raise ValueError("kv_cache longueur != n_layers")

        # checkpointing: disable cache mutation by forcing structure only
        if self.training and self.cfg.grad_checkpointing:
            kv_cache = [None] * self.cfg.n_layers

        # derive pos_offset only if cache is truly active across all layers
        if has_active_kv_cache(self.cfg, kv_cache):
            pos_offset = current_pos_offset(kv_cache, fallback=int(pos_offset))

        if enforce_max_seq_len is None:
            enforce_max_seq_len = bool(self.cfg.enforce_max_seq_len)

        rope_cos = rope_sin = None
        if self.cfg.use_rope:
            need_len = self._rope_need_len(int(pos_offset), int(Lq), kv_cache, global_indices)
            if enforce_max_seq_len and need_len > int(self.cfg.max_seq_len):
                raise ValueError("need_len exceeds max_seq_len; cap generation or increase max_seq_len.")
            self._ensure_rope(need_len, x.device, x.dtype)
            rope_cos, rope_sin = self._rope_cos, self._rope_sin

        gw = float(self.cfg.global_weight if global_weight is None else global_weight)
        gates: List[float] = []

        for i, blk in enumerate(self.blocks):
            layer_cache = None
            if self.cfg.use_kv_cache and kv_cache[i] is not None:
                layer_cache = kv_cache[i]

            x, gate = blk(
                x,
                cache_global=cache_global,
                global_indices=global_indices,
                global_valid_mask=global_valid_mask,
                global_weight=gw,
                kv_cache=layer_cache,
                pos_offset=int(pos_offset),
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            if gate is not None:
                gates.append(float(gate))

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if not return_aux:
            return logits
        return logits, {"gate_values": gates}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: int = 50256,
        cache_global: Optional[torch.Tensor] = None,
        global_indices: Optional[torch.Tensor] = None,
        global_valid_mask: Optional[torch.Tensor] = None,
        global_weight: Optional[float] = None,
        seed: Optional[int] = None,
        debug_cache_sync: bool = False,
    ) -> torch.Tensor:
        self.eval()
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        B = int(input_ids.size(0))
        device = input_ids.device

        kv_cache = self.init_kv_cache(B, device=device) if self.cfg.use_kv_cache else None

        _ = self(
            input_ids,
            kv_cache=kv_cache,
            cache_global=cache_global,
            global_indices=global_indices,
            global_valid_mask=global_valid_mask,
            global_weight=global_weight,
            enforce_max_seq_len=True,
        )

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(int(max_new_tokens)):
            if kv_cache is not None:
                pos_offset = current_pos_offset(kv_cache, fallback=int(input_ids.size(1)))
            else:
                pos_offset = int(input_ids.size(1))

            if int(pos_offset) >= int(self.cfg.max_seq_len):
                raise ValueError("Generation dépasse max_seq_len; augmenter max_seq_len ou limiter la génération.")

            if debug_cache_sync and kv_cache is not None:
                want = current_pos_offset(kv_cache, fallback=int(input_ids.size(1)))
                for lc in kv_cache:
                    k = lc.get("k", None)
                    v = lc.get("v", None)
                    if (k is None) != (v is None):
                        raise ValueError("KV cache corrompu: k/v mismatch.")
                    if k is None:
                        raise ValueError("KV cache désactivé en cours de route.")
                    got = int(lc.get("base_pos", 0)) + int(k.size(2))
                    if got != want:
                        raise ValueError("KV cache désynchronisé entre layers.")

            last = input_ids[:, -1:].contiguous()
            logits = self(
                last,
                kv_cache=kv_cache,
                cache_global=cache_global,
                global_indices=global_indices,
                global_valid_mask=global_valid_mask,
                global_weight=global_weight,
                enforce_max_seq_len=True,
            )[:, -1, :]

            next_tok = torch.empty((B, 1), device=device, dtype=torch.long)
            next_tok[finished] = int(eos_token_id)

            live = ~finished
            if live.any():
                live_logits = logits[live]
                if temperature == 0.0:
                    next_tok[live] = torch.argmax(live_logits, dim=-1, keepdim=True)
                else:
                    live_logits = live_logits / float(temperature)
                    if top_k is not None and top_k > 0:
                        k = min(int(top_k), int(live_logits.size(-1)))
                        vals, idxs = torch.topk(live_logits, k=k, dim=-1)
                        filt = torch.full_like(live_logits, float("-inf"))
                        filt.scatter_(1, idxs, vals)
                        live_logits = filt
                    probs = F.softmax(live_logits, dim=-1)
                    probs = torch.clamp(probs, min=1e-10)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    next_tok[live] = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_tok], dim=1)
            finished = finished | (next_tok.squeeze(-1) == eos_token_id)
            if finished.all():
                break

        return input_ids


__all__ = [
    "Config",
    "LLMTransformer",
    "TransformerBlock",
    "SLGAAttention",
    "RMSNorm",
    "SwiGLU",
    "has_active_kv_cache",
    "current_pos_offset",
    "build_rope_cache_pairwise",
    "apply_rope_pairwise_1dpos",
    "apply_rope_pairwise_2dpos",
]
