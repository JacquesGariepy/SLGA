"""
Sparse Local-Global Attention Module (SLGA)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

# Flash Attention support (optional)
try:
    from .flash_attention import (
        FLASH_ATTN_AVAILABLE,
        flash_attention_local_window,
        flash_attention_forward,
    )
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attention_local_window = None
    flash_attention_forward = None


# ======================================================================
# RoPE (Rotary Position Embedding)
# ======================================================================

class RotaryEmbedding(nn.Module):
    """
    RoPE optimisé pour application par tête.
    
    Appliqué sur tenseurs:
    - (B, H, L, Dh) format multi-head
    - (B, L, D) format standard (optionnel)
    
    Note: RoPE doit être appliqué sur Q et K, PAS sur V.
    """

    def __init__(self, dim: int, max_seq_len: int = 131072, theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RotaryEmbedding requiert une dimension paire"
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Fréquences inverses
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pré-calcul des cos/sin pour toutes les positions
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (max_seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)       # (max_seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Applique RoPE.
        
        Args:
            x: (B, H, L, Dh) ou (B, L, D)
            position_ids: (B, L) ou (B, G) positions entières
        
        Returns:
            x avec RoPE appliqué (même shape)
        """
        if x.dim() == 4:
            # Format multi-head: (B, H, L, Dh)
            B, H, L, Dh = x.shape
            assert Dh == self.dim, f"Dim mismatch: Dh={Dh}, expected={self.dim}"
            assert position_ids.shape[0] == B, "Batch size mismatch"
            assert position_ids.shape[1] == L, f"Seq len mismatch: {position_ids.shape[1]} vs {L}"
            
            # Récupérer cos/sin aux positions demandées
            # position_ids: (B, L) -> cos/sin: (B, L, Dh) -> (B, 1, L, Dh)
            cos = self.cos_cached[position_ids].unsqueeze(1)  # (B, 1, L, Dh)
            sin = self.sin_cached[position_ids].unsqueeze(1)  # (B, 1, L, Dh)
            
        elif x.dim() == 3:
            # Format standard: (B, L, D)
            B, L, D = x.shape
            assert D == self.dim, f"Dim mismatch: D={D}, expected={self.dim}"
            
            cos = self.cos_cached[position_ids]  # (B, L, D)
            sin = self.sin_cached[position_ids]  # (B, L, D)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D")

        # Rotation: x * cos + rotate_half(x) * sin
        half = self.dim // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        x_rot = torch.cat((-x2, x1), dim=-1)

        return x * cos + x_rot * sin


# ======================================================================
# SLGA Module
# ======================================================================

class SLGAModule(nn.Module):
    """
    Sparse Local-Global Attention Module.
    
    Architecture:
    1. Local attention: Fenêtre glissante causale de taille W
    2. Global attention: Top-K sur G landmarks avec diversité inter-têtes
    3. Fusion: Gated ou additive
    
    Paramètres:
        embed_dim: Dimension d'embedding
        num_heads: Nombre de têtes d'attention
        local_window: Taille de la fenêtre locale
        global_k: Top-K pour sélection globale par tête
        attn_drop: Dropout sur les poids d'attention
        proj_drop: Dropout sur la projection de sortie
        causal: Si True, masquage causal
        gated_fusion: Si True, fusion apprise (sinon additive)
        dilation: Facteur de dilation pour la fenêtre
        diverse_topk: Si True, diversité inter-têtes dans top-K
        max_seq_len: Longueur max pour RoPE
        rope_theta: Base theta pour RoPE
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        local_window: int = 128,
        global_k: int = 24,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        causal: bool = True,
        gated_fusion: bool = True,
        dilation: int = 1,
        diverse_topk: bool = True,
        joint_normalization: bool = False,
        max_seq_len: int = 131072,
        rope_theta: float = 10000.0,
        use_flash_attn: bool = True,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
        assert local_window > 0, f"local_window must be > 0, got {local_window}"
        assert global_k > 0, f"global_k must be > 0, got {global_k}"
        assert dilation >= 1, f"dilation must be >= 1, got {dilation}"

        # GQA support
        if num_kv_heads is None:
            num_kv_heads = num_heads  # Standard MHA

        assert num_kv_heads > 0 and num_kv_heads <= num_heads, \
            f"num_kv_heads={num_kv_heads} must be in range (0, num_heads={num_heads}]"
        assert num_heads % num_kv_heads == 0, \
            f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}"

        self.D = embed_dim
        self.H = num_heads
        self.H_kv = num_kv_heads
        self.Dh = embed_dim // num_heads
        self.n_rep = num_heads // num_kv_heads  # Queries per KV head
        self.use_gqa = num_kv_heads < num_heads

        self.W = local_window
        self.GK = global_k
        self.causal = causal
        self.gated = gated_fusion
        self.dilation = dilation
        self.diverse_topk = diverse_topk
        self.joint_norm = joint_normalization

        # Flash Attention configuration
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE
        if use_flash_attn and not FLASH_ATTN_AVAILABLE:
            print("⚠️  Flash Attention requested but not available. Install with: pip install flash-attn")
            print("    Falling back to standard PyTorch attention.")

        # Cache pour masques locaux (évite recalcul)
        self._mask_cache = {}

        # Projections QKV with GQA support
        if self.use_gqa:
            # Q: full heads, K/V: reduced heads
            self.q_proj = nn.Linear(self.D, self.D, bias=True)
            kv_dim = num_kv_heads * self.Dh
            self.k_proj = nn.Linear(self.D, kv_dim, bias=True)
            self.v_proj = nn.Linear(self.D, kv_dim, bias=True)
        else:
            # Standard unified QKV projection
            self.qkv_proj = nn.Linear(self.D, 3 * self.D, bias=True)

        self.out_proj = nn.Linear(self.D, self.D, bias=False)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.scale = self.Dh ** -0.5
        
        # RoPE interne (par tête, dimension Dh)
        self.rope = RotaryEmbedding(self.Dh, max_seq_len, rope_theta)
        
        # Gated fusion
        if self.gated:
            self.gate_proj = nn.Linear(2 * self.Dh, self.Dh)
        
        # Monitoring (détaché du graph)
        self.last_monitor: Dict[str, Optional[torch.Tensor]] = {
            "gate_scalar": None,
            "global_topk_indices": None,
            "global_attention_weights": None,
        }

        # Offsets pour fenêtrage dilaté
        base_offsets = torch.arange(self.W) - (self.W // 2)
        dilated_offsets = base_offsets * self.dilation
        self.register_buffer("offsets", dilated_offsets, persistent=False)

    def _split_heads(self, t: torch.Tensor, B: int, L: int, num_heads: Optional[int] = None) -> torch.Tensor:
        """(B, L, D) -> (B, H, L, Dh) or (B, H_kv, L, Dh) for GQA"""
        if num_heads is None:
            num_heads = self.H
        return t.view(B, L, num_heads, self.Dh).transpose(1, 2)

    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        """(B, H, L, Dh) -> (B, L, D)"""
        B, H, L, Dh = t.shape
        return t.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match Q heads for GQA.

        Args:
            x: (B, H_kv, L, Dh)

        Returns:
            (B, H, L, Dh)
        """
        if self.n_rep == 1:
            return x

        B, H_kv, L, Dh = x.shape
        # Repeat each KV head n_rep times
        x = x.unsqueeze(2).expand(B, H_kv, self.n_rep, L, Dh)
        x = x.reshape(B, H_kv * self.n_rep, L, Dh)
        return x
    
    def _safe_softmax(
        self, scores: torch.Tensor, mask: torch.Tensor, dim: int = -1
    ) -> torch.Tensor:
        """
        Softmax avec protection NaN pour lignes entièrement masquées.
        
        Args:
            scores: Scores d'attention
            mask: Masque booléen (True = invalide)
            dim: Dimension pour softmax
        """
        scores_masked = scores.masked_fill(mask, float('-inf'))
        all_masked = mask.all(dim=dim, keepdim=True)
        attn = F.softmax(scores_masked, dim=dim)
        
        # Remplacer NaN par zéros (pas de contribution)
        attn = torch.where(
            all_masked.expand_as(attn),
            torch.zeros_like(attn),
            attn
        )
        
        return attn
    
    def _diverse_topk(
        self, scores: torch.Tensor, k: int, diversity_penalty: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Top-K avec diversité inter-têtes.
        
        Encourage chaque tête à sélectionner des landmarks différents.
        """
        if not self.diverse_topk:
            return torch.topk(scores, k=k, dim=-1)

        B, H, L, G = scores.shape
        k_actual = min(k, G)

        # Compteur de sélections par position globale
        selection_counts = torch.zeros(B, L, G, device=scores.device, dtype=scores.dtype)

        topk_vals_list = []
        topk_idxs_list = []

        for h in range(H):
            scores_h = scores[:, h]  # (B, L, G)

            # Pénaliser les positions déjà sélectionnées
            if h > 0:
                penalty = diversity_penalty * selection_counts
                scores_h = scores_h - penalty

            # Top-K pour cette tête
            topk_val_h, topk_idx_h = torch.topk(scores_h, k=k_actual, dim=-1)

            # Mettre à jour le compteur
            selection_counts.scatter_add_(
                2, topk_idx_h, torch.ones_like(topk_idx_h, dtype=scores.dtype)
            )

            topk_vals_list.append(topk_val_h)
            topk_idxs_list.append(topk_idx_h)

        return torch.stack(topk_vals_list, dim=1), torch.stack(topk_idxs_list, dim=1)
    
    def _local_attention_flash(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flash Attention pour attention locale avec fenêtre glissante.

        2-3× plus rapide que l'implémentation vectorisée PyTorch.

        Args:
            q, k, v: (B, H, L, Dh) après RoPE

        Returns:
            ctx_local: (B, H, L, Dh)
        """
        # Flash Attention avec fenêtre locale
        # Dilation is handled by the caller (already in k, v)
        dropout_p = self.attn_drop.p if self.training else 0.0

        ctx_local = flash_attention_local_window(
            q, k, v,
            window_size=self.W,
            causal=self.causal,
            dropout_p=dropout_p,
            softmax_scale=self.scale,
        )

        return ctx_local

    def _local_attention_vectorized(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Attention locale ENTIÈREMENT VECTORISÉE (PyTorch fallback).

        ~10× plus rapide que la version avec boucle Python.

        Args:
            q: (B, H, L_q, Dh) query après RoPE
            k: (B, H, L_k, Dh) key après RoPE (may include cached)
            v: (B, H, L_k, Dh) value (may include cached)
            k_len: Total key length (L_k) including cache
            device: Device

        Returns:
            ctx_local: (B, H, L_q, Dh)
        """
        B, H, L_q, Dh = q.shape
        _, _, L_k, _ = k.shape
        W = self.W

        # For cached generation: query is new tokens, k/v include past
        # We need to attend to the appropriate window for each query position

        # 1. Calculer les indices de fenêtre pour chaque position de query
        # Note: positions are absolute (accounting for cache offset)
        # For simplicity, we'll use a different approach for cached case

        if L_q == L_k:
            # Standard case: no cache, query and key have same length
            i = torch.arange(L_q, device=device)      # (L_q,)
            offsets = self.offsets.to(device)        # (W,)

            # j[l, w] = position du w-ème élément de la fenêtre pour query l
            j = i.unsqueeze(1) + offsets.unsqueeze(0)  # (L_q, W)

            # 2. Créer le masque d'invalidité
            invalid = (j < 0) | (j >= L_k)
            if self.causal:
                invalid = invalid | (j > i.unsqueeze(1))

            # 3. Clamper les indices pour gather (invalides seront masqués après)
            j_clamped = j.clamp(0, L_k - 1)  # (L_q, W)

        else:
            # Cached case: L_q < L_k (new tokens only)
            # Query positions are at the end of the sequence
            offset = L_k - L_q
            i = torch.arange(offset, offset + L_q, device=device)  # (L_q,)
            offsets = self.offsets.to(device)  # (W,)

            j = i.unsqueeze(1) + offsets.unsqueeze(0)  # (L_q, W)

            # Masque d'invalidité
            invalid = (j < 0) | (j >= L_k)
            if self.causal:
                invalid = invalid | (j > i.unsqueeze(1))

            j_clamped = j.clamp(0, L_k - 1)  # (L_q, W)

        # 4. Gather K et V selon les indices de fenêtre
        # k: (B, H, L_k, Dh) -> k_win: (B, H, L_q, W, Dh)
        j_exp = j_clamped.view(1, 1, L_q, W).expand(B, H, L_q, W)
        j_gather = j_exp.unsqueeze(-1).expand(B, H, L_q, W, Dh)

        # Expand k et v pour gather sur dim=2
        k_exp = k.unsqueeze(2).expand(B, H, L_q, L_k, Dh)
        v_exp = v.unsqueeze(2).expand(B, H, L_q, L_k, Dh)

        # Gather
        k_win = torch.gather(k_exp, dim=3, index=j_gather)  # (B, H, L_q, W, Dh)
        v_win = torch.gather(v_exp, dim=3, index=j_gather)  # (B, H, L_q, W, Dh)

        # 5. Masquer les positions invalides
        invalid_exp = invalid.view(1, 1, L_q, W, 1).expand(B, H, L_q, W, Dh)
        k_win = k_win.masked_fill(invalid_exp, 0.0)
        v_win = v_win.masked_fill(invalid_exp, 0.0)

        # 6. Scores d'attention: Q · K^T
        scores = torch.einsum('bhld,bhlwd->bhlw', q, k_win) * self.scale  # (B, H, L_q, W)

        # 7. Masquer et softmax
        invalid_scores = invalid.view(1, 1, L_q, W).expand(B, H, L_q, W)
        attn = self._safe_softmax(scores, invalid_scores, dim=-1)
        attn = self.attn_drop(attn)

        # 8. Contexte local
        ctx_local = torch.einsum('bhlw,bhlwd->bhld', attn, v_win)  # (B, H, L_q, Dh)

        return ctx_local
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        cache_global: Optional[torch.Tensor] = None,
        cache_positions: Optional[torch.Tensor] = None,
        global_weight: float = 1.0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass de l'attention local-global.

        Args:
            x: (B, L, D) embeddings d'entrée
            position_ids: (B, L) positions pour RoPE
            cache_global: (B, G, D) états des landmarks
            cache_positions: (B, G) positions ORIGINALES des landmarks pour RoPE
            global_weight: poids pour warmup progressif (0→1)
            kv_cache: Optional[(B, H, past_L, Dh), (B, H, past_L, Dh)] cached K, V
            use_cache: If True, return updated K, V for caching

        Returns:
            out: (B, L, D)
            new_kv_cache: Optional[(B, H, total_L, Dh), (B, H, total_L, Dh)] if use_cache
        """
        B, L, D = x.shape
        device = x.device

        # === Projections QKV with GQA support ===
        if self.use_gqa:
            # Separate projections for GQA
            q = self.q_proj(x)  # (B, L, D)
            k = self.k_proj(x)  # (B, L, H_kv * Dh)
            v = self.v_proj(x)  # (B, L, H_kv * Dh)

            # Split heads
            q = self._split_heads(q, B, L, self.H)      # (B, H, L, Dh)
            k = self._split_heads(k, B, L, self.H_kv)   # (B, H_kv, L, Dh)
            v = self._split_heads(v, B, L, self.H_kv)   # (B, H_kv, L, Dh)

            # Apply RoPE before repeating KV
            q = self.rope(q, position_ids)  # (B, H, L, Dh)
            k = self.rope(k, position_ids)  # (B, H_kv, L, Dh)

            # Repeat KV heads to match Q heads
            k = self._repeat_kv(k)  # (B, H, L, Dh)
            v = self._repeat_kv(v)  # (B, H, L, Dh)
        else:
            # Standard unified QKV projection
            qkv = self.qkv_proj(x)  # (B, L, 3D)
            q, k, v = qkv.chunk(3, dim=-1)  # (B, L, D) each

            # Split heads
            q = self._split_heads(q, B, L)  # (B, H, L, Dh)
            k = self._split_heads(k, B, L)  # (B, H, L, Dh)
            v = self._split_heads(v, B, L)  # (B, H, L, Dh)

            # === CORRECTION MAJEURE: Appliquer RoPE sur Q et K (pas V!) ===
            q = self.rope(q, position_ids)  # (B, H, L, Dh)
            k = self.rope(k, position_ids)  # (B, H, L, Dh)

        # === KV Cache Management ===
        past_k: Optional[torch.Tensor] = None
        past_v: Optional[torch.Tensor] = None

        if kv_cache is not None:
            past_k, past_v = kv_cache
            # Concatenate past and current K, V
            k = torch.cat([past_k, k], dim=2)  # (B, H, past_L + L, Dh)
            v = torch.cat([past_v, v], dim=2)  # (B, H, past_L + L, Dh)

        # Store updated K, V for cache
        new_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if use_cache:
            new_kv_cache = (k, v)
        
        # =========================================
        # 1. ATTENTION LOCALE
        # =========================================
        # Note: For cached generation, we compute attention over full K, V
        # but only for the new query positions
        k_len = k.size(2)  # Total length (past + current)

        # Use Flash Attention if available and enabled
        if self.use_flash_attn and self.dilation == 1:
            # Flash Attention works best without dilation
            # For dilated windows, fall back to vectorized implementation
            ctx_local = self._local_attention_flash(q, k, v)
        else:
            # PyTorch vectorized fallback
            ctx_local = self._local_attention_vectorized(q, k, v, k_len, device)
        
        # =========================================
        # 2. ATTENTION GLOBALE
        # =========================================
        topk_idxs_safe: Optional[torch.Tensor] = None
        attn_g: Optional[torch.Tensor] = None
        ctx_global: Optional[torch.Tensor] = None

        if cache_global is not None and global_weight > 0.0:
            Bg, G, Dg = cache_global.shape
            assert Bg == B and Dg == D, f"Cache shape mismatch"

            # Projeter le cache global with GQA support
            if self.use_gqa:
                # Separate K/V projections for GQA
                kg = self.k_proj(cache_global)  # (B, G, H_kv * Dh)
                vg = self.v_proj(cache_global)  # (B, G, H_kv * Dh)

                # Split heads
                kg = kg.view(B, G, self.H_kv, self.Dh).transpose(1, 2)  # (B, H_kv, G, Dh)
                vg = vg.view(B, G, self.H_kv, self.Dh).transpose(1, 2)  # (B, H_kv, G, Dh)

                # Apply RoPE before repeating
                if cache_positions is not None:
                    kg = self.rope(kg, cache_positions)  # (B, H_kv, G, Dh)

                # Repeat KV heads
                kg = self._repeat_kv(kg)  # (B, H, G, Dh)
                vg = self._repeat_kv(vg)  # (B, H, G, Dh)
            else:
                # Standard unified projection
                kv_g = self.qkv_proj(cache_global)  # (B, G, 3D)
                _, kg, vg = kv_g.chunk(3, dim=-1)   # (B, G, D)

                # Split heads
                kg = kg.view(B, G, self.H, self.Dh).transpose(1, 2)  # (B, H, G, Dh)
                vg = vg.view(B, G, self.H, self.Dh).transpose(1, 2)  # (B, H, G, Dh)

                # === CORRECTION MAJEURE: RoPE sur K globaux avec positions ORIGINALES ===
                if cache_positions is not None:
                    kg = self.rope(kg, cache_positions)  # (B, H, G, Dh)
            
            # Scores globaux: Q @ Kg^T
            scores_g = torch.matmul(q, kg.transpose(-2, -1)) * self.scale  # (B, H, L, G)
            
            # Masque causal pour attention globale
            if self.causal and cache_positions is not None:
                # Position de chaque query: (B, 1, L, 1)
                pos_query = position_ids.view(B, 1, L, 1)
                # Position de chaque landmark: (B, 1, 1, G)
                pos_cache = cache_positions.view(B, 1, 1, G)
                # Masquer les landmarks du futur
                future_mask = pos_cache > pos_query  # (B, 1, L, G)
                scores_g = scores_g.masked_fill(future_mask, float('-inf'))
            
            # Top-K (avec ou sans diversité)
            k_sel = min(self.GK, G)
            if self.diverse_topk:
                topk_vals, topk_idxs = self._diverse_topk(scores_g, k=k_sel)
            else:
                topk_vals, topk_idxs = torch.topk(scores_g, k=k_sel, dim=-1)
            
            # Protection NaN (si tout masqué)
            all_masked = (topk_vals == float('-inf')).all(dim=-1)
            attn_g = F.softmax(topk_vals, dim=-1)
            if all_masked.any():
                attn_g = torch.where(
                    all_masked.unsqueeze(-1),
                    torch.zeros_like(attn_g),
                    attn_g
                )
            attn_g = self.attn_drop(attn_g)
            
            # Gather Vg selon top-K indices
            topk_idxs_safe = torch.clamp(topk_idxs, 0, G - 1)
            vg_exp = vg.unsqueeze(2).expand(B, self.H, L, G, self.Dh)
            topk_idxs_exp = topk_idxs_safe.unsqueeze(-1).expand(B, self.H, L, k_sel, self.Dh)
            vg_topk = torch.gather(vg_exp, dim=3, index=topk_idxs_exp)
            
            # Contexte global
            ctx_global = (attn_g.unsqueeze(-1) * vg_topk).sum(dim=3)  # (B, H, L, Dh)
        
        # =========================================
        # 3. FUSION LOCAL + GLOBAL
        # =========================================
        gate_scalar: Optional[torch.Tensor] = None

        if ctx_global is not None:
            ctx_global_weighted = ctx_global * global_weight

            if self.gated:
                # Fusion apprise avec gating
                ctx_cat = torch.cat([ctx_local, ctx_global_weighted], dim=-1)  # (B, H, L, 2*Dh)
                B_val, H_val, L_val, _ = ctx_cat.shape
                ctx_cat_reshaped = ctx_cat.reshape(B_val * H_val * L_val, 2 * self.Dh)
                
                gate_flat = torch.sigmoid(self.gate_proj(ctx_cat_reshaped))
                gate = gate_flat.view(B_val, H_val, L_val, self.Dh)
                gate_scalar = gate.mean(dim=-1)  # Pour monitoring
                
                ctx = gate * ctx_local + (1 - gate) * ctx_global_weighted
            else:
                # Fusion additive simple
                ctx = ctx_local + ctx_global_weighted
        else:
            ctx = ctx_local
        
        # =========================================
        # 4. PROJECTION DE SORTIE
        # =========================================
        ctx = self._merge_heads(ctx)
        out = self.out_proj(ctx)
        out = self.proj_drop(out)
        
        # Monitoring (détaché du graph)
        self.last_monitor = {
            "gate_scalar": gate_scalar.detach() if gate_scalar is not None else None,
            "global_topk_indices": topk_idxs_safe.detach() if topk_idxs_safe is not None else None,
            "global_attention_weights": attn_g.detach() if attn_g is not None else None,
        }

        if use_cache:
            return out, new_kv_cache
        return out, None


__all__ = ["SLGAModule", "RotaryEmbedding"]
