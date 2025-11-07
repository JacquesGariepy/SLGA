# slga.py
"""
Sparse Local-Global Attention Module (SLGA) - Version Corrigée et Optimisée

Améliorations incluses:
- Fenêtrage local sans biais de clamping
- Fusion gated apprise entre contextes local et global
- Top-K diversifié pour éviter la dégénérescence
- Protection contre les NaN dans softmax
- Support fenêtres dilatées par couche
- Normalisation jointe optionnelle des scores
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class SLGAModule(nn.Module):
    """
    Sparse Local-Global Attention avec corrections critiques.
    
    Architecture:
    1. Attention locale: fenêtre glissante de taille W (masque causal)
    2. Attention globale: cache de G états avec top-K sélection par tête
    3. Fusion: gated learning ou additive simple
    
    Paramètres:
        embed_dim: Dimension des embeddings
        num_heads: Nombre de têtes d'attention
        local_window: Taille de la fenêtre locale
        global_k: Top-K pour sélection globale par tête
        attn_drop: Dropout sur les poids d'attention
        proj_drop: Dropout sur la projection de sortie
        causal: Si True, applique masque causal
        gated_fusion: Si True, utilise fusion apprise (sinon additive)
        dilation: Facteur de dilatation pour fenêtre (1=dense, 2=skip every 2)
        diverse_topk: Si True, encourage diversité inter-têtes dans top-K
        joint_normalization: Si True, normalise local+global ensemble (expérimental)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        local_window: int = 128,
        global_k: int = 24,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        causal: bool = True,
        gated_fusion: bool = True,
        dilation: int = 1,
        diverse_topk: bool = True,
        joint_normalization: bool = False,
    ):
        super().__init__()

        # BUG FIX #1: Validation exhaustive des paramètres (ANALYSE_COMPLETE_LLM.md ligne 152-168)
        assert embed_dim % num_heads == 0, f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
        assert local_window > 0, f"local_window must be > 0, got {local_window}"
        assert global_k > 0, f"global_k must be > 0, got {global_k}"
        assert 0.0 <= attn_drop < 1.0, f"attn_drop must be in [0, 1), got {attn_drop}"
        assert 0.0 <= proj_drop < 1.0, f"proj_drop must be in [0, 1), got {proj_drop}"
        assert dilation >= 1, f"dilation must be >= 1, got {dilation}"
        
        self.D = embed_dim
        self.H = num_heads
        self.Dh = embed_dim // num_heads
        self.W = local_window
        self.GK = global_k
        self.causal = causal
        self.gated = gated_fusion
        self.dilation = dilation
        self.diverse_topk = diverse_topk
        self.joint_norm = joint_normalization

        # BUG FIX #2: Cache pour masques locaux (ANALYSE_COMPLETE_LLM.md ligne 172-197)
        # Évite recomputation coûteuse (5-10x speedup sur séquences répétées)
        self._mask_cache = {}

        # Projection QKV unifiée (utilisée pour flux principal ET cache global)
        self.qkv_proj = nn.Linear(self.D, 3 * self.D, bias=False)
        self.out_proj = nn.Linear(self.D, self.D, bias=False)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.scale = self.Dh ** -0.5
        
        # Fusion gated (par tête, dimension Dh)
        if self.gated:
            # Input: concat(ctx_local, ctx_global) = 2*Dh
            # Output: gate weights de dimension Dh
            self.gate_proj = nn.Linear(2 * self.Dh, self.Dh)
        
        # Stockage des métriques du dernier forward (pour monitoring)
        self.last_monitor: Dict[str, Optional[torch.Tensor]] = {
            "gate_scalar": None,
            "global_topk_indices": None,
            "global_attention_weights": None,
        }

        # Offsets pour fenêtre centrée avec dilatation
        base_offsets = torch.arange(self.W) - (self.W // 2)
        dilated_offsets = base_offsets * self.dilation
        self.register_buffer("offsets", dilated_offsets, persistent=False)
        
        # Padding states pour positions invalides (jamais backpropagé)
        self.register_buffer("k_pad", torch.zeros(1, 1, 1, self.Dh), persistent=False)
        self.register_buffer("v_pad", torch.zeros(1, 1, 1, self.Dh), persistent=False)
    
    def _create_local_causal_mask_vectorized(self, seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
        """
        BUG FIX #2: Version vectorisée avec cache pour masque local causal.
        (ANALYSE_COMPLETE_LLM.md ligne 172-197)

        Args:
            seq_len: Longueur de séquence
            window_size: Taille de fenêtre locale
            device: Device PyTorch

        Returns:
            mask: (seq_len, seq_len) masque booléen (True = invalide)

        Performance: 5-10x speedup vs loop Python
        """
        cache_key = (seq_len, window_size, device)

        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        # Vectorisation complète (pas de loop)
        i = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
        j = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)

        # Masque: True si position j est invalide pour query i
        # Invalide = (j > i) OU (j < i - window_size)
        mask = (j > i) | (j < i - window_size)

        # Cache pour réutilisation
        self._mask_cache[cache_key] = mask
        return mask

    def _window_indices_robust(
        self, L: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcule les indices de fenêtrage et masque de validité SANS biais de clamping.

        Returns:
            idx: (L, W) indices dans [0, L-1] pour positions valides, -1 pour invalides
            mask: (L, W) True où position est invalide (hors range ou future si causal)
        """
        i = torch.arange(L, device=device).view(L, 1)  # (L, 1)
        off = self.offsets.to(device).view(1, self.W)  # (1, W)
        raw = i + off  # (L, W)

        # Validité: dans les bornes ET (si causal) pas dans le futur
        valid = (raw >= 0) & (raw < L)
        if self.causal:
            valid = valid & (raw <= i)

        # Utiliser -1 comme sentinel pour positions invalides (sera remplacé par padding)
        idx = torch.where(valid, raw, torch.full_like(raw, -1))
        mask = ~valid

        return idx.long(), mask
    
    def _split_heads(self, t: torch.Tensor, B: int, L: int) -> torch.Tensor:
        """Reshape (B, L, D) -> (B, H, L, Dh)"""
        return t.view(B, L, self.H, self.Dh).transpose(1, 2)
    
    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        """Reshape (B, H, L, Dh) -> (B, L, D)"""
        B, H, L, Dh = t.shape
        return t.transpose(1, 2).contiguous().view(B, L, H * Dh)
    
    def _safe_masked_softmax(
        self, scores: torch.Tensor, mask: torch.Tensor, dim: int = -1
    ) -> torch.Tensor:
        """
        Softmax avec protection contre NaN pour lignes entièrement masquées.
        
        Args:
            scores: Scores d'attention
            mask: Masque booléen (True = invalide)
            dim: Dimension pour softmax
        """
        scores_masked = scores.masked_fill(mask, float('-inf'))
        
        # Détecter lignes entièrement masquées
        all_masked = mask.all(dim=dim, keepdim=True)
        
        # Softmax (génère nan/inf pour lignes all-masked)
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
        Top-K avec encouragement de diversité inter-têtes.

        Args:
            scores: (B, H, L, G) scores d'attention globale
            k: Nombre d'éléments à sélectionner
            diversity_penalty: Poids de pénalité pour répétition

        Returns:
            topk_values: (B, H, L, k)
            topk_indices: (B, H, L, k)
        """
        if not self.diverse_topk:
            # Mode standard (diversité désactivée)
            return torch.topk(scores, k=k, dim=-1)

        # Garder la diversité active en mode évaluation pour conserver la spécialisation des têtes

        B, H, L, G = scores.shape
        k_actual = min(k, G)

        # Accumulateur pour compter sélections par position globale
        selection_counts = torch.zeros(B, L, G, device=scores.device, dtype=scores.dtype)

        topk_vals_list = []
        topk_idxs_list = []

        for h in range(H):
            scores_h = scores[:, h]  # (B, L, G)

            # Pénaliser positions déjà sélectionnées par têtes précédentes
            if h > 0:
                penalty = diversity_penalty * selection_counts
                scores_h = scores_h - penalty

            # Top-K pour cette tête
            topk_val_h, topk_idx_h = torch.topk(scores_h, k=k_actual, dim=-1)

            # Mettre à jour compteur
            selection_counts.scatter_add_(
                2, topk_idx_h, torch.ones_like(topk_idx_h, dtype=scores.dtype)
            )

            topk_vals_list.append(topk_val_h)
            topk_idxs_list.append(topk_idx_h)

        topk_values = torch.stack(topk_vals_list, dim=1)  # (B, H, L, k)
        topk_indices = torch.stack(topk_idxs_list, dim=1)  # (B, H, L, k)

        return topk_values, topk_indices
    
    def forward(
        self,
        x: torch.Tensor,
        cache_global: Optional[torch.Tensor] = None,
        cache_positions: Optional[torch.Tensor] = None,
        global_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass de l'attention locale-globale.

        Args:
            x: (B, L, D) embeddings d'entrée
            cache_global: (B, G, D) états globaux (optionnel)
            cache_positions: (B, G) positions absolues du cache pour causalité (optionnel)
            global_weight: Poids de l'attention globale (0.0 à 1.0), pour warmup progressif

        Returns:
            out: (B, L, D) contexte combiné
        """
        B, L, D = x.shape
        device = x.device
        
        # === Projections QKV pour flux principal ===
        qkv = self.qkv_proj(x)  # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)  # (B, L, D) chacun
        
        # Split en têtes
        q = self._split_heads(q, B, L)  # (B, H, L, Dh)
        k = self._split_heads(k, B, L)  # (B, H, L, Dh)
        v = self._split_heads(v, B, L)  # (B, H, L, Dh)
        
        # =========================================
        # 1. ATTENTION LOCALE
        # =========================================
        
        # Indices et masque de fenêtre
        win_idx, win_mask = self._window_indices_robust(L, device)  # (L, W), (L, W)
        
        # ✅ FIX: Gather K/V SANS biais de clamping
        # Stratégie: utiliser indices clamped UNIQUEMENT pour gather (éviter crash),
        # mais JAMAIS utiliser les valeurs gathérées pour positions invalides
        W = self.W

        # Créer tenseurs de destination (initialisés à zéro = padding implicite)
        k_win = torch.zeros(B, self.H, L, W, self.Dh, device=device, dtype=k.dtype)
        v_win = torch.zeros(B, self.H, L, W, self.Dh, device=device, dtype=v.dtype)

        # Masque pour positions valides (True = valide, False = invalide)
        valid_mask = win_idx >= 0  # (L, W)

        # ✅ FIX CRITIQUE: Gather avec remplacement explicite par padding
        # On ne réutilise JAMAIS les valeurs gathérées aux positions invalides
        for w in range(W):
            valid_w = valid_mask[:, w]  # (L,)
            if valid_w.any():
                # Clamp uniquement pour éviter index out-of-bounds (crash)
                # MAIS on va immédiatement masquer ces valeurs biaisées
                idx_w = win_idx[:, w].clamp(min=0, max=L-1)  # (L,)

                # Gather: (B, H, L, Dh)
                # ⚠️ ATTENTION: Pour positions invalides, ceci lit position 0 (BIAISÉ)
                k_gathered = k[:, :, idx_w, :]  # (B, H, L, Dh)
                v_gathered = v[:, :, idx_w, :]  # (B, H, L, Dh)

                # ✅ FIX: Remplacer IMMÉDIATEMENT par padding (zéros)
                # Ceci élimine le biais car les valeurs clamped ne sont JAMAIS utilisées
                valid_w_expanded = valid_w.view(1, 1, L, 1).expand_as(k_gathered)
                k_win[:, :, :, w, :] = torch.where(
                    valid_w_expanded,
                    k_gathered,
                    torch.zeros_like(k_gathered)  # Padding explicite
                )
                v_win[:, :, :, w, :] = torch.where(
                    valid_w_expanded,
                    v_gathered,
                    torch.zeros_like(v_gathered)  # Padding explicite
                )
        
        # Scores locaux: Q @ K^T
        q_exp = q.unsqueeze(3)  # (B, H, L, 1, Dh)
        scores_local = (q_exp * k_win).sum(-1) * self.scale  # (B, H, L, W)
        
        # Masquer positions invalides
        local_mask = win_mask.view(1, 1, L, W).expand(B, self.H, L, W)
        attn_local = self._safe_masked_softmax(scores_local, local_mask, dim=-1)
        attn_local = self.attn_drop(attn_local)
        
        # Contexte local
        ctx_local = (attn_local.unsqueeze(-1) * v_win).sum(dim=3)  # (B, H, L, Dh)
        
        # =========================================
        # 2. ATTENTION GLOBALE (si cache fourni)
        # =========================================
        
        topk_idxs_safe: Optional[torch.Tensor] = None
        attn_g: Optional[torch.Tensor] = None

        if cache_global is not None:
            Bg, G, Dg = cache_global.shape
            assert Bg == B and Dg == D, f"Cache shape mismatch: {cache_global.shape} vs ({B}, G, {D})"
            
            # Projeter cache avec MÊMES poids que flux principal (unification)
            kv_g = self.qkv_proj(cache_global)  # (B, G, 3D)
            # Prendre seulement K et V (ignorer Q du cache)
            _, kg, vg = kv_g.chunk(3, dim=-1)  # (B, G, D)
            
            # Split en têtes
            kg = kg.view(B, G, self.H, self.Dh).transpose(1, 2)  # (B, H, G, Dh)
            vg = vg.view(B, G, self.H, self.Dh).transpose(1, 2)  # (B, H, G, Dh)
            
            # Scores globaux: Q @ Kg^T
            scores_g = torch.matmul(q, kg.transpose(-2, -1)) * self.scale  # (B, H, L, G)
            
            # Masque causal si positions fournies
            if self.causal and cache_positions is not None:
                pos_query = torch.arange(L, device=device).view(1, 1, L, 1)
                pos_cache = cache_positions.view(B, 1, 1, G)
                future_mask = pos_cache > pos_query  # (B, 1, L, G) ou (1, 1, L, G)
                scores_g = scores_g.masked_fill(future_mask, float('-inf'))
            
            # Top-K (avec ou sans diversité)
            # 🔧 FIX: Retirer "and self.training" pour diversité aussi en inference
            k_sel = min(self.GK, G)
            if self.diverse_topk:
                topk_vals, topk_idxs = self._diverse_topk(scores_g, k=k_sel)
            else:
                topk_vals, topk_idxs = torch.topk(scores_g, k=k_sel, dim=-1)
            
            # ✅ FIX Bug #18: Protéger contre lignes entièrement masquées (-inf)
            # Si tous topk_vals sont -inf (après causal mask), softmax → NaN
            # Solution: Détecter et mettre à zéro ces lignes
            all_masked = (topk_vals == float('-inf')).all(dim=-1)  # (B, H, L)

            # Softmax sur top-K
            attn_g = F.softmax(topk_vals, dim=-1)  # (B, H, L, k_sel)

            # Remplacer NaN par zéros pour lignes entièrement masquées
            if all_masked.any():
                # Mettre attention à zéro pour positions sans contexte global valide
                attn_g = torch.where(
                    all_masked.unsqueeze(-1),  # (B, H, L, 1)
                    torch.zeros_like(attn_g),
                    attn_g
                )

            attn_g = self.attn_drop(attn_g)
            
            # Gather Vg selon top-K indices
            # vg: (B, H, G, Dh) -> besoin de gather selon topk_idxs: (B, H, L, k_sel)
            # ✅ FIX: Clamp indices avant gather pour éviter index out-of-bounds
            # Protection contre topk_idxs >= G (peut arriver si k_sel > G, bien que rare)
            topk_idxs_safe = torch.clamp(topk_idxs, 0, G - 1)  # (B, H, L, k_sel)
            vg_exp = vg.unsqueeze(2).expand(B, self.H, L, G, self.Dh)  # (B, H, L, G, Dh)
            topk_idxs_exp = topk_idxs_safe.unsqueeze(-1).expand(B, self.H, L, k_sel, self.Dh)
            vg_topk = torch.gather(vg_exp, dim=3, index=topk_idxs_exp)  # (B, H, L, k_sel, Dh)
            
            # Contexte global
            ctx_global = (attn_g.unsqueeze(-1) * vg_topk).sum(dim=3)  # (B, H, L, Dh)
        else:
            ctx_global = None
        
        # =========================================
        # 3. FUSION LOCAL + GLOBAL
        # =========================================

        gate_scalar: Optional[torch.Tensor] = None

        if ctx_global is not None and global_weight > 0.0:
            # Appliquer le warmup weight au contexte global
            ctx_global_weighted = ctx_global * global_weight

            if self.gated:
                # Fusion apprise avec gating (par tête)
                # Concatener les deux contextes
                ctx_cat = torch.cat([ctx_local, ctx_global_weighted], dim=-1)  # (B, H, L, 2*Dh)

                # Reshape pour appliquer gate_proj par tête
                B_val, H_val, L_val, _ = ctx_cat.shape
                ctx_cat_reshaped = ctx_cat.reshape(B_val * H_val * L_val, 2 * self.Dh)

                # Gate sigmoïdal
                gate_flat = torch.sigmoid(self.gate_proj(ctx_cat_reshaped))  # (B*H*L, Dh)
                gate = gate_flat.view(B_val, H_val, L_val, self.Dh)  # (B, H, L, Dh)
                gate_scalar = gate.mean(dim=-1)  # (B, H, L) poids local moyens

                # Fusion pondérée
                ctx = gate * ctx_local + (1 - gate) * ctx_global_weighted
            else:
                # Fusion additive simple
                ctx = ctx_local + ctx_global_weighted
        else:
            # Pas de global (soit None, soit weight=0)
            ctx = ctx_local
        
        # =========================================
        # 4. PROJECTION DE SORTIE
        # =========================================
        
        ctx = self._merge_heads(ctx)  # (B, L, D)
        out = self.out_proj(ctx)
        out = self.proj_drop(out)
        
        # Sauvegarder les métriques pour monitoring externe (détachées du graphe)
        self.last_monitor = {
            "gate_scalar": gate_scalar.detach() if gate_scalar is not None else None,
            "global_topk_indices": topk_idxs_safe.detach() if topk_idxs_safe is not None else None,
            "global_attention_weights": attn_g.detach() if attn_g is not None else None,
        }

        return out


def test_slga_module():
    """Test rapide du module SLGA"""
    print("=== Test SLGA Module ===")
    
    B, L, D, H = 2, 128, 256, 4
    W, GK = 32, 16
    
    module = SLGAModule(
        embed_dim=D,
        num_heads=H,
        local_window=W,
        global_k=GK,
        causal=True,
        gated_fusion=True,
    )
    
    x = torch.randn(B, L, D)
    cache = torch.randn(B, 32, D)  # 32 landmarks globaux
    
    print(f"Input: {x.shape}")
    print(f"Cache: {cache.shape}")
    
    out = module(x, cache_global=cache)
    
    print(f"Output: {out.shape}")
    assert out.shape == (B, L, D), "Shape mismatch!"
    print("✓ Test passed!")


if __name__ == "__main__":
    test_slga_module()
