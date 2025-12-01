"""
Transformer LLM avec Sparse Local-Global Attention (SLGA)
Architecture:
- SLGA module pour attention efficace (locale + globale sparse)
- Landmarks appris via Gumbel-Softmax différentiable
- Fenêtres dilatées par couche
- Gradient checkpointing dynamique
- Weight tying embedding/lm_head
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .slga import SLGAModule
from .landmarks import LearnableLandmarkSelector
from .kv_cache import KVCache


# ======================================================================
# Config
# ======================================================================

@dataclass
class Config:
    """Configuration du modèle SLGA."""
    vocab_size: int = 128256          # LLaMA 3 vocab (sera override par tokenizer)
    max_seq_len: int = 131072         # RoPE 128k context
    embed_dim: int = 512
    num_heads: int = 8
    ff_hidden_multiplier: int = 4
    n_layers: int = 12
    dropout_rate: float = 0.1

    # SLGA config
    local_window: int = 128
    global_k: int = 24
    gated_fusion: bool = True
    learned_landmarks: bool = True
    dilated_windows: bool = True
    diverse_topk: bool = True

    # GQA (Grouped Query Attention) config
    num_kv_heads: Optional[int] = None  # If None, use num_heads (standard MHA)
                                        # For GQA: set < num_heads (e.g., num_heads=8, num_kv_heads=2)
                                        # Memory savings: ~(1 - num_kv_heads/num_heads) * 66%

    # Optionnel
    landmark_selector: Optional[Dict[str, Any]] = None
    grad_checkpointing: bool = False  # Dynamique selon seq_len


# ======================================================================
# RMSNorm
# ======================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Plus efficace que LayerNorm car pas de centrage (mean=0).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# ======================================================================
# Feed Forward
# ======================================================================

class FeedForward(nn.Module):
    """
    Standard Feed-Forward Network (FFN) de transformer.
    Deux linéaires avec GELU et dropout.
    """

    def __init__(self, embed_dim: int, hidden_multiplier: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = embed_dim * hidden_multiplier
        self.fc1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ======================================================================
# Transformer Block
# ======================================================================

class TransformerBlock(nn.Module):
    """
    Bloc Transformer avec SLGA (attention sparse locale-globale).
    
    Inclut:
    - Pre-norm (RMSNorm avant attention et FFN)
    - Résiduel connections
    - Gradient checkpointing optionnel
    """

    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx

        # Dilation progressive par couche
        if cfg.dilated_windows:
            # Couches 0-3: dilation=1, 4-7: dilation=2, 8-11: dilation=4
            dilation_factor = 2 ** (layer_idx // max(1, cfg.n_layers // 3))
        else:
            dilation_factor = 1

        # SLGA avec RoPE interne et GQA support
        self.attn = SLGAModule(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,  # GQA support
            local_window=cfg.local_window,
            global_k=cfg.global_k,
            attn_drop=cfg.dropout_rate,
            proj_drop=cfg.dropout_rate,
            causal=True,
            gated_fusion=cfg.gated_fusion,
            dilation=dilation_factor,
            diverse_topk=cfg.diverse_topk,
            max_seq_len=cfg.max_seq_len,  # Pour RoPE interne
        )

        self.ffn = FeedForward(cfg.embed_dim, cfg.ff_hidden_multiplier, cfg.dropout_rate)
        self.norm1 = RMSNorm(cfg.embed_dim)
        self.norm2 = RMSNorm(cfg.embed_dim)

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
        Forward pass du bloc.

        Args:
            x: (B, L, D) hidden states
            position_ids: (B, L) positions pour RoPE
            cache_global: (B, G, D) états des landmarks
            cache_positions: (B, G) positions originales des landmarks
            global_weight: poids pour warmup progressif de l'attention globale
            kv_cache: Optional cached K, V from previous step
            use_cache: If True, return updated K, V for caching

        Returns:
            x: (B, L, D) output hidden states
            new_kv_cache: Optional updated cache if use_cache=True
        """
        # Attention avec gradient checkpointing optionnel
        # Note: KV cache disabled during gradient checkpointing
        if self.cfg.grad_checkpointing and self.training:
            attn_out = checkpoint(
                self._attn_forward,
                self.norm1(x),
                position_ids,
                cache_global,
                cache_positions,
                global_weight,
                use_reentrant=False,
            )
            new_kv_cache = None
        else:
            attn_out, new_kv_cache = self.attn(
                self.norm1(x),
                position_ids=position_ids,
                cache_global=cache_global,
                cache_positions=cache_positions,
                global_weight=global_weight,
                kv_cache=kv_cache,
                use_cache=use_cache,
            )
        x = x + attn_out

        # FFN avec gradient checkpointing optionnel
        if self.cfg.grad_checkpointing and self.training:
            ffn_out = checkpoint(self.ffn, self.norm2(x), use_reentrant=False)
        else:
            ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out

        return x, new_kv_cache

    def _attn_forward(
        self,
        x_normed: torch.Tensor,
        position_ids: torch.Tensor,
        cache_global: Optional[torch.Tensor],
        cache_positions: Optional[torch.Tensor],
        global_weight: float,
    ) -> torch.Tensor:
        """Helper pour checkpoint - ne peut pas passer des kwargs."""
        out, _ = self.attn(
            x_normed,
            position_ids=position_ids,
            cache_global=cache_global,
            cache_positions=cache_positions,
            global_weight=global_weight,
            kv_cache=None,
            use_cache=False,
        )
        return out


# ======================================================================
# LLM Transformer
# ======================================================================

class LLMTransformer(nn.Module):
    """
    Transformer LLM complet avec SLGA.
    
    Architecture:
    - Token embedding (weight-tied avec lm_head)
    - N blocs Transformer avec SLGA
    - Landmark selector appris (optionnel)
    - RMSNorm final
    - LM head pour prédiction
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Embeddings
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.emb_dropout = nn.Dropout(cfg.dropout_rate)

        # Landmark selector (si learned_landmarks=True)
        if cfg.learned_landmarks:
            self.landmark_selector = LearnableLandmarkSelector(
                embed_dim=cfg.embed_dim,
                num_landmarks=cfg.global_k * 2,  # 2× pour marge
            )
        else:
            self.landmark_selector = None

        # Blocs Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, i) for i in range(cfg.n_layers)
        ])

        # Output
        self.final_norm = RMSNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

        # Weight tying: embedding et lm_head partagent les poids
        self.lm_head.weight = self.token_emb.weight

        # Initialisation
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialisation des poids (style GPT-2/LLaMA)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_global_ids: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        global_weight: float = 1.0,
    ):
        """
        Forward pass du modèle.
        
        Args:
            input_ids: (B, L) token IDs
            cache_global_ids: (B, G) indices des landmarks (si fournis manuellement)
            return_aux: si True, retourne aussi les infos auxiliaires
            global_weight: poids pour warmup progressif (0→1)
        
        Returns:
            logits: (B, L, V) logits de prédiction
            aux: dict avec landmark_indices, landmark_scores, gate_values (si return_aux)
        """
        B, L = input_ids.shape
        device = input_ids.device

        # === Validation des input_ids ===
        min_id = int(input_ids.min().item())
        max_id = int(input_ids.max().item())
        if min_id < 0:
            raise ValueError(f"input_ids contient des ids négatifs: min={min_id}")
        if max_id >= self.cfg.vocab_size:
            raise ValueError(
                f"input_ids contient des ids >= vocab_size: max={max_id}, "
                f"vocab_size={self.cfg.vocab_size}\n"
                f"💡 Fix: Ensure tokenizer vocab_size matches model. "
                f"Use len(tokenizer) instead of tokenizer.vocab_size."
            )

        # === Embedding (SANS RoPE - RoPE est dans SLGAModule) ===
        x = self.token_emb(input_ids)  # (B, L, D)
        x = self.emb_dropout(x)

        # === Positions pour RoPE (passées aux blocs) ===
        position_ids = torch.arange(L, device=device, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)  # (B, L)

        # === Sélection des Landmarks ===
        landmark_indices = None
        landmark_scores = None

        if self.landmark_selector is not None:
            # Landmarks appris
            landmark_indices, _, landmark_scores = self.landmark_selector(
                x, use_gumbel=self.training
            )

            # Nettoyage des indices
            L_cur = x.size(1)
            landmark_indices = landmark_indices.nan_to_num_(nan=0.0)
            landmark_indices = landmark_indices.clamp_(0, L_cur - 1)
            landmark_indices = landmark_indices.long()

        elif cache_global_ids is not None:
            # Landmarks fournis par l'utilisateur
            landmark_indices = cache_global_ids
            L_cur = x.size(1)
            landmark_indices = landmark_indices.clamp(0, L_cur - 1).long()

        # === Pré-validation des landmarks (hors boucle) ===
        if landmark_indices is not None:
            G = landmark_indices.size(1)
            L_cur = x.size(1)
            if G > L_cur:
                # Fallback: utiliser toutes les positions
                landmark_indices = torch.arange(
                    L_cur, device=device
                ).unsqueeze(0).expand(B, L_cur)

        # === Pile de blocs Transformer ===
        gate_values_layers = []

        for block in self.blocks:
            # Extraire les états des landmarks (actualisés à chaque couche)
            if landmark_indices is not None:
                B_cur, L_cur, D = x.shape
                G = landmark_indices.size(1)

                # Gather des états aux positions des landmarks
                indices_exp = landmark_indices.unsqueeze(-1).expand(B_cur, G, D)
                landmark_states = torch.gather(x, dim=1, index=indices_exp)

                # Positions originales des landmarks (pour RoPE)
                landmark_positions = landmark_indices  # (B, G)
            else:
                landmark_states = None
                landmark_positions = None

            # Forward du bloc
            x, _ = block(
                x,
                position_ids=position_ids,
                cache_global=landmark_states,
                cache_positions=landmark_positions,
                global_weight=global_weight,
                kv_cache=None,
                use_cache=False,
            )

            # Récupérer les gate values pour monitoring
            monitor = getattr(block.attn, "last_monitor", None)
            if monitor and monitor.get("gate_scalar") is not None:
                gate_values_layers.append(monitor["gate_scalar"])

        # === Output ===
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, L, V)

        if return_aux:
            aux = {
                "landmark_indices": landmark_indices,
                "landmark_scores": landmark_scores,
            }
            if gate_values_layers:
                aux["gate_values"] = torch.stack(gate_values_layers, dim=0).detach()
            return logits, aux

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        cache_global_ids: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        stop_on_eos: bool = True,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: Optional[int] = None,
        use_kv_cache: bool = True,
    ) -> torch.Tensor:
        """
        Génération autoregressive.

        Args:
            input_ids: (B, L) prompt tokens
            max_new_tokens: nombre max de tokens à générer
            temperature: température d'échantillonnage (0 = greedy)
            top_k: filtrage top-K
            top_p: filtrage nucleus (top-P)
            cache_global_ids: landmarks fixes (sinon appris)
            seed: graine aléatoire
            stop_on_eos: arrêter sur EOS
            eos_token_id: ID du token EOS (REQUIS si stop_on_eos=True)
            repetition_penalty: pénalité de répétition
            no_repeat_ngram_size: interdire les n-grams répétés
            use_kv_cache: Si True, utilise KV cache pour accélération (défaut: True)

        Returns:
            tokens: (B, L + generated) séquence complète
        """
        if stop_on_eos and eos_token_id is None:
            raise ValueError(
                "eos_token_id must be provided when stop_on_eos=True. "
                "Pass tokenizer.eos_token_id explicitly."
            )

        self.eval()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        batch_size = input_ids.size(0)
        device = input_ids.device
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        use_no_repeat = no_repeat_ngram_size is not None and no_repeat_ngram_size > 0
        user_provided_landmarks = cache_global_ids is not None

        # Initialize KV cache if enabled
        kv_cache: Optional[KVCache] = None
        if use_kv_cache:
            kv_cache = KVCache.create(
                batch_size=batch_size,
                num_layers=self.cfg.n_layers,
                num_heads=self.cfg.num_heads,
                head_dim=self.cfg.embed_dim // self.cfg.num_heads,
                max_seq_len=self.cfg.max_seq_len,
                device=device,
                dtype=next(self.parameters()).dtype,
            )

        # First pass: process entire prompt
        current_input = input_ids

        for step_idx in range(max_new_tokens):
            # Truncate si trop long
            if input_ids.size(1) > self.cfg.max_seq_len:
                input_ids = input_ids[:, -self.cfg.max_seq_len:]
                # Cache invalidated if we truncate
                if kv_cache is not None:
                    kv_cache.clear()
                current_input = input_ids

            # Landmarks heuristiques si pas de selector appris
            if not self.cfg.learned_landmarks and not user_provided_landmarks:
                L = input_ids.size(1)
                landmark_positions = torch.linspace(
                    0, L - 1, self.cfg.global_k, device=device
                ).long()
                cache_global_ids = landmark_positions.unsqueeze(0).expand(batch_size, -1)

            # Forward pass with KV cache
            if use_kv_cache and kv_cache is not None and step_idx > 0:
                # After first step: only process last token
                logits = self._forward_with_cache(
                    current_input, cache_global_ids, kv_cache
                )
            else:
                # First step: process entire prompt
                logits = self(current_input, cache_global_ids=cache_global_ids)

            logits = logits[:, -1, :]  # (B, V)

            # Forcer EOS pour les séquences terminées
            if stop_on_eos and finished.any():
                logits[finished] = float("-inf")
                logits[finished, eos_token_id] = 1e4

            # Repetition penalty
            if repetition_penalty is not None and repetition_penalty != 1.0:
                penalty = max(repetition_penalty, 1e-6)
                for b_idx in range(logits.size(0)):
                    seen_tokens = input_ids[b_idx].unique()
                    if seen_tokens.numel() == 0:
                        continue
                    token_logits = logits[b_idx, seen_tokens]
                    penalized = torch.where(
                        token_logits < 0,
                        token_logits * penalty,
                        token_logits / penalty,
                    )
                    logits[b_idx, seen_tokens] = penalized

            # No-repeat n-gram
            if use_no_repeat:
                n_size = no_repeat_ngram_size
                for b_idx in range(logits.size(0)):
                    prev_tokens = input_ids[b_idx]
                    seq_len = prev_tokens.size(0)
                    if seq_len < n_size:
                        continue

                    ngram_dict: Dict[Tuple[int, ...], set] = {}
                    prev_tokens_list = prev_tokens.tolist()
                    for idx in range(seq_len - n_size + 1):
                        ngram = tuple(prev_tokens_list[idx: idx + n_size])
                        prefix = ngram[:-1]
                        next_tok = ngram[-1]
                        if prefix not in ngram_dict:
                            ngram_dict[prefix] = set()
                        ngram_dict[prefix].add(next_tok)

                    if n_size > 1:
                        current_prefix = tuple(prev_tokens_list[-(n_size - 1):])
                    else:
                        current_prefix = tuple()

                    banned_tokens = ngram_dict.get(current_prefix, set())
                    if banned_tokens:
                        banned_indices = torch.tensor(
                            list(banned_tokens), device=device, dtype=torch.long
                        )
                        logits[b_idx, banned_indices] = float("-inf")

            # Sampling
            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                if temperature != 1.0:
                    logits = logits / temperature

                # Top-K filtering
                if top_k is not None and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    topk_vals, topk_idxs = torch.topk(logits, k=k, dim=-1)
                    logits_filtered = torch.full_like(logits, float("-inf"))
                    logits_filtered.scatter_(1, topk_idxs, topk_vals)
                    logits = logits_filtered

                # Top-P (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    sorted_logits[sorted_indices_to_remove] = float("-inf")
                    logits = logits.scatter(1, sorted_indices, sorted_logits)

                probs = F.softmax(logits, dim=-1)

                # Protection NaN
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    probs = torch.ones_like(probs) / probs.size(-1)

                probs = torch.clamp(probs, min=1e-10)
                probs = probs / probs.sum(dim=-1, keepdim=True)

                next_token = torch.multinomial(probs, num_samples=1)

            # Forcer EOS pour séquences terminées
            if stop_on_eos and finished.any():
                next_token[finished] = eos_token_id

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
            current_input = next_token  # For next iteration with cache

            # Check EOS
            if stop_on_eos:
                eos_mask = (next_token.squeeze(-1) == eos_token_id)
                finished = finished | eos_mask
                if finished.all():
                    break

        # Cleanup: remplacer tout après premier EOS par EOS
        if stop_on_eos:
            for b_idx in range(input_ids.size(0)):
                tokens = input_ids[b_idx]
                eos_positions = (tokens == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    first_eos_pos = int(eos_positions[0].item())
                    input_ids[b_idx, first_eos_pos + 1:] = eos_token_id

        return input_ids

    def _forward_with_cache(
        self,
        input_ids: torch.Tensor,
        cache_global_ids: Optional[torch.Tensor],
        kv_cache: KVCache,
    ) -> torch.Tensor:
        """
        Forward pass with KV cache for autoregressive generation.

        Only processes new tokens and uses cached K, V from previous steps.

        Args:
            input_ids: (B, new_L) new token IDs (typically new_L=1)
            cache_global_ids: (B, G) landmark indices
            kv_cache: KVCache object with previous K, V

        Returns:
            logits: (B, new_L, V) logits for new tokens
        """
        B, new_L = input_ids.shape
        device = input_ids.device

        # === Validation ===
        min_id = int(input_ids.min().item())
        max_id = int(input_ids.max().item())
        if min_id < 0:
            raise ValueError(f"input_ids contient des ids négatifs: min={min_id}")
        if max_id >= self.cfg.vocab_size:
            raise ValueError(
                f"input_ids contient des ids >= vocab_size: max={max_id}, "
                f"vocab_size={self.cfg.vocab_size}"
            )

        # === Embedding ===
        x = self.token_emb(input_ids)  # (B, new_L, D)
        x = self.emb_dropout(x)

        # === Positions pour RoPE ===
        # Positions absolues basées sur la longueur du cache
        start_pos = kv_cache.seq_len
        position_ids = torch.arange(
            start_pos, start_pos + new_L, device=device, dtype=torch.long
        )
        position_ids = position_ids.unsqueeze(0).expand(B, -1)  # (B, new_L)

        # === Sélection des Landmarks ===
        # Note: Pour la génération avec cache, on réutilise les landmarks
        # de la séquence originale (landmarks fixes pendant génération)
        landmark_indices = cache_global_ids
        landmark_positions = cache_global_ids if cache_global_ids is not None else None

        # === Pile de blocs avec KV cache ===
        for layer_idx, block in enumerate(self.blocks):
            # Récupérer le cache pour cette couche
            layer_kv_cache = kv_cache.get(layer_idx)

            # Extraire les états des landmarks si nécessaire
            # Note: En génération avec cache, les landmarks sont fixes
            # On ne les réextrait pas à chaque step
            landmark_states = None
            if landmark_indices is not None:
                # Pour simplification, on peut désactiver global attention
                # pendant génération cached (optionnel)
                pass

            # Forward du bloc avec cache
            x, new_kv = block(
                x,
                position_ids=position_ids,
                cache_global=landmark_states,
                cache_positions=landmark_positions,
                global_weight=1.0,
                kv_cache=layer_kv_cache,
                use_cache=True,
            )

            # Mettre à jour le cache pour cette couche
            if new_kv is not None:
                new_k, new_v = new_kv
                # Extraire seulement les nouvelles K, V (derniers new_L tokens)
                new_k_only = new_k[:, :, -new_L:, :]
                new_v_only = new_v[:, :, -new_L:, :]
                kv_cache.update(layer_idx, new_k_only, new_v_only)

        # Incrémenter la longueur du cache après tous les layers
        kv_cache.increment_seq_len(new_L)

        # === Output ===
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, new_L, V)

        return logits

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Compte le nombre de paramètres.
        
        Args:
            non_embedding: Si True, exclut les embeddings du compte.
                          Par défaut False = total params.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_emb.weight.numel()
        return n_params


__all__ = ["Config", "LLMTransformer", "TransformerBlock", "FeedForward", "RMSNorm", "KVCache"]