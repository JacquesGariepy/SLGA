"""
Transformer LLM avec Sparse Local-Global Attention (SLGA)

Architecture complète intégrant:
- SLGA module pour attention efficace
- Landmarks appris optionnels
- Fenêtres dilatées par couche
- Gradient checkpointing
- KV-cache pour génération
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


# ======================================================================
# Config
# ======================================================================

@dataclass
class Config:
    vocab_size: int = 128256          # LLaMA 3 vocab
    max_seq_len: int = 131072         # RoPE 128k
    embed_dim: int = 512
    num_heads: int = 8
    ff_hidden_multiplier: int = 4
    n_layers: int = 12
    dropout_rate: float = 0.1

    local_window: int = 128
    global_k: int = 24
    gated_fusion: bool = True
    learned_landmarks: bool = True
    dilated_windows: bool = True
    diverse_topk: bool = True

    landmark_selector: Optional[Dict[str, Any]] = None
    grad_checkpointing: bool = False  # dynamique


# ======================================================================
# RMSNorm
# ======================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# ======================================================================
# RoPE
# ======================================================================

class RotaryEmbedding(nn.Module):
    """
    RoPE appliqué sur un tenseur (B, L, D) avec D pair.
    position_ids: (B, L) contenant les positions entières.
    """

    def __init__(self, dim: int, max_seq_len: int = 131072, theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RotaryEmbedding requiert une dimension paire"
        self.dim = dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Précompute sur max_seq_len
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (max_seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)            # (max_seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        position_ids: (B, L) avec valeurs dans [0, max_seq_len)
        """
        assert x.dim() == 3, "RotaryEmbedding attend un tenseur (B, L, D)"
        B, L, D = x.shape
        assert D == self.dim, f"Dim RoPE incohérente: D={D}, attendu={self.dim}"
        assert position_ids.shape == (B, L), "position_ids doit être (B, L)"

        if position_ids.max() >= self.max_seq_len:
            raise ValueError(
                f"position_ids contient une position {int(position_ids.max().item())} "
                f"supérieure à max_seq_len={self.max_seq_len}"
            )

        cos = self.cos_cached[position_ids]  # (B, L, D)
        sin = self.sin_cached[position_ids]  # (B, L, D)

        half = D // 2
        x1, x2 = x[..., :half], x[..., half:]
        x_rot = torch.cat((-x2, x1), dim=-1)

        return x * cos + x_rot * sin


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
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx

        if cfg.dilated_windows:
            dilation_factor = 2 ** (layer_idx // max(1, cfg.n_layers // 3))
        else:
            dilation_factor = 1

        self.attn = SLGAModule(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            local_window=cfg.local_window,
            global_k=cfg.global_k,
            attn_drop=cfg.dropout_rate,
            proj_drop=cfg.dropout_rate,
            causal=True,
            gated_fusion=cfg.gated_fusion,
            dilation=dilation_factor,
            diverse_topk=cfg.diverse_topk,
        )

        self.ffn = FeedForward(cfg.embed_dim, cfg.ff_hidden_multiplier, cfg.dropout_rate)
        self.norm1 = RMSNorm(cfg.embed_dim)
        self.norm2 = RMSNorm(cfg.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        cache_global: Optional[torch.Tensor] = None,
        global_weight: float = 1.0,
    ) -> torch.Tensor:
        if self.cfg.grad_checkpointing and self.training:
            attn_out = checkpoint(
                self.attn,
                self.norm1(x),
                cache_global,
                global_weight,
                use_reentrant=False,
            )
        else:
            attn_out = self.attn(self.norm1(x), cache_global=cache_global, global_weight=global_weight)
        x = x + attn_out

        if self.cfg.grad_checkpointing and self.training:
            ffn_out = checkpoint(self.ffn, self.norm2(x), use_reentrant=False)
        else:
            ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out

        return x


# ======================================================================
# LLM Transformer
# ======================================================================

class LLMTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        # RoPE sur la dimension d embedding complète
        self.rope = RotaryEmbedding(cfg.embed_dim, max_seq_len=cfg.max_seq_len)
        self.emb_dropout = nn.Dropout(cfg.dropout_rate)

        if cfg.learned_landmarks:
            self.landmark_selector = LearnableLandmarkSelector(
                embed_dim=cfg.embed_dim,
                num_landmarks=cfg.global_k * 2,
            )
        else:
            self.landmark_selector = None

        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    # --------------------------------------------------------------
    # Init
    # --------------------------------------------------------------

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # --------------------------------------------------------------
    # RoPE helper pour QKV concat (facultatif)
    # --------------------------------------------------------------

    def apply_rope(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Applique RoPE sur un tenseur QKV concaténé.

        x: (B, L, 3 * D) avec D = cfg.embed_dim
        position_ids: (B, L)
        """
        B, L, threeD = x.shape
        D = self.cfg.embed_dim
        assert threeD == 3 * D, f"Shape inattendue pour x dans apply_rope: {x.shape}, attendu 3*D"
        assert position_ids.shape == (B, L), "position_ids doit être (B, L)"

        qkv = x.view(B, L, 3, D)              # (B, L, 3, D)
        qkv_flat = qkv.reshape(B * 3, L, D)   # (3B, L, D)

        # On répète les positions pour les 3 canaux (Q, K, V)
        pos_flat = position_ids.unsqueeze(2).expand(B, L, 3).reshape(B * 3, L)

        qkv_rot = self.rope(qkv_flat, pos_flat)  # (3B, L, D)
        return qkv_rot.reshape(B, L, 3 * D)

    # --------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_global_ids: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        global_weight: float = 1.0,
    ):
        """
        input_ids: (B, L) entiers dans [0, vocab_size)
        cache_global_ids: (B, G) indices pour les landmarks optionnels
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Sanity check sur la plage des ids
        min_id = int(input_ids.min().item())
        max_id = int(input_ids.max().item())
        assert min_id >= 0, f"input_ids contient des ids négatifs: min={min_id}"
        assert max_id < self.cfg.vocab_size, (
            f"input_ids contient des ids >= vocab_size: max={max_id}, vocab_size={self.cfg.vocab_size}\n"
            f"💡 Fix: Ensure tokenizer vocab_size matches model. Use len(tokenizer) instead of tokenizer.vocab_size.\n"
            f"   Some tokenizers (e.g., Qwen2) have special tokens that extend beyond vocab_size."
        )

        # Embedding
        x = self.token_emb(input_ids)  # (B, L, D)

        # Positions RoPE
        position_ids = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        x = self.rope(x, position_ids)  # (B, L, D)
        x = self.emb_dropout(x)

        # Landmarks
        landmark_indices = None
        landmark_scores = None

        if self.landmark_selector is not None:
            landmark_indices, _, landmark_scores = self.landmark_selector(x, use_gumbel=self.training)

            # Nettoyage agressif
            L_cur = x.size(1)
            landmark_indices = landmark_indices.nan_to_num_(nan=0.0)
            landmark_indices = landmark_indices.clamp_(0, L_cur - 1)
            landmark_indices = landmark_indices.long()

            min_l = int(landmark_indices.min().item())
            max_l = int(landmark_indices.max().item())
            assert 0 <= min_l <= max_l < L_cur, (
                f"landmark_indices hors bornes: [{min_l}, {max_l}] vs L_cur={L_cur}"
            )

        elif cache_global_ids is not None:
            # Landmarks fournis par l utilisateur ou le cache
            landmark_indices = cache_global_ids
            L_cur = x.size(1)
            landmark_indices = landmark_indices.clamp(0, L_cur - 1).long()

            min_l = int(landmark_indices.min().item())
            max_l = int(landmark_indices.max().item())
            assert 0 <= min_l <= max_l < L_cur, (
                f"cache_global_ids hors bornes: [{min_l}, {max_l}] vs L_cur={L_cur}"
            )

        gate_values_layers = []

        # Pile de blocs
        for block in self.blocks:
            if landmark_indices is not None:
                B_cur, L_cur, D = x.shape
                G = landmark_indices.size(1)

                if G > L_cur:
                    # Limitation safe
                    landmark_indices = torch.arange(L_cur, device=x.device).unsqueeze(0).expand(B_cur, L_cur)
                    G = L_cur

                min_l = int(landmark_indices.min().item())
                max_l = int(landmark_indices.max().item())
                assert 0 <= min_l <= max_l < L_cur, (
                    f"[block {block.layer_idx}] landmark_indices hors bornes: "
                    f"[{min_l}, {max_l}] vs L_cur={L_cur}"
                )

                indices_exp = landmark_indices.unsqueeze(-1).expand(B_cur, G, D)  # (B, G, D)
                landmark_states = torch.gather(x, dim=1, index=indices_exp)      # (B, G, D)
            else:
                landmark_states = None

            x = block(x, cache_global=landmark_states, global_weight=global_weight)

            monitor = getattr(block.attn, "last_monitor", None)
            if monitor and monitor.get("gate_scalar") is not None:
                gate_values_layers.append(monitor["gate_scalar"])

        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, L, V)

        if return_aux:
            aux = {"landmark_indices": landmark_indices, "landmark_scores": landmark_scores}
            if gate_values_layers:
                aux["gate_values"] = torch.stack(gate_values_layers, dim=0).detach()
            return logits, aux

        return logits

    # --------------------------------------------------------------
    # Generation
    # --------------------------------------------------------------

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
        eos_token_id: int = 50256,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generation autoregressive avec contrôle EOS et anti répétition.

        input_ids: (B, L)
        retourne: (B, L + tokens générés)
        """
        self.eval()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        batch_size = input_ids.size(0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        use_no_repeat = no_repeat_ngram_size is not None and no_repeat_ngram_size > 0
        user_provided_landmarks = cache_global_ids is not None

        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.cfg.max_seq_len:
                input_ids = input_ids[:, -self.cfg.max_seq_len:]

            if not self.cfg.learned_landmarks and not user_provided_landmarks:
                L = input_ids.size(1)
                landmark_positions = torch.linspace(0, L - 1, self.cfg.global_k, device=input_ids.device).long()
                cache_global_ids = landmark_positions.unsqueeze(0).expand(input_ids.size(0), -1)

            logits = self(input_ids, cache_global_ids=cache_global_ids)  # (B, L, V)
            logits = logits[:, -1, :]  # (B, V)

            if stop_on_eos and finished.any():
                logits[finished] = float("-inf")
                logits[finished, eos_token_id] = 1e4

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
                        banned_indices = torch.tensor(list(banned_tokens), device=logits.device, dtype=torch.long)
                        logits[b_idx, banned_indices] = float("-inf")

            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                if temperature != 1.0:
                    logits = logits / temperature

                if top_k is not None and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    topk_vals, topk_idxs = torch.topk(logits, k=k, dim=-1)
                    logits_filtered = torch.full_like(logits, float("-inf"))
                    logits_filtered.scatter_(1, topk_idxs, topk_vals)
                    logits = logits_filtered

                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    sorted_logits[sorted_indices_to_remove] = float("-inf")
                    logits = logits.scatter(1, sorted_indices, sorted_logits)

                probs = F.softmax(logits, dim=-1)

                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    probs = torch.ones_like(probs) / probs.size(-1)

                probs = torch.clamp(probs, min=1e-10)
                probs = probs / probs.sum(dim=-1, keepdim=True)

                next_token = torch.multinomial(probs, num_samples=1)

            if stop_on_eos and finished.any():
                next_token[finished] = eos_token_id

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if stop_on_eos:
                eos_mask = (next_token.squeeze(-1) == eos_token_id)
                finished = finished | eos_mask
                if finished.all():
                    break

        if stop_on_eos:
            for b_idx in range(input_ids.size(0)):
                tokens = input_ids[b_idx]
                eos_positions = (tokens == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    first_eos_pos = int(eos_positions[0].item())
                    input_ids[b_idx, first_eos_pos + 1:] = eos_token_id

        return input_ids

    # --------------------------------------------------------------
    # Utilitaires
    # --------------------------------------------------------------

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_emb.weight.numel()
        return n_params

__all__ = ["Config", "LLMTransformer", "TransformerBlock", "FeedForward", "RMSNorm"]
