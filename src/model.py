# model.py
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from torch.utils.checkpoint import checkpoint

from .slga import SLGAModule
from .landmarks import LearnableLandmarkSelector


@dataclass
class Config:
    vocab_size: int = 128256          # LLaMA-3 vocab
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
    grad_checkpointing: bool = False  # Dynamique maintenant

# ====================== RMSNorm ======================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weightc

# ====================== RoPE ======================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 131072, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self.cached_freqs = None

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # x: (B, L, H, Dh) or (B, L, D)
        if x.dim() == 4:  # (B, H, L, Dh)
            seq_len = x.size(2)
        else:
            seq_len = x.size(1)

        if self.cached_freqs is None or self.cached_freqs.size(0) < seq_len:
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.device))
            emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
            self.register_buffer("cached_cos", emb.cos(), persistent=False)
            self.register_buffer("cached_sin", emb.sin(), persistent=False)
        cos = self.cached_cos[:seq_len]
        sin = self.cached_sin[:seq_len]

        if x.dim() == 4:
            x1, x2 = x[..., : x.shape[-1]//2], x[..., x.shape[-1]//2 :]
            x_rot = torch.cat((-x2, x1), dim=-1)
            x_out = x * cos + x_rot * sin
        else:
            x1, x2 = x[..., : x.shape[-1]//2], x[..., x.shape[-1]//2 :]
            x_rot = torch.cat((-x2, x1), dim=-1)
            x_out = x * cos.unsqueeze(0) + x_rot * sin.unsqueeze(0)
        return x_out.float()

class FeedForward(nn.Module):
    """
    Standard Feed-Forward Network (FFN) used in transformer blocks.
    Consists of two linear layers with GELU activation and dropout.
    """
    def __init__(self, embed_dim: int, hidden_multiplier: int = 4, dropout: float = 0.1):
        super().__init__()
        # Hidden layer dimension is typically 4x the embedding dimension
        hidden_dim = embed_dim * hidden_multiplier
        # First linear layer projects from embedding dimension to hidden dimension
        self.fc1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        # Second linear layer projects back to embedding dimension
        self.fc2 = nn.Linear(hidden_dim, embed_dim, bias=False)
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FFN.
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        # Project input to hidden dimension
        x = self.fc1(x)
        # Apply GELU non-linearity
        x = F.gelu(x)
        # Project back to embedding dimension
        x = self.fc2(x)
        # Apply dropout
        x = self.dropout(x)
        return x


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

    def forward(self, x: torch.Tensor, cache_global: Optional[torch.Tensor] = None, global_weight: float = 1.0):
        if self.cfg.grad_checkpointing and self.training:
            attn_out = checkpoint(self.attn, self.norm1(x), cache_global, global_weight, use_reentrant=False)
        else:
            attn_out = self.attn(self.norm1(x), cache_global=cache_global, global_weight=global_weight)
        x = x + attn_out

        if self.cfg.grad_checkpointing and self.training:
            ffn_out = checkpoint(self.ffn, self.norm2(x), use_reentrant=False)
        else:
            ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x
        
class LLMTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.rope = RotaryEmbedding(cfg.embed_dim // cfg.num_heads)
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

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def apply_rope(self, x: torch.Tensor, position_ids: torch.Tensor):
        qkv = x.unflatten(-1, (3, self.cfg.embed_dim)).transpose(1, 2)  # simplifié pour SLGA
        return self.rope(qkv, position_ids).flatten(-2).transpose(1, 2)

    def forward(self, input_ids: torch.Tensor, cache_global_ids=None, return_aux=False, global_weight=1.0):
        B, L = input_ids.shape
        device = input_ids.device

        x = self.token_emb(input_ids)
        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = self.rope(x, position_ids)
        x = self.emb_dropout(x)

        # Landmark selection (inchangé)
        landmark_indices = None
        landmark_scores = None
        if self.landmark_selector is not None:
            landmark_indices, _, landmark_scores = self.landmark_selector(x, use_gembel=self.training)
        elif cache_global_ids is not None:
            landmark_indices = cache_global_ids

        gate_values_layers = []
        for block in self.blocks:
            if landmark_indices is not None:
                B_cur, L_cur, D = x.shape
                G = landmark_indices.size(1)
                idx_safe = torch.clamp(landmark_indices, 0, L_cur - 1)
                idx_exp = idx_safe.unsqueeze(-1).expand(B_cur, G, D)
                landmark_states = torch.gather(x, dim=1, index=idx_exp)
            else:
                landmark_states = None

            x = block(x, cache_global=landmark_states, global_weight=global_weight)
            monitor = getattr(block.attn, "last_monitor", None)
            if monitor and monitor.get("gate_scalar") is not None:
                gate_values_layers.append(monitor["gate_scalar"])

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if return_aux:
            aux = {"landmark_indices": landmark_indices, "landmark_scores": landmark_scores}
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
        eos_token_id: int = 50256,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with deterministic and corrected EOS handling.
        
        Args:
            input_ids: Prompt token IDs of shape (batch, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = greedy, >0.0 = stochastic)
            top_k: Top-K filtering threshold
            top_p: Nucleus (top-P) filtering threshold
            cache_global_ids: User-provided landmark indices (if learned=False)
            seed: Random seed for reproducibility
            stop_on_eos: Stop when all sequences generate EOS
            eos_token_id: EOS token ID (default: GPT-2 EOS)
            repetition_penalty: Penalty for repeating tokens (>1.0 penalizes repeats)
            no_repeat_ngram_size: Size of n-grams to avoid repeating
        
        Returns:
            Generated token IDs of shape (batch, seq_len + generated_tokens)
        """
        # Force evaluation mode
        self.eval()

        # Set random seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Batch-aware EOS tracking: track which sequences have finished
        batch_size = input_ids.size(0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        # Precompute for no-repeat n-gram blocking
        use_no_repeat = no_repeat_ngram_size is not None and no_repeat_ngram_size > 0

        # Preserve user-provided landmarks to avoid overwriting during generation
        user_provided_landmarks = cache_global_ids is not None

        for step in range(max_new_tokens):
            # Truncate if sequence exceeds max length
            if input_ids.size(1) > self.cfg.max_seq_len:
                input_ids = input_ids[:, -self.cfg.max_seq_len:]
            
            # Recalculate heuristic landmarks for each step (unless user provided)
            if not self.cfg.learned_landmarks and not user_provided_landmarks:
                L = input_ids.size(1)
                # Use linspace to ensure exactly global_k evenly spaced landmarks
                landmark_positions = torch.linspace(0, L-1, self.cfg.global_k, device=input_ids.device).long()
                cache_global_ids = landmark_positions.unsqueeze(0).expand(input_ids.size(0), -1)
            
            # Forward pass to get logits
            logits = self(input_ids, cache_global_ids=cache_global_ids)  # (B, L, V)
            
            # Take logits for the last token
            logits = logits[:, -1, :]  # (B, V)

            # Freeze finished sequences to only generate EOS
            if stop_on_eos and finished.any():
                logits[finished] = float('-inf')
                logits[finished, eos_token_id] = 1e4  # Force EOS

            # Anti-repetition controls before sampling
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

                    ngram_dict = {}
                    prev_tokens_list = prev_tokens.tolist()
                    for idx in range(seq_len - n_size + 1):
                        ngram = tuple(prev_tokens_list[idx : idx + n_size])
                        prefix = ngram[:-1]
                        next_token = ngram[-1]
                        if prefix not in ngram_dict:
                            ngram_dict[prefix] = set()
                        ngram_dict[prefix].add(next_token)

                    current_prefix = tuple(prev_tokens_list[-(n_size - 1) :]) if n_size > 1 else tuple()
                    banned_tokens = ngram_dict.get(current_prefix, set())
                    if banned_tokens:
                        banned_indices = torch.tensor(list(banned_tokens), device=logits.device, dtype=torch.long)
                        logits[b_idx, banned_indices] = float('-inf')

            # Correct order: Temperature → Top-K → Top-P → Sample
            if temperature == 0.0:
                # Greedy sampling: deterministic selection of best token
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
            else:
                # Stochastic sampling
                # 1. Apply temperature scaling
                if temperature != 1.0:
                    logits = logits / temperature

                # 2. Top-K filtering on temperature-scaled logits
                if top_k is not None and top_k > 0:
                    topk_vals, topk_idxs = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered.scatter_(1, topk_idxs, topk_vals)
                    logits = logits_filtered

                # 3. Top-P (nucleus) filtering on temperature-scaled logits
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens above cumulative probability threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False  # Keep at least the best token

                    sorted_logits[sorted_indices_to_remove] = float('-inf')
                    logits = logits.scatter(1, sorted_indices, sorted_logits)
                
                # Sample with NaN protection
                probs = F.softmax(logits, dim=-1)
                
                # Fallback to uniform if all logits are -inf
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    probs = torch.ones_like(probs) / probs.size(-1)
                
                # Clamp and renormalize probabilities
                probs = torch.clamp(probs, min=1e-10)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Replace finished sequences with EOS
            if stop_on_eos and finished.any():
                next_token[finished] = eos_token_id

            # Append new token to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Batch-aware EOS stopping
            if stop_on_eos:
                eos_mask = (next_token.squeeze(-1) == eos_token_id)  # (B,)
                finished = finished | eos_mask

                # Stop if all sequences have generated EOS
                if finished.all():
                    break

        # Post-processing: truncate at first EOS to avoid repeated EOS
        if stop_on_eos:
            for b_idx in range(input_ids.size(0)):
                tokens = input_ids[b_idx]
                eos_positions = (tokens == eos_token_id).nonzero(as_tuple=True)[0]

                if len(eos_positions) > 0:
                    first_eos_pos = eos_positions[0].item()
                    # Pad everything after first EOS with EOS for consistent batch length
                    input_ids[b_idx, first_eos_pos+1:] = eos_token_id

        return input_ids
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_emb.weight.numel()
        return n_params
    
    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float, device: str = "cuda") -> float:
        """
        Estimate Model FLOPs Utilization (MFU) as percentage of peak theoretical FLOPs.
        
        Args:
            fwdbwd_per_iter: Number of examples per iteration (batch_size * accum_steps)
            dt: Time per iteration in seconds
            device: Device name for peak FLOPs lookup
        
        Returns:
            MFU as a fraction (0.0 to 1.0)
        """
        # Approximate FLOPs per forward pass
        L = self.cfg.max_seq_len
        N = self.cfg.n_layers
        D = self.cfg.embed_dim
        V = self.cfg.vocab_size
        
        # Rough approximation: attention + FFN FLOPs
        flops_per_token = 6 * N * D * D
        flops_per_fwdbwd = fwdbwd_per_iter * L * flops_per_token * 3  # ×3 for backward
        
        flops_per_sec = flops_per_fwdbwd / dt
        
        # Peak theoretical FLOPs based on device
        if "3090" in device or "RTX 3090" in device:
            peak_flops = 35.6e12  # RTX 3090 peak TFLOPs
        elif "4090" in device:
            peak_flops = 82.6e12  # RTX 4090 peak TFLOPs
        elif "A100" in device:
            peak_flops = 312e12  # A100 peak TFLOPs
        else:
            peak_flops = 100e12  # Default fallback
        
        mfu = flops_per_sec / peak_flops
        return mfu

__all__ = ["Config", "LLMTransformer", "TransformerBlock", "FeedForward", "RMSNorm"]
