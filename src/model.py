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
    """Configuration du modèle SLGA"""
    vocab_size: int = 50257
    max_seq_len: int = 2048
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

    # Landmark selector config (v1.1) - optional, ignored if None
    landmark_selector: Optional[Dict[str, Any]] = None

    # Training config
    grad_checkpointing: bool = False


class FeedForward(nn.Module):
    """Feed-Forward Network (FFN) standard"""
    
    def __init__(self, embed_dim: int, hidden_multiplier: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = embed_dim * hidden_multiplier
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Bloc Transformer avec SLGA.
    
    Architecture: Pre-norm (LayerNorm avant attention/FFN)
    """
    
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        
        self.cfg = cfg
        self.layer_idx = layer_idx
        
        # Dilatation progressive par couche si activée
        if cfg.dilated_windows:
            # Couches basses: dense, couches hautes: dilatées
            dilation_factor = 2 ** (layer_idx // max(1, cfg.n_layers // 3))
        else:
            dilation_factor = 1
        
        # Attention SLGA
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
        
        # FFN
        self.ffn = FeedForward(
            cfg.embed_dim,
            cfg.ff_hidden_multiplier,
            cfg.dropout_rate,
        )
        
        # Layer norms (pre-norm)
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
    
    def _attn_forward(self, x: torch.Tensor, cache_global: Optional[torch.Tensor], global_weight: float = 1.0) -> torch.Tensor:
        """Wrapper pour attention (utilisé avec checkpointing)"""
        return self.attn(x, cache_global=cache_global, global_weight=global_weight)
    
    def _ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Wrapper pour FFN (utilisé avec checkpointing)"""
        return self.ffn(x)
    
    def forward(
        self,
        x: torch.Tensor,
        cache_global: Optional[torch.Tensor] = None,
        global_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
            cache_global: (B, G, D) landmarks globaux
            global_weight: Poids de l'attention globale (0.0 à 1.0)

        Returns:
            x: (B, L, D)
        """
        # Attention avec résiduelle (pre-norm)
        if self.cfg.grad_checkpointing and self.training:
            attn_out = checkpoint(self._attn_forward, self.norm1(x), cache_global, global_weight, use_reentrant=False)
        else:
            attn_out = self.attn(self.norm1(x), cache_global=cache_global, global_weight=global_weight)
        
        x = x + attn_out
        
        # FFN avec résiduelle
        if self.cfg.grad_checkpointing and self.training:
            ffn_out = checkpoint(self._ffn_forward, self.norm2(x), use_reentrant=False)
        else:
            ffn_out = self.ffn(self.norm2(x))
        
        x = x + ffn_out
        
        return x


class LLMTransformer(nn.Module):
    """
    Modèle Transformer causal complet avec SLGA.
    
    Architecture:
    1. Token + Position Embeddings
    2. N × TransformerBlock (SLGA + FFN)
    3. Final LayerNorm
    4. LM Head (projection vers vocab)
    
    Support:
    - Landmarks appris (optionnel)
    - Landmarks heuristiques via cache_global_ids
    - Génération avec sampling
    """
    
    def __init__(self, cfg: Config):
        super().__init__()
        
        self.cfg = cfg
        
        # Embeddings
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.embed_dim)
        self.emb_dropout = nn.Dropout(cfg.dropout_rate)
        
        # Sélecteur de landmarks (si activé)
        if cfg.learned_landmarks:
            self.landmark_selector = LearnableLandmarkSelector(
                embed_dim=cfg.embed_dim,
                num_landmarks=cfg.global_k * 2,  # Sélectionner plus, top-K restreint dans SLGA
            )
        else:
            self.landmark_selector = None
        
        # Blocs Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, layer_idx=i) for i in range(cfg.n_layers)
        ])
        
        # Final norm et LM head
        self.final_norm = nn.LayerNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        
        # Tie embeddings (partage poids token_emb et lm_head)
        self.lm_head.weight = self.token_emb.weight
        
        # Initialisation
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialisation des poids (GPT-2 style)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        cache_global_ids: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        global_weight: float = 1.0,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass du modèle avec métriques auxiliaires optionnelles."""
        B, L = input_ids.shape
        device = input_ids.device

        # Embeddings
        tok_emb = self.token_emb(input_ids)  # (B, L, D)
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        pos_emb = self.pos_emb(pos)  # (B, L, D)
        x = self.emb_dropout(tok_emb + pos_emb)

        # Sélection landmarks initiale
        landmark_indices = None
        landmark_scores = None

        if self.landmark_selector is not None:
            # Landmarks appris - sélectionner les indices une fois
            # ✅ FIX: Activer Gumbel-Softmax en mode training pour backprop
            landmark_indices, _, landmark_scores = self.landmark_selector(x, use_gumbel=self.training)
            # landmark_indices: (B, G)
        elif cache_global_ids is not None:
            # Landmarks heuristiques - utiliser les indices fournis
            landmark_indices = cache_global_ids  # (B, G)

        # Passer par les blocs Transformer
        # Mettre à jour les landmarks à chaque couche pour qu'ils évoluent avec x
        gate_values_layers: list[torch.Tensor] = []

        for block in self.blocks:
            # Extraire les états actuels des landmarks depuis x
            if landmark_indices is not None:
                B_cur, L_cur, D = x.shape
                G = landmark_indices.size(1)
                # ✅ FIX: Clamp indices pour éviter out-of-bounds (si landmark_indices ≥ L_cur)
                # Cela peut arriver lors de troncation de séquence ou erreurs de sélection
                landmark_indices_safe = torch.clamp(landmark_indices, 0, L_cur - 1)
                landmark_indices_exp = landmark_indices_safe.unsqueeze(-1).expand(B_cur, G, D)
                landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)  # (B, G, D)
            else:
                landmark_states = None

            # Forward du bloc avec landmarks mis à jour
            x = block(x, cache_global=landmark_states, global_weight=global_weight)

            # Récupérer métriques de gating (si disponibles)
            monitor = getattr(block.attn, "last_monitor", None)
            if monitor and monitor.get("gate_scalar") is not None:
                gate_values_layers.append(monitor["gate_scalar"])
        
        # Final norm et projection
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, L, V)
        
        if return_aux:
            aux = {
                "landmark_scores": landmark_scores,  # Scores softmax (B, L)
                "landmark_indices": landmark_indices,  # Indices sélectionnés (B, G)
            }

            if gate_values_layers:
                gate_stack = torch.stack(gate_values_layers, dim=0)  # (layers, B, H, L)
                aux["gate_values"] = gate_stack.detach()

            return logits, aux
        else:
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
        seed: Optional[int] = None,  # ← NOUVEAU: seed pour reproductibilité
        stop_on_eos: bool = True,  # 💡 NOUVEAU: arrêt sur EOS token
        eos_token_id: int = 50256,  # GPT-2 EOS token par défaut
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Génération auto-régressive DÉTERMINISTE et CORRIGÉE avec arrêt sur EOS.

        Args:
            input_ids: (B, L) prompt
            max_new_tokens: Nombre de tokens à générer
            temperature: Température de sampling
                - 0.0 = greedy DÉTERMINISTE (argmax)
                - >0 = sampling stochastique (multinomial)
            top_k: Si fourni, restreint aux top-K logits
            top_p: Si fourni, nucleus sampling
            cache_global_ids: Landmarks globaux (si learned=False)
            seed: Seed pour reproductibilité (optionnel)
            stop_on_eos: Si True, arrêt quand TOUS les samples ont généré EOS
            eos_token_id: ID du token EOS (défaut: 50256 pour GPT-2)
            repetition_penalty: >1.0 applique une pénalisation des tokens déjà vus
            no_repeat_ngram_size: Taille d'n-grammes interdits (None = désactivé)

        Returns:
            output_ids: (B, L + tokens_générés) où tokens_générés ≤ max_new_tokens
        """
        # Force eval mode
        self.eval()

        # Set seed si fourni (pour reproductibilité)
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # 💡 Batch-aware EOS tracking
        # Track which sequences have finished (pour chaque sample du batch)
        batch_size = input_ids.size(0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        # Precompute pour no-repeat n-gram
        use_no_repeat = no_repeat_ngram_size is not None and no_repeat_ngram_size > 0

        # ✅ FIX Bug #14: Sauvegarder si landmarks fournis par utilisateur
        # Pour ne pas les écraser avec autogeneration
        user_provided_landmarks = cache_global_ids is not None

        for step in range(max_new_tokens):
            # Tronquer si dépasse max_seq_len
            if input_ids.size(1) > self.cfg.max_seq_len:
                input_ids = input_ids[:, -self.cfg.max_seq_len:]
            
            # ✅ FIX Bug #11 + Bug #14: Recalculer landmarks pour heuristic mode
            # Avant: calculait seulement si cache_global_ids is None (figé après premier step)
            # Après: recalcule SAUF si utilisateur a fourni landmarks personnalisés
            if not self.cfg.learned_landmarks and not user_provided_landmarks:
                # Autogénération: recalculer à chaque step pour tracker nouveaux tokens
                L = input_ids.size(1)
                # Utiliser linspace pour garantir exactement global_k landmarks
                landmark_positions = torch.linspace(0, L-1, self.cfg.global_k, device=input_ids.device).long()
                cache_global_ids = landmark_positions.unsqueeze(0).expand(input_ids.size(0), -1)
            
            # Forward
            logits = self(input_ids, cache_global_ids=cache_global_ids)  # (B, L, V)
            
            # Prendre logits du dernier token
            logits = logits[:, -1, :]  # (B, V)

            # ✅ FIX Bug #12: Geler les séquences terminées pour éviter génération post-EOS
            # Forcer les séquences finies à générer seulement EOS
            if stop_on_eos and finished.any():
                # Pour séquences terminées: logits → -inf sauf EOS
                logits[finished] = float('-inf')
                logits[finished, eos_token_id] = 1e4  # Force EOS

            # ===================================================================
            # 🔧 Contrôles anti-répétition avant sampling
            # ===================================================================

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

            # ===================================================================
            # 🔧 FIX: ORDRE CORRECT - Temperature → Top-K → Top-P → Sample
            # ===================================================================

            # Cas spécial: Greedy (temperature=0)
            if temperature == 0.0:
                # GREEDY: Sélection déterministe du meilleur token
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)

            else:
                # SAMPLING: Comportement stochastique

                # 1. TEMPÉRATURE (PREMIER!) - Échelle les logits
                if temperature != 1.0:
                    logits = logits / temperature

                # 2. TOP-K filtering (sur logits tempérés)
                if top_k is not None and top_k > 0:
                    topk_vals, topk_idxs = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered.scatter_(1, topk_idxs, topk_vals)
                    logits = logits_filtered

                # 3. TOP-P (nucleus) filtering (sur logits tempérés)
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    # Softmax maintenant sur logits DÉJÀ tempérés
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    # Shift right to keep at least the best token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False  # Always keep the best token

                    # Set filtered logits to -inf
                    sorted_logits[sorted_indices_to_remove] = float('-inf')

                    # Scatter back to original positions
                    logits = logits.scatter(1, sorted_indices, sorted_logits)
                
                # Sample avec protection contre NaN
                probs = F.softmax(logits, dim=-1)
                
                # Protection: si tous les logits sont -inf, utiliser distribution uniforme
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    probs = torch.ones_like(probs) / probs.size(-1)
                
                # Clamp pour s'assurer que les probs sont valides
                probs = torch.clamp(probs, min=1e-10)
                probs = probs / probs.sum(dim=-1, keepdim=True)  # Re-normaliser
                
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # ✅ FIX Bug #12 (Part 2): Pour séquences terminées, remplacer par EOS
            # Empêcher ajout de tokens non-EOS après la fin
            if stop_on_eos and finished.any():
                next_token[finished] = eos_token_id

            # Ajouter à la séquence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # 💡 FEATURE: Arrêt sur EOS token (batch-aware)
            if stop_on_eos:
                # Marquer les séquences qui ont généré EOS
                eos_mask = (next_token.squeeze(-1) == eos_token_id)  # (B,)
                finished = finished | eos_mask

                # Arrêter si TOUTES les séquences ont généré EOS
                if finished.all():
                    break

        # ✅ FIX Bug #12 (Part 3): Post-processing - tronquer au premier EOS
        # Pour éviter tokens/EOS répétés après le premier EOS
        if stop_on_eos:
            for b_idx in range(input_ids.size(0)):
                tokens = input_ids[b_idx]
                eos_positions = (tokens == eos_token_id).nonzero(as_tuple=True)[0]

                if len(eos_positions) > 0:
                    # Garder jusqu'au premier EOS (inclus), pad le reste
                    first_eos_pos = eos_positions[0].item()
                    # Tout après le premier EOS devient EOS (pour longueur consistante dans batch)
                    input_ids[b_idx, first_eos_pos+1:] = eos_token_id

        return input_ids
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Compte le nombre de paramètres"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos_emb.weight.numel()
            n_params -= self.token_emb.weight.numel()
        return n_params
    
    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float, device: str = "cuda") -> float:
        """
        Estime Model FLOPs Utilization (MFU) en % de peak théorique.
        
        Args:
            fwdbwd_per_iter: Nombre d'exemples par itération (batch_size * accum_steps)
            dt: Temps par itération (secondes)
            device: Device utilisé
        """
        # Estimer FLOPs par forward pass (approximatif)
        L = self.cfg.max_seq_len
        N = self.cfg.n_layers
        D = self.cfg.embed_dim
        V = self.cfg.vocab_size
        
        # Attention: 2 * N * L * D^2 (QKV proj + output proj)
        # FFN: 2 * N * L * D * (4D) * 2 (up + down)
        # Embeddings: L * D * V (négligeable)
        
        flops_per_token = 6 * N * D * D  # Approximation simple
        flops_per_fwdbwd = fwdbwd_per_iter * L * flops_per_token * 3  # ×3 pour backward
        
        flops_per_sec = flops_per_fwdbwd / dt
        
        # Peak FLOPs théorique (TFLOPS) selon device
        if "3090" in device or "RTX 3090" in device:
            peak_flops = 35.6e12  # 35.6 TFLOPS (FP32), 142 TFLOPS (FP16 Tensor Cores)
        elif "4090" in device:
            peak_flops = 82.6e12
        elif "A100" in device:
            peak_flops = 312e12  # 312 TFLOPS (FP16 Tensor Cores)
        else:
            peak_flops = 100e12  # Valeur par défaut
        
        mfu = flops_per_sec / peak_flops
        return mfu


def test_model():
    """Test du modèle complet"""
    print("=== Test LLM Transformer ===")
    
    cfg = Config(
        vocab_size=50257,
        max_seq_len=512,
        embed_dim=256,
        num_heads=4,
        n_layers=4,
        local_window=64,
        global_k=16,
        learned_landmarks=True,
    )
    
    model = LLMTransformer(cfg)
    
    print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")
    
    B, L = 2, 128
    input_ids = torch.randint(0, cfg.vocab_size, (B, L))
    
    print(f"Input: {input_ids.shape}")
    
    # Forward
    logits = model(input_ids)
    
    print(f"Logits: {logits.shape}")
    assert logits.shape == (B, L, cfg.vocab_size), "Shape mismatch!"
    
    # Test avec aux
    logits, aux = model(input_ids, return_aux=True)
    print(f"Aux keys: {aux.keys()}")
    
    # Test génération
    prompt = torch.randint(0, cfg.vocab_size, (1, 10))
    output = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=40)
    print(f"Generated: {output.shape}")
    
    print("✓ Test passed!")


if __name__ == "__main__":
    test_model()
