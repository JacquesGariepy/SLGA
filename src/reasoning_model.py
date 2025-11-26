# reasoning_model.py
"""
SLGA-Reasoning: Modèle de Raisonnement Efficace

Intègre les composants de raisonnement avec l'architecture SLGA
pour créer un modèle capable de raisonnement multi-étapes efficace.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any

from .model import Config, LLMTransformer, TransformerBlock, FeedForward
from .slga import SLGAModule
from .landmarks import LearnableLandmarkSelector
from .reasoning import (
    ReasoningConfig,
    ThoughtTokenEmbedding,
    ReasoningController,
    ProcessRewardModel,
    ChainOfThoughtLoss,
    ReasoningLandmarkSelector,
    SelfConsistencyDecoder,
    extract_reasoning_steps,
)


@dataclass
class ReasoningModelConfig(Config):
    """Configuration étendue pour le modèle de raisonnement"""

    # Hérite de Config, ajoute:
    reasoning_enabled: bool = True

    # Tokens spéciaux
    think_token_id: int = 50257
    end_think_token_id: int = 50258
    step_token_id: int = 50259
    answer_token_id: int = 50260

    # Paramètres de raisonnement
    max_reasoning_steps: int = 16
    min_reasoning_steps: int = 2
    reasoning_depth_penalty: float = 0.01

    # Process Reward Model
    use_prm: bool = True
    prm_hidden_dim: int = 256

    # Self-Consistency
    use_self_consistency: bool = True
    num_reasoning_samples: int = 5
    consistency_temperature: float = 0.7


class SLGAReasoningModel(nn.Module):
    """
    Modèle de raisonnement basé sur SLGA.

    Architecture:
    1. Token + Position + Thought Embeddings
    2. N × TransformerBlock avec SLGA (landmarks appris pour raisonnement)
    3. Reasoning Controller (décide quand répondre)
    4. Process Reward Model (évalue les étapes)
    5. LM Head

    Capacités:
    - Raisonnement multi-étapes avec tokens <think>, <step>, <answer>
    - Évaluation des étapes intermédiaires (PRM)
    - Self-consistency pour améliorer la fiabilité
    - Attention sparse efficace pour longues chaînes de raisonnement
    """

    def __init__(self, cfg: ReasoningModelConfig):
        super().__init__()

        self.cfg = cfg

        # === Embeddings ===
        # Token embeddings (incluant tokens spéciaux)
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.embed_dim)
        self.emb_dropout = nn.Dropout(cfg.dropout_rate)

        # Embeddings spéciaux pour tokens de raisonnement
        self.thought_emb = ThoughtTokenEmbedding(
            embed_dim=cfg.embed_dim,
            num_special_tokens=4,  # <think>, </think>, <step>, <answer>
        )

        # === Landmark Selector (spécialisé raisonnement) ===
        if cfg.learned_landmarks:
            self.landmark_selector = ReasoningLandmarkSelector(
                embed_dim=cfg.embed_dim,
                num_landmarks=cfg.global_k * 2,
                num_reasoning_types=4,
            )
        else:
            self.landmark_selector = None

        # === Transformer Blocks ===
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, layer_idx=i) for i in range(cfg.n_layers)
        ])

        # === Reasoning Controller ===
        reasoning_cfg = ReasoningConfig(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            max_reasoning_steps=cfg.max_reasoning_steps,
            min_reasoning_steps=cfg.min_reasoning_steps,
            reasoning_depth_penalty=cfg.reasoning_depth_penalty,
            use_prm=cfg.use_prm,
            prm_hidden_dim=cfg.prm_hidden_dim,
        )
        self.reasoning_controller = ReasoningController(reasoning_cfg)

        # === Process Reward Model ===
        if cfg.use_prm:
            self.prm = ProcessRewardModel(reasoning_cfg)
        else:
            self.prm = None

        # === Self-Consistency Decoder ===
        if cfg.use_self_consistency:
            self.self_consistency = SelfConsistencyDecoder(
                reasoning_cfg,
                num_samples=cfg.num_reasoning_samples,
                temperature=cfg.consistency_temperature,
            )
        else:
            self.self_consistency = None

        # === Output ===
        self.final_norm = nn.LayerNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

        # Tie embeddings
        self.lm_head.weight = self.token_emb.weight

        # === Loss ===
        self.cot_loss = ChainOfThoughtLoss(reasoning_cfg)

        # Initialize
        self.apply(self._init_weights)

        # Cache pour les IDs des tokens spéciaux
        self._special_token_ids = torch.tensor([
            cfg.think_token_id,
            cfg.end_think_token_id,
            cfg.step_token_id,
            cfg.answer_token_id,
        ])

    def _init_weights(self, module: nn.Module):
        """Initialisation GPT-2 style"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _identify_special_tokens(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Identifie les tokens spéciaux dans la séquence.

        Returns:
            special_mask: (B, L) bool, True si token spécial
            special_local_ids: (B, L) IDs locaux (0-3) des tokens spéciaux
        """
        B, L = input_ids.shape
        device = input_ids.device

        special_ids = self._special_token_ids.to(device)

        # Créer masque pour chaque token spécial
        special_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        special_local_ids = torch.zeros(B, L, dtype=torch.long, device=device)

        for i, sid in enumerate(special_ids):
            matches = input_ids == sid
            special_mask = special_mask | matches
            special_local_ids = torch.where(matches, torch.full_like(special_local_ids, i), special_local_ids)

        return special_mask, special_local_ids

    def _find_step_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Trouve les positions des tokens <step>.

        Returns:
            step_positions: (B, max_steps) avec padding -1
        """
        B, L = input_ids.shape
        step_token = self.cfg.step_token_id
        max_steps = self.cfg.max_reasoning_steps

        step_positions = torch.full(
            (B, max_steps), -1, dtype=torch.long, device=input_ids.device
        )

        for b in range(B):
            step_mask = input_ids[b] == step_token
            step_indices = step_mask.nonzero(as_tuple=True)[0]
            num_steps = min(len(step_indices), max_steps)
            if num_steps > 0:
                step_positions[b, :num_steps] = step_indices[:num_steps]

        return step_positions

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_reasoning_info: bool = False,
        global_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass du modèle de raisonnement.

        Args:
            input_ids: (B, L) tokens d'entrée
            targets: (B, L) tokens cibles pour l'entraînement
            return_reasoning_info: Si True, retourne les infos de raisonnement
            global_weight: Poids de l'attention globale

        Returns:
            dict avec:
                - logits: (B, L, V)
                - loss: scalar (si targets fourni)
                - reasoning_info: dict (si return_reasoning_info)
        """
        B, L = input_ids.shape
        device = input_ids.device

        # === 1. Embeddings ===
        tok_emb = self.token_emb(input_ids)  # (B, L, D)
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        pos_emb = self.pos_emb(pos)

        x = tok_emb + pos_emb

        # Enrichir avec thought embeddings
        special_mask, special_local_ids = self._identify_special_tokens(input_ids)
        x = self.thought_emb(x, special_mask, special_local_ids)

        x = self.emb_dropout(x)

        # === 2. Landmark Selection ===
        landmark_indices = None
        landmark_type_probs = None

        if self.landmark_selector is not None:
            landmark_indices, landmark_states, landmark_type_probs = self.landmark_selector(
                x, return_types=True
            )

        # === 3. Transformer Blocks ===
        for block in self.blocks:
            if landmark_indices is not None:
                B_cur, L_cur, D = x.shape
                G = landmark_indices.size(1)
                landmark_indices_safe = torch.clamp(landmark_indices, 0, L_cur - 1)
                landmark_indices_exp = landmark_indices_safe.unsqueeze(-1).expand(B_cur, G, D)
                landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)
            else:
                landmark_states = None

            x = block(x, cache_global=landmark_states, global_weight=global_weight)

        # === 4. Final norm and LM head ===
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, L, V)

        # === 5. Reasoning Controller ===
        step_positions = self._find_step_positions(input_ids)
        valid_step_mask = step_positions >= 0

        reasoning_decision = self.reasoning_controller(
            x,
            step_positions=step_positions[valid_step_mask.any(dim=1)] if valid_step_mask.any() else None,
        )

        # === 6. Process Reward Model (si entraînement) ===
        step_rewards = None
        if self.prm is not None and valid_step_mask.any():
            # Gather états aux positions des étapes
            valid_positions = step_positions.clamp(min=0)
            step_pos_exp = valid_positions.unsqueeze(-1).expand(B, self.cfg.max_reasoning_steps, self.cfg.embed_dim)
            step_states = torch.gather(x, dim=1, index=step_pos_exp)

            # Masquer les positions invalides
            step_states = step_states * valid_step_mask.unsqueeze(-1).float()

            prm_output = self.prm(step_states)
            step_rewards = prm_output["step_rewards"]

        # === 7. Compute Loss (si targets) ===
        output = {"logits": logits}

        if targets is not None:
            # Compter le nombre d'étapes
            num_steps = (input_ids == self.cfg.step_token_id).sum(dim=1)

            loss_dict = self.cot_loss(
                logits=logits,
                targets=targets,
                step_rewards=step_rewards,
                step_labels=None,  # Non supervisé par défaut
                num_steps=num_steps,
                step_hidden_states=step_states if self.prm is not None and valid_step_mask.any() else None,
            )
            output["loss"] = loss_dict["total_loss"]
            output["loss_components"] = loss_dict

        # === 8. Reasoning Info ===
        if return_reasoning_info:
            output["reasoning_info"] = {
                "decision": reasoning_decision,
                "step_positions": step_positions,
                "step_rewards": step_rewards,
                "landmark_indices": landmark_indices,
                "landmark_types": landmark_type_probs,
                "num_steps": (input_ids == self.cfg.step_token_id).sum(dim=1),
            }

        return output

    @torch.no_grad()
    def generate_reasoning(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        use_self_consistency: bool = False,
        num_samples: int = 5,
    ) -> Dict[str, Any]:
        """
        Génère une chaîne de raisonnement complète.

        Args:
            input_ids: (B, L) prompt (question)
            max_new_tokens: Maximum de tokens à générer
            temperature: Température (0 = déterministe)
            top_k: Top-K sampling
            top_p: Nucleus sampling
            use_self_consistency: Si True, génère plusieurs chemins et vote
            num_samples: Nombre de chemins pour self-consistency

        Returns:
            dict avec:
                - generated_ids: (B, L + generated) tokens générés
                - reasoning_steps: List[str] étapes extraites
                - answer: str réponse finale
                - confidence: float confiance
        """
        self.eval()
        B = input_ids.size(0)
        device = input_ids.device

        if use_self_consistency and self.self_consistency is not None:
            # Générer plusieurs chemins
            all_generations = []
            all_answers = []

            for _ in range(num_samples):
                gen = self._generate_single(
                    input_ids.clone(),
                    max_new_tokens,
                    temperature=self.cfg.consistency_temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                all_generations.append(gen)

            # Voter pour la meilleure réponse
            # TODO: Implémenter le vote basé sur les embeddings
            best_gen = all_generations[0]  # Simplification

            return {
                "generated_ids": best_gen,
                "all_generations": all_generations,
                "num_samples": num_samples,
            }

        else:
            # Génération simple
            generated = self._generate_single(
                input_ids,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
            )

            return {
                "generated_ids": generated,
            }

    def _generate_single(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> torch.Tensor:
        """Génération d'une seule séquence."""

        for _ in range(max_new_tokens):
            # Truncate si nécessaire
            if input_ids.size(1) > self.cfg.max_seq_len:
                input_ids = input_ids[:, -self.cfg.max_seq_len:]

            # Forward
            outputs = self(input_ids, return_reasoning_info=True)
            logits = outputs["logits"][:, -1, :]  # (B, V)

            # Vérifier si le controller dit de s'arrêter
            decision = outputs["reasoning_info"]["decision"]
            if not decision["should_continue"].any():
                # Forcer génération de <answer>
                logits[:, self.cfg.answer_token_id] += 10.0

            # Sampling
            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None:
                    topk_vals, topk_idxs = torch.topk(logits, k=min(top_k, logits.size(-1)))
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered.scatter_(1, topk_idxs, topk_vals)
                    logits = logits_filtered

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    sorted_logits[sorted_indices_to_remove] = float('-inf')
                    logits = logits.scatter(1, sorted_indices, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Arrêter si </answer> ou EOS
            if next_token.item() in [self.cfg.answer_token_id, 50256]:
                break

        return input_ids

    def get_num_params(self) -> int:
        """Compte les paramètres du modèle."""
        return sum(p.numel() for p in self.parameters())


def create_reasoning_model(
    vocab_size: int = 50261,
    embed_dim: int = 768,
    n_layers: int = 16,
    **kwargs
) -> SLGAReasoningModel:
    """
    Factory pour créer un modèle de raisonnement.

    Args:
        vocab_size: Taille du vocabulaire (incluant tokens spéciaux)
        embed_dim: Dimension des embeddings
        n_layers: Nombre de couches transformer
        **kwargs: Arguments supplémentaires pour ReasoningModelConfig

    Returns:
        SLGAReasoningModel configuré
    """
    cfg = ReasoningModelConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        n_layers=n_layers,
        num_heads=embed_dim // 64,  # 64 dim per head
        **kwargs
    )
    return SLGAReasoningModel(cfg)


__all__ = [
    "ReasoningModelConfig",
    "SLGAReasoningModel",
    "create_reasoning_model",
]
