# reasoning.py
"""
Reasoning Components for SLGA-Reasoning Model

Composants pour transformer SLGA en un modèle de raisonnement:
1. ReasoningController - Contrôle la profondeur de raisonnement
2. ThoughtTokens - Gestion des tokens spéciaux <think>, <step>, <answer>
3. ProcessRewardModel - Récompense les étapes intermédiaires
4. ChainOfThoughtLoss - Loss pour l'entraînement CoT
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass


@dataclass
class ReasoningConfig:
    """Configuration pour le module de raisonnement"""
    # Tokens spéciaux
    think_token_id: int = 50257      # <think>
    end_think_token_id: int = 50258  # </think>
    step_token_id: int = 50259       # <step>
    answer_token_id: int = 50260     # <answer>

    # Contrôle du raisonnement
    max_reasoning_steps: int = 16     # Maximum d'étapes de raisonnement
    min_reasoning_steps: int = 1      # Minimum d'étapes
    reasoning_depth_penalty: float = 0.01  # Pénalité pour trop d'étapes

    # Récompenses
    step_reward: float = 0.1          # Récompense par étape correcte
    answer_reward: float = 1.0        # Récompense pour bonne réponse

    # Architecture
    embed_dim: int = 512
    num_heads: int = 8

    # Process Reward Model (PRM)
    use_prm: bool = True              # Activer le Process Reward Model
    prm_hidden_dim: int = 256


class ThoughtTokenEmbedding(nn.Module):
    """
    Embeddings spéciaux pour les tokens de raisonnement.

    Apprend des embeddings distincts pour:
    - <think>: début de réflexion
    - </think>: fin de réflexion
    - <step>: étape intermédiaire
    - <answer>: réponse finale
    """

    def __init__(self, embed_dim: int, num_special_tokens: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_special_tokens = num_special_tokens

        # Embeddings appris pour tokens spéciaux
        self.special_embeddings = nn.Embedding(num_special_tokens, embed_dim)

        # Projection pour fusionner avec embeddings existants
        self.fusion = nn.Linear(embed_dim * 2, embed_dim)

        # Initialisation
        nn.init.normal_(self.special_embeddings.weight, std=0.02)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        special_token_mask: torch.Tensor,
        special_token_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Fusionne les embeddings de tokens spéciaux avec les embeddings standards.

        Args:
            token_embeddings: (B, L, D) embeddings des tokens
            special_token_mask: (B, L) bool, True si token spécial
            special_token_ids: (B, L) IDs locaux (0-3) des tokens spéciaux

        Returns:
            enriched_embeddings: (B, L, D) embeddings enrichis
        """
        B, L, D = token_embeddings.shape

        # Récupérer embeddings spéciaux
        # Clamp pour éviter index out of bounds
        special_ids_clamped = torch.clamp(special_token_ids, 0, self.num_special_tokens - 1)
        special_emb = self.special_embeddings(special_ids_clamped)  # (B, L, D)

        # Fusionner seulement aux positions de tokens spéciaux
        fused = self.fusion(torch.cat([token_embeddings, special_emb], dim=-1))

        # Masquer: utiliser fusion seulement pour tokens spéciaux
        mask_expanded = special_token_mask.unsqueeze(-1).expand_as(token_embeddings)
        output = torch.where(mask_expanded, fused, token_embeddings)

        return output


class ReasoningController(nn.Module):
    """
    Contrôleur de raisonnement adaptatif.

    Décide dynamiquement:
    1. Faut-il continuer à raisonner ou donner la réponse?
    2. Quelle profondeur de raisonnement est nécessaire?
    3. Le raisonnement actuel est-il correct?

    Utilise un mécanisme d'attention sur l'historique pour décider.
    """

    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.config = config
        embed_dim = config.embed_dim

        # Encodeur du contexte de raisonnement
        self.context_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )

        # Tête de décision: continuer ou répondre
        self.decision_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 2),  # [continue, answer]
        )

        # Estimateur de confiance
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Compteur d'étapes (buffer, pas de gradients)
        self.register_buffer("step_count", torch.tensor(0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        step_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Évalue l'état du raisonnement et décide de la prochaine action.

        Args:
            hidden_states: (B, L, D) états cachés actuels
            step_positions: (B, num_steps) positions des <step> tokens

        Returns:
            dict avec:
                - decision: (B, 2) logits [continuer, répondre]
                - confidence: (B, 1) confiance dans le raisonnement actuel
                - should_continue: (B,) bool, continuer ou non
        """
        B, L, D = hidden_states.shape

        # Pooling sur la séquence (mean pooling)
        context = hidden_states.mean(dim=1)  # (B, D)

        # Si positions des étapes fournies, enrichir avec attention
        if step_positions is not None and step_positions.numel() > 0:
            # Gather les états aux positions des étapes
            num_steps = step_positions.size(1)
            step_positions_clamped = torch.clamp(step_positions, 0, L - 1)
            step_pos_exp = step_positions_clamped.unsqueeze(-1).expand(B, num_steps, D)
            step_states = torch.gather(hidden_states, dim=1, index=step_pos_exp)  # (B, num_steps, D)

            # Attention sur les étapes précédentes
            step_context = step_states.mean(dim=1)  # (B, D)
            context = context + step_context

        # Encoder le contexte
        encoded = self.context_encoder(context)  # (B, D)

        # Décision
        decision_logits = self.decision_head(encoded)  # (B, 2)
        decision_probs = F.softmax(decision_logits, dim=-1)

        # Confiance
        confidence = self.confidence_head(encoded)  # (B, 1)

        # Seuil adaptatif basé sur le nombre d'étapes
        # Plus on a d'étapes, plus on favorise "répondre"
        step_bias = min(self.step_count.item() / self.config.max_reasoning_steps, 1.0)

        # Décision finale: continue si prob(continue) > 0.5 - step_bias * 0.3
        threshold = 0.5 - step_bias * 0.3
        should_continue = decision_probs[:, 0] > threshold

        return {
            "decision_logits": decision_logits,
            "decision_probs": decision_probs,
            "confidence": confidence,
            "should_continue": should_continue,
        }


class ProcessRewardModel(nn.Module):
    """
    Process Reward Model (PRM) pour le raisonnement.

    Évalue la qualité de chaque étape de raisonnement, pas seulement
    la réponse finale. Inspiré de "Let's Verify Step by Step" (OpenAI).

    Avantages vs Outcome Reward Model (ORM):
    - Feedback plus granulaire
    - Détecte les erreurs tôt
    - Meilleur crédit assignment
    """

    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.config = config
        embed_dim = config.embed_dim
        hidden_dim = config.prm_hidden_dim

        # Encodeur d'étape
        self.step_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Tête de récompense par étape
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Tête de classification: étape correcte/incorrecte
        self.correctness_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),  # [incorrect, correct]
        )

    def forward(
        self,
        step_hidden_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Évalue chaque étape de raisonnement.

        Args:
            step_hidden_states: (B, num_steps, D) états des tokens <step>

        Returns:
            dict avec:
                - step_rewards: (B, num_steps) récompense par étape
                - step_correctness: (B, num_steps, 2) logits correct/incorrect
        """
        B, S, D = step_hidden_states.shape

        # Encoder chaque étape
        encoded = self.step_encoder(step_hidden_states)  # (B, S, hidden)

        # Récompenses
        rewards = self.reward_head(encoded).squeeze(-1)  # (B, S)

        # Correctness
        correctness_logits = self.correctness_head(encoded)  # (B, S, 2)

        return {
            "step_rewards": rewards,
            "step_correctness": correctness_logits,
        }

    def compute_process_reward(
        self,
        step_rewards: torch.Tensor,
        step_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calcule la récompense totale du processus de raisonnement.

        Args:
            step_rewards: (B, num_steps) récompenses prédites
            step_labels: (B, num_steps) labels (1=correct, 0=incorrect) si supervisé

        Returns:
            total_reward: (B,) récompense totale par séquence
        """
        if step_labels is not None:
            # Mode supervisé: récompense = sum des étapes correctes
            masked_rewards = step_rewards * step_labels
            return masked_rewards.sum(dim=-1)
        else:
            # Mode non-supervisé: somme simple
            return step_rewards.sum(dim=-1)


class ChainOfThoughtLoss(nn.Module):
    """
    Loss function pour l'entraînement Chain-of-Thought.

    Combine:
    1. Language modeling loss (cross-entropy)
    2. Process reward loss (récompenser les bonnes étapes)
    3. Reasoning depth loss (pénaliser trop/peu d'étapes)
    4. Consistency loss (étapes doivent être cohérentes)
    """

    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.config = config

        # Poids des différentes pertes
        self.lm_weight = 1.0
        self.process_weight = 0.5
        self.depth_weight = 0.1
        self.consistency_weight = 0.1

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        step_rewards: Optional[torch.Tensor] = None,
        step_labels: Optional[torch.Tensor] = None,
        num_steps: Optional[torch.Tensor] = None,
        step_hidden_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Calcule la loss totale pour l'entraînement CoT.

        Args:
            logits: (B, L, V) logits du modèle
            targets: (B, L) tokens cibles
            step_rewards: (B, S) récompenses par étape (optionnel)
            step_labels: (B, S) labels des étapes (optionnel)
            num_steps: (B,) nombre d'étapes par séquence (optionnel)
            step_hidden_states: (B, S, D) pour consistency loss (optionnel)

        Returns:
            dict avec toutes les losses et la loss totale
        """
        B, L, V = logits.shape

        # 1. Language modeling loss
        lm_loss = F.cross_entropy(
            logits.view(-1, V),
            targets.view(-1),
            ignore_index=-100,  # Padding token
        )

        total_loss = self.lm_weight * lm_loss
        losses = {"lm_loss": lm_loss}

        # 2. Process reward loss (si fourni)
        if step_rewards is not None and step_labels is not None:
            # Binary cross-entropy pour prédire si étape correcte
            process_loss = F.binary_cross_entropy_with_logits(
                step_rewards,
                step_labels.float(),
            )
            total_loss = total_loss + self.process_weight * process_loss
            losses["process_loss"] = process_loss

        # 3. Reasoning depth loss
        if num_steps is not None:
            # Pénaliser si trop loin de la cible optimale
            target_steps = (self.config.min_reasoning_steps + self.config.max_reasoning_steps) / 2
            depth_loss = ((num_steps.float() - target_steps) ** 2).mean()
            total_loss = total_loss + self.depth_weight * depth_loss
            losses["depth_loss"] = depth_loss

        # 4. Consistency loss (étapes successives doivent être cohérentes)
        if step_hidden_states is not None and step_hidden_states.size(1) > 1:
            # Cosine similarity entre étapes adjacentes
            step_i = step_hidden_states[:, :-1, :]  # (B, S-1, D)
            step_j = step_hidden_states[:, 1:, :]   # (B, S-1, D)

            # Normaliser
            step_i_norm = F.normalize(step_i, dim=-1)
            step_j_norm = F.normalize(step_j, dim=-1)

            # Similarité (voulons qu'elle soit élevée mais pas 1.0)
            similarity = (step_i_norm * step_j_norm).sum(dim=-1)  # (B, S-1)

            # Cible: similarité autour de 0.7 (cohérent mais pas identique)
            target_sim = 0.7
            consistency_loss = ((similarity - target_sim) ** 2).mean()
            total_loss = total_loss + self.consistency_weight * consistency_loss
            losses["consistency_loss"] = consistency_loss

        losses["total_loss"] = total_loss
        return losses


class ReasoningLandmarkSelector(nn.Module):
    """
    Sélecteur de landmarks spécialisé pour le raisonnement.

    Apprend à identifier les positions importantes pour le raisonnement:
    - Prémisses (données du problème)
    - Définitions et contraintes
    - Étapes intermédiaires clés
    - Conclusions partielles
    """

    def __init__(
        self,
        embed_dim: int,
        num_landmarks: int,
        num_reasoning_types: int = 4,  # prémisse, définition, étape, conclusion
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_landmarks = num_landmarks
        self.num_types = num_reasoning_types

        # Classifieur de type de token
        self.type_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_reasoning_types),
        )

        # Importance scorer par type
        self.importance_scorers = nn.ModuleList([
            nn.Linear(embed_dim, 1) for _ in range(num_reasoning_types)
        ])

        # Pondération apprise des types
        self.type_weights = nn.Parameter(torch.ones(num_reasoning_types))

    def forward(
        self,
        x: torch.Tensor,
        return_types: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Sélectionne les landmarks importants pour le raisonnement.

        Args:
            x: (B, L, D) séquence d'entrée
            return_types: si True, retourne aussi les types prédits

        Returns:
            landmark_indices: (B, G) indices des landmarks
            landmark_states: (B, G, D) états correspondants
            type_predictions: (B, L, num_types) si return_types=True
        """
        B, L, D = x.shape

        # Classifier les types
        type_logits = self.type_classifier(x)  # (B, L, num_types)
        type_probs = F.softmax(type_logits, dim=-1)  # (B, L, num_types)

        # Scorer l'importance par type
        scores_per_type = []
        for i, scorer in enumerate(self.importance_scorers):
            score = scorer(x).squeeze(-1)  # (B, L)
            scores_per_type.append(score)

        scores_stacked = torch.stack(scores_per_type, dim=-1)  # (B, L, num_types)

        # Score final = somme pondérée par probabilité de type et poids appris
        weights = F.softmax(self.type_weights, dim=-1)  # (num_types,)
        final_scores = (scores_stacked * type_probs * weights).sum(dim=-1)  # (B, L)

        # Top-K
        k = min(self.num_landmarks, L)
        _, landmark_indices = torch.topk(final_scores, k=k, dim=-1)  # (B, k)

        # Gather states
        landmark_indices_safe = torch.clamp(landmark_indices, 0, L - 1)
        landmark_indices_exp = landmark_indices_safe.unsqueeze(-1).expand(B, k, D)
        landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)

        if return_types:
            return landmark_indices, landmark_states, type_probs
        return landmark_indices, landmark_states, None


class SelfConsistencyDecoder(nn.Module):
    """
    Décodeur avec Self-Consistency pour améliorer le raisonnement.

    Génère plusieurs chaînes de raisonnement et vote pour la réponse
    finale. Implémente "Self-Consistency Improves Chain of Thought
    Reasoning in Language Models" (Wang et al., 2022).
    """

    def __init__(
        self,
        config: ReasoningConfig,
        num_samples: int = 5,
        temperature: float = 0.7,
    ):
        super().__init__()
        self.config = config
        self.num_samples = num_samples
        self.temperature = temperature

        # Agrégateur de votes
        self.vote_aggregator = nn.Sequential(
            nn.Linear(config.embed_dim * num_samples, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )

    def aggregate_answers(
        self,
        answer_embeddings: torch.Tensor,
        answer_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Agrège plusieurs réponses par vote majoritaire.

        Args:
            answer_embeddings: (B, num_samples, D) embeddings des réponses
            answer_tokens: (B, num_samples, answer_len) tokens des réponses

        Returns:
            best_answer_embedding: (B, D)
            vote_distribution: (B, num_samples) poids de chaque réponse
        """
        B, S, D = answer_embeddings.shape

        # Calculer similarité entre toutes les paires de réponses
        # Réponses similaires = même "vote"
        answer_norm = F.normalize(answer_embeddings, dim=-1)  # (B, S, D)
        similarity = torch.bmm(answer_norm, answer_norm.transpose(1, 2))  # (B, S, S)

        # Score de chaque réponse = somme des similarités (popularité)
        vote_scores = similarity.sum(dim=-1)  # (B, S)
        vote_distribution = F.softmax(vote_scores, dim=-1)  # (B, S)

        # Réponse finale = moyenne pondérée
        best_answer = (answer_embeddings * vote_distribution.unsqueeze(-1)).sum(dim=1)  # (B, D)

        return best_answer, vote_distribution


# === Fonctions utilitaires ===

def create_reasoning_tokens(tokenizer, vocab_size: int) -> Dict[str, int]:
    """
    Crée les tokens spéciaux pour le raisonnement.

    Returns:
        dict avec les IDs des tokens spéciaux
    """
    special_tokens = {
        "<think>": vocab_size,
        "</think>": vocab_size + 1,
        "<step>": vocab_size + 2,
        "<answer>": vocab_size + 3,
    }
    return special_tokens


def format_cot_example(
    question: str,
    reasoning_steps: List[str],
    answer: str,
) -> str:
    """
    Formate un exemple Chain-of-Thought.

    Args:
        question: Question posée
        reasoning_steps: Liste des étapes de raisonnement
        answer: Réponse finale

    Returns:
        formatted: Texte formaté avec tokens spéciaux
    """
    parts = [question, "<think>"]

    for i, step in enumerate(reasoning_steps):
        parts.append(f"<step> Step {i+1}: {step}")

    parts.append("</think>")
    parts.append(f"<answer>{answer}</answer>")

    return " ".join(parts)


def extract_reasoning_steps(
    generated_text: str,
) -> Tuple[List[str], str]:
    """
    Extrait les étapes de raisonnement et la réponse d'un texte généré.

    Args:
        generated_text: Texte généré par le modèle

    Returns:
        steps: Liste des étapes de raisonnement
        answer: Réponse finale
    """
    steps = []
    answer = ""

    # Extraire entre <think> et </think>
    import re
    think_match = re.search(r"<think>(.*?)</think>", generated_text, re.DOTALL)
    if think_match:
        think_content = think_match.group(1)
        # Extraire chaque <step>
        step_matches = re.findall(r"<step>\s*(.*?)(?=<step>|$)", think_content, re.DOTALL)
        steps = [s.strip() for s in step_matches if s.strip()]

    # Extraire la réponse
    answer_match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()

    return steps, answer


__all__ = [
    "ReasoningConfig",
    "ThoughtTokenEmbedding",
    "ReasoningController",
    "ProcessRewardModel",
    "ChainOfThoughtLoss",
    "ReasoningLandmarkSelector",
    "SelfConsistencyDecoder",
    "create_reasoning_tokens",
    "format_cot_example",
    "extract_reasoning_steps",
]
