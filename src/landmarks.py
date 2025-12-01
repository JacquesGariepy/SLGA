"""
Learnable Landmark Selection Module
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LearnableLandmarkSelector(nn.Module):
    """
    Sélectionne G landmarks de manière différentiable parmi L positions.
    
    Utilise un scorer neural pour assigner des scores d'importance,
    puis sélectionne top-K via:
    - Entraînement: Gumbel-Softmax pour différentiabilité
    - Inférence: Top-K déterministe
    
    Args:
        embed_dim: Dimension d'embedding d'entrée
        num_landmarks: Nombre de landmarks à sélectionner (G)
        hidden_dim: Dimension cachée du scorer (défaut: embed_dim // 2)
        temperature: Température Gumbel initiale (plus bas = plus dur)
        temperature_decay: Facteur de decay par step
        min_temperature: Température minimale
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_landmarks: int,
        hidden_dim: Optional[int] = None,
        temperature: float = 1.0,
        temperature_decay: float = 0.999,   # Plus agressif que 0.9999
        min_temperature: float = 0.3,        # Plus discriminatif que 0.5
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_landmarks = num_landmarks
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        
        # Scorer: embed -> hidden -> 1 (score d'importance)
        hidden = hidden_dim or (embed_dim // 2)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )
        
        # Compteur de steps pour decay de température
        self.register_buffer("step_count", torch.tensor(0), persistent=False)
    
    def _get_temperature(self) -> float:
        """Calcule la température courante avec decay."""
        if self.training:
            temp = self.temperature * (self.temperature_decay ** self.step_count.item())
            return max(temp, self.min_temperature)
        else:
            return self.min_temperature
    
    def _gumbel_topk(
        self, scores: torch.Tensor, k: int, temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Approximation différentiable de top-K via Gumbel-Softmax.
        
        Args:
            scores: (B, L) scores d'importance
            k: Nombre d'éléments à sélectionner
            temperature: Température pour relaxation
        
        Returns:
            soft_selection: (B, L) poids continus
            hard_indices: (B, k) indices durs
        """
        B, L = scores.shape
        
        # CORRECTION: Générer bruit Gumbel en float32 pour éviter NaN en AMP
        eps = 1e-10
        original_dtype = scores.dtype
        
        # Bruit en float32 (stable)
        uniform_noise = torch.rand(scores.shape, dtype=torch.float32, device=scores.device)
        uniform_noise = uniform_noise.clamp(min=eps, max=1.0 - eps)  # Éviter 0 et 1
        gumbel_noise = -torch.log(-torch.log(uniform_noise))
        
        # Cast vers dtype original
        gumbel_noise = gumbel_noise.to(original_dtype)
        
        # Vérification NaN (sécurité)
        if torch.isnan(gumbel_noise).any() or torch.isinf(gumbel_noise).any():
            gumbel_noise = torch.zeros_like(scores)
        
        perturbed_scores = (scores + gumbel_noise) / temperature
        
        # Top-K dur (pour forward)
        _, hard_indices = torch.topk(perturbed_scores, k=k, dim=-1)
        
        # Sélection soft via softmax (pour backward)
        soft_scores = F.softmax(perturbed_scores, dim=-1)
        
        # Protection NaN
        if torch.isnan(soft_scores).any():
            soft_scores = torch.ones_like(soft_scores) / L
        
        return soft_scores, hard_indices
    
    def _straight_through_topk(
        self, scores: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Top-K avec straight-through estimator amélioré.
        
        Forward: Top-K dur (one-hot)
        Backward: Gradient via sigmoid soft-thresholding
        """
        B, L = scores.shape
        
        # Forward: Top-K dur
        topk_vals, topk_indices = torch.topk(scores, k=k, dim=-1)
        
        # One-hot des sélections
        selection_onehot = torch.zeros_like(scores)
        selection_onehot.scatter_(1, topk_indices, 1.0)
        
        # Sélection soft pour backward consistant
        threshold = topk_vals[:, -1:].detach()  # k-ème score
        temp = 0.1
        selection_soft = torch.sigmoid((scores - threshold) / temp)
        
        # Straight-through: forward=hard, backward=soft
        selection = selection_onehot + selection_soft - selection_soft.detach()
        
        return selection, topk_indices
    
    def forward(
        self, x: torch.Tensor, use_gumbel: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sélectionne les landmarks de manière différentiable.
        
        Args:
            x: (B, L, D) séquence d'entrée
            use_gumbel: Si True, utilise Gumbel-Softmax (sinon straight-through)
        
        Returns:
            landmark_indices: (B, G) indices des landmarks sélectionnés
            landmark_states: (B, G, D) états correspondants
            selection_scores: (B, L) scores de sélection pour loss auxiliaire
        """
        B, L, D = x.shape
        
        # Scorer chaque position
        scores = self.scorer(x).squeeze(-1)  # (B, L)
        
        # Protection NaN: clamp avant softmax/gumbel
        scores = torch.clamp(scores, min=-20, max=20)
        
        # Sélection différentiable
        k = min(self.num_landmarks, L)
        
        if self.training:
            if use_gumbel:
                temp = self._get_temperature()
                selection_soft, landmark_indices = self._gumbel_topk(scores, k, temp)
                self.step_count += 1
            else:
                selection_soft, landmark_indices = self._straight_through_topk(scores, k)
        else:
            # Inférence: top-K déterministe
            _, landmark_indices = torch.topk(scores, k=k, dim=-1)
            selection_soft = None
        
        # Gather des états correspondants
        landmark_indices_safe = torch.clamp(landmark_indices, 0, L - 1)
        landmark_indices_exp = landmark_indices_safe.unsqueeze(-1).expand(B, k, D)
        landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)
        
        # Scores de sélection normalisés
        selection_scores = F.softmax(scores, dim=-1)
        
        # Protection NaN
        if torch.isnan(selection_scores).any():
            selection_scores = torch.ones_like(selection_scores) / L
        
        return landmark_indices, landmark_states, selection_scores


class PositionalLandmarkSelector(nn.Module):
    """
    Sélecteur de landmarks basé sur des patterns positionnels appris.
    
    Au lieu de scorer chaque token individuellement, ce module apprend
    des patterns de positions importantes.
    """
    
    def __init__(
        self,
        max_seq_len: int,
        num_landmarks: int,
        embed_dim: int,
    ):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.num_landmarks = num_landmarks
        
        self.pos_embeddings = nn.Parameter(torch.randn(max_seq_len, embed_dim))
        self.scorer = nn.Linear(embed_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        
        pos_emb = self.pos_embeddings[:L]
        scores = self.scorer(pos_emb).squeeze(-1)
        scores = scores.unsqueeze(0).expand(B, L)
        
        k = min(self.num_landmarks, L)
        _, landmark_indices = torch.topk(scores, k=k, dim=-1)
        
        landmark_indices_safe = torch.clamp(landmark_indices, 0, L - 1)
        landmark_indices_exp = landmark_indices_safe.unsqueeze(-1).expand(B, k, D)
        landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)
        
        selection_scores = F.softmax(scores, dim=-1)
        
        return landmark_indices, landmark_states, selection_scores


class HybridLandmarkSelector(nn.Module):
    """
    Combine sélection contenu-based et positionnelle.
    Utilise un gating pour décider dynamiquement de la combinaison.
    """
    
    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int,
        num_landmarks: int,
    ):
        super().__init__()
        
        self.num_landmarks = num_landmarks
        self.content_selector = LearnableLandmarkSelector(embed_dim, num_landmarks)
        self.position_selector = PositionalLandmarkSelector(max_seq_len, num_landmarks, embed_dim)
        self.gate = nn.Linear(embed_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        
        _, _, scores_content = self.content_selector(x)
        _, _, scores_position = self.position_selector(x)
        
        x_pooled = x.mean(dim=1)
        gate_weight = torch.sigmoid(self.gate(x_pooled))
        
        scores_combined = gate_weight * scores_content + (1 - gate_weight) * scores_position
        
        k = min(self.num_landmarks, L)
        _, landmark_indices = torch.topk(scores_combined, k=k, dim=-1)
        
        landmark_indices_safe = torch.clamp(landmark_indices, 0, L - 1)
        landmark_indices_exp = landmark_indices_safe.unsqueeze(-1).expand(B, k, D)
        landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)
        
        return landmark_indices, landmark_states, scores_combined


# ======================================================================
# Auxiliary Losses
# ======================================================================

def landmark_spacing_loss(
    landmark_indices: torch.Tensor,
    seq_len: int,
    lambda_reg: float = 0.01,
    selection_scores: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Pénalise les gaps non-uniformes entre landmarks.
    
    Encourage un espacement uniforme pour maximiser la couverture spatiale.
    
    Args:
        landmark_indices: (B, G) indices des landmarks
        seq_len: Longueur de séquence L
        lambda_reg: Poids de régularisation
        selection_scores: (B, L) scores différentiables (REQUIS pour gradients!)
    
    Returns:
        loss: Scalaire encourageant espacement uniforme
    """
    B, G = landmark_indices.shape
    
    if selection_scores is not None:
        # Mode différentiable avec scores
        positions = torch.arange(seq_len, device=selection_scores.device, dtype=selection_scores.dtype)
        
        # Masque pour positions sélectionnées
        mask = torch.zeros(B, seq_len, device=selection_scores.device, dtype=selection_scores.dtype)
        mask.scatter_(1, landmark_indices, 1.0)
        
        # Scores masqués et normalisés
        masked_scores = selection_scores * mask
        normalized_scores = masked_scores / (masked_scores.sum(dim=1, keepdim=True) + 1e-8)
        
        # Poids par segment
        segment_size = seq_len / G
        segment_indices = (torch.arange(seq_len, device=selection_scores.device).float() / segment_size).long()
        segment_indices = segment_indices.clamp(max=G-1)
        segment_indices_exp = segment_indices.unsqueeze(0).expand(B, seq_len)
        
        segment_weights = torch.zeros(B, G, device=selection_scores.device, dtype=selection_scores.dtype)
        segment_weights.scatter_add_(1, segment_indices_exp, normalized_scores)
        
        # Pénaliser déviation de 1/G
        ideal_weight = 1.0 / G
        loss = lambda_reg * ((segment_weights - ideal_weight) ** 2).mean()
        
    else:
        # Fallback non-différentiable
        if G < 2:
            return torch.tensor(0.0, device=landmark_indices.device, dtype=torch.float32)
        
        sorted_idx, _ = torch.sort(landmark_indices, dim=-1)
        gaps = sorted_idx[:, 1:] - sorted_idx[:, :-1]
        ideal_gap = seq_len / G
        loss = lambda_reg * ((gaps - ideal_gap) ** 2).mean()
    
    return loss


def landmark_diversity_loss(
    selection_scores: torch.Tensor, lambda_reg: float = 0.01
) -> torch.Tensor:
    """
    [DEPRECATED] Loss basée sur l'entropie.
    Utilisez landmark_spacing_loss() à la place.
    """
    B, L = selection_scores.shape
    entropy = -(selection_scores * torch.log(selection_scores + 1e-10)).sum(dim=-1)
    max_entropy = math.log(L)
    normalized_entropy = entropy / max_entropy
    loss = lambda_reg * (1 - normalized_entropy).mean()
    return loss


def landmark_sparsity_loss(
    selection_scores: torch.Tensor,
    num_landmarks: int,
    lambda_reg: float = 0.001
) -> torch.Tensor:
    """
    Mesure la concentration de masse dans les top-G landmarks.
    
    Pénalise si les scores sont trop dispersés.
    
    Args:
        selection_scores: (B, L) scores bruts
        num_landmarks: Nombre de landmarks G
        lambda_reg: Poids de régularisation
    
    Returns:
        loss: Scalaire (0 si bonne concentration)
    """
    B, L = selection_scores.shape
    
    # Normaliser via softmax
    probs = F.softmax(selection_scores, dim=-1)
    
    # Top-G indices
    _, top_g_indices = torch.topk(selection_scores, k=num_landmarks, dim=-1)
    
    # Masse dans top-G
    top_g_probs = torch.gather(probs, dim=1, index=top_g_indices)
    mass_in_top_g = top_g_probs.sum(dim=-1).mean()
    
    # Target adaptatif
    ideal_ratio = num_landmarks / L
    target_mass = 0.60 + ideal_ratio * 0.40
    
    # Pénaliser si masse insuffisante
    loss = lambda_reg * F.relu(target_mass - mass_in_top_g)
    
    return loss


__all__ = [
    "LearnableLandmarkSelector",
    "PositionalLandmarkSelector", 
    "HybridLandmarkSelector",
    "landmark_spacing_loss",
    "landmark_diversity_loss",
    "landmark_sparsity_loss",
]
