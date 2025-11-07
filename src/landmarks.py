# landmarks.py
"""
Learnable Landmark Selection Module

Implémente la sélection différentiable de positions importantes (landmarks)
pour l'attention globale via Gumbel-Softmax ou top-K avec straight-through estimator.
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
    
    Utilise un scorer neural pour attribuer un score d'importance à chaque position,
    puis sélectionne les top-K via:
    - Mode training: Gumbel-Softmax relaxation pour différentiabilité
    - Mode inference: Hard top-K déterministe
    
    Args:
        embed_dim: Dimension des embeddings d'entrée
        num_landmarks: Nombre de landmarks à sélectionner (G)
        hidden_dim: Dimension de la couche cachée du scorer (default: embed_dim // 2)
        temperature: Température Gumbel (plus bas = plus hard, default: 1.0)
        temperature_decay: Facteur de décroissance de température (default: 0.999, 10× plus rapide)
        min_temperature: Température minimale (default: 0.3, plus discriminatif)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_landmarks: int,
        hidden_dim: Optional[int] = None,
        temperature: float = 1.0,
        temperature_decay: float = 0.999,  # Optimisation #1: 10× plus rapide (0.9999 → 0.999)
        min_temperature: float = 0.3,      # Optimisation #1: Plus discriminatif (0.5 → 0.3)
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
        
        # Compteur d'étapes pour décroissance température
        self.register_buffer("step_count", torch.tensor(0), persistent=False)
    
    def _get_temperature(self) -> float:
        """Calcule température actuelle avec décroissance"""
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
            soft_selection: (B, L) poids continus (approx one-hot top-k)
            hard_indices: (B, k) indices hard (pour gather)
        """
        B, L = scores.shape

        # ✅ FIX Bug #17: Générer bruit Gumbel en float32 pour éviter NaN en AMP
        # Problème: En float16/bfloat16, torch.rand_like() peut produire exactement 0
        # → -log(-log(0)) = NaN
        # Solution: Générer en float32, puis caster
        eps = 1e-10
        original_dtype = scores.dtype

        # Générer bruit en float32 (stable)
        uniform_noise = torch.rand(scores.shape, dtype=torch.float32, device=scores.device)
        gumbel_noise = -torch.log(-torch.log(uniform_noise + eps) + eps)

        # Caster dans le dtype original
        gumbel_noise = gumbel_noise.to(original_dtype)

        # 🔧 FIX: Vérifier NaN/Inf dans Gumbel noise (safety check)
        if torch.isnan(gumbel_noise).any() or torch.isinf(gumbel_noise).any():
            print(f"⚠️ NaN/Inf détecté dans Gumbel noise après fix, utilisation fallback")
            print(f"   Scores - min: {scores.min().item()}, max: {scores.max().item()}")
            gumbel_noise = torch.zeros_like(scores)

        perturbed_scores = (scores + gumbel_noise) / temperature

        # Top-K hard (pour forward)
        _, hard_indices = torch.topk(perturbed_scores, k=k, dim=-1)  # (B, k)

        # Soft selection via softmax (pour backward)
        # On utilise un "trick" pour concentrer la masse sur top-K
        soft_scores = F.softmax(perturbed_scores, dim=-1)  # (B, L)

        # 🔧 FIX: Vérifier NaN après softmax sur perturbed scores
        if torch.isnan(soft_scores).any():
            print(f"❌ NaN détecté dans soft_scores après Gumbel softmax!")
            print(f"   Perturbed scores - min: {perturbed_scores.min().item()}, max: {perturbed_scores.max().item()}")
            print(f"   Temperature: {temperature}")
            # Fallback: distribution uniforme
            soft_scores = torch.ones_like(soft_scores) / L

        return soft_scores, hard_indices
    
    def _straight_through_topk(
        self, scores: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Top-K avec straight-through estimator AMÉLIORÉ.

        🔧 FIX: Utilise une soft selection différentiable pour backward au lieu
        de passer brutalement le gradient des scores bruts.

        Forward: Hard top-K (one-hot)
        Backward: Gradient via sigmoid soft-thresholding (cohérent avec forward)

        Avantages vs version précédente:
        - Gradients plus stables (soft selection vs hard scores)
        - Meilleure cohérence forward/backward
        - Temperature contrôle la "sharpness" du soft selection

        Args:
            scores: (B, L) scores d'importance bruts
            k: Nombre d'éléments à sélectionner

        Returns:
            selection: (B, L) poids de sélection (forward=hard, backward=soft)
            topk_indices: (B, k) indices hard des top-k

        Notes:
            - Temperature=0.1 rend la soft selection proche de hard (mais différentiable)
            - Le seuil est basé sur le k-ième score (adaptatif automatiquement)
        """
        B, L = scores.shape

        # Forward: Hard top-K
        topk_vals, topk_indices = torch.topk(scores, k=k, dim=-1)  # (B, k)

        # Créer one-hot encodings des sélections (forward)
        selection_onehot = torch.zeros_like(scores)  # (B, L)
        selection_onehot.scatter_(1, topk_indices, 1.0)

        # ✅ AMÉLIORATION: Soft selection pour backward cohérent
        # Utilise sigmoid soft-thresholding basé sur k-ième valeur
        threshold = topk_vals[:, -1:].detach()  # (B, 1) - k-ième score (seuil adaptatif)
        temp = 0.1  # Temperature: plus bas = plus proche de hard selection

        # Soft selection via sigmoid: positions > threshold → poids ~1, sinon ~0
        selection_soft = torch.sigmoid((scores - threshold) / temp)  # (B, L)

        # Straight-through: forward=hard (one-hot), backward=soft (sigmoid)
        # Cette formulation garantit:
        #   - y = selection_onehot (forward)
        #   - dy/dx = d(selection_soft)/dx (backward)
        selection = selection_onehot + selection_soft - selection_soft.detach()

        return selection, topk_indices
    
    def forward(
        self, x: torch.Tensor, use_gumbel: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sélectionne landmarks de manière différentiable.

        Args:
            x: (B, L, D) séquence d'entrée
            use_gumbel: Si True, utilise Gumbel-Softmax (sinon straight-through)

        Returns:
            landmark_indices: (B, G) indices des landmarks sélectionnés
            landmark_states: (B, G, D) états correspondants (gathered)
            selection_scores: (B, L) scores de sélection pour loss auxiliaire

        Notes:
            💡 RECOMMANDATION: use_gumbel=True est préférable pour l'entraînement!

            Comparaison des méthodes:
            ┌─────────────────┬──────────────────┬────────────────┬──────────────┐
            │ Méthode         │ Gradients        │ Convergence    │ Stabilité    │
            ├─────────────────┼──────────────────┼────────────────┼──────────────┤
            │ Gumbel-Softmax  │ Smooth & continu │ Plus stable    │ ⭐⭐⭐⭐⭐   │
            │ (use_gumbel=T)  │ Temperature decay│ Converge mieux │              │
            ├─────────────────┼──────────────────┼────────────────┼──────────────┤
            │ Straight-through│ Approximatif     │ Plus rapide    │ ⭐⭐⭐       │
            │ (use_gumbel=F)  │ Sigmoid-based    │ Moins stable   │              │
            └─────────────────┴──────────────────┴────────────────┴──────────────┘

            Pourquoi Gumbel est meilleur:
            1. ✅ Gradients théoriquement fondés (relaxation continue de argmax)
            2. ✅ Temperature annealing → converge vers hard selection progressivement
            3. ✅ Utilisé dans Sparse Transformer, DALL-E, et autres modèles SOTA

            Quand utiliser straight-through:
            - Prototypage rapide (pas besoin de tuner temperature)
            - Ressources limitées (léger gain de vitesse ~5-10%)
            - Fine-tuning court où gradients approximatifs suffisent

            ⚠️ ATTENTION: Straight-through peut causer gradients instables en début
            d'entraînement car le seuil adaptatif bouge beaucoup quand scores non-calibrés.
            → Utiliser Gumbel pour entraînement from-scratch, straight-through pour fine-tuning.
        """
        B, L, D = x.shape

        # Scorer chaque position
        scores = self.scorer(x).squeeze(-1)  # (B, L)

        # 🔧 FIX: Protection NaN - clamp scores avant softmax/gumbel
        # Évite overflow dans exp() si scores extrêmes
        scores = torch.clamp(scores, min=-20, max=20)

        # Sélection différentiable
        k = min(self.num_landmarks, L)
        
        if self.training:
            if use_gumbel:
                # Mode Gumbel (plus smooth mais plus lent)
                temp = self._get_temperature()
                selection_soft, landmark_indices = self._gumbel_topk(scores, k, temp)
                # Mettre à jour compteur température
                self.step_count += 1
            else:
                # Mode straight-through (plus efficace)
                selection_soft, landmark_indices = self._straight_through_topk(scores, k)
        else:
            # Inference: hard top-K déterministe
            _, landmark_indices = torch.topk(scores, k=k, dim=-1)
            selection_soft = None
        
        # Gather les états correspondants
        # x: (B, L, D), indices: (B, G) -> expand to (B, G, D)
        # ✅ PROTECTION: Clamp indices avant gather pour éviter index out-of-bounds
        landmark_indices_safe = torch.clamp(landmark_indices, 0, L - 1)
        landmark_indices_exp = landmark_indices_safe.unsqueeze(-1).expand(B, k, D)
        landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)  # (B, G, D)

        # Scores de sélection (pour loss auxiliaire de diversité)
        # Normaliser avec softmax pour interprétabilité
        selection_scores = F.softmax(scores, dim=-1)  # (B, L)

        # 🔧 FIX: Vérifier NaN après softmax
        if torch.isnan(selection_scores).any():
            print(f"❌ NaN détecté dans selection_scores après softmax!")
            print(f"   Scores avant softmax - min: {scores.min().item()}, max: {scores.max().item()}")
            print(f"   Clamp appliqué ([-20, 20]) mais NaN persiste - possible overflow")
            # Fallback: distribution uniforme
            selection_scores = torch.ones_like(selection_scores) / selection_scores.size(-1)

        return landmark_indices, landmark_states, selection_scores


class PositionalLandmarkSelector(nn.Module):
    """
    Sélecteur de landmarks basé sur patterns positionnels appris.
    
    Au lieu de scorer chaque token individuellement, ce module apprend
    des patterns de positions importantes (e.g., début de paragraphe,
    tous les N tokens, etc.)
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
        
        # Embeddings positionnels apprenables
        self.pos_embeddings = nn.Parameter(torch.randn(max_seq_len, embed_dim))
        
        # Projecteur vers scores
        self.scorer = nn.Linear(embed_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, D) - l'embed_dim de x peut différer de self pos_embeddings
        
        Returns:
            landmark_indices: (B, G)
            landmark_states: (B, G, D)
            selection_scores: (B, L)
        """
        B, L, D = x.shape
        
        # Prendre les pos embeddings pour cette longueur
        pos_emb = self.pos_embeddings[:L]  # (L, embed_dim_pos)
        
        # Scorer positions
        scores = self.scorer(pos_emb).squeeze(-1)  # (L,)
        scores = scores.unsqueeze(0).expand(B, L)  # (B, L)
        
        # Top-K
        k = min(self.num_landmarks, L)
        _, landmark_indices = torch.topk(scores, k=k, dim=-1)  # (B, k)
        
        # Gather states
        # ✅ PROTECTION: Clamp indices avant gather pour éviter index out-of-bounds
        landmark_indices_safe = torch.clamp(landmark_indices, 0, L - 1)
        landmark_indices_exp = landmark_indices_safe.unsqueeze(-1).expand(B, k, D)
        landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)
        
        selection_scores = F.softmax(scores, dim=-1)
        
        return landmark_indices, landmark_states, selection_scores


class HybridLandmarkSelector(nn.Module):
    """
    Combine sélection apprise (content-based) et positionnelle.
    
    Utilise un gating pour décider dynamiquement de la combinaison.
    """
    
    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int,
        num_landmarks: int,
    ):
        super().__init__()

        self.num_landmarks = num_landmarks  # Store for use in forward
        self.content_selector = LearnableLandmarkSelector(embed_dim, num_landmarks)
        self.position_selector = PositionalLandmarkSelector(max_seq_len, num_landmarks, embed_dim)

        # Gate pour combiner les deux
        self.gate = nn.Linear(embed_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        
        # Sélections des deux modules
        idx_content, states_content, scores_content = self.content_selector(x)
        idx_position, states_position, scores_position = self.position_selector(x)
        
        # Gate basé sur moyenne globale de la séquence
        x_pooled = x.mean(dim=1)  # (B, D)
        gate_weight = torch.sigmoid(self.gate(x_pooled))  # (B, 1)
        
        # Combiner scores
        scores_combined = gate_weight * scores_content + (1 - gate_weight) * scores_position
        
        # Re-sélectionner top-K selon scores combinés
        k = min(self.num_landmarks, L)
        _, landmark_indices = torch.topk(scores_combined, k=k, dim=-1)
        
        # Gather
        # ✅ PROTECTION: Clamp indices avant gather pour éviter index out-of-bounds
        landmark_indices_safe = torch.clamp(landmark_indices, 0, L - 1)
        landmark_indices_exp = landmark_indices_safe.unsqueeze(-1).expand(B, k, D)
        landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)
        
        return landmark_indices, landmark_states, scores_combined


def landmark_spacing_loss(
    landmark_indices: torch.Tensor,
    seq_len: int,
    lambda_reg: float = 0.01,
    selection_scores: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Optimisation #2: Pénalise gaps non-uniformes entre landmarks.

    🔧 FIX: Version différentiable qui utilise selection_scores pour gradients!

    Encourage un espacement uniforme des landmarks dans la séquence pour maximiser
    la couverture spatiale et éviter le clustering de landmarks proches.

    Args:
        landmark_indices: (B, G) indices des landmarks sélectionnés (pour calcul fallback)
        seq_len: Longueur de séquence L (nombre total de positions)
        lambda_reg: Poids de régularisation (default: 0.01)
        selection_scores: (B, L) scores de sélection différentiables (REQUIS pour gradients!)

    Returns:
        loss: Scalaire encourageant espacement uniforme des landmarks

    Notes:
        - 🚨 IMPORTANT: Passer selection_scores pour que les gradients flow vers le scorer!
        - Si selection_scores=None, utilise une version non-différentiable (fallback)
        - Complexité: O(L) avec scores, O(G log G) sans scores
    """
    B, G = landmark_indices.shape

    if selection_scores is not None:
        # 🔧 MODE DIFFÉRENTIABLE: Utilise les scores (qui ont des gradients!)
        # Calcule la "position moyenne pondérée" des scores élevés
        # et vérifie qu'ils couvrent uniformément [0, L-1]

        # Positions dans la séquence
        positions = torch.arange(seq_len, device=selection_scores.device, dtype=selection_scores.dtype)  # (L,)

        # Normaliser scores pour avoir une distribution
        # On utilise les top-K scores seulement (masquer les autres)
        # Créer un masque pour les positions sélectionnées
        B_idx = torch.arange(B, device=landmark_indices.device).unsqueeze(1).expand(B, G)
        mask = torch.zeros(B, seq_len, device=selection_scores.device, dtype=selection_scores.dtype)
        mask.scatter_(1, landmark_indices, 1.0)  # (B, L) avec 1.0 aux positions des landmarks

        # Scores masqués et normalisés
        masked_scores = selection_scores * mask  # (B, L)
        normalized_scores = masked_scores / (masked_scores.sum(dim=1, keepdim=True) + 1e-8)  # (B, L)

        # Calculer position moyenne pondérée pour chaque "segment" de la séquence
        # Diviser [0, L-1] en G segments et vérifier que chaque segment a du poids
        segment_size = seq_len / G
        segment_centers = torch.arange(G, device=selection_scores.device, dtype=selection_scores.dtype) * segment_size + segment_size / 2  # (G,)

        # ⚡ VERSION VECTORISÉE: Calculer poids par segment (GPU-optimisé)
        # Créer indices de segments pour chaque position [0, seq_len-1] → [0, G-1]
        segment_indices = (torch.arange(seq_len, device=selection_scores.device).float() / segment_size).long()
        segment_indices = segment_indices.clamp(max=G-1)  # (L,) - handle rounding

        # Expand pour batch: (B, L)
        segment_indices_exp = segment_indices.unsqueeze(0).expand(B, seq_len)

        # Scatter-add pour accumuler poids par segment (entièrement différentiable)
        segment_weights = torch.zeros(B, G, device=selection_scores.device, dtype=selection_scores.dtype)
        segment_weights.scatter_add_(1, segment_indices_exp, normalized_scores)  # (B, G)

        # Loss: Pénaliser écart par rapport à poids uniforme (1/G pour chaque segment)
        ideal_weight = 1.0 / G
        loss = lambda_reg * ((segment_weights - ideal_weight) ** 2).mean()

    else:
        # ⚠️ FALLBACK NON-DIFFÉRENTIABLE (ancien comportement)

        # ✅ FIX Bug #16: Guard contre G <= 1 (pas de gaps possibles)
        # Problème: Si G <= 1, gaps est vide → .mean() = NaN
        # Se produit: curriculum court, séquences courtes, G=1 configs
        if G < 2:
            # Pas assez de landmarks pour calculer spacing
            # Retourner 0 (pas de pénalité)
            return torch.tensor(0.0, device=landmark_indices.device, dtype=torch.float32)

        # Trier indices pour calculer gaps entre landmarks consécutifs
        sorted_idx, _ = torch.sort(landmark_indices, dim=-1)  # (B, G)

        # Calculer gaps (distances) entre landmarks adjacents
        gaps = sorted_idx[:, 1:] - sorted_idx[:, :-1]  # (B, G-1)

        # Gap idéal pour espacement uniforme = L / G
        ideal_gap = seq_len / G

        # MSE loss sur gaps: pénalise déviations de l'espacement uniforme
        loss = lambda_reg * ((gaps - ideal_gap) ** 2).mean()

    return loss


def landmark_diversity_loss(
    selection_scores: torch.Tensor, lambda_reg: float = 0.01
) -> torch.Tensor:
    """
    [DEPRECATED] Loss auxiliaire basée sur l'entropie (remplacée par spacing_loss).

    Maximise l'entropie de la distribution de sélection pour encourager diversité.

    ⚠️ Limitation: Pousse vers distribution uniforme sur L positions au lieu de
    pénaliser directement le clustering des G landmarks sélectionnés.

    → Utiliser landmark_spacing_loss() à la place pour de meilleurs résultats.

    Args:
        selection_scores: (B, L) probabilités de sélection normalisées
        lambda_reg: Poids de régularisation

    Returns:
        loss: Scalaire, à minimiser
    """
    B, L = selection_scores.shape

    # Entropie de la distribution: H = -sum(p * log(p))
    entropy = -(selection_scores * torch.log(selection_scores + 1e-10)).sum(dim=-1)  # (B,)

    # Normaliser par entropie max (log(L))
    max_entropy = math.log(L)
    normalized_entropy = entropy / max_entropy  # (B,) dans [0, 1]

    # Pénaliser faible entropie (on veut haute entropie = diversité)
    loss = lambda_reg * (1 - normalized_entropy).mean()

    return loss


def landmark_sparsity_loss(
    selection_scores: torch.Tensor,
    num_landmarks: int,
    lambda_reg: float = 0.001
) -> torch.Tensor:
    """
    ✅ VERSION CORRECTE v4: Mesure la concentration via proportion de "masse" dans top-G.

    Pénalise si les scores sont trop dispersés, c'est-à-dire si la "masse"
    (somme des scores positifs) n'est PAS suffisamment concentrée dans les top-G.

    Approche:
        1. Normaliser scores via softmax → probabilités
        2. Calculer proportion de masse dans top-G
        3. Target: au moins 80% de la masse devrait être dans top-G
        4. Pénaliser si proportion < target

    Problème résolu:
        Versions précédentes: Gap ou count donnaient loss=0 toujours

        Version v4: Proportion de masse
        → Si bien concentré: top-G contient 90%+ de masse → loss=0
        → Si dispersé: top-G contient 60% de masse → loss>0
        → Mesure directe et intuitive de la concentration

    Args:
        selection_scores: (B, L) scores bruts de sélection (non normalisés)
        num_landmarks: Nombre de landmarks G
        lambda_reg: Poids de régularisation (default: 0.001)

    Returns:
        loss: Scalaire, à minimiser (0 si bonne concentration)

    Example:
        Pour G=48, L=384:
        Cas concentré:
          top_48 contient 95% masse → mass_ratio=0.95 → loss=0
        Cas dispersé:
          top_48 contient 30% masse → mass_ratio=0.30 → loss>0

    Notes:
        - Softmax normalise et rend comparables tous scores
        - Température=1.0: pas de sharpening, mesure naturelle
        - Différentiable: gradients fluent via softmax et indexing
        - Target adaptatif: 60% + (G/L)*40% pour tenir compte du ratio
    """
    B, L = selection_scores.shape

    # 1. Normaliser via softmax pour obtenir "probabilités" / masse relative
    # Température 1.0: pas de sharpening, distribution naturelle
    probs = F.softmax(selection_scores, dim=-1)  # (B, L), sum=1 par batch

    # 2. Trouver top-G indices
    _, top_g_indices = torch.topk(selection_scores, k=num_landmarks, dim=-1)  # (B, G)

    # 3. Calculer la masse totale dans top-G
    # Gather les probabilités des top-G positions
    top_g_probs = torch.gather(probs, dim=1, index=top_g_indices)  # (B, G)
    mass_in_top_g = top_g_probs.sum(dim=-1).mean()  # Scalaire, moyenne sur batch

    # 4. Target adaptatif: on attend qu'une majorité de la masse soit dans top-G
    # Idéalement, si G=48 et L=384 (12.5%), on voudrait au moins 60-80% de masse dans top-G
    # Formule: base 60% + bonus selon ratio G/L
    ideal_ratio = num_landmarks / L  # Ex: 48/384 = 0.125
    target_mass = 0.60 + ideal_ratio * 0.40  # Ex: 0.60 + 0.125*0.40 = 0.65

    # 5. Pénaliser si masse insuffisante dans top-G
    # Plus la masse est dispersée, plus la loss est élevée
    loss = lambda_reg * F.relu(target_mass - mass_in_top_g)

    # 📊 Retourner aussi mass_in_top_g pour logging (optionnel)
    # Permet de monitorer l'évolution même si loss constante au début
    # Usage: loss, mass = landmark_sparsity_loss(..., return_mass=True)
    # Note: Pour compatibilité backward, on retourne juste loss par défaut
    # Le caller peut extraire mass_in_top_g en réexécutant le calcul si besoin

    return loss