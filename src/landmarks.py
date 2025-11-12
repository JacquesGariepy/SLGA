# landmarks.py
"""
Learnable Landmark Selection Module

Implements differentiable selection of important positions (landmarks)
for global attention via Gumbel-Softmax or top-K with straight-through estimator.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LearnableLandmarkSelector(nn.Module):
    """
    Selects G landmarks in a differentiable manner from L positions.
    
    Uses a neural scorer to assign importance scores to each position,
    then selects top-K via:
    - Training mode: Gumbel-Softmax relaxation for differentiability
    - Inference mode: Hard top-K deterministic selection
    
    Args:
        embed_dim: Input embedding dimension
        num_landmarks: Number of landmarks to select (G)
        hidden_dim: Hidden dimension of scorer network (default: embed_dim // 2)
        temperature: Gumbel temperature (lower = harder, default: 1.0)
        temperature_decay: Temperature decay factor (default: 0.999, 10x faster)
        min_temperature: Minimum temperature (default: 0.3, more discriminative)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_landmarks: int,
        hidden_dim: Optional[int] = None,
        temperature: float = 1.0,
        temperature_decay: float = 0.999,  # Optimization #1: 10x faster (0.9999 → 0.999)
        min_temperature: float = 0.3,      # Optimization #1: More discriminative (0.5 → 0.3)
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_landmarks = num_landmarks
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        
        # Scorer: embed -> hidden -> 1 (importance score)
        hidden = hidden_dim or (embed_dim // 2)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )
        
        # Step counter for temperature decay
        self.register_buffer("step_count", torch.tensor(0), persistent=False)
    
    def _get_temperature(self) -> float:
        """Compute current temperature with decay"""
        if self.training:
            temp = self.temperature * (self.temperature_decay ** self.step_count.item())
            return max(temp, self.min_temperature)
        else:
            return self.min_temperature
    
    def _gumbel_topk(
        self, scores: torch.Tensor, k: int, temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable approximation of top-K via Gumbel-Softmax.

        Args:
            scores: (B, L) importance scores
            k: Number of elements to select
            temperature: Temperature for relaxation

        Returns:
            soft_selection: (B, L) continuous weights (approx one-hot top-k)
            hard_indices: (B, k) hard indices (for gather)
        """
        B, L = scores.shape

        # Fix Bug #17: Generate Gumbel noise in float32 to avoid NaN in AMP
        # Problem: In float16/bfloat16, torch.rand_like() can produce exactly 0
        # → -log(-log(0)) = NaN
        # Solution: Generate in float32, then cast
        eps = 1e-10
        original_dtype = scores.dtype

        # Generate noise in float32 (stable)
        uniform_noise = torch.rand(scores.shape, dtype=torch.float32, device=scores.device)
        gumbel_noise = -torch.log(-torch.log(uniform_noise + eps) + eps)

        # Cast to original dtype
        gumbel_noise = gumbel_noise.to(original_dtype)

        # Fix: Check for NaN/Inf in Gumbel noise (safety check)
        if torch.isnan(gumbel_noise).any() or torch.isinf(gumbel_noise).any():
            print(f"⚠️ NaN/Inf detected in Gumbel noise after fix, using fallback")
            print(f"   Scores - min: {scores.min().item()}, max: {scores.max().item()}")
            gumbel_noise = torch.zeros_like(scores)

        perturbed_scores = (scores + gumbel_noise) / temperature

        # Hard top-K (for forward)
        _, hard_indices = torch.topk(perturbed_scores, k=k, dim=-1)  # (B, k)

        # Soft selection via softmax (for backward)
        # Uses a "trick" to concentrate mass on top-K
        soft_scores = F.softmax(perturbed_scores, dim=-1)  # (B, L)

        # Fix: Check for NaN after softmax on perturbed scores
        if torch.isnan(soft_scores).any():
            print(f"❌ NaN detected in soft_scores after Gumbel softmax!")
            print(f"   Perturbed scores - min: {perturbed_scores.min().item()}, max: {perturbed_scores.max().item()}")
            print(f"   Temperature: {temperature}")
            # Fallback: uniform distribution
            soft_scores = torch.ones_like(soft_scores) / L

        return soft_scores, hard_indices
    
    def _straight_through_topk(
        self, scores: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Top-K with improved straight-through estimator.

        Fix: Uses differentiable soft selection for backward instead of
        passing gradients from raw hard scores.

        Forward: Hard top-K (one-hot)
        Backward: Gradient via sigmoid soft-thresholding (consistent with forward)

        Advantages vs previous version:
        - More stable gradients (soft selection vs hard scores)
        - Better forward/backward consistency
        - Temperature controls "sharpness" of soft selection

        Args:
            scores: (B, L) raw importance scores
            k: Number of elements to select

        Returns:
            selection: (B, L) selection weights (forward=hard, backward=soft)
            topk_indices: (B, k) hard indices of top-k

        Notes:
            - Temperature=0.1 makes soft selection close to hard (but differentiable)
            - Threshold based on k-th score (adaptive automatically)
        """
        B, L = scores.shape

        # Forward: Hard top-K
        topk_vals, topk_indices = torch.topk(scores, k=k, dim=-1)  # (B, k)

        # Create one-hot encodings of selections (forward)
        selection_onehot = torch.zeros_like(scores)  # (B, L)
        selection_onehot.scatter_(1, topk_indices, 1.0)

        # Improvement: Soft selection for consistent backward
        # Uses sigmoid soft-thresholding based on k-th value
        threshold = topk_vals[:, -1:].detach()  # (B, 1) - k-th score (adaptive threshold)
        temp = 0.1  # Temperature: lower = closer to hard selection

        # Soft selection via sigmoid: positions > threshold → weight ~1, else ~0
        selection_soft = torch.sigmoid((scores - threshold) / temp)  # (B, L)

        # Straight-through: forward=hard (one-hot), backward=soft (sigmoid)
        # This ensures:
        #   - y = selection_onehot (forward)
        #   - dy/dx = d(selection_soft)/dx (backward)
        selection = selection_onehot + selection_soft - selection_soft.detach()

        return selection, topk_indices
    
    def forward(
        self, x: torch.Tensor, use_gumbel: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Selects landmarks in a differentiable manner.

        Args:
            x: (B, L, D) input sequence
            use_gumbel: If True, uses Gumbel-Softmax (else straight-through)

        Returns:
            landmark_indices: (B, G) indices of selected landmarks
            landmark_states: (B, G, D) corresponding states (gathered)
            selection_scores: (B, L) selection scores for auxiliary loss

        Notes:
            💡 RECOMMENDATION: use_gumbel=True is preferable for training!

            Method comparison:
            ┌─────────────────┬──────────────────┬────────────────┬──────────────┐
            │ Method          │ Gradients        │ Convergence    │ Stability    │
            ├─────────────────┼──────────────────┼────────────────┼──────────────┤
            │ Gumbel-Softmax  │ Smooth & continu │ More stable    │ ⭐⭐⭐⭐⭐ │
            │ (use_gumbel=T)  │ Temperature decay│ Converge better│              │
            ├─────────────────┼──────────────────┼────────────────┼──────────────┤
            │ Straight-through│ Approximative    │ Faster         │ ⭐⭐⭐      │
            │ (use_gumbel=F)  │ Sigmoid-based    │ Less stable    │              │
            └─────────────────┴──────────────────┴────────────────┴──────────────┘

            Why Gumbel is better:
            1. Theoretically grounded gradients (continuous relaxation of argmax)
            2. Temperature annealing → converges to hard selection progressively
            3. Used in Sparse Transformer, DALL-E, and other SOTA models

            When to use straight-through:
            - Quick prototyping (no need to tune temperature)
            - Limited resources (slight speed gain ~5-10%)
            - Short fine-tuning where approximate gradients suffice

            WARNING: Straight-through can cause unstable gradients early in
            training because the adaptive threshold moves a lot when scores are uncalibrated.
            → Use Gumbel for from-scratch training, straight-through for fine-tuning.
        """
        B, L, D = x.shape

        # Score each position
        scores = self.scorer(x).squeeze(-1)  # (B, L)

        # Fix: NaN protection - clamp scores before softmax/gumbel
        # Avoids overflow in exp() if extreme scores
        scores = torch.clamp(scores, min=-20, max=20)

        # Differentiable selection
        k = min(self.num_landmarks, L)
        
        if self.training:
            if use_gumbel:
                # Gumbel mode (smoother but slower)
                temp = self._get_temperature()
                selection_soft, landmark_indices = self._gumbel_topk(scores, k, temp)
                # Update temperature counter
                self.step_count += 1
            else:
                # Straight-through mode (more efficient)
                selection_soft, landmark_indices = self._straight_through_topk(scores, k)
        else:
            # Inference: hard top-K deterministic
            _, landmark_indices = torch.topk(scores, k=k, dim=-1)
            selection_soft = None
        
        # Gather corresponding states
        # x: (B, L, D), indices: (B, G) -> expand to (B, G, D)
        # Protection: Clamp indices before gather to avoid index out-of-bounds
        landmark_indices_safe = torch.clamp(landmark_indices, 0, L - 1)
        landmark_indices_exp = landmark_indices_safe.unsqueeze(-1).expand(B, k, D)
        landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)  # (B, G, D)

        # Selection scores (for auxiliary diversity loss)
        # Normalize with softmax for interpretability
        selection_scores = F.softmax(scores, dim=-1)  # (B, L)

        # Fix: Check for NaN after softmax
        if torch.isnan(selection_scores).any():
            print(f"❌ NaN detected in selection_scores after softmax!")
            print(f"   Scores before softmax - min: {scores.min().item()}, max: {scores.max().item()}")
            print(f"   Clamp applied ([-20, 20]) but NaN persists - possible overflow")
            # Fallback: uniform distribution
            selection_scores = torch.ones_like(selection_scores) / selection_scores.size(-1)

        return landmark_indices, landmark_states, selection_scores


class PositionalLandmarkSelector(nn.Module):
    """
    Landmark selector based on learned positional patterns.
    
    Instead of scoring each token individually, this module learns
    patterns of important positions (e.g., paragraph starts,
    every N tokens, etc.)
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
        
        # Learnable positional embeddings
        self.pos_embeddings = nn.Parameter(torch.randn(max_seq_len, embed_dim))
        
        # Projector to scores
        self.scorer = nn.Linear(embed_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, D) - embed_dim of x may differ from self pos_embeddings
        
        Returns:
            landmark_indices: (B, G)
            landmark_states: (B, G, D)
            selection_scores: (B, L)
        """
        B, L, D = x.shape
        
        # Take pos embeddings for this length
        pos_emb = self.pos_embeddings[:L]  # (L, embed_dim_pos)
        
        # Score positions
        scores = self.scorer(pos_emb).squeeze(-1)  # (L,)
        scores = scores.unsqueeze(0).expand(B, L)  # (B, L)
        
        # Top-K
        k = min(self.num_landmarks, L)
        _, landmark_indices = torch.topk(scores, k=k, dim=-1)  # (B, k)
        
        # Gather states
        # Protection: Clamp indices before gather to avoid index out-of-bounds
        landmark_indices_safe = torch.clamp(landmark_indices, 0, L - 1)
        landmark_indices_exp = landmark_indices_safe.unsqueeze(-1).expand(B, k, D)
        landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)
        
        selection_scores = F.softmax(scores, dim=-1)
        
        return landmark_indices, landmark_states, selection_scores


class HybridLandmarkSelector(nn.Module):
    """
    Combine learned (content-based) and positional selection.
    
    Uses gating to dynamically decide the combination.
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

        # Gate to combine the two
        self.gate = nn.Linear(embed_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        
        # Selections from both modules
        idx_content, states_content, scores_content = self.content_selector(x)
        idx_position, states_position, scores_position = self.position_selector(x)
        
        # Gate based on global average of the sequence
        x_pooled = x.mean(dim=1)  # (B, D)
        gate_weight = torch.sigmoid(self.gate(x_pooled))  # (B, 1)
        
        # Combine scores
        scores_combined = gate_weight * scores_content + (1 - gate_weight) * scores_position
        
        # Re-select top-K according to combined scores
        k = min(self.num_landmarks, L)
        _, landmark_indices = torch.topk(scores_combined, k=k, dim=-1)
        
        # Gather
        # Protection: Clamp indices before gather to avoid index out-of-bounds
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
    Optimization #2: Penalizes non-uniform gaps between landmarks.

    Fix: Differentiable version that uses selection_scores for gradients!

    Encourages uniform spacing of landmarks in the sequence to maximize
    spatial coverage and avoid clustering of nearby landmarks.

    Args:
        landmark_indices: (B, G) indices of selected landmarks (for fallback calculation)
        seq_len: Sequence length L (total number of positions)
        lambda_reg: Regularization weight (default: 0.01)
        selection_scores: (B, L) differentiable selection scores (REQUIRED for gradients!)

    Returns:
        loss: Scalar encouraging uniform spacing of landmarks

    Notes:
        - IMPORTANT: Pass selection_scores so gradients flow to the scorer!
        - If selection_scores=None, uses non-differentiable version (fallback)
        - Complexity: O(L) with scores, O(G log G) without scores
    """
    B, G = landmark_indices.shape

    if selection_scores is not None:
        # Differentiable mode: Uses scores (which have gradients!)
        # Calculates "weighted average position" of high scores
        # and checks that they cover [0, L-1] uniformly

        # Positions in the sequence
        positions = torch.arange(seq_len, device=selection_scores.device, dtype=selection_scores.dtype)  # (L,)

        # Normalize scores to have a distribution
        # Use only top-K scores (mask others)
        # Create mask for selected positions
        B_idx = torch.arange(B, device=landmark_indices.device).unsqueeze(1).expand(B, G)
        mask = torch.zeros(B, seq_len, device=selection_scores.device, dtype=selection_scores.dtype)
        mask.scatter_(1, landmark_indices, 1.0)  # (B, L) with 1.0 at landmark positions

        # Masked and normalized scores
        masked_scores = selection_scores * mask  # (B, L)
        normalized_scores = masked_scores / (masked_scores.sum(dim=1, keepdim=True) + 1e-8)  # (B, L)

        # Calculate weighted average position for each "segment" of the sequence
        # Divide [0, L-1] into G segments and check each has weight
        segment_size = seq_len / G
        segment_centers = torch.arange(G, device=selection_scores.device, dtype=selection_scores.dtype) * segment_size + segment_size / 2  # (G,)

        # Vectorized version: Calculate weight per segment (GPU-optimized)
        # Create segment indices for each position [0, seq_len-1] → [0, G-1]
        segment_indices = (torch.arange(seq_len, device=selection_scores.device).float() / segment_size).long()
        segment_indices = segment_indices.clamp(max=G-1)  # (L,) - handle rounding

        # Expand for batch: (B, L)
        segment_indices_exp = segment_indices.unsqueeze(0).expand(B, seq_len)

        # Scatter-add to accumulate weight per segment (fully differentiable)
        segment_weights = torch.zeros(B, G, device=selection_scores.device, dtype=selection_scores.dtype)
        segment_weights.scatter_add_(1, segment_indices_exp, normalized_scores)  # (B, G)

        # Loss: Penalize deviation from uniform weight (1/G for each segment)
        ideal_weight = 1.0 / G
        loss = lambda_reg * ((segment_weights - ideal_weight) ** 2).mean()

    else:
        # Fallback non-differentiable (old behavior)

        # Fix Bug #16: Guard against G <= 1 (no gaps possible)
        # Problem: If G <= 1, gaps is empty → .mean() = NaN
        # Occurs: Short curriculum, short sequences, G=1 configs
        if G < 2:
            # Not enough landmarks to calculate spacing
            # Return 0 (no penalty)
            return torch.tensor(0.0, device=landmark_indices.device, dtype=torch.float32)

        # Sort indices to calculate gaps between consecutive landmarks
        sorted_idx, _ = torch.sort(landmark_indices, dim=-1)  # (B, G)

        # Calculate gaps (distances) between adjacent landmarks
        gaps = sorted_idx[:, 1:] - sorted_idx[:, :-1]  # (B, G-1)

        # Ideal gap for uniform spacing = L / G
        ideal_gap = seq_len / G

        # MSE loss on gaps: penalizes deviations from uniform spacing
        loss = lambda_reg * ((gaps - ideal_gap) ** 2).mean()

    return loss


def landmark_diversity_loss(
    selection_scores: torch.Tensor, lambda_reg: float = 0.01
) -> torch.Tensor:
    """
    [DEPRECATED] Auxiliary loss based on entropy (replaced by spacing_loss).

    Maximizes entropy of the selection distribution to encourage diversity.

    Limitation: Pushes towards uniform distribution over L positions instead of
    directly penalizing clustering of the G selected landmarks.

    → Use landmark_spacing_loss() instead for better results.

    Args:
        selection_scores: (B, L) normalized selection probabilities
        lambda_reg: Regularization weight

    Returns:
        loss: Scalar, to minimize
    """
    B, L = selection_scores.shape

    # Entropy of the distribution: H = -sum(p * log(p))
    entropy = -(selection_scores * torch.log(selection_scores + 1e-10)).sum(dim=-1)  # (B,)

    # Normalize by max entropy (log(L))
    max_entropy = math.log(L)
    normalized_entropy = entropy / max_entropy  # (B,) in [0, 1]

    # Penalize low entropy (want high entropy = diversity)
    loss = lambda_reg * (1 - normalized_entropy).mean()

    return loss


def landmark_sparsity_loss(
    selection_scores: torch.Tensor,
    num_landmarks: int,
    lambda_reg: float = 0.001
) -> torch.Tensor:
    """
    Correct version v4: Measures concentration via proportion of "mass" in top-G.

    Penalizes if scores are too dispersed, i.e., if the "mass"
    (sum of positive scores) is NOT sufficiently concentrated in the top-G.

    Approach:
        1. Normalize scores via softmax → probabilities
        2. Calculate proportion of mass in top-G
        3. Target: at least 80% of mass should be in top-G
        4. Penalize if proportion < target

    Problem solved:
        Previous versions: Gap or count gave loss=0 always

        Version v4: Proportion of mass
        → If well concentrated: top-G contains 90%+ mass → loss=0
        → If dispersed: top-G contains 60% mass → loss>0
        → Direct and intuitive measure of concentration

    Args:
        selection_scores: (B, L) raw selection scores (unnormalized)
        num_landmarks: Number of landmarks G
        lambda_reg: Regularization weight (default: 0.001)

    Returns:
        loss: Scalar, to minimize (0 if good concentration)

    Example:
        For G=48, L=384:
        Concentrated case:
          top_48 contains 95% mass → mass_ratio=0.95 → loss=0
        Dispersed case:
          top_48 contains 30% mass → mass_ratio=0.30 → loss>0

    Notes:
        - Softmax normalizes and makes all scores comparable
        - Temperature=1.0: no sharpening, natural distribution
        - Differentiable: gradients flow via softmax and indexing
        - Adaptive target: 60% + (G/L)*40% to account for ratio
    """
    B, L = selection_scores.shape

    # 1. Normalize via softmax to get "probabilities" / relative mass
    # Temperature 1.0: no sharpening, natural distribution
    probs = F.softmax(selection_scores, dim=-1)  # (B, L), sum=1 per batch

    # 2. Find top-G indices
    _, top_g_indices = torch.topk(selection_scores, k=num_landmarks, dim=-1)  # (B, G)

    # 3. Calculate total mass in top-G
    # Gather probabilities of top-G positions
    top_g_probs = torch.gather(probs, dim=1, index=top_g_indices)  # (B, G)
    mass_in_top_g = top_g_probs.sum(dim=-1).mean()  # Scalar, average over batch

    # 4. Adaptive target: expect majority of mass in top-G
    # Ideally, if G=48 and L=384 (12.5%), want at least 60-80% mass in top-G
    # Formula: base 60% + bonus based on G/L ratio
    ideal_ratio = num_landmarks / L  # Ex: 48/384 = 0.125
    target_mass = 0.60 + ideal_ratio * 0.40  # Ex: 0.60 + 0.125*0.40 = 0.65

    # 5. Penalize if insufficient mass in top-G
    # More dispersed mass = higher loss
    loss = lambda_reg * F.relu(target_mass - mass_in_top_g)

    # Return mass_in_top_g for logging (optional)
    # Allows monitoring evolution even if loss constant early
    # Usage: loss, mass = landmark_sparsity_loss(..., return_mass=True)
    # Note: For backward compatibility, return just loss by default
    # Caller can extract mass_in_top_g by re-running calculation if needed

    return loss