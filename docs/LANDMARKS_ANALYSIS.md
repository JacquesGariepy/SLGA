# Analyse exhaustive: `src/landmarks.py` (376 lignes)

**Système de sélection de landmarks appris pour SLGA-Plus**

---

## 📋 Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [LearnableLandmarkSelector](#1-learnablelandmarkselector)
3. [PositionalLandmarkSelector](#2-positionallandmarkselector)
4. [HybridLandmarkSelector](#3-hybridlandmarkselector)
5. [Loss auxiliaires](#4-loss-auxiliaires)
6. [Analyse de complexité](#5-analyse-de-complexité)
7. [Trade-offs et recommandations](#6-trade-offs-et-recommandations)

---

## Vue d'ensemble

Ce module implémente **3 stratégies de sélection de landmarks différentiables** pour l'attention globale:

| Sélecteur | Type | Différentiabilité | Complexité |
|-----------|------|-------------------|------------|
| **LearnableLandmarkSelector** | Content-based | Gumbel-Softmax / Straight-through | O(L·D) |
| **PositionalLandmarkSelector** | Position-based | Top-K déterministe | O(L) |
| **HybridLandmarkSelector** | Fusion adaptive | Gating appris | O(L·D) |

---

## 1. LearnableLandmarkSelector

**Lignes 17-173** : Sélection content-based avec mécanismes différentiables

### 1.1 Architecture (lignes 35-62)

```python
# Lignes 52-59: Scorer neural 2-couches
self.scorer = nn.Sequential(
    nn.Linear(embed_dim, hidden),      # D → D/2
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(hidden, 1),              # D/2 → 1
)
```

**Design**:
- **Input**: (B, L, D) séquence d'embeddings
- **Output**: (B, L) scores d'importance
- **Hidden dim**: D/2 par défaut (compromis capacité/efficacité)

**Justification GELU**:
- Plus smooth que ReLU → gradients moins bruités
- Meilleure convergence observée empiriquement

### 1.2 Temperature decay (lignes 64-70)

```python
# Lignes 66-68: Décroissance exponentielle
temp = self.temperature * (self.temperature_decay ** step_count)
return max(temp, self.min_temperature)
```

**Mécanisme**:
1. **Début training**: temp = 1.0 (soft selection)
2. **Progressivement**: temp → 0.5 (harder selection)
3. **Inference**: temp = 0.5 (presque déterministe)

**Paramètres**:
- `temperature_decay = 0.9999` → très lent (intentionnel)
- À 10k steps: temp ≈ 0.905
- À 100k steps: temp ≈ 0.367
- Min clampé à 0.5

**⚠️ Observation**: Décroissance peut-être **trop conservatrice** pour convergence rapide.

### 1.3 Gumbel-Softmax (lignes 72-100)

```python
# Lignes 89-91: Injection de bruit Gumbel
gumbel_noise = -log(-log(U)) where U ~ Uniform(0,1)
perturbed_scores = (scores + gumbel_noise) / temperature

# Lignes 94: Top-K hard pour forward
_, hard_indices = torch.topk(perturbed_scores, k=k)

# Lignes 98: Soft distribution pour backward
soft_scores = F.softmax(perturbed_scores, dim=-1)
```

**Avantages**:
- ✅ **Fully differentiable** : gradients fluent naturellement
- ✅ **Stochastique** : exploration pendant training
- ✅ **Annealing** : converge vers hard selection

**Inconvénients**:
- ❌ **Biais stochastique** : variance dans gradients
- ❌ **Complexité** : calcul softmax sur L positions
- ❌ **Memory** : stocke soft_scores (B, L)

**Quand utiliser**:
- Tâches nécessitant **exploration** (RL-like)
- Training de **scratch** sans pré-training
- Quand **variance acceptable** (grands batch sizes)

### 1.4 Straight-through estimator (lignes 102-124)

```python
# Lignes 114: Forward = hard top-K
topk_vals, topk_indices = torch.topk(scores, k=k)

# Lignes 117-118: Créer one-hot hard selection
selection_onehot = zeros_like(scores)
selection_onehot.scatter_(1, topk_indices, 1.0)

# Ligne 122: TRICK - gradient passthrough
selection = selection_onehot + scores - scores.detach()
#           ^hard forward       ^soft backward^
```

**Fonctionnement**:
1. **Forward pass**: One-hot dur → pas d'ambiguïté
2. **Backward pass**: Gradient de `scores` (continu) passe à travers
3. **Trick mathématique**: `scores - scores.detach() = 0` en valeur, mais gradient préservé

**Avantages**:
- ✅ **Déterministe** : pas de variance
- ✅ **Efficace** : pas de softmax, juste top-K
- ✅ **Gradients propres** : biaisés mais low-variance

**Inconvénients**:
- ❌ **Biaisé** : gradient ne reflète pas vraiment top-K
- ❌ **Moins explorateur** : sélection greedy
- ❌ **Théoriquement douteux** : gradient approximé

**Quand utiliser**:
- **Fine-tuning** de modèles pré-entraînés
- Training avec **petits batches**
- Production où **stabilité > exploration**

### 1.5 Forward pass (lignes 126-173)

```python
# Lignes 143-144: Scorer toutes positions
scores = self.scorer(x).squeeze(-1)  # (B, L)

# Lignes 149-158: Routing selon mode
if self.training:
    if use_gumbel:
        selection_soft, landmark_indices = self._gumbel_topk(...)
    else:
        selection_soft, landmark_indices = self._straight_through_topk(...)
else:
    _, landmark_indices = torch.topk(scores, k=k)  # Hard inference

# Lignes 164-167: Gather les états
landmark_indices_exp = landmark_indices.unsqueeze(-1).expand(B, k, D)
landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)
```

**Décisions de design**:
1. **Inference = hard top-K** : pas de Gumbel noise, déterministe
2. **Training = configurable** : `use_gumbel` flag pour flexibilité
3. **Gather efficace** : une seule opération, pas de loops

**Complexité gather**:
- Time: O(G·D) avec G landmarks
- Memory: O(B·G·D) pour stockage

### 1.6 Outputs (lignes 169-173)

```python
# Ligne 171: Normalisation pour interprétabilité
selection_scores = F.softmax(scores, dim=-1)  # (B, L)

return landmark_indices, landmark_states, selection_scores
```

**Trois sorties**:
1. **landmark_indices** (B, G): Positions sélectionnées (pour attention)
2. **landmark_states** (B, G, D): États gathered (prêts pour attention)
3. **selection_scores** (B, L): Distribution normalisée (pour loss auxiliaire)

---

## 2. PositionalLandmarkSelector

**Lignes 176-231** : Patterns positionnels appris

### 2.1 Architecture (lignes 185-200)

```python
# Ligne 197: Embeddings positionnels apprenables
self.pos_embeddings = nn.Parameter(torch.randn(max_seq_len, embed_dim))

# Ligne 200: Projecteur simple
self.scorer = nn.Linear(embed_dim, 1)
```

**Concept**:
- **Hypothèse**: Certaines positions sont **structurellement importantes**
  - Ex: début de paragraphe, tous les N tokens, fin de section
- **Pas de dépendance au contenu**: scores basés uniquement sur position

**Avantages**:
- ✅ **Très efficace**: scores identiques pour tous les exemples du batch
- ✅ **Interprétable**: peut révéler patterns structurels
- ✅ **Stable**: pas de variance due au contenu

**Limitations**:
- ❌ **Rigide**: ne s'adapte pas au contenu
- ❌ **Capacité limitée**: ne peut capturer des patterns complexes

### 2.2 Forward pass (lignes 202-231)

```python
# Lignes 214-215: Scores uniquement positionnels
pos_emb = self.pos_embeddings[:L]
scores = self.scorer(pos_emb).squeeze(-1)  # (L,)

# Ligne 219: Broadcast aux batchs
scores = scores.unsqueeze(0).expand(B, L)

# Lignes 222-223: Top-K déterministe
_, landmark_indices = torch.topk(scores, k=k)
```

**Observation critique**:
- **Pas de différentiabilité complexe**: juste Linear + top-K
- **Gradients fluent** vers `pos_embeddings` via scorer
- **Économie computationnelle**: scorer une seule fois par longueur

**Complexité**:
- Scoring: O(L·D) une fois par longueur unique
- Top-K: O(L log k)
- **Total par batch: O(L log k)** si scores cachés

---

## 3. HybridLandmarkSelector

**Lignes 234-277** : Fusion content + position avec gating

### 3.1 Architecture (lignes 241-253)

```python
# Lignes 249-250: Deux sélecteurs
self.content_selector = LearnableLandmarkSelector(...)
self.position_selector = PositionalLandmarkSelector(...)

# Ligne 253: Gate pour combiner
self.gate = nn.Linear(embed_dim, 1)
```

**Design pattern**: **Mixture-of-Experts** light
- Deux "experts" spécialisés
- Gating appris pour routing

### 3.2 Fusion adaptative (lignes 255-277)

```python
# Lignes 259-260: Sélections des deux modules
idx_content, states_content, scores_content = self.content_selector(x)
idx_position, states_position, scores_position = self.position_selector(x)

# Lignes 262-264: Gating sur moyenne globale
x_pooled = x.mean(dim=1)  # (B, D)
gate_weight = torch.sigmoid(self.gate(x_pooled))  # (B, 1)

# Ligne 267: Combinaison linéaire
scores_combined = gate_weight * scores_content + (1 - gate_weight) * scores_position

# Lignes 270-271: Re-sélection sur scores combinés
_, landmark_indices = torch.topk(scores_combined, k=k)
```

**Interprétation du gating**:
- **gate_weight ≈ 1**: Séquence complexe → privilégier content
- **gate_weight ≈ 0**: Séquence structurée → privilégier position
- **gate_weight ≈ 0.5**: Combiner équitablement

**Analyse critique**:

✅ **Avantages**:
- Adaptatif au type de séquence
- Meilleur des deux mondes
- Gating appris automatiquement

❌ **Limitations**:
- **Pooling global simpliste**: perd information locale
- **Re-sélection coûteuse**: top-K supplémentaire
- **Pas de backprop vers indices**: seuls scores_combined différentiables

**Alternative potentielle**:
```python
# Au lieu de combiner scores, combiner indices directement
# Ex: k//2 de chaque sélecteur
indices_hybrid = torch.cat([idx_content[:, :k//2], idx_position[:, :k//2]], dim=1)
```
→ Évite re-sélection, mais perd fluidité de gating

---

## 4. Loss auxiliaires

### 4.1 Diversity loss (lignes 280-307)

```python
# Lignes 297-298: Entropie de Shannon
entropy = -(selection_scores * log(selection_scores)).sum(dim=-1)

# Lignes 300-302: Normalisation
max_entropy = log(L)
normalized_entropy = entropy / max_entropy  # [0, 1]

# Ligne 305: Pénaliser faible entropie
loss = lambda_reg * (1 - normalized_entropy).mean()
```

**Objectif**: Encourager **diversité spatiale** des landmarks

**Mécanisme**:
- **Haute entropie** (normalized ≈ 1): Distribution uniforme → landmarks espacés
- **Faible entropie** (normalized ≈ 0): Pics concentrés → landmarks groupés

**Exemple numérique**:
```
L = 256, G = 32

Cas 1 (diversity haute):
  selection_scores ≈ uniform(1/256) partout
  entropy ≈ log(256) = 5.54
  normalized_entropy ≈ 1.0
  loss ≈ 0 → ✅ pas de pénalité

Cas 2 (diversity faible):
  selection_scores ≈ [0.9, 0.1, 0, 0, ..., 0] (concentré)
  entropy ≈ 0.47
  normalized_entropy ≈ 0.08
  loss ≈ 0.92 → ❌ forte pénalité
```

**Paramètres**:
- `lambda_reg = 0.01` (ligne 281): **Très faible** → impact minimal
- Raison: éviter conflit avec loss principale

**⚠️ Limitation**:
- **Biais théorique**: Maximiser entropie pousse vers **uniform distribution**
- Or on veut G landmarks espacés, pas L landmarks uniformes
- **Meilleure alternative**: Pénaliser proximité pairwise des landmarks

**Alternative proposée**:
```python
def landmark_spacing_loss(landmark_indices, L, lambda_reg=0.01):
    """Pénalise landmarks trop proches"""
    # landmark_indices: (B, G)
    sorted_indices, _ = torch.sort(landmark_indices, dim=-1)
    gaps = sorted_indices[:, 1:] - sorted_indices[:, :-1]  # (B, G-1)

    # Idéalement, gaps uniformes = L / G
    ideal_gap = L / landmark_indices.size(1)
    gap_variance = ((gaps - ideal_gap) ** 2).mean()

    return lambda_reg * gap_variance
```

### 4.2 Sparsity loss (lignes 310-331)

```python
# Lignes 323-325: Compter positions actives
threshold = 0.01
active_fraction = (selection_scores > threshold).float().mean()

# Lignes 327-329: Pénaliser si trop actif
target_active = 1 - target_sparsity  # default: 1 - 0.95 = 0.05
loss = lambda_reg * F.relu(active_fraction - target_active)
```

**Objectif**: **Concentrer la masse** sur peu de positions

**Mécanisme**:
- Compte % de positions avec score > 0.01
- Si > 5% positions actives → pénalité
- Sinon → loss = 0 (via ReLU)

**Paramètres par défaut**:
- `target_sparsity = 0.95` → max 5% positions actives
- `lambda_reg = 0.001` → très faible

**Analyse critique**:

❌ **Problème de design**:
```
Si G = 32 et L = 256:
  Fraction idéale active = G/L = 32/256 = 0.125 (12.5%)
  target_active = 0.05 (5%)

  → Incompatible ! Le top-K sélectionne 12.5%, mais loss pénalise au-delà de 5%
```

**Conséquence**:
- Loss **toujours active** si G/L > target_active
- Gradient **constant** → pas d'effet d'apprentissage utile

**Fix suggéré**:
```python
# Ligne 328 (fix):
target_active = max(1 - target_sparsity, num_landmarks / L * 1.2)
#                                         ^laisse marge 20%
```

---

## 5. Analyse de complexité

### 5.1 LearnableLandmarkSelector

| Opération | Complexité | Détails |
|-----------|-----------|---------|
| **Scoring** | O(L·D·(D/2)) | Linear D→D/2, puis D/2→1 |
| **Gumbel-Softmax** | O(L) | Sampling + softmax |
| **Straight-through** | O(L log k) | Top-K heap |
| **Gather** | O(G·D) | G landmarks × D dimensions |
| **Total (Gumbel)** | **O(L·D² + L)** | Dominé par scorer |
| **Total (ST)** | **O(L·D² + L log k)** | Dominé par scorer |

**Bottleneck**: Scorer neural (O(L·D²))

**Optimisation possible**:
```python
# Réduire hidden_dim pour accélérer
hidden = embed_dim // 4  # au lieu de // 2
# Trade-off: capacité vs vitesse
```

### 5.2 PositionalLandmarkSelector

| Opération | Complexité | Détails |
|-----------|-----------|---------|
| **Scoring** | O(L·D) | Linear sur pos_embeddings |
| **Top-K** | O(L log k) | Heap |
| **Gather** | O(G·D) | G landmarks |
| **Total** | **O(L·D + L log k)** | Linéaire en L |

**Très efficace** : Pas de dépendance quadratique

### 5.3 HybridLandmarkSelector

| Opération | Complexité | Détails |
|-----------|-----------|---------|
| **Content selector** | O(L·D²) | LearnableLandmarkSelector |
| **Position selector** | O(L·D) | PositionalLandmarkSelector |
| **Gating** | O(D) | Linear sur x_pooled |
| **Re-selection** | O(L log k) | Top-K sur scores combinés |
| **Total** | **O(L·D²)** | Dominé par content selector |

**Overhead**: ~2× plus lent que LearnableLandmarkSelector seul

### 5.4 Loss auxiliaires

| Loss | Complexité | Notes |
|------|-----------|-------|
| **Diversity** | O(L) | Somme entropie |
| **Sparsity** | O(L) | Comptage threshold |
| **Total** | **O(L)** | Négligeable |

---

## 6. Trade-offs et recommandations

### 6.1 Gumbel vs Straight-through

| Critère | Gumbel-Softmax | Straight-through |
|---------|---------------|------------------|
| **Différentiabilité** | ✅ Vraie | ⚠️ Approximée |
| **Variance gradients** | ❌ Haute | ✅ Faible |
| **Exploration** | ✅ Stochastique | ❌ Greedy |
| **Stabilité** | ⚠️ Moyenne | ✅ Haute |
| **Vitesse** | ⚠️ Softmax overhead | ✅ Rapide |
| **Convergence** | ⚠️ Lente | ✅ Rapide |

**Recommandations**:

1. **Training from scratch**:
   ```python
   # Phase 1 (0-10k steps): Gumbel avec temp=1.0
   selector(x, use_gumbel=True)

   # Phase 2 (10k+ steps): Switch to straight-through
   selector(x, use_gumbel=False)
   ```

2. **Fine-tuning pré-entraîné**:
   ```python
   # Toujours straight-through
   selector(x, use_gumbel=False)
   ```

3. **Production inference**:
   ```python
   # Mode eval → déterministe
   selector.eval()
   with torch.no_grad():
       indices, states, scores = selector(x)
   ```

### 6.2 Temperature decay

**Problème actuel**:
- `temperature_decay = 0.9999` → **trop lent**
- À 15k steps (dataset actuel): temp ≈ 0.78
- Pas assez bas pour hard selection

**Recommandations**:

1. **Decay plus agressif**:
   ```python
   temperature_decay = 0.999  # 10× plus rapide
   # À 5k steps: temp ≈ 0.50 (min)
   ```

2. **Linear decay** (alternatif):
   ```python
   def _get_temperature(self):
       if not self.training:
           return self.min_temperature

       progress = min(1.0, self.step_count / self.total_steps)
       return self.temperature - (self.temperature - self.min_temperature) * progress
   ```

3. **Cosine annealing** (recommandé):
   ```python
   def _get_temperature(self):
       progress = min(1.0, self.step_count / self.total_steps)
       cos_decay = 0.5 * (1 + math.cos(math.pi * progress))
       return self.min_temperature + (self.temperature - self.min_temperature) * cos_decay
   ```

### 6.3 Loss auxiliaires

**Fixes prioritaires**:

1. **Diversity loss** → Remplacer par spacing loss:
   ```python
   def landmark_spacing_loss(landmark_indices, seq_len, lambda_reg=0.01):
       B, G = landmark_indices.shape
       sorted_idx, _ = torch.sort(landmark_indices, dim=-1)

       # Gaps entre landmarks
       gaps = sorted_idx[:, 1:] - sorted_idx[:, :-1]
       ideal_gap = seq_len / G

       # Variance des gaps (0 = uniform spacing)
       gap_variance = ((gaps - ideal_gap) ** 2).mean()

       return lambda_reg * gap_variance
   ```

2. **Sparsity loss** → Ajuster target dynamiquement:
   ```python
   def landmark_sparsity_loss(selection_scores, num_landmarks, lambda_reg=0.001):
       L = selection_scores.size(1)

       # Target adaptatif
       min_active_fraction = num_landmarks / L
       target_active = min_active_fraction * 1.2  # Marge 20%

       threshold = 0.01
       active_fraction = (selection_scores > threshold).float().mean()

       loss = lambda_reg * F.relu(active_fraction - target_active)
       return loss
   ```

### 6.4 Choix du sélecteur

| Scénario | Sélecteur recommandé | Justification |
|----------|---------------------|---------------|
| **Training scratch** | Learnable (ST) | Flexibilité max, stable |
| **Fine-tuning** | Learnable (ST) | S'adapte au contenu |
| **Séquences structurées** | Positional | Efficace, interprétable |
| **Données hétérogènes** | Hybrid | Adaptatif, robuste |
| **Production** | Learnable (eval) | Meilleur équilibre |

### 6.5 Hyperparamètres recommandés

**LearnableLandmarkSelector**:
```python
selector = LearnableLandmarkSelector(
    embed_dim=384,
    num_landmarks=32,
    hidden_dim=96,              # D/4 pour efficacité
    temperature=1.0,
    temperature_decay=0.999,     # Plus rapide
    min_temperature=0.3,         # Plus bas pour hard selection
)
```

**Loss auxiliaires**:
```python
# Dans training loop:
diversity_loss = landmark_spacing_loss(indices, L, lambda_reg=0.005)  # Ajusté
sparsity_loss = landmark_sparsity_loss(scores, G, lambda_reg=0.002)   # Adaptatif

total_loss = main_loss + diversity_loss + sparsity_loss
```

---

## 7. Résumé et points critiques

### ✅ Points forts du design

1. **Flexibilité**: 3 sélecteurs pour différents scénarios
2. **Différentiabilité**: Gumbel + ST = best of both worlds
3. **Efficacité**: Complexity O(L·D²) acceptable
4. **Modularité**: Facile de swapper sélecteurs

### ⚠️ Points d'amélioration

1. **Temperature decay trop lent**:
   - Impact: Convergence lente, sélection reste "molle"
   - Fix: decay=0.999 ou cosine annealing

2. **Diversity loss suboptimal**:
   - Impact: Ne pénalise pas vraiment clustering
   - Fix: Spacing loss sur landmark_indices

3. **Sparsity loss incompatible**:
   - Impact: Gradient constant, pas d'apprentissage
   - Fix: Target adaptatif = G/L × 1.2

4. **Hybrid selector re-selection**:
   - Impact: Overhead computationnel
   - Alternative: Combiner indices directement

### 🎯 Recommandations pour training actuel

**Configuration SLGA-Plus** (d'après configs/):

1. **Sélecteur**: Utiliser **LearnableLandmarkSelector** en mode **straight-through**
   ```python
   use_gumbel=False  # Plus stable pour fine-tuning
   ```

2. **Hyperparamètres**:
   ```python
   temperature_decay = 0.999      # Plus rapide
   min_temperature = 0.3          # Plus hard
   lambda_diversity = 0.005       # Avec spacing loss
   lambda_sparsity = 0.002        # Adaptatif
   ```

3. **Monitoring**:
   ```python
   # À logger pendant training:
   - selection_scores.entropy()           # Diversité
   - (selection_scores > 0.01).float().mean()  # Sparsité
   - landmark_indices.std()               # Variance positions
   ```

4. **Diagnostic si problèmes**:
   - **Loss plateau**: Vérifier temperature (doit décroître)
   - **Landmarks groupés**: Augmenter lambda_diversity
   - **Trop dispersés**: Réduire lambda_diversity

---

## Annexe: Complexité détaillée

### Forward pass complet (LearnableLandmarkSelector + Attention)

```python
# 1. Landmark selection
scores = scorer(x)                    # O(L·D²)
indices = topk(scores, k=G)           # O(L log G)
landmarks = gather(x, indices)        # O(G·D)
# Total: O(L·D²) + O(G·D)

# 2. Global attention (dans slga_layer.py)
Q_global = x @ W_q                    # O(L·D²)
K_global = landmarks @ W_k            # O(G·D²)
V_global = landmarks @ W_v            # O(G·D²)
attn = softmax(Q @ K.T / sqrt(d))     # O(L·G·d)
out = attn @ V                        # O(L·G·d)
# Total: O(L·D²) + O(L·G·d)

# 3. Total pipeline
# = O(L·D²) + O(L·G·d)
# Avec d = D/h (h heads): O(L·D²) + O(L·G·D/h)
# Si G << L: Dominé par O(L·D²) du MLP
```

**Comparaison avec Full Attention**:
- Full: O(L²·D)
- SLGA avec landmarks: O(L·D² + L·G·D)
- Si G = 32, L = 256, D = 384:
  - Full: 256² × 384 = 25M ops
  - SLGA: 256 × 384² + 256 × 32 × 384 = 40M ops

**⚠️ Observation**: SLGA **plus coûteux** que full attention pour petites séquences !
→ Savings apparaissent seulement si **L > 1024** environ

---

**Fichier analysé**: `/mnt/d/ai/SLGA/src/landmarks.py`
**Date**: 2025-10-24
**Lignes**: 376
**Version**: SLGA-Plus v1.0
