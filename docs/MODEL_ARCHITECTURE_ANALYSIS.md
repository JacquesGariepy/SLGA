# Analyse Approfondie de l'Architecture LLMTransformer (model.py)

## 📋 Résumé Exécutif

**Date**: 2025-10-28
**Fichier Analysé**: `/mnt/d/ai/SLGA/src/model.py`
**Lignes de Code**: 490 lignes
**Statut Global**: ✅ **Architecture Correcte** avec quelques optimisations mineures possibles

---

## 🏗️ Architecture Globale

### Composants Principaux

```
LLMTransformer
├── Embeddings Layer
│   ├── token_emb (vocab_size → embed_dim)
│   ├── pos_emb (max_seq_len → embed_dim)
│   └── emb_dropout
├── Landmark Selector (optionnel)
│   └── LearnableLandmarkSelector
├── Transformer Blocks (N layers)
│   ├── SLGA Attention Module
│   ├── Feed-Forward Network
│   ├── norm1 (pre-norm)
│   └── norm2 (pre-norm)
├── Final LayerNorm
└── LM Head (embed_dim → vocab_size, tied weights)
```

---

## 🔍 Analyse Ligne par Ligne

### 1. Configuration (Lignes 26-50)

```python
@dataclass
class Config:
    vocab_size: int = 50257
    max_seq_len: int = 2048
    embed_dim: int = 512
    # ... autres paramètres
```

✅ **Correct**:
- Tous les hyperparamètres nécessaires sont présents
- Types annotés correctement
- Valeurs par défaut raisonnables

⚠️ **Remarque**:
- `landmark_selector: Optional[Dict[str, Any]] = None` (ligne 46) n'est **jamais utilisé** dans le code
- Ce paramètre était prévu pour une config v1.1 mais reste un placeholder

**Recommandation**: Retirer ce champ inutilisé ou l'implémenter.

---

### 2. FeedForward Network (Lignes 52-68)

```python
class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_multiplier: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = embed_dim * hidden_multiplier
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
```

✅ **Architecture Standard**:
- Expansion 4x (GPT-2 style)
- GELU activation
- Dropout après chaque projection
- **CORRECT**: Pas de biais (implicitement False par défaut)

❌ **BUG POTENTIEL #1**: **Double Dropout**
- Ligne 65: dropout après GELU
- Ligne 67: dropout après fc2
- **Impact**: Sur-régularisation possible (dropout = 0.1 → ~19% total)

**Solution**: Choisir dropout soit après GELU OU après fc2, pas les deux.

```python
# Version recommandée
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.fc1(x)
    x = F.gelu(x)
    x = self.fc2(x)
    x = self.dropout(x)  # Un seul dropout à la fin
    return x
```

---

### 3. TransformerBlock (Lignes 71-155)

#### 3.1 Initialisation (Lignes 78-114)

```python
# Dilatation progressive par couche si activée
if cfg.dilated_windows:
    dilation_factor = 2 ** (layer_idx // max(1, cfg.n_layers // 3))
else:
    dilation_factor = 1
```

✅ **Correct**:
- Dilatation exponentielle (1, 2, 4, 8...)
- Protection contre division par zéro avec `max(1, ...)`
- Les couches basses restent denses, les couches hautes dilatent

**Exemple pour 12 couches**:
- Couches 0-3: dilation = 1
- Couches 4-7: dilation = 2
- Couches 8-11: dilation = 4

#### 3.2 Forward Pass (Lignes 124-155)

```python
def forward(
    self,
    x: torch.Tensor,
    cache_global: Optional[torch.Tensor] = None,
    global_weight: float = 1.0,
) -> torch.Tensor:
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
```

✅ **Architecture Correcte**:
- **Pre-norm**: LayerNorm AVANT attention/FFN (moderne, stable)
- **Residual connections** bien placées
- **Gradient checkpointing** conditionnel (training only)
- `use_reentrant=False` pour compatibilité PyTorch 2.x

⚠️ **PROBLÈME SUBTIL #2**: **Wrapper Functions pour Checkpointing**

Lignes 116-122:
```python
def _attn_forward(self, x: torch.Tensor, cache_global: Optional[torch.Tensor], global_weight: float = 1.0) -> torch.Tensor:
    return self.attn(x, cache_global=cache_global, global_weight=global_weight)

def _ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.ffn(x)
```

**Problème**: Ces wrappers sont nécessaires car `checkpoint()` attend une fonction, pas un module.

**MAIS**: La signature de `_attn_forward` a un **problème de compatibilité**:

```python
# Ligne 141: Checkpoint appelle avec cache_global ET global_weight
checkpoint(self._attn_forward, self.norm1(x), cache_global, global_weight, use_reentrant=False)
```

❌ **BUG #2**: Si `cache_global=None`, `checkpoint()` passe `None` en 2e argument, mais `global_weight` devient le 3e argument **positionnel**.

**Solution**: Utiliser kwargs explicitement:

```python
def _attn_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
    return self.attn(x, **kwargs)

# Dans forward():
attn_out = checkpoint(
    lambda x: self._attn_forward(x, cache_global=cache_global, global_weight=global_weight),
    self.norm1(x),
    use_reentrant=False
)
```

---

### 4. LLMTransformer.__init__ (Lignes 174-206)

#### 4.1 Embeddings (Lignes 179-182)

```python
self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.embed_dim)
self.emb_dropout = nn.Dropout(cfg.dropout_rate)
```

✅ **Correct**: Architecture standard.

#### 4.2 Landmark Selector (Lignes 185-191)

```python
if cfg.learned_landmarks:
    self.landmark_selector = LearnableLandmarkSelector(
        embed_dim=cfg.embed_dim,
        num_landmarks=cfg.global_k * 2,  # Sélectionner plus, top-K restreint dans SLGA
    )
else:
    self.landmark_selector = None
```

✅ **Correct**:
- Sélectionne 2x plus de landmarks que nécessaire
- SLGA fera le top-K final dans chaque tête

**Justification**: Permet plus de diversité inter-têtes.

#### 4.3 Tied Weights (Lignes 202-203)

```python
self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
self.lm_head.weight = self.token_emb.weight
```

✅ **CORRECT**: Weight tying standard pour économiser mémoire et améliorer convergence.

⚠️ **Attention**: Cette ligne **partage** le tenseur (pas de copie). Toute mise à jour de `token_emb.weight` affecte aussi `lm_head.weight`.

---

### 5. LLMTransformer.forward() (Lignes 220-287)

#### 5.1 Embeddings (Lignes 240-247)

```python
B, L = input_ids.shape
device = input_ids.device

tok_emb = self.token_emb(input_ids)  # (B, L, D)
pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
pos_emb = self.pos_emb(pos)  # (B, L, D)
x = self.emb_dropout(tok_emb + pos_emb)
```

✅ **Correct**:
- Positions générées dynamiquement (pas de cache)
- Device handling correct
- Dropout après addition

#### 5.2 Landmark Selection (Lignes 249-259)

```python
landmark_indices = None
landmark_scores = None

if self.landmark_selector is not None:
    # Landmarks appris - sélectionner les indices une fois
    landmark_indices, _, landmark_scores = self.landmark_selector(x)
    # landmark_indices: (B, G)
elif cache_global_ids is not None:
    # Landmarks heuristiques - utiliser les indices fournis
    landmark_indices = cache_global_ids  # (B, G)
```

✅ **Correct**:
- Sélection UNE FOIS au début (pas recalculée à chaque couche)
- Support deux modes: appris vs heuristique

#### 5.3 Passage dans les Blocs (Lignes 262-274)

```python
for block in self.blocks:
    # Extraire les états actuels des landmarks depuis x
    if landmark_indices is not None:
        B_cur, L_cur, D = x.shape
        G = landmark_indices.size(1)
        landmark_indices_exp = landmark_indices.unsqueeze(-1).expand(B_cur, G, D)
        landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)  # (B, G, D)
    else:
        landmark_states = None

    # Forward du bloc avec landmarks mis à jour
    x = block(x, cache_global=landmark_states, global_weight=global_weight)
```

✅ **EXCELLENT DESIGN**:
- Les **landmark_indices** sont fixes (sélectionnés une fois)
- Mais les **landmark_states** sont **extraits à chaque couche** depuis `x` mis à jour
- Cela permet aux landmarks d'**évoluer** avec la représentation

**Pourquoi c'est important**:
- Layer 0: landmarks = embeddings bruts
- Layer 6: landmarks = représentations intermédiaires enrichies
- Layer 11: landmarks = contexte sémantique de haut niveau

❌ **BUG POTENTIEL #3**: **Gather sans protection**

Si `landmark_indices` contient des **indices hors limites** (≥ L), `torch.gather()` va crasher ou retourner des valeurs indéfinies.

**Scénario problématique**:
1. Séquence initiale: L=512, landmarks sélectionnés dans [0, 511]
2. Dans `generate()`, séquence tronquée à max_seq_len (ligne 329)
3. Si landmarks pointent vers positions > L après troncation → **crash**

**Solution**: Clipper les indices avant gather:

```python
if landmark_indices is not None:
    B_cur, L_cur, D = x.shape
    G = landmark_indices.size(1)
    # PROTECTION: Clipper indices dans [0, L_cur-1]
    landmark_indices_clipped = torch.clamp(landmark_indices, 0, L_cur - 1)
    landmark_indices_exp = landmark_indices_clipped.unsqueeze(-1).expand(B_cur, G, D)
    landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)
```

#### 5.4 Final Projection (Lignes 276-287)

```python
x = self.final_norm(x)
logits = self.lm_head(x)  # (B, L, V)

if return_aux:
    aux = {
        "landmark_scores": landmark_scores,  # Scores softmax (B, L)
        "landmark_indices": landmark_indices,  # Indices sélectionnés (B, G)
    }
    return logits, aux
else:
    return logits
```

✅ **Correct**:
- Final norm avant projection (standard)
- Return types corrects

⚠️ **REMARQUE**: Si `learned_landmarks=False` et `cache_global_ids=None`, `landmark_scores=None` dans aux. Les scripts d'entraînement doivent gérer ce cas.

---

### 6. LLMTransformer.generate() (Lignes 289-398)

#### 6.1 Seed Handling (Lignes 320-324)

```python
if seed is not None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

✅ **Correct**: Seed pour reproductibilité.

⚠️ **LIMITATION**: Ne garantit PAS la reproductibilité complète si:
- Opérations non-déterministes activées (cuDNN benchmarking)
- Multi-GPU avec nccl
- Operations atomiques sur GPU

**Pour reproductibilité 100%**:
```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
```

#### 6.2 Séquence Truncation (Lignes 327-329)

```python
if input_ids.size(1) > self.cfg.max_seq_len:
    input_ids = input_ids[:, -self.cfg.max_seq_len:]
```

✅ **Correct**: Sliding window pour séquences longues.

❌ **BUG #4**: **Pas de recalcul des landmarks après troncation**

Si `learned_landmarks=True`, les landmarks ont été calculés sur la séquence AVANT troncation. Après troncation, ces indices peuvent être **invalides**.

**Solution**: Recalculer landmarks après troncation OU utiliser positions relatives.

#### 6.3 Heuristic Landmarks (Lignes 332-336)

```python
if not self.cfg.learned_landmarks and cache_global_ids is None:
    L = input_ids.size(1)
    stride = max(1, L // self.cfg.global_k)
    landmark_positions = torch.arange(0, L, stride, device=input_ids.device)
    cache_global_ids = landmark_positions.unsqueeze(0).expand(input_ids.size(0), -1)
```

✅ **Correct**: Landmarks uniformément espacés.

⚠️ **PROBLÈME #5**: **Nombre de landmarks variable**

Si `L=130` et `global_k=32`:
- stride = 130 // 32 = 4
- landmarks = [0, 4, 8, ..., 128] = **33 landmarks** (pas 32!)

**Impact**:
- Shape mismatch possible avec cache pré-alloué
- SLGA attend `global_k` landmarks, reçoit `global_k+1`

**Solution**: Forcer exactement G landmarks:

```python
if not self.cfg.learned_landmarks and cache_global_ids is None:
    L = input_ids.size(1)
    # Créer exactement global_k landmarks uniformément espacés
    indices = torch.linspace(0, L-1, self.cfg.global_k, device=input_ids.device).long()
    cache_global_ids = indices.unsqueeze(0).expand(input_ids.size(0), -1)
```

#### 6.4 Sampling (Lignes 338-393)

```python
# Forward
logits = self(input_ids, cache_global_ids=cache_global_ids)  # (B, L, V)

# Prendre logits du dernier token (RAW, sans temperature)
logits = logits[:, -1, :]  # (B, V)

# Top-K filtering (sur logits RAW)
if top_k is not None and top_k > 0:
    topk_vals, topk_idxs = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
    logits_filtered = torch.full_like(logits, float('-inf'))
    logits_filtered.scatter_(1, topk_idxs, topk_vals)
    logits = logits_filtered
```

✅ **EXCELLENT**:
- Filtering AVANT temperature (correct)
- Protection `min(top_k, vocab_size)`

```python
# Top-P (nucleus) filtering (sur logits RAW)
if top_p is not None and top_p < 1.0:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False  # Always keep the best token

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    logits = logits.scatter(1, sorted_indices, sorted_logits)
```

✅ **CORRECT**: Implémentation nucleus sampling standard.

⚠️ **REMARQUE**: Le shift `sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()` est nécessaire pour **garder au moins 1 token** (le meilleur).

```python
if temperature == 0.0:
    # GREEDY: Sélection déterministe du meilleur token
    next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)

else:
    # SAMPLING: Comportement stochastique
    if temperature != 1.0:
        logits = logits / temperature

    probs = F.softmax(logits, dim=-1)

    # Protection: si tous les logits sont -inf, utiliser distribution uniforme
    if torch.isnan(probs).any() or torch.isinf(probs).any():
        probs = torch.ones_like(probs) / probs.size(-1)

    probs = torch.clamp(probs, min=1e-10)
    probs = probs / probs.sum(dim=-1, keepdim=True)

    next_token = torch.multinomial(probs, num_samples=1)
```

✅ **EXCELLENT**:
- Greedy mode déterministe avec `temperature=0`
- Protection contre NaN/Inf
- Re-normalisation explicite

❌ **BUG SUBTIL #6**: **Protection NaN excessive**

Ligne 386-387:
```python
if torch.isnan(probs).any() or torch.isinf(probs).any():
    probs = torch.ones_like(probs) / probs.size(-1)
```

**Problème**: Si filtering top-k/top-p est très agressif, TOUS les logits peuvent être `-inf`, ce qui donne `probs=NaN`. Le code fallback à uniforme, mais cela **ignore complètement le filtering**.

**Meilleur fallback**: Revert to greedy si NaN détecté.

```python
if torch.isnan(probs).any() or torch.isinf(probs).any():
    # Fallback: greedy sur logits originaux (avant filtering)
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    input_ids = torch.cat([input_ids, next_token], dim=1)
    continue
```

---

### 7. Méthodes Utilitaires

#### 7.1 get_num_params() (Lignes 400-406)

```python
def get_num_params(self, non_embedding: bool = True) -> int:
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding:
        n_params -= self.pos_emb.weight.numel()
        n_params -= self.token_emb.weight.numel()
    return n_params
```

❌ **BUG #7**: **Double-comptage avec tied weights**

Puisque `lm_head.weight = token_emb.weight` (ligne 203), les deux partagent le MÊME tenseur. Mais `self.parameters()` itère sur tous les modules, donc compte `lm_head.weight` séparément.

**Impact**: `token_emb.weight` est compté **2 fois** si `non_embedding=False`.

**Solution**:
```python
def get_num_params(self, non_embedding: bool = True) -> int:
    # Utiliser un set pour dédupliquer les tenseurs partagés
    param_set = {id(p): p.numel() for p in self.parameters()}
    n_params = sum(param_set.values())

    if non_embedding:
        n_params -= self.pos_emb.weight.numel()
        # token_emb est déjà dédupliqué via id()
    return n_params
```

#### 7.2 estimate_mfu() (Lignes 408-443)

✅ **Approximation Raisonnable**: Calcul FLOPs simplifié pour MFU.

⚠️ **Remarque**: Le calcul est **très approximatif** car ignore:
- FLOPs de SLGA (local + global attention)
- FLOPs des landmarks selector
- Memory bandwidth bottlenecks

Pour MFU précis, utiliser un profiler (e.g., `torch.profiler`).

---

## 🐛 Résumé des Bugs Identifiés

### Bugs Critiques (À corriger immédiatement)

| # | Ligne | Sévérité | Description | Impact |
|---|-------|----------|-------------|--------|
| **#3** | 268 | 🔴 **CRITIQUE** | `torch.gather()` sans protection contre indices hors limites | Crash pendant génération si séquence tronquée |
| **#4** | 329 | 🔴 **CRITIQUE** | Pas de recalcul landmarks après troncation | Indices invalides → comportement indéfini |
| **#5** | 332-336 | 🟠 **MAJEUR** | Nombre de landmarks heuristiques variable (G vs G+1) | Shape mismatch potentiel dans SLGA |

### Bugs Mineurs (Optimisations recommandées)

| # | Ligne | Sévérité | Description | Impact |
|---|-------|----------|-------------|--------|
| **#1** | 65-67 | 🟡 **MINEUR** | Double dropout dans FFN | Sur-régularisation (~19% au lieu de 10%) |
| **#2** | 141 | 🟡 **MINEUR** | Arguments positionnels dans checkpoint avec cache_global=None | Erreur subtile si global_weight mal passé |
| **#6** | 386-387 | 🟡 **MINEUR** | Fallback uniforme ignore filtering | Qualité de génération dégradée en edge case |
| **#7** | 400-406 | 🟡 **MINEUR** | Double-comptage tied weights | Métriques incorrectes (cosmétique) |

### Non-Utilisé (À nettoyer)

- **Ligne 46**: `landmark_selector: Optional[Dict[str, Any]]` jamais utilisé

---

## ✅ Points Forts de l'Architecture

1. **Pre-Norm Transformer**: Architecture moderne et stable
2. **SLGA Integration**: Seamless intégration avec module externe
3. **Landmark Evolution**: Les landmarks évoluent à travers les couches (ligne 268-269)
4. **Gradient Checkpointing**: Support optionnel pour économiser mémoire
5. **Generation**: Sampling robuste avec protection NaN
6. **Tied Weights**: Économise mémoire et améliore convergence
7. **Dilated Windows**: Dilatation progressive intelligente
8. **Warmup Support**: `global_weight` pour warmup progressif de l'attention globale

---

## 🔧 Patches Recommandés

### Patch 1: Protection Gather (Critique)

```python
# Ligne 268 - AVANT
landmark_indices_exp = landmark_indices.unsqueeze(-1).expand(B_cur, G, D)
landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)

# APRÈS (avec protection)
landmark_indices_clipped = torch.clamp(landmark_indices, 0, L_cur - 1)
landmark_indices_exp = landmark_indices_clipped.unsqueeze(-1).expand(B_cur, G, D)
landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)
```

### Patch 2: Fix Heuristic Landmarks (Majeur)

```python
# Ligne 332-336 - AVANT
stride = max(1, L // self.cfg.global_k)
landmark_positions = torch.arange(0, L, stride, device=input_ids.device)

# APRÈS (exactement G landmarks)
landmark_positions = torch.linspace(0, L-1, self.cfg.global_k, device=input_ids.device).long()
```

### Patch 3: Remove Double Dropout (Mineur)

```python
# FFN forward - AVANT
x = self.dropout(x)  # après GELU
x = self.fc2(x)
x = self.dropout(x)  # après fc2

# APRÈS (un seul dropout)
# x = self.dropout(x)  # REMOVED
x = self.fc2(x)
x = self.dropout(x)
```

### Patch 4: Fix Tied Weights Counting (Cosmétique)

```python
def get_num_params(self, non_embedding: bool = True) -> int:
    # Dédupliquer par ID de tenseur
    unique_params = {id(p): p.numel() for p in self.parameters()}
    n_params = sum(unique_params.values())

    if non_embedding:
        n_params -= self.pos_emb.weight.numel()
    return n_params
```

---

## 📊 Compatibilité avec SLGA et Landmarks

### SLGA Module (slga.py)

✅ **Intégration Correcte**:
- `cache_global` reçoit les bons landmarks extraits (ligne 269)
- `global_weight` passé correctement pour warmup
- Dimensions compatibles: `(B, G, D)` comme attendu par SLGA

⚠️ **Dépendance Critique**:
- SLGA suppose `cache_global` a shape `(B, G, D)` avec G = `global_k` (ou proche)
- Si nombre de landmarks variable (Bug #5), SLGA peut crasher dans `_diverse_topk`

### Landmark Selector (landmarks.py)

✅ **Intégration Correcte**:
- `LearnableLandmarkSelector` retourne `(indices, states, scores)` comme attendu
- `num_landmarks = global_k * 2` permet diversité (ligne 188)
- Extraction via `torch.gather()` cohérente

⚠️ **Forward-Only Selection**:
- Landmarks sélectionnés UNE FOIS au début (ligne 255)
- Pas de re-sélection dynamique dans les couches suivantes
- **Justification**: Évite instabilité et coût computationnel

---

## 🎯 Recommandations Finales

### Priorités

1. **🔴 URGENT**: Appliquer Patch 1 (protection gather) et Patch 2 (landmarks fixes)
2. **🟠 IMPORTANT**: Gérer troncation dans `generate()` (recalculer landmarks ou passer en mode heuristique)
3. **🟡 RECOMMANDÉ**: Retirer double dropout FFN (Patch 3)
4. **🟢 OPTIONNEL**: Nettoyer `landmark_selector` inutilisé dans Config

### Tests de Validation

Après application des patches, tester:

```python
# Test 1: Génération avec séquence longue (trigger truncation)
model = LLMTransformer(cfg)
prompt = torch.randint(0, 50257, (1, 2500))  # > max_seq_len
output = model.generate(prompt, max_new_tokens=50)  # Devrait marcher sans crash

# Test 2: Landmarks heuristiques avec différentes longueurs
for L in [50, 100, 127, 128, 129, 256, 511, 512, 513]:
    input_ids = torch.randint(0, 50257, (2, L))
    logits = model(input_ids, cache_global_ids=None)
    assert logits.shape == (2, L, 50257), f"Failed for L={L}"

# Test 3: Count parameters avec tied weights
n_params = model.get_num_params(non_embedding=False)
# Vérifier manuellement qu'il n'y a pas de double-comptage
```

---

## 📝 Conclusion

L'architecture `LLMTransformer` est **globalement solide** avec une intégration SLGA bien pensée. Les **3 bugs critiques/majeurs** identifiés concernent principalement la **génération avec séquences longues** et doivent être corrigés pour éviter crashes en production.

Les **points forts** (landmarks évolutifs, pre-norm, tied weights, warmup support) montrent une conception mature. Après application des patches, le modèle sera **production-ready**.

**Score Architecture**: **8.5/10** ⭐⭐⭐⭐ (deviendra 9.5/10 après patches)
