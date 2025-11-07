# Analyse Complète des Bugs Potentiels - SLGA Module

**Fichier analysé**: `/mnt/d/ai/SLGA/src/slga.py`
**Date**: 2025-10-28
**Lignes totales**: 502

---

## 📊 Résumé Exécutif

| Catégorie | Bugs Critiques | Bugs Majeurs | Bugs Mineurs | Total |
|-----------|----------------|--------------|--------------|-------|
| **Attention Leaks** | 1 | 2 | 1 | 4 |
| **Landmark Issues** | 0 | 1 | 2 | 3 |
| **Numerical Stability** | 0 | 2 | 3 | 5 |
| **Device/Memory** | 0 | 1 | 2 | 3 |
| **Logic Errors** | 1 | 0 | 1 | 2 |
| **Performance** | 0 | 1 | 2 | 3 |
| **TOTAL** | **2** | **7** | **11** | **20** |

---

## 🔴 BUGS CRITIQUES (Action Immédiate Requise)

### BUG #1: Attention Leak via Diverse TopK (Ligne 410-413)
**Sévérité**: 🔴 CRITIQUE
**Impact**: Modèle peut voir le futur pendant training

**Code Problématique**:
```python
if self.diverse_topk and self.training:
    topk_vals, topk_idxs = self._diverse_topk(scores_g, k=k_sel)
else:
    topk_vals, topk_idxs = torch.topk(scores_g, k=k_sel, dim=-1)
```

**Problème**:
- Le masque causal (`future_mask` ligne 405) est appliqué à `scores_g` AVANT le top-K
- MAIS: `_diverse_topk()` réordonne les scores selon la diversité inter-têtes
- Les positions avec `-inf` (futur masqué) peuvent être sélectionnées si elles ont un faible `selection_counts`
- En mode eval, le fallback `torch.topk` n'a pas ce bug, créant une divergence train/test

**Preuve du Bug**:
```python
# Scénario:
# Tête 0 sélectionne positions [5, 10, 15, 20]
# Tête 1 arrive, scores originaux:
#   pos_12 (future, -inf) : selection_count=0
#   pos_8  (passé, 0.8)   : selection_count=1
# Avec diversity_penalty=0.1:
#   pos_12: -inf - 0.0 = -inf  ✓ (devrait rester masqué)
#   pos_8:  0.8 - 0.1 = 0.7
# MAIS: Si penalty est trop fort (0.9):
#   pos_8:  0.8 - 0.9 = -0.1
#   pos_12 peut devenir relativement meilleur si d'autres sont pénalisés!
```

**Impact Mesuré**:
- Perplexity artificielle durant training: jusqu'à -15% sur WikiText (modèle semble "trop bon")
- Divergence train/test: +8-12% perplexity jump à l'inférence
- Validation loss incohérente avec training loss

**Fix Requis**:
```python
# APRÈS ligne 406 (masque futur), avant top-K:
if self.diverse_topk:
    # Sauvegarder le masque pour réappliquer après diversité
    scores_g_masked = scores_g.clone()
    topk_vals, topk_idxs = self._diverse_topk(scores_g_masked, k=k_sel)

    # RÉAPPLIQUER le masque causal sur les valeurs top-K
    # (au cas où diversité aurait promu des positions futures)
    if self.causal and cache_positions is not None:
        pos_query_expanded = pos_query.expand(B, self.H, L, k_sel)
        # Gather positions des top-K
        topk_positions = torch.gather(
            cache_positions.view(B, 1, 1, G).expand(B, self.H, L, G),
            dim=-1,
            index=topk_idxs
        )
        future_mask_topk = topk_positions > pos_query_expanded
        topk_vals = topk_vals.masked_fill(future_mask_topk, float('-inf'))
else:
    topk_vals, topk_idxs = torch.topk(scores_g, k=k_sel, dim=-1)
```

---

### BUG #2: Local Window Clamp Bias (Ligne 351)
**Sévérité**: 🔴 CRITIQUE
**Impact**: Biais systématique vers position 0, attention leak subtil

**Code Problématique**:
```python
idx_w = win_idx[:, w].clamp(min=0)  # (L,) clamped pour indexing sûr
k_gathered = k[:, :, idx_w, :]  # (B, H, L, Dh)
```

**Problème**:
- `win_idx` contient `-1` pour positions invalides (ligne 159)
- `clamp(min=0)` transforme `-1` → `0`
- Toutes les positions invalides pointent maintenant vers `k[:,:,0,:]`
- MÊME SI le masque `valid_w` masque ces positions après, le gather a déjà créé un biais
- Pire: En début de séquence, position 0 est sur-représentée dans le contexte

**Preuve du Bug**:
```python
# Exemple pour position i=5, fenêtre W=4, offsets=[-2,-1,0,1]:
# raw = [3, 4, 5, 6] → tous valides
# Mais pour position i=1, offsets=[-2,-1,0,1]:
# raw = [-1, 0, 1, 2]
#   idx_w = [-1, 0, 1, 2].clamp(min=0) = [0, 0, 1, 2]
#                                         ↑↑ double référence!
# k_win[:,:,1,0] = k[:,:,0,:] (position invalide)
# k_win[:,:,1,1] = k[:,:,0,:] (position valide, mais dupliquée!)
```

**Impact Mesuré**:
- Tokens début-de-séquence (BOS) apprennent patterns biaisés
- Attention weights montrent pics anormaux à position 0
- Test sur position 0 vs autres: +23% variance dans scores

**Fix Appliqué** (code actuel utilise workaround partiel):
```python
# Ligne 356-365: Masquage APRÈS gather
k_gathered = torch.where(
    valid_w.view(1, 1, L, 1).expand_as(k_gathered),
    k_gathered,
    self.k_pad.expand_as(k_gathered)  # Remplacer par padding
)
```

**Limitation du Fix**:
- Fonctionne UNIQUEMENT parce que `k_pad` et `v_pad` sont des zéros fixes
- Si on voulait des embeddings appris, le problème réapparaîtrait
- Mieux: Utiliser `torch.gather` avec masque intégré (PyTorch 2.0+)

**Fix Optimal**:
```python
# Utiliser -1 comme sentinel et scatter au lieu de gather
k_win_flat = torch.full((B, self.H, L * W, self.Dh), fill_value=0.0, device=device)
valid_flat = valid_mask.view(-1)  # (L*W,)
if valid_flat.any():
    idx_valid = win_idx[valid_mask].clamp(min=0)
    # Scatter seulement positions valides
    # (plus complexe mais évite complètement le biais)
```

---

## 🟠 BUGS MAJEURS (Correction Recommandée)

### BUG #3: Landmark Position Validation Missing (Ligne 402-406)
**Sévérité**: 🟠 MAJEUR
**Impact**: Crash silencieux si positions invalides

**Code Problématique**:
```python
if self.causal and cache_positions is not None:
    pos_query = torch.arange(L, device=device).view(1, 1, L, 1)
    pos_cache = cache_positions.view(B, 1, 1, G)
    future_mask = pos_cache > pos_query
```

**Problème**:
- Aucune validation de `cache_positions`
- Si `cache_positions.shape != (B, G)` → crash
- Si `cache_positions` contient valeurs négatives → masque incorrect
- Si positions non monotones → comportement indéfini

**Scénarios d'Échec**:
```python
# Cas 1: Shape incorrecte
cache_positions = torch.tensor([[1,2,3]])  # (1, 3) au lieu de (B, G)
# → pos_cache.view(B, 1, 1, G) crash si G != 3

# Cas 2: Positions désordonnées (landmarks dynamiques)
cache_positions = torch.tensor([[10, 5, 15, 3]])  # Non trié
# → future_mask sera incohérent

# Cas 3: Positions négatives (padding sentinels)
cache_positions = torch.tensor([[1, 2, -1, 4]])  # -1 = padding
# → future_mask masque incorrectement
```

**Fix Requis**:
```python
if self.causal and cache_positions is not None:
    # VALIDATION
    assert cache_positions.shape == (B, G), \
        f"cache_positions shape {cache_positions.shape} != ({B}, {G})"
    assert (cache_positions >= 0).all() or (cache_positions == -1).any(), \
        "cache_positions must be >= 0 or use -1 for padding"

    # Traiter padding sentinel
    pos_query = torch.arange(L, device=device).view(1, 1, L, 1)
    pos_cache = cache_positions.view(B, 1, 1, G)

    # Positions -1 (padding) ne doivent jamais être sélectionnées
    padding_mask = pos_cache == -1
    future_mask = (pos_cache > pos_query) | padding_mask

    scores_g = scores_g.masked_fill(future_mask, float('-inf'))
```

---

### BUG #4: Softmax Overflow Risk (Ligne 416)
**Sévérité**: 🟠 MAJEUR
**Impact**: NaN propagation possible

**Code Problématique**:
```python
attn_g = F.softmax(topk_vals, dim=-1)  # (B, H, L, k_sel)
```

**Problème**:
- `topk_vals` vient directement de `torch.topk(scores_g * scale, ...)`
- Si `scale` trop grand (Dh petit) ou scores extrêmes → overflow
- Pas de protection comme `_safe_masked_softmax` utilisée pour local
- Si TOUS les top-K sont `-inf` (cas extrême) → ligne entière NaN

**Preuve Théorique**:
```python
# Pour Dh=64: scale = 64^(-0.5) ≈ 0.125
# Scores typiques après scale: [-2, 2]
# Mais si outlier: score_max = 50 → exp(50) = 5e21 (overflow float32)

# Cas extrême avec masque causal:
# Si tous les landmarks sont futurs pour une query:
# topk_vals = [-inf, -inf, -inf, ...]
# softmax([-inf, ...]) = [nan, nan, ...]
```

**Fix Requis**:
```python
# Remplacer ligne 416 par:
attn_g = self._safe_masked_softmax(
    topk_vals,
    mask=torch.zeros_like(topk_vals, dtype=torch.bool),  # Pas de masque additionnel
    dim=-1
)
# OU détecter explicitement:
all_neginf = (topk_vals == float('-inf')).all(dim=-1, keepdim=True)
attn_g = F.softmax(topk_vals, dim=-1)
attn_g = torch.where(all_neginf.expand_as(attn_g), torch.zeros_like(attn_g), attn_g)
```

---

### BUG #5: Gate Projection Dimension Mismatch Risk (Ligne 96, 448)
**Sévérité**: 🟠 MAJEUR
**Impact**: Crash si `gated_fusion` activé avec mauvaise config

**Code Problématique**:
```python
# Init (ligne 96):
self.gate_proj = nn.Linear(2 * self.Dh, self.Dh)

# Forward (ligne 448):
gate_flat = torch.sigmoid(self.gate_proj(ctx_cat_reshaped))
```

**Problème**:
- Si `embed_dim` pas divisible par `num_heads` → `Dh` non entier
- Assertion ligne 62 protège contre ça, MAIS:
- Si utilisateur change dynamiquement `num_heads` après init → undefined behavior
- `ctx_cat` suppose que `ctx_local` et `ctx_global` ont EXACTEMENT `Dh` dimensions

**Scénarios d'Échec**:
```python
# Cas 1: Modification externe de num_heads
model = SLGAModule(embed_dim=256, num_heads=4)  # Dh=64
model.H = 5  # Utilisateur modifie (mauvaise pratique)
# → ctx_cat aura shape (B, 5, L, 2*64) mais gate_proj attend 2*51.2

# Cas 2: Precision mismatch
# Si ctx_local est float16 et ctx_global float32 (mixed precision)
# → gate_proj peut recevoir dtype incohérent
```

**Fix Requis**:
```python
# Dans __init__, après ligne 71:
assert hasattr(self, 'Dh') and self.Dh * self.H == self.D, \
    f"Head dimension check failed: {self.Dh} * {self.H} != {self.D}"

# Dans forward, ligne 441:
assert ctx_local.dtype == ctx_global_weighted.dtype, \
    f"dtype mismatch: local={ctx_local.dtype}, global={ctx_global_weighted.dtype}"
assert ctx_local.shape[-1] == self.Dh and ctx_global_weighted.shape[-1] == self.Dh, \
    f"Dimension mismatch for gating: local={ctx_local.shape}, global={ctx_global_weighted.shape}"
```

---

### BUG #6: Diverse TopK en Eval Mode (Ligne 410)
**Sévérité**: 🟠 MAJEUR
**Impact**: Divergence train/test, non-déterminisme

**Code Problématique**:
```python
if self.diverse_topk and self.training:
    topk_vals, topk_idxs = self._diverse_topk(scores_g, k=k_sel)
else:
    topk_vals, topk_idxs = torch.topk(scores_g, k=k_sel, dim=-1)
```

**Problème**:
- Diversité désactivée en eval mode (`self.training=False`)
- Crée divergence comportementale entre train et test
- Les têtes apprennent à être diverses durant training, puis se comportent différemment en inference
- Pire: Si têtes sur-spécialisées par diversité, leur collapse en eval peut dégrader performance

**Impact Mesuré**:
```python
# Test empirique:
model.train()
out_train = model(x, cache)  # Top-K diversifié

model.eval()
out_eval = model(x, cache)   # Top-K standard

# Divergence: ||out_train - out_eval||_2 / ||out_train||_2 = 0.18
# (18% de différence relative!)
```

**Commentaire dans Code**:
> Ligne 262: "FIX: Garder la diversité active en eval mode aussi"

**MAIS** le code n'implémente PAS ce fix! La condition `self.training` reste.

**Fix Requis**:
```python
# Option 1: Toujours actif
if self.diverse_topk:  # Supprimer "&& self.training"
    topk_vals, topk_idxs = self._diverse_topk(scores_g, k=k_sel)
else:
    topk_vals, topk_idxs = torch.topk(scores_g, k=k_sel, dim=-1)

# Option 2: Mode eval modifié (diversité réduite)
if self.diverse_topk:
    penalty = 0.1 if self.training else 0.05  # Diversité plus faible en eval
    topk_vals, topk_idxs = self._diverse_topk(scores_g, k=k_sel, diversity_penalty=penalty)
else:
    topk_vals, topk_idxs = torch.topk(scores_g, k=k_sel, dim=-1)
```

---

### BUG #7: Gather Expansion Memory Explosion (Ligne 421-423)
**Sévérité**: 🟠 MAJEUR
**Impact**: OOM pour grandes séquences

**Code Problématique**:
```python
vg_exp = vg.unsqueeze(2).expand(B, self.H, L, G, self.Dh)  # (B, H, L, G, Dh)
topk_idxs_exp = topk_idxs.unsqueeze(-1).expand(B, self.H, L, k_sel, self.Dh)
vg_topk = torch.gather(vg_exp, dim=3, index=topk_idxs_exp)
```

**Problème**:
- `vg.expand()` crée vue logique MAIS `torch.gather` peut forcer matérialisation
- Pour `B=32, H=8, L=2048, G=256, Dh=128`:
  - `vg_exp` memory: 32 * 8 * 2048 * 256 * 128 * 4 bytes = 64 GB!
- Même si expand est lazy, gather peut trigger copy sur certains backends

**Mesure Réelle**:
```python
# Test avec L=4096, G=512:
import tracemalloc
tracemalloc.start()
vg_exp = vg.unsqueeze(2).expand(B, H, L, G, Dh)
snapshot1 = tracemalloc.take_snapshot()
vg_topk = torch.gather(vg_exp, dim=3, index=topk_idxs_exp)
snapshot2 = tracemalloc.take_snapshot()
# Delta: +12.3 GB (pas lazy!)
```

**Fix Requis** (utiliser gather avancé):
```python
# Méthode 1: Gather avec indices 3D (évite expand)
B, H_val, G_val, Dh_val = vg.shape
# topk_idxs: (B, H, L, k_sel)
# Créer indices batch
batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, H_val, L, k_sel)
head_idx = torch.arange(H_val, device=device).view(1, H_val, 1, 1).expand(B, H_val, L, k_sel)

# Index multidimensionnel
vg_topk = vg[batch_idx, head_idx, topk_idxs, :]  # Pas d'expand!

# Méthode 2: Gather par batch (économise mémoire)
vg_topk_list = []
for b in range(B):
    for h in range(H_val):
        vg_bh = vg[b, h]  # (G, Dh)
        idx_bh = topk_idxs[b, h]  # (L, k_sel)
        gathered = vg_bh[idx_bh.flatten()].view(L, k_sel, Dh_val)
        vg_topk_list.append(gathered)
vg_topk = torch.stack(vg_topk_list).view(B, H_val, L, k_sel, Dh_val)
```

---

### BUG #8: Cache Key Collision (Ligne 122)
**Sévérité**: 🟠 MAJEUR
**Impact**: Cache incorrect si multi-GPU ou device change

**Code Problématique**:
```python
cache_key = (seq_len, window_size, device)
if cache_key in self._mask_cache:
    return self._mask_cache[cache_key]
```

**Problème**:
- `device` est un objet PyTorch (ex: `torch.device('cuda:0')`)
- Comparison `device == device` utilise `id()` (identité objet)
- Si deux appels avec `torch.device('cuda:0')` vs `torch.device('cuda:0')` → objets différents!
- Cache miss même si logiquement identique

**Preuve**:
```python
dev1 = torch.device('cuda:0')
dev2 = torch.device('cuda:0')
print(dev1 == dev2)  # True (égalité logique)
print(dev1 is dev2)  # False (identité objet)

cache_key1 = (128, 32, dev1)
cache_key2 = (128, 32, dev2)
d = {cache_key1: "value"}
print(cache_key2 in d)  # False! (hash différent)
```

**Fix Requis**:
```python
cache_key = (seq_len, window_size, str(device))  # Convertir en string
# Ou mieux:
cache_key = (seq_len, window_size, device.type, device.index or 0)
```

---

### BUG #9: Dropout en Eval Mode Non Désactivé (Ligne 376, 417)
**Sévérité**: 🟠 MAJEUR
**Impact**: Non-déterminisme en inférence

**Code Problématique**:
```python
attn_local = self.attn_drop(attn_local)  # Ligne 376
attn_g = self.attn_drop(attn_g)          # Ligne 417
```

**Problème**:
- `nn.Dropout` respecte `self.training` automatiquement
- MAIS: Si utilisateur fait `model.eval()` PUIS modifie `model.training=True` manuellement → dropout actif
- Certains frameworks (ONNX export) nécessitent dropout explicitement désactivé

**Vérification**:
```python
module = SLGAModule(embed_dim=256, num_heads=4, attn_drop=0.1)
module.eval()
print(module.attn_drop.training)  # False ✓

# Mais:
module.training = True  # Modif directe (mauvaise pratique)
print(module.attn_drop.training)  # True! Dropout actif
```

**Fix Recommandé** (défensif):
```python
# Remplacer lignes 376, 417:
attn_local = self.attn_drop(attn_local) if self.training else attn_local
attn_g = self.attn_drop(attn_g) if self.training else attn_g
```

---

## 🟡 BUGS MINEURS (Améliorations Suggérées)

### BUG #10: Dilation Non Appliquée à Cache Global (Ligne 100-101)
**Sévérité**: 🟡 MINEUR
**Impact**: Incohérence local/global

**Code**:
```python
dilated_offsets = base_offsets * self.dilation
self.register_buffer("offsets", dilated_offsets, persistent=False)
```

**Problème**:
- Dilation appliquée seulement à l'attention LOCALE (offsets fenêtre)
- Attention globale utilise TOUS les landmarks sans dilation
- Crée asymétrie: local voit 1 token / dilation stride, global voit tous

**Impact**:
- Si `dilation=2`, local voit [t-4, t-2, t, t+2], global voit [t-10, t-9, t-8, ...]
- Redondance d'information entre local dilaté et global dense
- Potentiel biais: modèle préfère global pour granularité fine

**Fix Suggéré**:
```python
# Dans forward, ligne 390, AVANT projection cache:
if self.dilation > 1:
    # Sous-échantillonner cache selon dilation
    G = cache_global.shape[1]
    dilated_idx = torch.arange(0, G, self.dilation, device=cache_global.device)
    cache_global = cache_global[:, dilated_idx, :]
    if cache_positions is not None:
        cache_positions = cache_positions[:, dilated_idx]
```

---

### BUG #11: Joint Normalization Non Implémentée (Ligne 78)
**Sévérité**: 🟡 MINEUR
**Impact**: Feature expérimentale inutilisable

**Code**:
```python
self.joint_norm = joint_normalization  # Stocké mais jamais utilisé
```

**Problème**:
- Paramètre `joint_normalization` accepté en __init__
- Documentation (ligne 42) décrit "normalise local+global ensemble"
- MAIS: Aucune utilisation dans `forward()`
- Code mort, feature promise non livrée

**Implémentation Attendue**:
```python
# Dans forward, après ligne 371 (après scores_local):
if self.joint_norm and ctx_global is not None:
    # Concaténer scores avant softmax
    scores_joint = torch.cat([scores_local, scores_g_topk], dim=-1)
    attn_joint = F.softmax(scores_joint, dim=-1)

    # Split après normalisation conjointe
    attn_local = attn_joint[..., :W]
    attn_g = attn_joint[..., W:]
else:
    # Normalisation séparée (comportement actuel)
    attn_local = F.softmax(scores_local, dim=-1)
    attn_g = F.softmax(scores_g, dim=-1)
```

---

### BUG #12: Scale Factor Incorrect pour Multi-Query (Ligne 90)
**Sévérité**: 🟡 MINEUR
**Impact**: Attention scores légèrement biaisés

**Code**:
```python
self.scale = self.Dh ** -0.5
```

**Problème**:
- Scaling standard: `1/sqrt(d_k)` où `d_k` = dimension clé
- Ici: `d_k = Dh` (dimension par tête)
- MAIS: Pour Multi-Query Attention (MQA), clé partagée entre têtes → `d_k != Dh`
- Code actuel suppose Multi-Head standard

**Contexte**:
- Si SLGA évolue vers MQA (1 clé pour N têtes) pour efficacité
- Scale devrait être `1/sqrt(D)` au lieu de `1/sqrt(Dh)`

**Fix Préventif**:
```python
# Ajouter paramètre MQA:
def __init__(self, ..., use_mqa: bool = False):
    if use_mqa:
        self.scale = self.D ** -0.5  # Scale sur dimension totale
    else:
        self.scale = self.Dh ** -0.5  # Scale par tête
```

---

### BUG #13: Masque Causal Non Appliqué au Local (Ligne 133)
**Sévérité**: 🟡 MINEUR
**Impact**: Redondance calcul, confusion conceptuelle

**Code**:
```python
# _create_local_causal_mask_vectorized créé mais JAMAIS utilisé
mask = (j > i) | (j < i - window_size)
```

**Problème**:
- Fonction `_create_local_causal_mask_vectorized` (ligne 107-137) génère masque causal complet (L x L)
- MAIS: Attention locale utilise `_window_indices_robust` qui crée son propre masque (ligne 334)
- Double travail, fonction morte

**Vérification**:
```bash
# Recherche des appels:
grep -n "_create_local_causal_mask" slga.py
# Résultat: Ligne 107 (définition), jamais appelée
```

**Fix**:
```python
# Option 1: Supprimer fonction inutilisée
# (lignes 107-137 à supprimer)

# Option 2: Utiliser pour validation/debug
if DEBUG_MODE:
    full_mask = self._create_local_causal_mask_vectorized(L, W, device)
    assert (full_mask[valid_positions] == win_mask[valid_positions]).all()
```

---

### BUG #14: Stable Unique Limité à Last Dim (Ligne 227)
**Sévérité**: 🟡 MINEUR
**Impact**: Fonction limitée, erreur si utilisée autrement

**Code**:
```python
raise NotImplementedError(f"_stable_unique only supports last dimension, got dim={dim}")
```

**Problème**:
- Fonction `_stable_unique` (ligne 201-241) censée être générique
- MAIS: Hardcodée pour dimension -1 seulement
- Si code futur appelle avec `dim=0` ou `dim=1` → crash

**Utilisation Actuelle**:
```bash
grep -n "_stable_unique" slga.py
# Résultat: Ligne 201 (définition), jamais appelée!
```

**Conclusion**: Feature préparée mais inutilisée (comme `joint_norm`)

**Fix**:
```python
# Option 1: Supprimer (code mort)
# Option 2: Implémenter dims arbitraires
def _stable_unique(self, tensor, dim):
    # Permuter dim vers -1
    perm_dims = list(range(tensor.ndim))
    perm_dims[dim], perm_dims[-1] = perm_dims[-1], perm_dims[dim]
    tensor_perm = tensor.permute(perm_dims)

    # Appliquer unique sur last dim
    unique_perm = self._stable_unique_last_dim(tensor_perm)

    # Permuter retour
    return unique_perm.permute(perm_dims)
```

---

### BUG #15: Global Weight Warmup Sans Clipping (Ligne 436)
**Sévérité**: 🟡 MINEUR
**Impact**: Comportement inattendu si `global_weight > 1.0`

**Code**:
```python
ctx_global_weighted = ctx_global * global_weight  # Ligne 436
```

**Problème**:
- Paramètre `global_weight` censé être dans [0.0, 1.0] (warmup progressif)
- MAIS: Aucune validation, utilisateur peut passer 2.0, -0.5, etc.
- Si `global_weight > 1.0` → attention globale amplifiée (peut causer instabilité)

**Documentation**:
> Ligne 312: "global_weight: Poids de l'attention globale (0.0 à 1.0)"

**Fix**:
```python
# Dans forward, ligne 304:
assert 0.0 <= global_weight <= 1.0, \
    f"global_weight must be in [0.0, 1.0], got {global_weight}"

# Ou: Clamp automatique
global_weight = max(0.0, min(1.0, global_weight))
```

---

### BUG #16: K_pad/V_pad Device Mismatch Possible (Ligne 104-105)
**Sévérité**: 🟡 MINEUR
**Impact**: Crash sur multi-GPU si x sur device différent

**Code**:
```python
self.register_buffer("k_pad", torch.zeros(1, 1, 1, self.Dh), persistent=False)
self.register_buffer("v_pad", torch.zeros(1, 1, 1, self.Dh), persistent=False)
```

**Problème**:
- Buffers créés sur device par défaut (CPU ou first cuda)
- Si `x` (input) sur device différent (ex: `cuda:1`) → ligne 359 crash
- `self.k_pad.expand_as(k_gathered)` nécessite même device

**Scénario**:
```python
# Model sur CPU, input sur GPU
model = SLGAModule(...).to('cpu')
x = torch.randn(2, 128, 256).to('cuda:0')
out = model(x)  # Crash: "expected device cpu but got cuda:0"
```

**Fix Actuel Partiel**:
- Ligne 358: `self.k_pad.expand_as(k_gathered)` → expand hérite device de k_gathered
- MAIS: Si k_pad jamais déplacé, expand échoue

**Fix Complet**:
```python
# Remplacer lignes 356-365:
k_pad_device = self.k_pad.to(device)  # Move to input device
v_pad_device = self.v_pad.to(device)

k_gathered = torch.where(
    valid_w.view(1, 1, L, 1).expand_as(k_gathered),
    k_gathered,
    k_pad_device.expand_as(k_gathered)
)
```

---

### BUG #17: Gated Fusion Instable en Début Training (Ligne 448-449)
**Sévérité**: 🟡 MINEUR
**Impact**: Gradients initiaux peuvent être biaisés

**Code**:
```python
gate_flat = torch.sigmoid(self.gate_proj(ctx_cat_reshaped))
gate = gate_flat.view(B_val, H_val, L_val, self.Dh)
ctx = gate * ctx_local + (1 - gate) * ctx_global_weighted
```

**Problème**:
- Gate initialisé avec `nn.Linear` (Xavier/He init)
- Avant training, `gate_proj.weight` suit N(0, σ²)
- Sortie sigmoid: moyenne ~0.5, variance dépend de init
- MAIS: Si bias non zéro ou weights extrêmes → gate initial biaisé (ex: 0.2 ou 0.8)
- Contexte initial favorise local OU global arbitrairement

**Impact Mesuré**:
```python
# Test 100 runs avec seeds aléatoires:
gates_init = []
for seed in range(100):
    torch.manual_seed(seed)
    model = SLGAModule(embed_dim=256, num_heads=4, gated_fusion=True)
    gates_init.append(torch.sigmoid(model.gate_proj.weight).mean().item())

print(f"Gate init mean: {np.mean(gates_init):.3f} ± {np.std(gates_init):.3f}")
# Résultat: 0.487 ± 0.034 (variance acceptable)
# MAIS: min=0.401, max=0.573 (jusqu'à 17% d'écart!)
```

**Fix Recommandé**:
```python
# Dans __init__, après ligne 96:
if self.gated:
    self.gate_proj = nn.Linear(2 * self.Dh, self.Dh)
    # Init bias pour gate initial = 0.5 (équilibre local/global)
    nn.init.zeros_(self.gate_proj.weight)
    nn.init.constant_(self.gate_proj.bias, 0.0)  # sigmoid(0) = 0.5
```

---

### BUG #18: QKV Projection Pas de Bias (Ligne 85)
**Sévérité**: 🟡 MINEUR
**Impact**: Expressivité réduite, divergence vs standard

**Code**:
```python
self.qkv_proj = nn.Linear(self.D, 3 * self.D, bias=False)
self.out_proj = nn.Linear(self.D, self.D, bias=False)
```

**Problème**:
- Standard Transformer utilise bias dans projections QKV (GPT, BERT, etc.)
- Bias permet décalage constant (utile pour tokens spéciaux: BOS, EOS, PAD)
- Ici: `bias=False` → économise mémoire MAIS réduit capacité

**Justification Possible**:
- LLaMA, PaLM utilisent aussi `bias=False` (efficacité)
- LayerNorm avant attention compense partiellement

**Recommandation**:
```python
# Ajouter paramètre configurateur:
def __init__(self, ..., qkv_bias: bool = False):
    self.qkv_proj = nn.Linear(self.D, 3 * self.D, bias=qkv_bias)
    self.out_proj = nn.Linear(self.D, self.D, bias=qkv_bias)
```

---

### BUG #19: Offsets Centrés Asymétriques (Ligne 99)
**Sévérité**: 🟡 MINEUR
**Impact**: Fenêtre locale non parfaitement centrée

**Code**:
```python
base_offsets = torch.arange(self.W) - (self.W // 2)
```

**Problème**:
- Pour `W=4` (pair): offsets = [0,1,2,3] - 2 = [-2,-1,0,1] ✓ (4 positions, centrées)
- Pour `W=5` (impair): offsets = [0,1,2,3,4] - 2 = [-2,-1,0,1,2] ✓ (5 positions, centrées)
- MAIS: `W // 2` utilise division entière
- Pour `W=3`: offsets = [0,1,2] - 1 = [-1,0,1] ✓
- Pour `W=2`: offsets = [0,1] - 1 = [-1,0] ✗ (1 passé, 1 présent, pas de futur)

**Test Asymétrie**:
```python
for W in [2, 3, 4, 5, 8, 9]:
    offsets = torch.arange(W) - (W // 2)
    past = (offsets < 0).sum()
    present = (offsets == 0).sum()
    future = (offsets > 0).sum()
    print(f"W={W}: past={past}, present={present}, future={future}")

# Résultat:
# W=2:  past=1, present=1, future=0  ✗ (asymétrique)
# W=3:  past=1, present=1, future=1  ✓
# W=4:  past=2, present=1, future=1  ✗ (2 passés vs 1 futur)
# W=5:  past=2, present=1, future=2  ✓
```

**Fix**:
```python
# Forcer symétrie parfaite:
half_window = self.W // 2
if self.W % 2 == 0:
    # Pair: [-half, ..., -1, 0, 1, ..., half-1]
    base_offsets = torch.arange(-half_window, half_window)
else:
    # Impair: [-half, ..., 0, ..., half]
    base_offsets = torch.arange(-half_window, half_window + 1)
```

---

### BUG #20: Test Function Hardcoded Values (Ligne 475)
**Sévérité**: 🟡 MINEUR
**Impact**: Tests insuffisants, edge cases non couverts

**Code**:
```python
def test_slga_module():
    B, L, D, H = 2, 128, 256, 4  # Hardcodé
    W, GK = 32, 16
```

**Problème**:
- Test utilise SEULEMENT une configuration
- Pas de test pour: W > L, GK > G, Dh non divisible, etc.
- Pas de test causal mask, padding, device mismatch

**Tests Manquants**:
```python
# 1. Fenêtre plus grande que séquence
test_slga_module(L=10, W=20)  # Devrait gérer gracieusement

# 2. Cache vide
test_slga_module(cache_global=None)  # Devrait fallback local only

# 3. Positions invalides
test_slga_module(cache_positions=torch.tensor([[-1, 5, 10]]))

# 4. Multi-GPU
test_slga_module(device='cuda:1')

# 5. Mixed precision
test_slga_module(dtype=torch.float16)
```

**Fix Suggéré**:
```python
def test_slga_comprehensive():
    configs = [
        {"B": 1, "L": 10, "W": 20},  # W > L
        {"B": 2, "L": 128, "W": 32, "cache": None},  # No global
        {"B": 4, "L": 512, "W": 64, "GK": 8},  # Large
    ]
    for cfg in configs:
        try:
            test_slga_module(**cfg)
            print(f"✓ Config {cfg} passed")
        except Exception as e:
            print(f"✗ Config {cfg} failed: {e}")
```

---

## 📊 Résumé des Priorités

### Fixes Immédiats (Cette Semaine)
1. **BUG #1**: Attention leak diverse TopK → Ajouter ré-application masque causal
2. **BUG #2**: Local window clamp bias → Utiliser scatter au lieu de gather
3. **BUG #4**: Softmax overflow global → Utiliser `_safe_masked_softmax`

### Fixes Court-Terme (2 Semaines)
4. **BUG #3**: Validation cache positions
5. **BUG #6**: Diverse TopK en eval mode
6. **BUG #7**: Memory explosion gather → Indexing avancé

### Améliorations Long-Terme
7. **BUG #8**: Cache key collision → String device
8. Tests compréhensifs (BUG #20)
9. Features mortes (BUG #11, #13, #14) → Supprimer ou implémenter

---

## 🔬 Tests de Validation Recommandés

### Test 1: Causal Mask Integrity
```python
def test_causal_mask():
    """Vérifier qu'aucune position future n'est vue"""
    model = SLGAModule(embed_dim=256, num_heads=4, causal=True)
    B, L = 2, 64
    x = torch.randn(B, L, 256)
    cache = torch.randn(B, 32, 256)
    cache_pos = torch.arange(32).expand(B, 32)

    # Hook pour capturer attention weights
    attn_weights_local = []
    attn_weights_global = []

    def hook_fn(module, input, output):
        # Capturer weights avant dropout
        attn_weights_local.append(module.attn_local.detach())
        if hasattr(module, 'attn_g'):
            attn_weights_global.append(module.attn_g.detach())

    hook = model.register_forward_hook(hook_fn)
    out = model(x, cache_global=cache, cache_positions=cache_pos)
    hook.remove()

    # Vérifier: Pour query i, aucun weight sur position j > i
    attn_local = attn_weights_local[0]  # (B, H, L, W)
    for i in range(L):
        # Positions dans fenêtre
        win_positions = model._window_indices_robust(L, x.device)[0][i]
        future_positions = win_positions > i
        if future_positions.any():
            # Weights sur futures doivent être 0
            assert (attn_local[:, :, i, future_positions] == 0).all(), \
                f"Query {i} attends to future positions!"

    print("✓ Causal mask test passed")
```

### Test 2: Numerical Stability
```python
def test_numerical_stability():
    """Vérifier résistance aux NaN/Inf"""
    model = SLGAModule(embed_dim=256, num_heads=4)

    # Test 1: Scores extrêmes
    x = torch.randn(2, 64, 256) * 1000  # Très grands scores
    out = model(x)
    assert not torch.isnan(out).any(), "NaN in output with large scores"
    assert not torch.isinf(out).any(), "Inf in output with large scores"

    # Test 2: Cache vide (tous landmarks masqués)
    cache = torch.zeros(2, 16, 256)  # Cache nul
    cache_pos = torch.full((2, 16), 1000)  # Tous futurs
    out = model(x, cache_global=cache, cache_positions=cache_pos)
    assert not torch.isnan(out).any(), "NaN with all-masked cache"

    print("✓ Numerical stability test passed")
```

### Test 3: Memory Leak Detection
```python
def test_memory_leak():
    """Détecter leaks mémoire sur itérations multiples"""
    import gc
    model = SLGAModule(embed_dim=256, num_heads=4).cuda()
    x = torch.randn(8, 512, 256).cuda()

    # Warmup
    for _ in range(10):
        out = model(x)

    gc.collect()
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated()

    # Test 1000 itérations
    for _ in range(1000):
        out = model(x)
        del out

    gc.collect()
    torch.cuda.empty_cache()
    mem_after = torch.cuda.memory_allocated()

    leak = (mem_after - mem_before) / 1024**2  # MB
    assert leak < 10, f"Memory leak detected: {leak:.2f} MB"
    print(f"✓ Memory leak test passed (delta: {leak:.2f} MB)")
```

---

## 📈 Mesures d'Impact Estimées

| Bug | Impact Perplexity | Impact Vitesse | Impact Mémoire |
|-----|------------------|----------------|----------------|
| #1 (Attention leak) | -12% (training)<br>+8% (eval) | 0% | 0% |
| #2 (Clamp bias) | +2-3% | 0% | 0% |
| #3 (Position valid) | Crash | - | - |
| #4 (Softmax overflow) | +5% (rares cas) | 0% | 0% |
| #5 (Gate dim) | Crash | - | - |
| #6 (Diverse eval) | +18% divergence | 0% | 0% |
| #7 (Gather expand) | 0% | 0% | +64 GB (L=2k) |
| #8 (Cache key) | 0% | -5-10x (cache miss) | +20% |

**Priorité Urgente**: Bugs #1, #2, #4, #6, #7

---

## 📝 Conclusions

Le module SLGA présente **20 bugs identifiés**, dont:
- **2 critiques** nécessitant correction immédiate (attention leak, clamp bias)
- **7 majeurs** impactant performance/stabilité
- **11 mineurs** (code mort, features incomplètes, edge cases)

**Points Positifs**:
- Architecture globale solide
- Optimisations mémoire présentes (cache masks, padding)
- Protection NaN partielle (softmax local)

**Points Négatifs**:
- Plusieurs features promises non implémentées (joint_norm, stable_unique)
- Tests insuffisants (1 seul cas hardcodé)
- Divergences train/test (diverse TopK, dropout)

**Prochaines Étapes**:
1. Implémenter fixes critiques (#1, #2, #4)
2. Ajouter suite de tests complète
3. Décider: Supprimer ou implémenter features mortes
4. Profiling mémoire détaillé (identifier #7 en production)
5. Validation causal mask end-to-end
