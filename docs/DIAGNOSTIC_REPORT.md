# Rapport d'Analyse - Perplexité Critique du Modèle SLGA

## Résumé
Après 30,000 steps d'entraînement, le modèle SLGA présente une **perplexité catastrophique de 5448** en moyenne, indiquant qu'il n'a pratiquement rien appris. Cette analyse identifie les causes probables.

---

## Bugs Critiques Identifiés

### 🚨 BUG CRITIQUE #1: `global_warmup_weight` Non Utilisé

**Localisation**: `scripts/train.py:354`

**Problème**:
```python
# Le poids est calculé...
global_weight = get_global_warmup_weight(step, cfg)

# ...mais JAMAIS appliqué au modèle !
# Il est seulement loggé
```

**Impact**:
- Le warmup progressif de l'attention globale n'est PAS implémenté
- Avec `global_warmup_start: 30000` dans la config, l'attention globale devrait être désactivée jusqu'au step 30000
- Mais comme le poids n'est jamais utilisé, l'attention globale est active dès le début
- Cela peut causer de l'instabilité et empêcher le modèle d'apprendre progressivement

**Solution Requise**:
Le `global_warmup_weight` doit être passé au modèle et utilisé pour pondérer la contribution de l'attention globale.

---

### ⚠️  BUG #2: Landmarks Fixes à Travers les Couches

**Localisation**: `src/model.py:248, 267`

**Problème**:
```python
# Les landmarks sont sélectionnés UNE SEULE FOIS
_, landmark_states, landmark_scores = self.landmark_selector(x)  # Ligne 248

# Puis utilisés à TOUTES les couches
for block in self.blocks:
    x = block(x, cache_global=landmark_states)  # Ligne 267
    # x évolue, mais landmark_states reste fixe !
```

**Impact**:
- Les representations principales évoluent à chaque couche
- Mais les landmarks restent les mêmes embeddings initiaux
- Cela crée un décalage sémantique croissant entre les deux
- L'attention globale devient de moins en moins pertinente dans les couches profondes

**Question**: Est-ce intentionnel (style cross-attention) ou un bug ?

**Solutions Possibles**:
1. **Option A**: Re-sélectionner les landmarks à chaque couche (coûteux mais précis)
2. **Option B**: Mettre à jour les landmark embeddings à chaque couche sans re-sélection
3. **Option C**: Si c'est intentionnel, documentez pourquoi

---

### 📊 PROBLÈME #3: Logs TensorBoard Non Écrits

**Localisation**: `out_slga/tensorboard/`

**Observation**:
```bash
-rwxrwxrwx 1 jac jac   88 Oct 22 18:34 events.out.tfevents...
```

Les fichiers TensorBoard ne font que 88 bytes, ce qui signifie qu'ils sont quasiment vides.

**Impact**:
- Impossible de visualiser la courbe d'entraînement
- Pas de monitoring du progrès
- Difficile de debugger

**Vérification Nécessaire**:
Le `SummaryWriter` est-il correctement utilisé dans la boucle d'entraînement ?

---

## Architecture SLGA - Points de Vigilance

### 1. Fusion Gated
**Localisation**: `src/slga.py:345-359`

La fusion entre attention locale et globale utilise un gate appr is. Vérifiez que:
- Les gradients passent correctement
- Le gate n'est pas saturé (toujours 0 ou toujours 1)
- La dimension est correcte (corrigée récemment)

### 2. Diverse Top-K
**Localisation**: `src/slga.py:181`

Le `selection_counts` doit avoir le même dtype que `scores` (corrigé récemment pour AMP).

### 3. Landmark Selector avec Straight-Through
**Localisation**: `src/landmarks.py:120-124`

Le straight-through estimator permet les gradients mais peut être instable:
```python
selection = selection_onehot + scores - scores.detach()
```

Vérifiez que les gradients passent correctement.

---

## Tests de Diagnostic Recommandés

### Test 1: Forward Pass sur Modèle Non-Entraîné
```bash
python scripts/diagnose.py
```

Ce script teste:
- ✅ Forward pass sans erreurs
- ✅ Pas de NaN/Inf dans les logits
- ✅ Attention mechanism fonctionnel
- ✅ Landmark selector fonctionnel

### Test 2: Overfitting Test
Créer un test pour voir si le modèle PEUT overfit sur un tiny batch:
- Si OUI → le modèle peut apprendre, problème dans l'entraînement
- Si NON → le modèle a un problème architectural

### Test 3: Vérifier les Gradients
```python
# Dans la boucle d'entraînement
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm == 0:
            print(f"WARNING: {name} has zero gradient!")
        if math.isnan(grad_norm):
            print(f"ERROR: {name} has NaN gradient!")
```

---

## Recommandations Prioritaires

### 🔴 URGENT - À Corriger Immédiatement

1. **Implémenter le warmup de l'attention globale**
   - Modifier `src/slga.py` pour accepter un paramètre `global_weight`
   - Appliquer ce poids à `ctx_global` avant la fusion
   - Passer le poids depuis le training script

2. **Décider sur la stratégie des landmarks**
   - Si fixes = documenter pourquoi
   - Sinon, mettre à jour les landmarks à chaque couche

3. **Fixer le logging TensorBoard**
   - Vérifier que `writer.add_scalar()` est appelé correctement
   - Ajouter `writer.flush()` périodiquement

### 🟡 IMPORTANT - À Investiguer

4. **Vérifier la qualité des données**
   - S'assurer que le dataset Wikipedia charge correctement
   - Vérifier que les textes sont sensés
   - Vérifier la tokenization

5. **Tester l'overfitting capacity**
   - Prendre 10 exemples
   - Entraîner jusqu'à loss < 0.1
   - Si impossible → problème architectural

6. **Analyser la distribution des landmarks**
   - Visualiser quelles positions sont sélectionnées
   - Vérifier la diversité spatiale
   - S'assurer qu'ils ne sont pas tous au même endroit

### 🟢 NICE TO HAVE - Améliorations

7. **Ajouter plus de logging**
   - Log des activations moyennes par couche
   - Log du nombre de landmarks sélectionnés
   - Log des normes de gradients

8. **Validation périodique**
   - Calculer la perplexity sur validation set
   - Comparer avec un modèle baseline

---

## Code à Ajouter: Global Warmup

### Dans `src/slga.py`

```python
def forward(
    self,
    x: torch.Tensor,
    cache_global: Optional[torch.Tensor] = None,
    cache_positions: Optional[torch.Tensor] = None,
    global_weight: float = 1.0,  # NOUVEAU
) -> torch.Tensor:
    # ... code existant ...

    # Dans la section fusion (ligne ~344)
    if ctx_global is not None:
        # Appliquer le warmup weight
        ctx_global = ctx_global * global_weight  # NOUVEAU

        if self.gated:
            # ... reste du code de fusion ...
```

### Dans `src/model.py`

```python
def forward(
    self,
    input_ids: torch.Tensor,
    cache_global_ids: Optional[torch.Tensor] = None,
    return_aux: bool = False,
    global_weight: float = 1.0,  # NOUVEAU
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
    # ... code existant ...

    # Passer aux blocs
    for block in self.blocks:
        x = block(x, cache_global=landmark_states, global_weight=global_weight)  # MODIFIÉ
```

### Dans `src/model.py` - TransformerBlock

```python
def forward(
    self,
    x: torch.Tensor,
    cache_global: Optional[torch.Tensor] = None,
    global_weight: float = 1.0,  # NOUVEAU
) -> torch.Tensor:
    attn_out = self.attn(
        self.norm1(x),
        cache_global=cache_global,
        global_weight=global_weight  # NOUVEAU
    )
    # ... reste ...
```

### Dans `scripts/train.py`

```python
# Dans la boucle d'entraînement (ligne ~368)
logits, aux = model(
    input_ids,
    cache_global_ids=cache_ids,
    return_aux=True,
    global_weight=global_weight  # NOUVEAU - utiliser la variable calculée
)
```

---

## Conclusion

Le modèle SLGA a plusieurs problèmes qui empêchent l'apprentissage:

1. **Le warmup global n'est pas implémenté** → cause principale probable
2. **Les landmarks sont statiques** → réduit l'efficacité
3. **Le monitoring est cassé** → impossible de diagnostiquer

**Prochaines étapes**:
1. Exécuter `python scripts/diagnose.py` pour confirmer que l'architecture fonctionne
2. Implémenter le global warmup (code fourni ci-dessus)
3. Relancer l'entraînement depuis le début
4. Surveiller la perplexity toutes les 1000 steps

**Résultat attendu après corrections**:
- Step 1000: PPL ~500-1000
- Step 5000: PPL ~200-400
- Step 10000: PPL ~100-200
- Step 20000: PPL ~50-100
