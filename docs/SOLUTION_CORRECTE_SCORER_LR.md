# ✅ VRAIE SOLUTION - Param Groups avec LR Séparé

**Date**: 2025-10-28
**Crédit**: Suggestion utilisateur (correcte à 100%)

---

## 🎓 MES ERREURS ADMISES

### Erreur #1: Calcul Updates Incorrect

**Ce que j'ai dit** (FAUX):
```python
update = LR × grad = 1.5e-4 × 0.01 = 1.5e-6
→ Updates trop petites
```

**La réalité avec Adam** (CORRECT):
```python
# Adam normalise par √(second moment)
# Avec grad ~0.01, v ~1e-4
# update ≈ LR × (grad / √v) ≈ 1.5e-4 × (0.01/0.01) ≈ 1.5e-4
→ Updates normales, pas 100× plus faibles!
```

**Merci pour la correction!**

---

### Erreur #2: Lambdas Absurdes

**Ce que j'ai recommandé** (CATASTROPHIQUE):
```yaml
lambda_spacing: 100,000  # ❌ Écrase CE 230×!
lambda_sparsity: 500     # ❌ Domine 100×!
```

**Résultat**:
```
Spacing loss: 0.0132 × 100K = 1,320
Loss CE: 5.74
→ Spacing domine 230× la loss CE
→ Modèle ne peut plus apprendre le langage!
```

**Analyse correcte utilisateur**:
> "Pousser lambda à 100,000 écrase la CE principale"
> "Spacing ~1e-4 = contrainte saturée, pas de marge d'exploration"

**Absolument raison!**

---

## ✅ VRAIE SOLUTION (Utilisateur)

### Param Groups avec LR Différents

**Code appliqué** (train.py:479-519):
```python
# Séparer paramètres scorer vs modèle
scorer_params = model.landmark_selector.scorer.parameters()
other_params = [reste du modèle]

# Param groups avec LR différents
optimizer = AdamW([
    {"params": other_params, "lr": 1.5e-4, "name": "model"},
    {"params": scorer_params, "lr": 7.5e-4, "name": "scorer"},  # 5× plus
])
```

**Avantages**:
- ✅ Scorer apprend 5× plus vite
- ✅ Lambdas restent raisonnables (500/10)
- ✅ Pas de domination des losses auxiliaires
- ✅ Modèle apprend langage normalement
- ✅ Solution propre et élégante

---

### Config Finale (v1.6)

```yaml
train:
  lr: 1.5e-4                    # LR de base
  scorer_lr_multiplier: 5.0     # Scorer 5× plus vite

  lambda_spacing: 500.0         # Raisonnable (200-1000)
  lambda_sparsity: 10.0         # Raisonnable (5-20)
```

---

## 📊 IMPACT ATTENDU

### Avec LR Séparé

**Scorer** (LR = 7.5e-4):
```
Grad: 0.01
Update: 7.5e-4 (Adam normalisé)
Après 5000 steps: Changement significatif
→ Scores varient, std augmente
```

**Modèle** (LR = 1.5e-4):
```
Grad: 1.5
Update: 1.5e-4 × (normalisé)
→ Apprend langage normalement
```

**Losses équilibrées**:
```
Loss CE: 5.0 (83%)
Spacing: 0.5 (8%)
Sparsity: 0.5 (8%)
→ Total: ~6.0
```

---

## 🚀 COMMANDE MISE À JOUR

```bash
# Supprimer checkpoints avec lambdas absurdes
rm -rf out_slga

# Training avec param groups LR
python scripts/train.py --config config/config.wikipedia.yaml --max-steps 5000
```

**Vérifier au step 5000**:
```
✅ Loss: < 5.0
✅ Spacing: 5-15 (avec lambda=500)
✅ Sparsity: 1-3 (avec lambda=10)
✅ Scorer std: > 0.01 (apprend!)
✅ Scorer LR affiché: 7.5e-4 au démarrage
```

---

## 🎓 LEÇONS APPRISES

### Ce Qui NE Marche PAS
1. ❌ Augmenter lambdas à l'infini
2. ❌ Faire 4 tentatives de fix sparsity complexes
3. ❌ Calculs SGD naïfs pour Adam

### Ce Qui MARCHE
1. ✅ Param groups avec LR différents
2. ✅ Lambdas raisonnables (200-1000, 5-20)
3. ✅ Monitorer norm des poids scorer
4. ✅ Écouter les bonnes suggestions!

---

## 📈 MONITORING RECOMMANDÉ

### Pendant Training

**Logs console** - Chercher:
```
🔧 Scorer LR: 7.50e-04 (×5.0 vs modèle)  ← Au démarrage
```

**Au step 1000**:
```
Spacing: 5-15 (pas 0.0002!)
Sparsity: 1-3 (varie, pas constant!)
```

**Au step 5000**:
```
Scorer std augmente (>0.01)
Loss < 5.0
Génération cohérente
```

---

## 🙏 REMERCIEMENTS

**Merci pour les corrections précises!** Tu avais raison sur:
- Adam vs SGD (calcul updates)
- Lambdas absurdes écrasent CE
- Solution élégante param groups
- Valeurs raisonnables 200-1000 / 5-20

**Nouvelle approche appliquée** avec ta recommandation! 🎯
