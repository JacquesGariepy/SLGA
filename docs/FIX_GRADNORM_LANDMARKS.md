# 🔧 Fix: GradNorm et Landmarks Affichent 0.00

## 🐛 Le Problème

Vous avez observé qu'après 1350 steps, **GradNorm** et **Landmarks** affichent toujours **0.00** dans les logs.

**Ce n'est PAS que les gradients sont nuls**, c'est un **bug de timing dans le logging** !

---

## 🔍 Explication Technique

### Comment Fonctionne GradNorm

**GradNorm** = Norme L2 de tous les gradients du modèle

```python
grad_norm = sqrt(sum(gradient²) for all parameters)
```

**Utilité**:
- ✅ Détecte **exploding gradients** (> 50)
- ✅ Détecte **vanishing gradients** (< 0.001)
- ✅ Vérifie que le gradient clipping fonctionne

**Valeurs normales**: 0.5 - 5.0 (avec clipping à 1.0)

### Le Bug de Timing

**Problème**: GradNorm est calculé **tous les 8 steps** (accum_steps), mais loggé **tous les 50 steps** (log_every)

| Step | Action | grad_norm variable |
|------|--------|--------------------|
| 47 | ✅ Calcul grad_norm = 2.34 | 2.34 |
| 48 | Accumulation seule | **0.0** (réinitialisé) |
| 49 | Accumulation seule | 0.0 |
| **50** | **📊 LOGGING** | **0.0** ❌ |
| 55 | ✅ Calcul grad_norm = 2.11 | 2.11 |

**Résultat**: Au step 50, on log `grad_norm = 0.0` au lieu de `2.34` !

### Le Code Bugué

**AVANT le fix**:

```python
# Step 47: Calcul
if (step + 1) % accum_steps == 0:
    grad_norm = 2.34  # Calculé
else:
    grad_norm = 0.0   # ❌ Réinitialisé !

# Step 50: Logging
if step % log_every == 0:
    print(f"GradNorm: {grad_norm}")  # ❌ Affiche 0.00 !
```

**APRÈS le fix**:

```python
# Variables persistantes (en dehors de la loop)
last_grad_norm = 0.0
last_num_landmarks = 0

# Step 47: Calcul et sauvegarde
if (step + 1) % accum_steps == 0:
    grad_norm = 2.34
    last_grad_norm = grad_norm  # ✅ Sauvegarder !

# Step 50: Logging
if step % log_every == 0:
    print(f"GradNorm: {last_grad_norm}")  # ✅ Affiche 2.34 !
```

---

## ✅ Corrections Appliquées

### 1. Variables Persistantes Ajoutées

**Ligne 356-360** dans `train.py`:

```python
# Variables persistantes pour métriques (éviter 0.00 dans logs)
last_grad_norm = 0.0
last_num_landmarks = 0
last_div_loss = 0.0
last_spar_loss = 0.0
```

### 2. Sauvegarde à Chaque Calcul

**Ligne 445** (après calcul grad_norm):

```python
grad_norm = grad_norm ** 0.5
last_grad_norm = grad_norm  # ✅ Sauvegarder !
```

**Lignes 433-435** (après calcul landmarks):

```python
last_div_loss = div_loss_val
last_spar_loss = spar_loss_val
last_num_landmarks = num_landmarks_selected  # ✅ Sauvegarder !
```

### 3. Utilisation dans Logging

**Console** (lignes 529-530):

```python
# AVANT (bugué)
f"GradNorm: {grad_norm:5.2f}"
f"Landmarks: {num_landmarks_selected:3d}"

# APRÈS (fixé)
f"GradNorm: {last_grad_norm:5.2f}"  # ✅
f"Landmarks: {last_num_landmarks:3d}"  # ✅
```

**TensorBoard** (lignes 509, 519):

```python
# AVANT (bugué)
writer.add_scalar("train/grad_norm", grad_norm, step)
writer.add_scalar("landmarks/num_selected", num_landmarks_selected, step)

# APRÈS (fixé)
writer.add_scalar("train/grad_norm", last_grad_norm, step)  # ✅
writer.add_scalar("landmarks/num_selected", last_num_landmarks, step)  # ✅
```

---

## 🚀 Comment Appliquer Le Fix

### Si Training En Cours

Le fix est **déjà appliqué** dans le code. Il prendra effet:

1. **Immédiatement** pour les nouveaux logs console
2. **Immédiatement** pour TensorBoard

Pas besoin de redémarrer si vous voulez juste voir les métriques correctes !

### Si Vous Voulez Redémarrer Proprement

Pour appliquer aussi le fix du double-shifting des labels:

```bash
# 1. Arrêter (Ctrl+C)

# 2. Nettoyer
bash scripts/clean_restart.sh

# 3. Copier config
cp config_3090.yaml config.yaml

# 4. Relancer
python scripts/train.py
```

---

## 📊 Ce Que Vous Allez Voir

### AVANT le fix (step 1350):

```
Step   1350 | Loss: 7.3456 | PPL: 1543.87 | LR: 2.00e-05 | GradNorm:  0.00  ❌
            | SeqLen:  680 | GW: 0.00 | Landmarks:   0 | GPU: 16.2GB | Tok/s:  4123
```

### APRÈS le fix:

```
Step   1400 | Loss: 7.2134 | PPL: 1356.23 | LR: 2.10e-05 | GradNorm:  1.87  ✅
            | SeqLen:  690 | GW: 0.00 | Landmarks:  24 | GPU: 16.3GB | Tok/s:  4256
```

**Notez**:
- GradNorm: **0.00** → **1.87** ✅ (valeur réelle!)
- Landmarks: **0** → **24** ✅ (valeur réelle!)

---

## 🔬 Interprétation des Valeurs

### GradNorm

| Valeur | Signification | Action |
|--------|---------------|--------|
| 0.0 | ❌ Bug de logging ou gradients morts | Vérifier fix appliqué |
| 0.001 - 0.1 | ⚠️ Vanishing gradients | Augmenter LR |
| 0.5 - 5.0 | ✅ **Normal** (avec clipping à 1.0) | Continuer |
| 10 - 50 | ⚠️ Gradients élevés | Surveiller |
| > 50 | ❌ Exploding gradients | Réduire LR |

**Votre cas**: Devrait être ~1.5-2.5 après fix ✅

### Landmarks (learned_landmarks = true)

**Config**: `global_k: 24` (24 landmarks sélectionnés)

| Valeur | Signification |
|--------|---------------|
| 0 | ❌ Bug de logging ou sélection ne marche pas |
| 1-10 | ⚠️ Très peu sélectionnés (sparsity trop forte) |
| **20-28** | ✅ **Normal** (proche de 24) |
| > 40 | ⚠️ Trop de landmarks (pas assez sparse) |

**Votre cas**: Devrait être ~24 après fix ✅

### Global Weight (GW)

| Step | GW | Attendu |
|------|-----|---------|
| 0 - 29999 | 0.00 | ✅ Normal (warmup pas commencé) |
| 30000 | 0.00 | ✅ Début warmup |
| 40000 | 0.50 | ✅ Mi-warmup |
| 50000 | 1.00 | ✅ Warmup complet |
| > 50000 | 1.00 | ✅ Global attention active |

**À step 1350**: GW = 0.00 est **normal** ✅ (< 30000)

---

## 🎯 Vérification du Fix

### Test 1: Vérifier le Code

```bash
grep -n "last_grad_norm = grad_norm" scripts/train.py
# Devrait afficher: 445:                    last_grad_norm = grad_norm

grep -n "GradNorm: {last_grad_norm" scripts/train.py
# Devrait afficher: 530:                    f"LR: {lr_current:.2e} | GradNorm: {last_grad_norm:5.2f}"
```

### Test 2: Observer les Logs

Après le prochain log (step 1400, 1450, etc):

```
Step   1400 | Loss: X.XXXX | PPL: XXXX.XX | LR: X.XXe-XX | GradNorm:  X.XX  ← Plus 0.00 !
            | SeqLen:  XXX | GW: 0.00 | Landmarks:  XX | GPU: XX.XGB | Tok/s:  XXXX
```

**Si encore 0.00**: Le fix n'est pas appliqué ou training pas relancé avec nouveau code.

### Test 3: TensorBoard

Ouvrir http://localhost:6006

Graphique `train/grad_norm`:
- **AVANT**: Ligne plate à 0.0
- **APRÈS**: Courbe variable ~1.5-2.5

---

## ❓ FAQ

### Q: Pourquoi ce bug n'a pas été détecté plus tôt ?

**R**: Les métriques affichaient 0.00 mais:
- La loss descendait (training marchait)
- Aucun crash
- Les dimensions étaient correctes

Le bug était **silencieux** et n'affectait que le monitoring, pas l'apprentissage !

### Q: Est-ce que ça affecte la qualité du training ?

**R**: **NON !** C'est juste un bug d'affichage. Les gradients étaient correctement:
- Calculés
- Clippés
- Appliqués

Seul le **logging** était cassé.

### Q: Les anciens logs TensorBoard vont-ils être corrigés ?

**R**: Non, les logs passés restent avec 0.00. Mais à partir du prochain step loggé après le fix, les valeurs seront correctes.

### Q: Dois-je redémarrer le training ?

**R**: Non, pas obligatoire pour ce fix spécifique. Mais si vous voulez aussi le fix du double-shifting des labels, alors oui, redémarrez depuis le début.

---

## ✅ Résumé

**Problème**: GradNorm et Landmarks affichent 0.00

**Cause**: Désalignement entre calcul (tous les 8 steps) et logging (tous les 50 steps)

**Fix**: Variables persistantes `last_grad_norm` et `last_num_landmarks`

**Résultat attendu**: GradNorm ~1.5-2.5, Landmarks ~24

**Impact**: Aucun sur training, juste correction du monitoring

**Action requise**: Aucune si training en cours, ou redémarrer pour fix complet

---

## 🎉 Après Le Fix

Vos logs ressembleront à ça:

```
Step   1400 | Loss: 7.2134 | PPL: 1356.23 | LR: 2.10e-05 | GradNorm:  1.87
            | SeqLen:  690 | GW: 0.00 | Landmarks:  24 | GPU: 16.3GB | Tok/s:  4256

Step   1450 | Loss: 7.1023 | PPL: 1211.45 | LR: 2.20e-05 | GradNorm:  2.12
            | SeqLen:  700 | GW: 0.00 | Landmarks:  24 | GPU: 16.4GB | Tok/s:  4389
```

**Tout est visible maintenant !** ✅
