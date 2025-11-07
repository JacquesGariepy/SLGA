# 🎯 GUIDE PRATIQUE - Validation Training & Métriques

**Date**: 2025-10-28
**Objectif**: Savoir EXACTEMENT quoi vérifier et quand c'est "bon"

---

## 📋 PARTIE 1: CHANGEMENTS CONFIG RECOMMANDÉS

### ✅ CE QUI EST DÉJÀ BON

```yaml
# config/config.wikipedia.yaml

model:
  embed_dim: 512              # ✅ OK
  num_heads: 8                # ✅ OK (512/8 = 64 head_dim)
  n_layers: 12                # ✅ OK (38M params)
  local_window: 128           # ✅ OK
  global_k: 24                # ✅ OK (48 landmarks total)
  learned_landmarks: true     # ✅ OK (essentiel!)

train:
  batch_size: 8               # ✅ OK pour RTX 3090
  accum_steps: 4              # ✅ OK (effective batch = 32)
  lr: 2.0e-4                  # ✅ OK (dans la plage)
  warmup_steps: 2000          # ✅ OK (2% du total)
  max_steps: 100000           # ✅ OK

  # FIXES APPLIQUÉS
  lambda_spacing: 50.0        # ✅ OK (signal fort)
  lambda_sparsity: 5.0        # ✅ OK (signal fort)
```

### ⚠️ CHANGEMENTS OPTIONNELS (SI PROBLÈMES)

Si le training **diverge** ou si les **métriques sont mauvaises** après 1000 steps:

```yaml
train:
  # Option A: LR plus conservateur (recommandation autre agent)
  lr: 1.5e-4                  # Au lieu de 2.0e-4 (réduire 25%)
  warmup_steps: 5000          # Au lieu de 2000 (augmenter)

  # Option B: Lambdas réduits (si loss auxiliaires dominent)
  lambda_spacing: 5.0         # Au lieu de 50.0 (réduire 10x)
  lambda_sparsity: 0.5        # Au lieu de 5.0 (réduire 10x)

  # Option C: Sequence length plus conservateur (si OOM)
  seq_len_start: 256          # Au lieu de 384
  seq_len_final: 1536         # Au lieu de 2048
  grad_checkpointing: true    # Activer si OOM
```

### 🎯 MA RECOMMANDATION

**COMMENCE AVEC CONFIG ACTUELLE** (pas de changements)

**Pourquoi**:
- LR=2e-4 est standard pour batch=32
- Warmup=2000 (2%) est suffisant
- Lambda=50/5 donne signal fort (c'est voulu après le fix!)
- Ton training précédent au step 18,020 montrait Loss=4.0, stable

**Change SEULEMENT SI** validation 1000 steps échoue (voir métriques ci-dessous).

---

## 📊 PARTIE 2: MÉTRIQUES À VÉRIFIER

### 🎯 CRITÈRES DE SUCCÈS (après 1000 steps)

Voici **exactement** ce que tu dois vérifier:

---

### ✅ MÉTRIQUE #1: Loss Auxiliaires (LOGS CONSOLE)

**Où regarder**: Logs console pendant training

**Chercher lignes comme**:
```
📊 Step 1000 | Loss: X.XX (best: Y.YY) | PPL: ZZ.Z
   ...
   Landmarks: 48 │ Spacing: X.XX │ Sparsity: Y.YY
```

#### Spacing Loss
```
✅ BON: 0.5 ≤ Spacing ≤ 2.0
⚠️ LIMITE: 0.1 ≤ Spacing < 0.5  → Lambda peut-être trop faible
❌ MAUVAIS: Spacing < 0.1      → Gradients encore bloqués
❌ MAUVAIS: Spacing > 2.0      → Lambda trop élevé, domine
```

**Action si hors limites**:
- Si < 0.5: Augmenter `lambda_spacing` à 100.0
- Si > 2.0: Réduire `lambda_spacing` à 25.0

#### Sparsity Loss
```
✅ BON: 0.05 ≤ Sparsity ≤ 0.2
⚠️ LIMITE: 0.01 ≤ Sparsity < 0.05  → Lambda faible
❌ MAUVAIS: Sparsity < 0.01         → Gradients bloqués
❌ MAUVAIS: Sparsity > 0.2          → Lambda trop élevé
```

**Action si hors limites**:
- Si < 0.05: Augmenter `lambda_sparsity` à 10.0
- Si > 0.2: Réduire `lambda_sparsity` à 2.5

---

### ✅ MÉTRIQUE #2: Landmarks Actifs (LOGS CONSOLE)

**Où regarder**: Ligne de progression

```
Step 1000 | ... | LM: XX→YY | ...
```

**Critères**:
```
✅ BON: LM: 48→48        → Tous landmarks utilisés
⚠️ LIMITE: LM: 48→30-47  → Certains filtrés (acceptable)
❌ MAUVAIS: LM: 48→0     → Bug comptage ou landmarks invalides
```

**Action si mauvais**:
- Si 48→0: Vérifier fix ligne 723 de train.py appliqué
- Si 48→<20: Problème avec landmark selection

---

### ✅ MÉTRIQUE #3: Loss Principale (LOGS + TENSORBOARD)

**Où regarder**: Logs console ET TensorBoard

**Console**:
```
Step 1000 | Loss: X.XX | PPL: YY.Y
```

**TensorBoard**:
- Graph `train/loss`
- Graph `train/perplexity`

**Critères après 1000 steps**:
```
✅ BON: Loss entre 6.0-8.5, descend smoothly
   PPL entre 400-5000

⚠️ LIMITE: Loss entre 8.5-10.0
   Descend mais lentement

❌ MAUVAIS: Loss > 10.0 ou augmente
   NaN/Inf détecté
   Oscillations >0.5
```

**Courbe attendue** (loss vs step):
```
Step    0: 10.9 (random)
Step  100: 10.0-10.5
Step  500: 8.0-9.0
Step 1000: 6.0-8.5  ← TARGET
```

**Action si mauvais**:
- Si Loss > 10 au step 1000: Réduire LR à 1.5e-4
- Si NaN: Le code arrêtera automatiquement (fix #15)
- Si oscillations: Réduire lambdas de moitié

---

### ✅ MÉTRIQUE #4: Scorer Apprend (TENSORBOARD REQUIS)

**Où regarder**: TensorBoard (http://localhost:6006)

**Métriques importantes**:

#### 4a. Scorer Std (écart-type des scores)
```
Section: SCALARS
Chercher: landmarks/scorer_std ou similaire

✅ BON: Augmente de 0.001 → 0.01+ au fil des 1000 steps
   Courbe ascendante = scorer apprend!

❌ MAUVAIS: Reste < 0.001 (plat)
   Scorer n'apprend toujours pas
```

**Si pas de métrique scorer_std dans TensorBoard**:
- C'est OK, vérifie les autres métriques
- Le fix des gradients suffit

#### 4b. Spacing Mean (espacement moyen)
```
Chercher: landmarks/spacing_mean

✅ BON: Augmente progressivement
   Step 100: ~5-8
   Step 500: ~10-15
   Step 1000: ~15-20

❌ MAUVAIS: Reste < 5 ou ne bouge pas
```

---

### ✅ MÉTRIQUE #5: Pas de Problèmes Techniques

**Dans les logs console, vérifier ABSENCE de**:

```
❌ "RuntimeError"           → Crash
❌ "CUDA out of memory"     → OOM
❌ "IndexError"             → Gather hors limites
❌ "NaN/Inf detected"       → Divergence
❌ Loss = 0.000             → Bug calcul
❌ Grad norm = 0.000        → Pas d'apprentissage
```

**Si tu vois un de ces messages**:
- Training arrêtera automatiquement (nos fixes)
- Checkpoint debug sera sauvegardé
- Consulte les docs correspondantes dans `/docs/`

---

### ✅ MÉTRIQUE #6: Gradient Norm (LOGS CONSOLE)

**Où regarder**: Logs détaillés tous les 10 steps

```
Step 1000 | ... | Grad: X.XX | ...
```

**Critères**:
```
✅ BON: 1.0 ≤ Grad ≤ 5.0
   Stable, pas d'oscillations sauvages

⚠️ LIMITE: 5.0 < Grad < 10.0
   Peut-être LR un peu haut

❌ MAUVAIS: Grad > 10.0
   Gradients explosent → réduire LR

❌ MAUVAIS: Grad < 0.5
   Gradients vanish → augmenter LR
```

---

### ✅ MÉTRIQUE #7: Throughput & GPU (LOGS CONSOLE)

**Où regarder**: Ligne de progression

```
Step 1000 | ... | Throughput: X,XXX tok/s | GPU: X.X/25.8 GB (Y%)
```

**Critères**:
```
✅ BON: Throughput 5,000-8,000 tok/s
   GPU utilisation 60-85%

⚠️ ACCEPTABLE: Throughput 3,000-5,000 tok/s
   GPU 40-60% (sous-utilisé mais OK)

❌ PROBLÈME: GPU > 90%
   Risque OOM, réduire batch ou seq_len
```

---

## 🎯 CHECKLIST RAPIDE (1000 steps)

Copie-colle cette checklist et coche au fur et à mesure:

```
APRÈS 1000 STEPS:

□ Spacing loss: 0.5-2.0 (console logs)
□ Sparsity loss: 0.05-0.2 (console logs)
□ Loss principale: 6.0-8.5 (console + TensorBoard)
□ LM: 48→48 (console, pas 48→0)
□ Grad norm: 1.0-5.0 (console)
□ Aucun NaN/Inf détecté (console)
□ Aucun crash/erreur (console)
□ GPU < 90% (console)

TensorBoard (optionnel mais recommandé):
□ train/loss descend smoothly (courbe)
□ landmarks/scorer_std augmente (courbe)
□ train/loss_spacing descend (courbe)

SI TOUS COCHÉS → ✅ SUCCÈS, continuer training!
SI 1-2 MANQUENT → ⚠️ Investiguer mais possiblement OK
SI 3+ MANQUENT → ❌ Problème, consulter docs
```

---

## 📊 EXEMPLE CONCRET

Voici à quoi devraient ressembler tes logs au step 1000:

### ✅ EXEMPLE BON
```
Step   1000/100000 [███░░░░░░░░░░░░░░░░░░░░░░░░░░]   1.0%
│ ⏱ 00:08:30 │ ETA 14:10:00 │ Loss  7.234 │ PPL  1386.2
│ LR 5.0e-05 │ SeqLen  450 │ GW 0.00 │ LM    48→48
│ 6,543 tok/s │ GPU  4.2/25.8 GB (75%)

────────────────────────────────────────────────────
📊 Step 1000 │ Loss: 7.2340 (best: 6.9123) │ PPL: 1386.23
   LR: 5.000000e-05 │ Grad: 2.3456 │ SeqLen: 450 │ GW: 0.00
   Throughput: 6,543 tok/s │ GPU: 4.2/25.8 GB (75%)
   Landmarks: 48 │ Spacing: 0.8234 │ Sparsity: 0.0723
────────────────────────────────────────────────────
```

**Vérification**:
- ✅ Loss 7.23 → entre 6-8.5 ✓
- ✅ Spacing 0.82 → entre 0.5-2.0 ✓
- ✅ Sparsity 0.07 → entre 0.05-0.2 ✓
- ✅ LM 48→48 ✓
- ✅ Grad 2.35 → entre 1-5 ✓
- ✅ GPU 75% → < 90% ✓

**Conclusion**: ✅ **TOUT EST BON, CONTINUER!**

---

### ❌ EXEMPLE MAUVAIS

```
Step   1000/100000 [███░░░░░░░░░░░░░░░░░░░░░░░░░░]   1.0%
│ Loss 10.567 │ PPL 38926.4 │ LR 2.0e-04
│ LM 48→0 │ Grad: 0.234

────────────────────────────────────────────────────
📊 Step 1000 │ Loss: 10.5670 (best: 10.5234)
   Landmarks: 48 │ Spacing: 0.0234 │ Sparsity: 0.0000
────────────────────────────────────────────────────
```

**Problèmes détectés**:
- ❌ Loss 10.57 → trop haut (devrait être 6-8.5)
- ❌ Spacing 0.023 → trop faible (devrait être 0.5+)
- ❌ Sparsity 0.000 → pas de gradients!
- ❌ LM 48→0 → bug comptage
- ❌ Grad 0.23 → trop faible

**Actions**:
1. ARRÊTER le training
2. Vérifier que TOUS les fixes sont appliqués
3. Relancer avec lambda × 10
4. Consulter `/docs/` pour diagnostics

---

## 🔍 OÙ TROUVER LES MÉTRIQUES

### 1. **Logs Console** (temps réel)

```bash
python scripts/train.py --config config/config.wikipedia.yaml --max-steps 1000
```

**Regarde directement**:
- Ligne de progression: `LM: XX→YY`
- Ligne détaillée (tous les 10 steps): `Spacing: X.XX │ Sparsity: Y.YY`
- Grad norm: `Grad: X.XX`

### 2. **TensorBoard** (graphiques détaillés)

```bash
# Dans un autre terminal
tensorboard --logdir=out_slga/tensorboard

# Ouvrir navigateur
http://localhost:6006/?darkMode=true#timeseries
```

**Graphiques importants**:

#### Onglet SCALARS:
- `train/loss` - Doit descendre smoothly
- `train/loss_spacing` - Doit être 0.5-2.0
- `train/loss_sparsity` - Doit être 0.05-0.2
- `train/perplexity` - Doit descendre
- `train/learning_rate` - Doit augmenter pendant warmup
- `landmarks/num_selected` - Doit rester ~48
- `landmarks/scorer_std` - **Doit augmenter** (preuve apprentissage!)

#### Onglet HISTOGRAMS (optionnel):
- `gradients/*` - Distribution gradients

---

## 📸 SCREENSHOT GUIDE

### À quoi ressemble TensorBoard "BON"

**Graph train/loss** (courbe descendante smooth):
```
10.0 ┤╮
 9.0 ┤ ╰╮
 8.0 ┤   ╰╮
 7.0 ┤     ╰─╮     ← Step 1000 ici
 6.0 ┤        ╰─
     └──────────────
     0   500  1000
```

**Graph landmarks/scorer_std** (courbe ASCENDANTE):
```
0.01 ┤         ╭─╮   ← Step 1000 ici
     ┤       ╭─╯  ╰
0.005┤     ╭─╯
     ┤  ╭──╯
0.001┤──╯
     └──────────────
     0   500  1000
```

**Graph train/loss_spacing** (commence haut, descend):
```
1.5 ┤╮
1.0 ┤ ╰╮
0.8 ┤   ╰─╮        ← Step 1000 ici
0.5 ┤      ╰───
    └──────────────
    0   500  1000
```

---

## ⏱️ TIMELINE DE VALIDATION

### Pendant les 1000 Steps (~8-12 minutes)

**Step 0-100**:
- ✅ Loss devrait descendre de 10.9 → 10.0
- ✅ Spacing/Sparsity apparaissent (non-zéro)
- ⚠️ Grad norm peut être élevé (5-10) - normal début

**Step 100-500**:
- ✅ Loss descend vers 8.0-9.0
- ✅ Spacing stabilise autour 0.8-1.5
- ✅ Grad norm se stabilise (2-4)

**Step 500-1000**:
- ✅ Loss atteint 6.0-8.5
- ✅ Scorer std commence à augmenter (>0.005)
- ✅ Tout stable

### Point de Décision (Step 1000)

**SI TOUS LES CRITÈRES OK**:
```bash
# CONTINUER! Lance training complet
# Option 1: Continuer depuis step 1000
Ctrl+C (arrêter)
python scripts/train.py --config config/config.wikipedia.yaml

# Option 2 (recommandé): Scratch
rm -rf out_slga
python scripts/train.py --config config/config.wikipedia.yaml
```

**SI 1-2 CRITÈRES LIMITE**:
```bash
# Ajuster config (ex: LR, lambdas)
# Relancer 1000 steps
# Vérifier amélioration
```

**SI 3+ CRITÈRES MAUVAIS**:
```bash
# STOP! Diagnostiquer
# Consulter docs dans /docs/
# Vérifier tous les fixes appliqués
# Demander aide avec logs précis
```

---

## 🎯 COMMANDES PRATIQUES

### Lancer Validation

```bash
# Terminal 1: Training
python scripts/train.py \
    --config config/config.wikipedia.yaml \
    --max-steps 1000 2>&1 | tee validation_1000.log

# Terminal 2: TensorBoard
tensorboard --logdir=out_slga/tensorboard --port 6006

# Terminal 3: Monitoring GPU (optionnel)
watch -n 2 nvidia-smi
```

### Vérifier Métriques Après

```bash
# 1. Grep dans logs pour step 1000
grep "Step.*1000 " validation_1000.log

# 2. Vérifier spacing/sparsity
grep "Spacing:" validation_1000.log | tail -10

# 3. Chercher erreurs
grep -i "error\|nan\|inf" validation_1000.log

# 4. Vérifier best loss
grep "best:" validation_1000.log | tail -1
```

---

## 📋 DECISION MATRIX

Utilise ce tableau pour décider:

| Spacing | Sparsity | Loss@1K | LM | Grad | Décision |
|---------|----------|---------|-----|------|----------|
| 0.5-2.0 | 0.05-0.2 | 6-8.5 | 48→48 | 1-5 | ✅ **PARFAIT! Continue** |
| 0.5-2.0 | 0.05-0.2 | 6-8.5 | 48→30+ | 1-5 | ✅ Acceptable, continue |
| 0.1-0.5 | 0.01-0.05 | 7-9 | 48→48 | 1-5 | ⚠️ Augmente lambdas 2× |
| <0.1 | <0.01 | >9 | 48→<20 | - | ❌ Problème! Vérifie fixes |
| >2.0 | >0.2 | - | - | >10 | ❌ Lambdas trop hauts, réduis |

---

## 🎓 CONCLUSION

### Config Actuelle: ✅ BONNE

**NE CHANGE RIEN** pour l'instant. Lance avec config actuelle:

```yaml
# Garder ces valeurs
lr: 2.0e-4
warmup_steps: 2000
lambda_spacing: 50.0
lambda_sparsity: 5.0
```

### Métriques à Vérifier: **7 Critères Clairs**

1. ✅ Spacing: 0.5-2.0
2. ✅ Sparsity: 0.05-0.2
3. ✅ Loss@1K: 6.0-8.5
4. ✅ LM: 48→48
5. ✅ Grad: 1.0-5.0
6. ✅ Pas d'erreurs
7. ✅ GPU < 90%

### Validation: **Simple**

```bash
# 1. Lance
python scripts/train.py --config config/config.wikipedia.yaml --max-steps 1000

# 2. Attends 8-12 min

# 3. Regarde step 1000 dans logs

# 4. Vérifie checklist ci-dessus

# 5. Si ≥5/7 critères OK → Continue training complet!
```

**C'est aussi simple que ça!** 🎯

Veux-tu que je lance le training validation maintenant pour te montrer ? 🚀
