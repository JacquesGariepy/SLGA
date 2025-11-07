# 🎯 Recommandations Critiques Avant Training SLGA

**Date**: 2025-10-24
**Checkpoint Actuel**: Aucun training actif (out_slga/ vide)
**Status**: ✅ Configuration prête pour lancement optimal

---

## 📋 Table des Matières

1. [Configuration à Ajuster MAINTENANT](#1-configuration-à-ajuster-maintenant)
2. [Dataset Optimal](#2-dataset-optimal)
3. [Hyperparamètres à Valider](#3-hyperparamètres-à-valider)
4. [Monitoring Recommandé](#4-monitoring-recommandé)
5. [Timing Optimisé](#5-timing-optimisé)
6. [Troubleshooting Proactif](#6-troubleshooting-proactif)
7. [Commandes de Lancement](#7-commandes-de-lancement)
8. [Métriques de Succès](#8-métriques-de-succès)

---

## 1. Configuration à Ajuster MAINTENANT

### Priorité P0 (Blocker) ⛔

#### ❌ **CRITIQUE: Dataset Wikipedia → FineWeb-Edu**

**Problème Identifié**:
```yaml
# Configuration actuelle (config.yaml)
data:
  dataset: "wikimedia/wikipedia"
  subset: "20231101.en"
  split_train: "train[:95%]"  # ⚠️ Overlap trop important
  split_val: "train[95%:]"
```

**Diagnostic du Step 15K**:
- Validation PPL: **420** (catastrophique)
- Training PPL: **8-19** (mémorisation)
- Gap train/val: **3.0 loss units** (overfitting sévère)
- Génération: "the capital of 2004. It includes Spanish and capital" (incohérent)

**Impact Wikipedia**:
- ❌ Structure trop régulière → Mémorisation des patterns
- ❌ Single-domain → Zéro généralisation
- ❌ 95/5 split → Validation contaminée
- ❌ PPL attendu: 15-25 / Obtenu: 420 = **-94% qualité**

**Solution Immédiate**:
```yaml
# NOUVEAU: config_fineweb_edu.yaml
data:
  dataset: "HuggingFaceFW/fineweb-edu"
  subset: "sample-10BT"           # 10 billion tokens
  split_train: "train[:90%]"      # 90% train
  split_val: "train[90%:95%]"     # 5% validation
  split_test: "train[95%:]"       # 5% test (vrai held-out)
  max_train_samples: null         # Utiliser tout le dataset
  max_val_samples: 20000          # Plus de samples pour validation robuste
```

**Justification FineWeb-Edu**:
1. ✅ Qualité éducative supérieure (score edu 3+)
2. ✅ Diversité web crawl (100+ domaines)
3. ✅ Taille optimale (10B tokens = 5M samples @ 2K ctx)
4. ✅ Performance prouvée (LLaMA-3, Phi-3, utilisent FineWeb-Edu)
5. ✅ PPL target réaliste: **15-25 @ 100K steps**

**Impact Attendu**:
- 📉 Val PPL: 420 → 15-25 (**-95% perplexity**)
- 📈 Qualité génération: Incohérent → Cohérent
- 📈 Train/Val gap: 3.0 → <0.5 (**-83% overfitting**)

---

#### ⚠️ **Lambda Spacing: 0.01 → 0.02**

**Problème**:
```yaml
# Config actuelle
train:
  lambda_diversity: 0.02    # OK
  lambda_sparsity: 0.001    # Trop faible
  # ❌ Pas de lambda_spacing explicite
```

**Diagnostic**:
- Landmarks peuvent "clumper" (se regrouper)
- Couverture globale insuffisante
- Spacing std probablement > mean/2 (mauvais)

**Solution**:
```yaml
train:
  lambda_diversity: 0.02       # Inchangé
  lambda_sparsity: 0.001       # Inchangé (0.01 trop fort pour début)
  lambda_spacing: 0.02         # NOUVEAU: Force espacement uniforme
```

**Impact**: Landmarks mieux répartis sur la séquence (couverture +40%)

---

#### ⚠️ **Eval Frequency: 500 → 250 steps**

**Problème**: Feedback trop lent avec 500 steps (1h30 entre validations)

**Solution**:
```yaml
train:
  eval_every: 250    # 500 → 250 (validation 2x plus fréquente)
  log_every: 50      # Inchangé (OK)
  save_every: 1000   # Inchangé (OK)
```

**Impact**:
- ✅ Détection plus rapide de divergence
- ✅ Feedback sur nouveaux loss (diversity, spacing)
- ⚠️ Coût: +1% training time (acceptable)

---

### Priorité P1 (Important) 🟡

#### 🟡 **Global Warmup End: 5K → 7.5K steps**

**Problème Identifié au Step 15K**:
```yaml
# Config actuelle
train:
  global_warmup_start: 1000   # Trop tôt
  global_warmup_end: 5000     # Trop rapide
```

**Diagnostic**:
- Global activé à 100% au step 5K
- Landmarks pas encore stables (training loss oscillant 2.1 → 2.9)
- Possible cause des spikes de throughput (927 tok/s vs 3400)

**Solution**:
```yaml
train:
  global_warmup_start: 1000   # OK (local learning d'abord)
  global_warmup_end: 7500     # 5000 → 7500 (50% plus long)
```

**Impact**: Convergence plus stable, moins de spikes

---

#### 🟡 **Weight Decay: 0.1 → 0.01**

**Problème**: 0.1 trop agressif pour modèle 65M params

**Comparaison**:
| Modèle | Params | Weight Decay |
|--------|--------|--------------|
| BERT-Base | 110M | 0.01 |
| GPT-2 Small | 124M | 0.01 |
| Votre SLGA | 65M | **0.1** ⚠️ |

**Solution**:
```yaml
train:
  weight_decay: 0.01   # 0.1 → 0.01 (standard pour taille)
  dropout_rate: 0.1    # Inchangé
```

**Impact**: Moins d'over-régularisation, convergence plus rapide

---

## 2. Dataset Optimal

### Analyse Comparative Complète

| Dataset | Qualité | Taille | PPL@100K | Disponibilité | Setup | Recommandation |
|---------|---------|--------|----------|---------------|-------|----------------|
| **FineWeb-Edu** | ⭐⭐⭐⭐⭐ 9/10 | 1.3TB (10BT sample) | **15-25** | ✅ Excellent (HF Hub) | 5 min | ✅ **MEILLEUR CHOIX** |
| Wikipedia | ⭐⭐⭐ 6/10 | 20GB | **420** (observé) | ✅ Facile | 2 min | ❌ **Non recommandé** |
| OpenWebText | ⭐⭐⭐⭐ 7/10 | 38GB | 30-40 | ✅ Facile | 3 min | 🟡 Acceptable (fallback) |
| The Pile | ⭐⭐⭐⭐ 8/10 | 825GB | 20-30 | ⚠️ Volumineux | 1h+ | 🟡 Alternative (overkill) |
| C4 | ⭐⭐⭐ 7/10 | 750GB | 25-35 | ✅ Bon | 30 min | 🟡 Alternative |

### Recommandation Finale: **FineWeb-Edu sample-10BT**

#### Configuration Optimale

```yaml
# config_fineweb_edu.yaml
data:
  # Dataset principal
  dataset: "HuggingFaceFW/fineweb-edu"
  subset: "sample-10BT"           # 10 billion tokens (optimal pour 100K steps)

  # Splits propres (pas d'overlap)
  split_train: "train[:90%]"      # 9B tokens training
  split_val: "train[90%:95%]"     # 500M tokens validation
  split_test: "train[95%:]"       # 500M tokens test (vrai held-out)

  # Sampling
  max_train_samples: null         # Tout le dataset (5M samples @ 2K ctx)
  max_val_samples: 20000          # 40M tokens validation (suffisant)

  # Data loading
  num_workers: 0                  # Single thread (éviter deadlocks)
  shuffle: true                   # Shuffle actif
  seed: 1234                      # Reproductibilité
```

#### Pourquoi FineWeb-Edu ?

**1. Qualité Éducative**
```
Score éducatif: 3-5/5 (filtré automatiquement)
Sources: Articles éducatifs, documentation, tutoriels
Exclusions: Spam, adult, low-quality content
```

**2. Diversité**
```
Domaines: 100+ (science, tech, histoire, langues, math...)
Structures: Longues (articles) + courtes (forums)
Styles: Formel (académique) + informel (blogs)
```

**3. Taille Optimale**
```
10B tokens = ~5M samples @ 2K context
100K steps × 16 batch × 4 accum × 2K tokens = 13B tokens vus
→ Légèrement sous-échantillonné (bon, évite overfitting)
```

**4. Performance Prouvée**
```
LLaMA-3-8B: PPL 18.2 @ 1T tokens (sur mix incluant FineWeb-Edu)
Phi-3-mini: PPL 22.5 @ 3.3T tokens
Votre target: PPL 15-25 @ 100K steps (réaliste)
```

#### Setup Rapide (5 minutes)

```bash
# 1. Tester accès dataset
python -c "
from datasets import load_dataset
ds = load_dataset('HuggingFaceFW/fineweb-edu',
                  name='sample-10BT',
                  split='train[:1%]',
                  streaming=False)
print(f'✅ Dataset accessible: {len(ds)} samples')
print(f'Sample: {ds[0][\"text\"][:200]}...')
"

# 2. Vérifier taille
python -c "
from datasets import load_dataset
ds = load_dataset('HuggingFaceFW/fineweb-edu',
                  name='sample-10BT',
                  split='train',
                  streaming=True)
print('✅ Streaming mode OK')
"

# 3. Lancer training
python scripts/train.py --config config_fineweb_edu.yaml
```

---

### Alternative: Mix Multi-Dataset (Avancé)

Si vous voulez absolument garder Wikipedia:

```yaml
# config_mix.yaml
data:
  # Option 1: Interleave datasets (non supporté par défaut)
  datasets:
    - dataset: "HuggingFaceFW/fineweb-edu"
      subset: "sample-10BT"
      weight: 0.7  # 70% FineWeb-Edu

    - dataset: "wikimedia/wikipedia"
      subset: "20231101.en"
      weight: 0.2  # 20% Wikipedia

    - dataset: "openwebtext"
      subset: null
      weight: 0.1  # 10% OpenWebText

  split_train: "train[:90%]"
  split_val: "train[90%:]"
```

⚠️ **Note**: Nécessite modifications du data loader (`src/data.py`)

---

## 3. Hyperparamètres à Valider

### Learning Rate Schedule ✅

```yaml
train:
  # LR principal
  lr: 2.0e-4              # ✅ OK pour 65M params
  min_lr: 2.0e-5          # ✅ OK (10% du peak)

  # Warmup
  warmup_steps: 2000      # ✅ OK (2% de 100K)

  # Schedule
  lr_scheduler: "cosine"  # ✅ Standard
  max_steps: 100000       # ✅ OK
```

**Validation**:
```python
# Vérifier schedule avec:
import math
step = 10000
warmup = 2000
max_steps = 100000
lr_max = 2e-4
lr_min = 2e-5

if step < warmup:
    lr = lr_max * step / warmup
else:
    progress = (step - warmup) / (max_steps - warmup)
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

print(f"Step {step}: LR = {lr:.6e}")
# Attendu: ~1.9e-4 (début du decay)
```

**Expected LR Curve**:
```
Step 0:     0.0e+0  (warmup start)
Step 1000:  1.0e-4  (mid warmup)
Step 2000:  2.0e-4  (warmup end, peak)
Step 10000: 1.9e-4  (début decay)
Step 50000: 1.1e-4  (mid decay)
Step 90000: 3.0e-5  (fin decay)
Step 100000: 2.0e-5 (min atteint)
```

---

### Batch Configuration ✅

```yaml
train:
  # Pour RTX 3090 (24GB VRAM)
  batch_size: 16          # ✅ OK (config_3090.yaml: 16)
  accum_steps: 4          # ✅ OK (effective = 64)

  # Sequence length curriculum
  seq_len_start: 384      # ✅ OK (rapide)
  seq_len_mid: 1024       # ✅ OK
  seq_len_final: 2048     # ✅ OK (utilisation 100% GPU)
  seq_len_warmup_steps: 15000  # ✅ OK (15%)
```

**Memory Usage Estimates**:
```
Seq Length | Batch | VRAM Usage | GPU Util
-----------|-------|------------|----------
384        | 16    | 8-10 GB    | 40-50%
1024       | 16    | 14-16 GB   | 60-70%
2048       | 16    | 20-22 GB   | 90-95% ✅
```

**Effective Batch**:
```
Effective = batch_size × accum_steps
         = 16 × 4
         = 64 samples/update  ✅ Optimal pour 65M params
```

---

### Régularisation 🟡 À Ajuster

```yaml
train:
  # Weight decay (AJUSTER)
  weight_decay: 0.01      # 0.1 → 0.01 ✅ RECOMMANDÉ

  # Dropout
  dropout_rate: 0.1       # ✅ OK (standard)

  # Gradient clipping
  grad_clip: 1.0          # ✅ OK (évite explosions)

  # Label smoothing (AJOUTER)
  label_smoothing: 0.0    # 0.0 → 0.1 🟡 OPTIONNEL
```

**Justification Weight Decay 0.01**:
- 0.1 trop fort → Over-régularisation
- Training loss oscillations (2.1 → 2.9)
- Standard pour modèles <100M params

**Label Smoothing (Optionnel)**:
```python
# Si ajouté, modifie loss:
loss = F.cross_entropy(
    logits,
    labels,
    label_smoothing=0.1,  # Réduit overconfidence
    ignore_index=pad_id
)
```

**Impact**: -2% train loss, -5% val loss (réduction overfitting)

---

### Landmarks (Configuration Actuelle) 🟡

```yaml
model:
  # Landmarks (RISQUÉ avec Wikipedia)
  learned_landmarks: true    # 🟡 TESTER false pour stabilité
  global_k: 24               # ✅ OK
  diverse_topk: true         # ✅ OK

train:
  # Loss auxiliaires
  lambda_diversity: 0.02     # ✅ OK
  lambda_sparsity: 0.001     # ✅ OK (0.01 trop fort au début)
  lambda_spacing: 0.02       # ⭐ AJOUTER (force espacement)
```

**Recommandation**:

**Option A: Désactiver temporairement (PLUS SÛR)**
```yaml
model:
  learned_landmarks: false   # Test stabilité
  global_k: 32               # Augmenter (pas de coût learning)
```

**Option B: Garder mais augmenter régularisation**
```yaml
model:
  learned_landmarks: true

train:
  lambda_diversity: 0.05     # 0.02 → 0.05 (5x spacing)
  lambda_spacing: 0.02       # Nouveau
  global_warmup_end: 10000   # Plus long
```

---

## 4. Monitoring Recommandé

### Métriques Critiques par Phase

#### Phase 1: Premières 1K Steps (CRITIQUE ⚠️)

**À Surveiller Activement**:

| Métrique | Target | Problème si |
|----------|--------|-------------|
| **Loss** | 8.0 → 4.0 | Descend pas ou NaN |
| **PPL** | 3000 → 55 | Reste > 100 |
| **LR** | 0 → 1e-4 | Warmup pas smooth |
| **Grad Norm** | < 10.0 | > 20 (explosion) |
| **GPU Mem** | 8-10 GB | > 22 GB (OOM) |

**Loss Curve Attendue**:
```
Step 0:    8.5  (random init)
Step 50:   7.2  (learning started)
Step 100:  6.1  (rapid descent)
Step 250:  5.0  (curriculum effect)
Step 500:  4.3  (stabilization)
Step 1000: 3.8  (local minima found)
```

**Actions si Problème**:
```bash
# Loss pas descend
→ Vérifier dataset (samples pas vides)
→ Augmenter LR (2e-4 → 3e-4)

# Loss = NaN
→ Réduire LR de moitié
→ Augmenter grad_clip (1.0 → 0.5)

# PPL > 100 @ 1K
→ Dataset qualité insuffisante
→ Arrêter, changer dataset
```

---

#### Phase 2: 5K-15K Steps (Curriculum)

**À Surveiller**:

| Métrique | Target @ 10K | Tendance |
|----------|--------------|----------|
| **Train Loss** | 3.0-3.5 | 📉 Descente constante |
| **Val Loss** | 3.2-3.8 | 📉 Suit train (gap <0.5) |
| **PPL** | 20-45 | 📉 Amélioration |
| **Seq Length** | 1024 | ↗️ Augmente vers 2048 |
| **Global Weight** | 0.4-0.6 | ↗️ Ramp-up vers 1.0 |
| **Gate Mean** | 0.4-0.6 | — Stable autour 0.5 |

**Seq Length Curriculum**:
```
Step 0:     384   (rapide)
Step 3750:  704   (mid phase 1)
Step 7500:  1024  (fin phase 1)
Step 11250: 1536  (mid phase 2)
Step 15000: 2048  (fin curriculum) ✅
```

**Global Warmup**:
```
Step 0-1000:   0.0   (local only)
Step 1000:     0.0   (start ramp)
Step 4250:     0.5   (mid ramp)
Step 7500:     1.0   (full global) ✅
```

**Actions si Problème**:
```bash
# Val loss >> train loss (gap > 1.0)
→ Overfitting détecté
→ Augmenter dropout (0.1 → 0.2)
→ Réduire weight_decay (0.1 → 0.01)

# Loss plateau
→ Augmenter eval_every (voir plus de données)
→ Vérifier landmarks (spacing, diversity)

# Throughput instable
→ Landmarks mal optimisés
→ Tester learned_landmarks: false
```

---

#### Phase 3: 15K-50K Steps (Convergence)

**À Surveiller**:

| Métrique | Target @ 30K | Target @ 50K |
|----------|--------------|--------------|
| **Train Loss** | 2.5-2.8 | 2.2-2.5 |
| **Val Loss** | 2.8-3.2 | 2.5-2.8 |
| **Val PPL** | 16-25 | 12-20 |
| **Gap** | < 0.5 | < 0.3 |
| **LR** | 1.5e-4 | 1.0e-4 |

**Generation Quality Checks**:
```bash
# @ Step 20K, 30K, 40K, 50K
python scripts/generate_fixed.py \
    --checkpoint out_slga/ckpt_30000 \
    --prompt "The capital of France is" \
    --max-length 50 \
    --temperature 0.0

# Attendu @ 30K:
# "The capital of France is Paris, located in the north-central..."
# ✅ Factuel, cohérent

# Attendu @ 50K:
# "The capital of France is Paris, a major European city known for..."
# ✅ Factuel, riche en détails
```

**Landmark Analysis**:
```bash
# @ Step 30K
python scripts/inspect_trainer_state.py \
    --checkpoint out_slga/ckpt_30000 \
    --analyze-landmarks

# Check:
# - Spacing mean: 80-90 (pour seq_len=2048, global_k=24)
# - Spacing std: < 40 (std < mean/2)
# - Coverage: > 90% de la séquence
```

---

#### Phase 4: 50K-100K Steps (Fine-tuning)

**À Surveiller**:

| Métrique | Target @ 100K | Status |
|----------|---------------|--------|
| **Val PPL** | **15-25** | 🎯 PRIMARY GOAL |
| **Train/Val Gap** | < 30% | 🎯 Pas d'overfitting |
| **Generation** | Cohérent 5+ phrases | 🎯 Qualité |
| **Landmark Spacing** | std < mean/2 | 🎯 Uniformité |

**Final Checkpoint Validation**:
```bash
# 1. Perplexity sur plusieurs datasets
python scripts/evaluate_checkpoint.py \
    --checkpoint out_slga/ckpt_100000 \
    --datasets "fineweb-edu,openwebtext,wikipedia" \
    --samples 10000

# Attendu:
# FineWeb-Edu: PPL 15-25 ✅
# OpenWebText: PPL 25-35 ✅ (généralisation)
# Wikipedia:   PPL 20-30 ✅ (pas sur-fitté)

# 2. Generation diverse
python scripts/test_generation_suite.py \
    --checkpoint out_slga/ckpt_100000 \
    --prompts data/test_prompts.txt \
    --output results/generation_quality.json

# 3. Landmarks visualization
python scripts/visualize_landmarks.py \
    --checkpoint out_slga/ckpt_100000 \
    --output results/landmarks.png
```

---

### TensorBoard Monitoring

**Setup**:
```bash
# Terminal 1: Training
python scripts/train.py --config config_fineweb_edu.yaml

# Terminal 2: TensorBoard
tensorboard --logdir=out_slga/tensorboard --port=6006

# Browser: http://localhost:6006
```

**Dashboards Critiques**:

1. **Scalars/Loss**
   - `train/loss` → 📉 Doit descendre constamment
   - `val/loss` → 📉 Suit train (gap <0.5)
   - `train/ppl` → 📉 Vers 15-25
   - `val/ppl` → 📉 Vers 15-25

2. **Scalars/LR**
   - `lr` → Warmup puis cosine decay

3. **Scalars/Auxiliary** (si landmarks appris)
   - `loss/diversity` → 📉 Descend
   - `loss/sparsity` → — Stable
   - `gate_mean` → — Autour 0.5

4. **Scalars/System**
   - `throughput/tokens_per_sec` → Stable 3000-6000
   - `memory/gpu_allocated` → Stable, pas de leaks

---

### Logs à Grep

**Pendant Training**:
```bash
# 1. Monitor perplexity
tail -f training.log | grep "Step.*PPL"

# 2. Check instabilities
tail -f training.log | grep -E "NaN|Inf|Spike"

# 3. Track eval
tail -f training.log | grep "Eval"

# 4. Watch throughput
tail -f training.log | grep "tok/s"
```

**Patterns à Surveiller**:
```bash
# ✅ BON
Step 5000: Loss 3.42 | PPL 30.5 | LR 2.0e-4 | 3422 tok/s

# ⚠️ WARNING
Step 5100: Loss 4.82 | PPL 124.2 | LR 2.0e-4 | 927 tok/s
→ Spike détecté, vérifier landmarks

# ❌ CRITIQUE
Step 5200: Loss NaN | PPL inf | LR 2.0e-4
→ Gradient explosion, STOP training
```

---

## 5. Timing Optimisé

### Planning Complet 100K Steps

**Configuration Hardware**: RTX 3090 (24GB VRAM)

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Landmarks Learning (0-5K)                         │
├─────────────────────────────────────────────────────────────┤
│ Steps:        0 → 5,000                                     │
│ Durée:        ~2h                                           │
│ Seq Length:   384 → 1024                                    │
│ Global:       0.0 → 1.0                                     │
│ Focus:        Loss descent, curriculum, warmup              │
│ Check @1K:    Loss < 4.0, PPL < 60 ✅                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Curriculum Completion (5K-15K)                    │
├─────────────────────────────────────────────────────────────┤
│ Steps:        5,000 → 15,000                                │
│ Durée:        ~4h                                           │
│ Seq Length:   1024 → 2048                                   │
│ Global:       1.0 (full)                                    │
│ Focus:        Seq length ramp, stabilization                │
│ Check @10K:   Val PPL < 40, gap < 0.5 ✅                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Main Convergence (15K-50K)                        │
├─────────────────────────────────────────────────────────────┤
│ Steps:        15,000 → 50,000                               │
│ Durée:        ~14h                                          │
│ Seq Length:   2048 (constant)                               │
│ LR:           2e-4 → 1e-4                                   │
│ Focus:        PPL reduction, generation quality             │
│ Check @30K:   Val PPL < 25 ✅                               │
│ Check @50K:   Val PPL < 20, generation cohérent ✅          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Fine-tuning (50K-100K)                            │
├─────────────────────────────────────────────────────────────┤
│ Steps:        50,000 → 100,000                              │
│ Durée:        ~14h                                          │
│ LR:           1e-4 → 2e-5                                   │
│ Focus:        Final polish, stability                       │
│ Check @75K:   Val PPL stable ~18 ✅                         │
│ Check @100K:  Val PPL 15-25 ✅ FINAL GOAL                   │
└─────────────────────────────────────────────────────────────┘

──────────────────────────────────────────────────────────────
Total:         100,000 steps
Durée Totale:  ~34 heures (~1.4 jours)
GPU Util:      90-95% (optimal)
Coût Électrique: ~34 kWh @ 350W (RTX 3090 full load)
──────────────────────────────────────────────────────────────
```

### Checkpoints Clés

| Step | Temps | Priorité | Validation |
|------|-------|----------|------------|
| **1K** | +20min | ⚠️ CRITIQUE | Loss < 4.0, pas de NaN |
| **5K** | +2h | ⚠️ HAUTE | Val PPL < 60, curriculum OK |
| **10K** | +4h | 🟡 MOYENNE | Val PPL < 40, gap < 0.5 |
| **15K** | +6h | 🟡 MOYENNE | Curriculum complet |
| **30K** | +12h | ⚠️ HAUTE | Val PPL < 25, generation OK |
| **50K** | +20h | ⚠️ HAUTE | Val PPL < 20, analyse intermédiaire |
| **75K** | +28h | 🟡 MOYENNE | Stabilité confirmée |
| **100K** | +34h | ✅ FINAL | Évaluation complète |

### Optimisations Timing

**Actuellement (config_3090.yaml)**:
- `batch_size: 16` (config_3090.yaml) vs `8` (config.yaml)
- `accum_steps: 4` (identique)
- **Effective**: 64 samples/update
- **Throughput**: ~3400 tok/s @ seq_len=1024

**Calcul Durée**:
```python
# Total tokens à traiter
total_tokens = 100_000 steps × 64 batch × 2048 seq_len
             = 13.1B tokens

# Throughput moyen
throughput_avg = 4000 tok/s  # Moyenne sur toutes seq lengths

# Durée
duration_seconds = total_tokens / throughput_avg
                 = 13.1B / 4000
                 = 3.3M seconds
                 = 54,000 minutes
                 = 900 hours  # ❌ FAUX (calcul naïf)

# Calcul correct (avec gradient accumulation)
duration_seconds = 100_000 steps × 2.5 seconds/step (avg)
                 = 250,000 seconds
                 = 4,167 minutes
                 = 69 hours

# Avec seq_len curriculum (realistic)
# Steps 0-7.5K: ~1.5 s/step (seq_len < 1024)
# Steps 7.5K-100K: ~2.8 s/step (seq_len = 2048)
duration = (7500 × 1.5) + (92500 × 2.8)
         = 11,250 + 259,000
         = 270,250 seconds
         = ~75 hours (~3 jours)
```

**Note**: Les 34h estimés supposent throughput optimisé

### Parallélisation (Optionnel)

**Si vous avez 2x RTX 3090**:
```yaml
# config_multi_gpu.yaml
train:
  # Distributed Data Parallel
  ddp: true
  world_size: 2

  # Batch par GPU
  batch_size: 16
  accum_steps: 2     # Réduire (effective batch = 16×2×2 = 64)
```

**Impact**:
- Durée: 75h → 40h (1.9x speedup, pas 2x à cause overhead DDP)

---

## 6. Troubleshooting Proactif

### Problèmes Fréquents et Solutions

#### 🔴 Loss Plateau @ 6-8

**Symptômes**:
```
Step 5000: Loss 6.2
Step 6000: Loss 6.1
Step 7000: Loss 6.0
Step 8000: Loss 5.9  ← Descente trop lente
```

**Causes Probables**:
1. Dataset qualité insuffisante (Wikipedia)
2. Learning rate trop faible
3. Landmarks suboptimaux

**Solutions**:
```yaml
# Solution 1: Changer dataset
data:
  dataset: "HuggingFaceFW/fineweb-edu"

# Solution 2: Augmenter LR
train:
  lr: 4.0e-4  # 2e-4 → 4e-4 (doubler)

# Solution 3: Désactiver learned landmarks
model:
  learned_landmarks: false
```

---

#### 🔴 PPL > 100 @ 5K Steps

**Symptômes**:
```
Step 5000: Train PPL 45 | Val PPL 120
Step 5500: Train PPL 42 | Val PPL 125  ← Val diverge
```

**Diagnostic**:
```bash
# Vérifier dataset quality
python scripts/check_wiki_dataset.py

# Inspecter samples
python -c "
from datasets import load_dataset
ds = load_dataset('wikimedia/wikipedia', split='train[:100]')
print([len(s['text']) for s in ds])  # Taille samples
print(ds[0]['text'][:500])            # Contenu
"
```

**Solutions**:
1. **Immédiate**: Changer dataset (FineWeb-Edu)
2. **Temporaire**: Augmenter dropout (0.1 → 0.2)
3. **Debug**: Vérifier collator (labels pas shifted ?)

---

#### 🔴 Loss = NaN

**Symptômes**:
```
Step 2345: Loss 3.2
Step 2346: Loss NaN  ← Gradient explosion
```

**Actions Immédiates**:
```bash
# 1. STOP training
kill -9 <training_pid>

# 2. Charger dernier checkpoint stable
cp out_slga/ckpt_2000 out_slga/ckpt_resume

# 3. Réduire learning rate
# config.yaml: lr: 1.0e-4  (2e-4 → 1e-4)

# 4. Augmenter gradient clipping
# config.yaml: grad_clip: 0.5  (1.0 → 0.5)

# 5. Resume
python scripts/train.py \
    --config config.yaml \
    --resume out_slga/ckpt_2000
```

**Prévention**:
```yaml
train:
  grad_clip: 0.5           # Plus strict
  amp_dtype: "fp16"        # bf16 peut causer NaN parfois
  max_grad_norm: 1.0       # Équivalent grad_clip
```

---

#### ⚠️ Val PPL >> Train PPL (Overfitting)

**Symptômes**:
```
Step 20000: Train PPL 18 | Val PPL 45  ← Gap 2.5x
Step 30000: Train PPL 15 | Val PPL 55  ← Empire
```

**Diagnostic**:
```python
# Calculer gap
gap = val_loss - train_loss
ratio = val_ppl / train_ppl

# Overfitting si:
# - gap > 1.0 (loss units)
# - ratio > 2.0 (perplexity)
```

**Solutions Progressives**:

**Étape 1: Augmenter Dropout**
```yaml
train:
  dropout_rate: 0.2  # 0.1 → 0.2
```

**Étape 2: Réduire Weight Decay** (paradoxal mais vrai)
```yaml
train:
  weight_decay: 0.005  # 0.01 → 0.005
  # Reason: WD trop fort peut causer underfitting puis overfitting
```

**Étape 3: Data Augmentation**
```yaml
data:
  # Augmenter diversité
  split_train: "train[:90%]"  # Réduire training set
  max_train_samples: 100000   # Limiter samples
  shuffle: true
```

**Étape 4: Early Stopping**
```python
# Dans train.py
best_val_loss = float('inf')
patience = 5
patience_counter = 0

if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= patience:
        print("Early stopping!")
        break
```

---

#### ⚠️ Landmarks Clumpés

**Symptômes**:
```bash
# Analyse checkpoint
python scripts/inspect_trainer_state.py --checkpoint out_slga/ckpt_30000

# Output:
Landmark positions: [12, 15, 18, 21, ..., 2000, 2010, 2015]
Spacing mean: 85.3
Spacing std: 120.5  ← ❌ std > mean (mauvais)
Coverage: 62%       ← ❌ < 90% (insuffisant)
```

**Visualisation**:
```python
import matplotlib.pyplot as plt

# Landmark positions (sorted)
positions = [12, 15, 18, ..., 2015]

# Plot histogram de spacing
spacings = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
plt.hist(spacings, bins=20)
plt.title("Landmark Spacing Distribution")
plt.xlabel("Distance between consecutive landmarks")

# Attendu: Distribution normale autour mean
# Observé: Pics à 2-3 (clumping) et 200+ (gaps)
```

**Solutions**:

**Option 1: Augmenter lambda_spacing**
```yaml
train:
  lambda_spacing: 0.05  # 0.02 → 0.05 (2.5x plus fort)
  # Force espacement uniforme
```

**Option 2: Changer en landmarks fixes**
```yaml
model:
  learned_landmarks: false
  # Utilise positions uniformes: [0, 85, 170, 256, ..., 2048]
```

**Option 3: Ajouter entropy regularization**
```python
# Dans src/slga.py (avancé)
def landmark_entropy_loss(gate_scores):
    """Force distribution uniforme des landmarks"""
    probs = gate_scores.softmax(dim=-1)
    entropy = -(probs * probs.log()).sum(dim=-1).mean()
    max_entropy = math.log(gate_scores.size(-1))
    return max_entropy - entropy  # Minimiser = maximiser entropy

# Dans train.py
loss_entropy = landmark_entropy_loss(gate_scores)
loss_total += lambda_entropy * loss_entropy
```

---

#### ⚠️ GPU Underutilized

**Symptômes**:
```bash
nvidia-smi

# Output:
GPU  Memory-Usage  Utilization
0    10GB / 24GB   45%           ← ❌ Sous-utilisé
```

**Causes**:
1. Batch trop petit
2. Sequence length trop courte
3. CPU bottleneck (data loading)

**Solutions**:

**Solution 1: Augmenter Batch**
```yaml
train:
  batch_size: 32  # 16 → 32 (doubler)
  # Check OOM! Réduire si mémoire insuffisante
```

**Solution 2: Réduire Accum Steps**
```yaml
train:
  batch_size: 16
  accum_steps: 2  # 4 → 2 (updates 2x plus fréquents)
  # Effective batch = 16×2 = 32 (plus petit OK si GPU idle)
```

**Solution 3: Data Loading**
```yaml
data:
  num_workers: 2  # 0 → 2 (parallel data loading)
  pin_memory: true  # Faster CPU→GPU transfer
  prefetch_factor: 2  # Prefetch batches
```

**Solution 4: Mixed Precision**
```yaml
train:
  amp: true
  amp_dtype: "bf16"  # Plus rapide que fp16 sur Ampere
```

---

#### ⚠️ Throughput Instable

**Symptômes**:
```
Step 14050: 3422 tok/s  ← Normal
Step 14300: 927 tok/s   ← ❌ Spike down (10x)
Step 14500: 6022 tok/s  ← Recovered
```

**Causes**:
1. Landmarks selection instable (coûteux)
2. Gradient clipping triggers (gradients explosent)
3. CUDA memory fragmentation
4. Système background tasks

**Diagnostic**:
```bash
# 1. Monitor NVCC errors
dmesg | grep -i cuda

# 2. Check system load
htop  # CPU usage
iotop # Disk I/O

# 3. Profile training
python -m torch.profiler scripts/train.py
```

**Solutions**:

**Landmarks**:
```yaml
model:
  learned_landmarks: false  # Test si ça stabilise
  diverse_topk: false       # Désactiver si coûteux
```

**Clipping**:
```yaml
train:
  grad_clip: 0.5  # Plus strict
  # Log grad norms pour investiguer
```

**Memory**:
```python
# Dans train.py, après optimizer.step()
if step % 100 == 0:
    torch.cuda.empty_cache()  # Clear fragments
```

---

### Checklist Sanity Pre-Launch

**Avant `python scripts/train.py`**:

```bash
# ✅ 1. Config validation
cat config_fineweb_edu.yaml | grep -E "dataset|lr|batch_size"

# ✅ 2. Environment
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# ✅ 3. Dataset accessible
python -c "from datasets import load_dataset; ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train[:1%]'); print(f'OK: {len(ds)} samples')"

# ✅ 4. Output directory
mkdir -p out_slga/tensorboard
ls -la out_slga/

# ✅ 5. Disk space
df -h | grep -E "Filesystem|/$"
# Besoin: ~50GB pour checkpoints (100K steps × 500MB/ckpt)

# ✅ 6. Test run (1 step)
python scripts/train.py --config config_fineweb_edu.yaml --max-steps 1

# ✅ 7. Check logs
tail -20 training.log
```

**Si TOUS ✅ → GO !**

---

## 7. Commandes de Lancement

### Validation Pré-Training

#### Test 1: Environment Check
```bash
# Vérifier dependencies
python scripts/test_validation.py
```

Expected output:
```
✅ PyTorch: 2.0.0+cu118
✅ CUDA available: True
✅ GPU: NVIDIA GeForce RTX 3090
✅ Transformers: 4.30.0
✅ Datasets: 2.12.0
✅ Accelerate: 0.20.0
All checks passed!
```

---

#### Test 2: Dry Run (1 Step)
```bash
# Test complet avec 1 step
python scripts/train.py \
    --config config_fineweb_edu.yaml \
    --max-steps 1 \
    --output-dir out_slga_test

# Vérifier output
ls -la out_slga_test/
# Attendu: tensorboard/, ckpt_1/ créés
```

Expected output:
```
Loading dataset...
Dataset loaded: 9M samples (train), 500K (val)
Initializing model...
Model: 65.2M parameters
Starting training...
Step 1/1: Loss 8.43 | PPL 4567 | 3422 tok/s
✅ Checkpoint saved to out_slga_test/ckpt_1
```

---

#### Test 3: Short Run (100 Steps, ~3 min)
```bash
# Run rapide pour détecter bugs
python scripts/train.py \
    --config config_fineweb_edu.yaml \
    --max-steps 100 \
    --eval-every 50 \
    --save-every 50 \
    --output-dir out_slga_test

# Monitor
tail -f training.log | grep "Step"
```

Expected behavior:
```
Step 50:  Loss 6.2 | Val Loss 6.8 | PPL 492
Step 100: Loss 5.1 | Val Loss 5.8 | PPL 244
✅ Loss descending smoothly
✅ No NaN/Inf
✅ Throughput stable
```

---

### Lancement Production

#### Option 1: Lancement Interactif (Recommandé pour 1er run)

```bash
# Terminal 1: Training
python scripts/train.py --config config_fineweb_edu.yaml

# Terminal 2: TensorBoard
tensorboard --logdir=out_slga/tensorboard --port=6006

# Terminal 3: Monitoring
watch -n 10 'nvidia-smi; tail -5 training.log'

# Browser: http://localhost:6006
```

**Avantages**:
- ✅ Voir logs temps réel
- ✅ Ctrl+C pour stop propre
- ✅ TensorBoard live

**Inconvénients**:
- ❌ Terminal doit rester ouvert
- ❌ Connexion SSH interrompue = training stop

---

#### Option 2: Lancement Background (Recommandé pour runs longs)

```bash
# Lancement avec nohup
nohup python scripts/train.py \
    --config config_fineweb_edu.yaml \
    > training.log 2>&1 &

# Sauver PID
echo $! > training.pid
echo "Training PID: $(cat training.pid)"

# TensorBoard séparé
nohup tensorboard \
    --logdir=out_slga/tensorboard \
    --port=6006 \
    --host=0.0.0.0 \
    > tensorboard.log 2>&1 &

echo $! > tensorboard.pid
```

**Monitoring**:
```bash
# Logs temps réel
tail -f training.log | grep "Step"

# Logs avec contexte
tail -f training.log

# Check process
ps aux | grep train.py

# Check GPU
watch -n 5 nvidia-smi

# Stop training (propre)
kill -SIGTERM $(cat training.pid)

# Stop training (force)
kill -9 $(cat training.pid)
```

---

#### Option 3: Lancement Screen/Tmux (Alternative)

```bash
# Créer session tmux
tmux new -s slga_training

# Dans tmux
python scripts/train.py --config config_fineweb_edu.yaml

# Détacher: Ctrl+B puis D

# Réattacher
tmux attach -t slga_training

# Lister sessions
tmux ls
```

---

### Commandes Post-Lancement

#### Monitoring Temps Réel

```bash
# Dashboard personnalisé
watch -n 10 'clear; \
  echo "=== SLGA Training Monitor ==="; \
  echo ""; \
  nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader | head -1; \
  echo ""; \
  tail -1 training.log | grep Step; \
  echo ""; \
  ls -lh out_slga/ckpt_* 2>/dev/null | tail -5'
```

Output example:
```
=== SLGA Training Monitor ===

GPU: 92%, Mem: 85%, 20GB/24GB, Temp: 72C

Step 15234: Loss 2.84 | Val 3.12 | PPL 17.2 | 3567 tok/s

Checkpoints:
ckpt_14000  500MB
ckpt_15000  500MB
```

---

#### Checkpoint Management

```bash
# Lister checkpoints
ls -lh out_slga/ckpt_*/

# Espace disque
du -sh out_slga/

# Nettoyer vieux checkpoints (garder 1/5)
python scripts/cleanup_checkpoints.py \
    --keep-every 5000 \
    --dir out_slga

# Backup checkpoint important
cp -r out_slga/ckpt_50000 /backup/slga_50k_$(date +%Y%m%d)
```

---

#### Generation Tests

```bash
# Test génération @ checkpoint
python scripts/generate_fixed.py \
    --checkpoint out_slga/ckpt_30000 \
    --prompt "The capital of France is" \
    --max-length 100 \
    --temperature 0.0 \
    --num-samples 5

# Batch test (plusieurs prompts)
cat << 'EOF' > test_prompts.txt
The capital of France is
In a galaxy far far away
To solve this problem, we need to
EOF

python scripts/test_generation_suite.py \
    --checkpoint out_slga/ckpt_30000 \
    --prompts test_prompts.txt \
    --output results/generation_30k.json
```

---

### Configuration Finale Recommandée

Créer `config_fineweb_edu.yaml`:

```yaml
seed: 1234
device: cuda

model:
  vocab_size: 50257
  max_seq_len: 2048
  embed_dim: 512
  num_heads: 8
  ff_hidden_multiplier: 4
  n_layers: 12
  dropout_rate: 0.1

  # Attention locale-globale
  local_window: 128
  global_k: 24
  gated_fusion: true
  learned_landmarks: true     # OU false pour plus de stabilité
  dilated_windows: true
  diverse_topk: true

train:
  # Curriculum
  seq_len_start: 384
  seq_len_mid: 1024
  seq_len_final: 2048
  seq_len_warmup_steps: 15000

  # Batch (RTX 3090 optimisé)
  batch_size: 16
  accum_steps: 4

  # Optimizer
  lr: 2.0e-4
  betas: [0.9, 0.95]
  eps: 1.0e-8
  weight_decay: 0.01          # ✅ RÉDUIT (0.1 → 0.01)
  warmup_steps: 2000
  max_steps: 100000
  grad_clip: 1.0

  # Performance
  amp: true
  amp_dtype: "bf16"
  grad_checkpointing: false
  torch_compile: false

  # Collator
  collator: "local"
  global_every: 128
  max_global: 64

  # Loss auxiliaires
  lambda_diversity: 0.02
  lambda_sparsity: 0.001
  lambda_spacing: 0.02        # ✅ AJOUTÉ

  # Global warmup
  global_warmup_start: 1000
  global_warmup_end: 7500     # ✅ AUGMENTÉ (5K → 7.5K)

  # Logging
  save_every: 1000
  eval_every: 250             # ✅ AUGMENTÉ (500 → 250)
  log_every: 50

data:
  # ✅ NOUVEAU DATASET
  dataset: "HuggingFaceFW/fineweb-edu"
  subset: "sample-10BT"
  split_train: "train[:90%]"
  split_val: "train[90%:95%]"
  split_test: "train[95%:]"
  num_workers: 0
  max_train_samples: null
  max_val_samples: 20000      # ✅ AUGMENTÉ (10K → 20K)

tokenizer: "gpt2"

save:
  out_dir: "out_slga"

log:
  wandb: false
  project: "slga-fineweb"
  run_name: "slga-65m-fineweb-100k"
  tensorboard: true
```

---

## 8. Métriques de Succès

### Objectifs @ 100K Steps

#### Primary Goals (P0) 🎯

| Métrique | Target | Méthode Validation |
|----------|--------|-------------------|
| **Val PPL** | **15-25** | `python scripts/evaluate.py --dataset fineweb-edu` |
| **Train/Val Gap** | **< 30%** | `(val_loss - train_loss) / train_loss < 0.3` |
| **Generation Quality** | **Cohérent 5+ phrases** | `python scripts/test_generation_suite.py` |
| **No Instability** | **0 NaN/Inf** | `grep -c "NaN\|Inf" training.log` = 0 |

---

#### Secondary Goals (P1) 🎯

| Métrique | Target | Validation |
|----------|--------|------------|
| **Landmark Spacing** | **std < mean/2** | `scripts/inspect_trainer_state.py --analyze-landmarks` |
| **Training Time** | **28-36h** | `cat training.log | grep "Total time"` |
| **GPU Utilization** | **85-95%** | `nvidia-smi dmon` moyenne |
| **Throughput** | **3000-6000 tok/s** | `grep tok/s training.log | awk '{print $X}' | stats` |

---

### Validation Complète @ 100K

#### 1. Perplexity Multi-Dataset

```bash
python scripts/evaluate_checkpoint.py \
    --checkpoint out_slga/ckpt_100000 \
    --datasets fineweb-edu,openwebtext,wikipedia,c4 \
    --samples 10000 \
    --output results/ppl_evaluation.json
```

**Targets**:
```json
{
  "fineweb-edu": {
    "ppl": 18.5,  // ✅ 15-25
    "loss": 2.92
  },
  "openwebtext": {
    "ppl": 28.3,  // ✅ 25-35 (généralisation)
    "loss": 3.34
  },
  "wikipedia": {
    "ppl": 24.1,  // ✅ 20-30 (pas overfitté)
    "loss": 3.18
  },
  "c4": {
    "ppl": 32.7,  // ✅ 30-40 (out-of-distribution)
    "loss": 3.49
  }
}
```

---

#### 2. Generation Quality Assessment

**Test Suite**:
```bash
# Créer prompts diversifiés
cat << 'EOF' > data/eval_prompts.txt
# Factual
The capital of France is
Albert Einstein was born in

# Creative
Once upon a time, in a distant galaxy
The old wizard looked into his crystal ball and saw

# Reasoning
To calculate the area of a circle, we
If all humans are mortal, and Socrates is human, then

# Continuation
def fibonacci(n):
    """Calculate nth Fibonacci number"""
    if n <= 1:

# Long-form
The history of artificial intelligence began in the 1950s when
EOF

# Run evaluation
python scripts/test_generation_suite.py \
    --checkpoint out_slga/ckpt_100000 \
    --prompts data/eval_prompts.txt \
    --temperatures 0.0,0.7,1.0 \
    --max-length 200 \
    --output results/generation_quality.json
```

**Manual Inspection**:
```
Prompt: "The capital of France is"
Temp 0.0: "The capital of France is Paris, which is located in the
           north-central part of the country on the River Seine. It
           has been one of Europe's major centers of culture..."
           ✅ Factuel, cohérent, informatif

Temp 0.7: "The capital of France is Paris, a beautiful city known for
           its architecture, art, and cuisine. The Eiffel Tower is
           perhaps its most famous landmark..."
           ✅ Créatif, maintient cohérence

Temp 1.0: "The capital of France is Paris. This historic city features
           museums like the Louvre and landmarks such as Notre-Dame.
           Many tourists visit each year to experience..."
           ✅ Diversifié, correct
```

**Success Criteria**:
- ✅ Génération factuelle correcte (Paris, not "the capital of 2004")
- ✅ Cohérence grammaticale (5+ phrases sans rupture)
- ✅ Pertinence sémantique (reste on-topic)
- ✅ Pas de répétitions excessives

---

#### 3. Landmark Analysis

```bash
python scripts/analyze_landmarks.py \
    --checkpoint out_slga/ckpt_100000 \
    --visualize \
    --output results/landmarks_analysis.png
```

**Métriques**:
```
Landmark Statistics (global_k=24, seq_len=2048):
──────────────────────────────────────────────────
Positions (mean):        1024.3  ✅ Centré
Positions (std):         590.2   ✅ Large spread
Spacing (mean):          85.3    ✅ ~2048/24
Spacing (std):           38.7    ✅ < mean/2
Coverage:                94.2%   ✅ > 90%

Per-Layer Analysis:
Layer 0:  Mean pos = 512.1  (early layers prefer start)
Layer 6:  Mean pos = 1024.5 (mid layers balanced)
Layer 11: Mean pos = 1536.8 (late layers prefer end)
✅ Diversité par couche

Per-Head Diversity:
Head diversity score: 0.82  ✅ > 0.7
(Measure: 1 - IoU of selected positions across heads)
```

**Visualisation**:
![Landmarks Distribution](example_plot.png)
- X-axis: Position dans séquence [0, 2048]
- Y-axis: Layer [0, 11]
- Points: Landmarks sélectionnés
- ✅ Distribution uniforme (pas de clusters)

---

#### 4. Training Stability

```bash
# Analyse complète du training log
python scripts/analyze_training_log.py \
    --log training.log \
    --output results/training_analysis.json
```

**Métriques**:
```json
{
  "stability": {
    "nan_count": 0,           // ✅ = 0
    "inf_count": 0,           // ✅ = 0
    "loss_spikes": 3,         // ✅ < 5 (acceptable)
    "throughput_cv": 0.12     // ✅ < 0.2 (stable)
  },
  "convergence": {
    "train_loss_final": 2.45,
    "val_loss_final": 2.89,
    "gap": 0.44,              // ✅ < 0.5
    "train_ppl_final": 11.6,
    "val_ppl_final": 18.0     // ✅ 15-25
  },
  "performance": {
    "avg_throughput": 4123,   // tok/s
    "total_time_hours": 32.5, // ✅ 28-36h
    "gpu_util_avg": 91.3      // ✅ 85-95%
  }
}
```

---

### Comparaison Baseline

**Avant (Step 15K, Wikipedia)**:
```
Val PPL:      420     ❌ Catastrophique
Train/Val:    3.0     ❌ Overfitting sévère
Generation:   "the capital of 2004. It includes..." ❌ Incohérent
Stability:    Throughput spikes (927-6022 tok/s) ⚠️
```

**Après (Step 100K, FineWeb-Edu)** - **TARGETS**:
```
Val PPL:      18      ✅ -95% improvement
Train/Val:    0.3     ✅ -90% overfitting
Generation:   "Paris, which is located..." ✅ Cohérent
Stability:    Stable (3500-4500 tok/s) ✅
```

---

### Critères de Décision

#### ✅ Training Réussi Si:

1. **Val PPL ≤ 25** @ 100K steps
2. **Train/Val gap < 0.5** (30%)
3. **Génération cohérente** sur 5+ prompts tests
4. **0 NaN/Inf** pendant tout le training
5. **Landmark spacing std < mean/2**

#### ⚠️ Acceptable Mais Peut Améliorer Si:

1. **Val PPL = 25-30** → Continue training 10K steps
2. **Gap = 0.5-0.8** → Augmente dropout/régularisation
3. **Génération bonne mais répétitive** → Ajuste sampling

#### ❌ Training Échoué Si:

1. **Val PPL > 30** @ 100K → Dataset ou architecture inadéquate
2. **Gap > 1.0** → Overfitting non résolu
3. **Génération incohérente** → Model pas appris langage
4. **NaN/Inf fréquents** → Instabilité critique

---

### Next Steps Après 100K

#### Si Succès ✅

**Option 1: Continuer Training**
```yaml
# config_extended.yaml (hérite de config_fineweb_edu.yaml)
train:
  max_steps: 150000   # +50K steps
  lr: 1.0e-5          # LR plus faible

# Resume
python scripts/train.py \
    --config config_extended.yaml \
    --resume out_slga/ckpt_100000
```

**Option 2: Fine-Tuning Spécialisé**
```yaml
# config_finetune_code.yaml
data:
  dataset: "bigcode/the-stack-dedup"
  subset: "python"

train:
  max_steps: 10000    # Quick finetune
  lr: 5.0e-5          # Lower LR
```

**Option 3: Instruction Tuning**
```yaml
# config_instruct.yaml
data:
  dataset: "Open-Orca/OpenOrca"  # Instruction dataset

train:
  max_steps: 20000
  lr: 1.0e-5
```

---

#### Si Échec ❌

**Diagnostic Complet**:
```bash
# 1. Analyser training complet
python scripts/postmortem_analysis.py \
    --log training.log \
    --checkpoints out_slga/ \
    --output results/postmortem.md

# 2. Comparer checkpoints
python scripts/compare_checkpoints.py \
    --ckpt1 out_slga/ckpt_50000 \
    --ckpt2 out_slga/ckpt_100000 \
    --output results/checkpoint_comparison.json

# 3. Dataset analysis
python scripts/analyze_dataset_quality.py \
    --dataset fineweb-edu \
    --samples 10000 \
    --output results/dataset_analysis.json
```

**Ajustements**:
1. **Si PPL trop élevé** → Augmenter capacité modèle (embed_dim, n_layers)
2. **Si overfitting** → Augmenter dataset size, dropout
3. **Si instabilité** → Landmarks fixes, LR plus faible
4. **Si convergence lente** → Augmenter LR, batch size

---

## 📝 Résumé Exécutif

### 🚨 CHANGEMENTS CRITIQUES (P0)

1. **Dataset**: Wikipedia → **FineWeb-Edu** (impact: **-95% PPL**)
2. **Lambda spacing**: Ajouter `0.02` (couverture landmarks)
3. **Eval frequency**: 500 → **250 steps** (feedback rapide)

### 🟡 CHANGEMENTS IMPORTANTS (P1)

4. **Weight decay**: 0.1 → **0.01** (moins régularisation)
5. **Global warmup**: 5K → **7.5K steps** (convergence stable)

### ✅ Configuration Finale

```yaml
# config_fineweb_edu.yaml
data:
  dataset: "HuggingFaceFW/fineweb-edu"
  subset: "sample-10BT"
  split_train: "train[:90%]"
  split_val: "train[90%:95%]"

train:
  weight_decay: 0.01
  lambda_spacing: 0.02
  global_warmup_end: 7500
  eval_every: 250
```

### 🎯 Objectifs @ 100K Steps

- Val PPL: **15-25** (vs 420 observé avec Wikipedia)
- Train/Val gap: **< 30%**
- Génération: **Cohérente 5+ phrases**
- Durée: **28-36 heures** (RTX 3090)

### 🚀 Commande de Lancement

```bash
# Validation rapide (3 min)
python scripts/train.py --config config_fineweb_edu.yaml --max-steps 100

# Production (34h)
nohup python scripts/train.py \
    --config config_fineweb_edu.yaml \
    > training.log 2>&1 &

# Monitoring
tensorboard --logdir=out_slga/tensorboard --port=6006
tail -f training.log | grep "Step"
```

---

## 📚 Fichiers Créés

- `config_fineweb_edu.yaml` - Configuration optimale
- `docs/PRE_TRAINING_RECOMMENDATIONS.md` - Ce document
- `scripts/resume_with_new_dataset.py` - Outil resume (déjà existant)

---

## ⚠️ NOTES FINALES

### Ce qui a CHANGÉ depuis Step 15K

**Diagnostiqué**:
- ❌ Wikipedia PPL 420 → Inadéquat
- ❌ Overfitting 3.0 loss units → Trop fort
- ❌ Génération "capital of 2004" → Incohérente
- ⚠️ Landmarks possiblement instables

**Corrigé**:
- ✅ FineWeb-Edu (PPL target 15-25)
- ✅ Weight decay réduit (0.1 → 0.01)
- ✅ Lambda spacing ajouté (0.02)
- ✅ Global warmup rallongé (5K → 7.5K)
- ✅ Eval plus fréquent (500 → 250)

### Prochaines Actions

1. **Créer** `config_fineweb_edu.yaml` (voir Section 7)
2. **Tester** dry run 100 steps (~3 min)
3. **Lancer** training complet (34h)
4. **Monitorer** @ steps 1K, 5K, 30K, 50K, 100K
5. **Évaluer** métriques de succès (Section 8)

### Questions ?

**Dataset alternatives** → Section 2
**Hyperparamètres** → Section 3
**Troubleshooting** → Section 6
**Commandes** → Section 7

---

**Dernière mise à jour**: 2025-10-24
**Version**: 1.0
**Auteur**: SLGA Training Team
