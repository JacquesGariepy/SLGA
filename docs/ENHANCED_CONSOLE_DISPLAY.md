# 📊 Console Display Amélioré - Guide Complet

**Status**: ✅ Prêt à utiliser
**Temps d'intégration**: 10-15 minutes

---

## 🎯 Objectif

Remplacer l'affichage console basique actuel:
```
Step  15000 | Loss: 2.5213 | PPL:   12.45 | LR: 2.00e-04 | GradNorm:  1.38
```

Par un affichage riche et visuel:
```
================================================================================
🚀 SLGA Training Live Metrics
Step 15,000 / 100,000 (15.0%) │ Elapsed: 11:37:45 │ ETA: 62:14:07
[████████████████████──────────────────] 15.0%

📊 Core Metrics
────────────────────────────────────────────────────────────────────────────────
  Loss:         2.5213 ↓  │  Best: 1.9679
  Perplexity:    12.45 ↓  │  Best: 7.16
  Learn Rate: 2.000e-04  │  Grad Norm: 1.3800

🎯 Validation
────────────────────────────────────────────────────────────────────────────────
  Val Loss:     6.0310  │  Val PPL:  416.12
  Train/Val Gap: +3.5097  │  Best Val: 6.0310

🚀 Performance
────────────────────────────────────────────────────────────────────────────────
  Throughput:    6,267 tok/s  │    3.3 samples/s
  Seq Length:    1,918 tokens
  GPU Memory:    12.45 / 24.0 GB  ( 51.9%)

🧠 Landmarks
────────────────────────────────────────────────────────────────────────────────
  Selected:       24  │  Global Weight: 100.0%
  Spacing Loss: 0.0023  │  Sparsity Loss: 0.0005

📊 Loss Trend (last 50 steps)
────────────────────────────────────────────────────────────────────────────────
  ██████████████████████████████████████████████████
  ██████████████▌
                 ▌██████████████████████████████████
```

---

## 📦 Fichiers Créés

### 1. Module Principal
**`src/live_metrics.py`** (700+ lignes)

Features:
- ✅ Classe `LiveMetricsDisplay` pour affichage riche
- ✅ Couleurs ANSI (rouge/vert/jaune selon valeurs)
- ✅ Symboles Unicode (↑↓→ ✓ ⚠ 🚀 📊)
- ✅ Barres de progression visuelles
- ✅ Mini graphiques ASCII pour tendances
- ✅ Best values tracking automatique
- ✅ Détection d'anomalies (NaN, explosions)
- ✅ ETA intelligent
- ✅ Historique de métriques avec deque

### 2. Guide d'Intégration
**`docs/LIVE_METRICS_INTEGRATION.md`**

Contient:
- ✅ Instructions pas-à-pas
- ✅ Exemples de code
- ✅ Customisation options
- ✅ Troubleshooting

### 3. Script de Patching
**`scripts/patch_train_with_live_metrics.py`**

Permet:
- ✅ Patching automatique de `train.py`
- ✅ Backup automatique
- ✅ Mode --dry-run pour preview

---

## 🚀 Quick Start (3 étapes)

### Étape 1: Tester le Module (1 minute)

```bash
cd /mnt/d/ai/SLGA
python -c "from src.live_metrics import test_display; test_display()"
```

**Résultat attendu**: Affichage animé avec métriques simulées pendant ~30 secondes.

### Étape 2: Intégration Manuelle (10 minutes)

**Option A**: Modification manuelle de `train.py`

Voir `docs/LIVE_METRICS_INTEGRATION.md` pour instructions détaillées.

**Résumé**:
```python
# 1. Import
from src.live_metrics import LiveMetricsDisplay

# 2. Initialiser (remplacer progress_bar)
live_display = LiveMetricsDisplay(
    max_steps=total_steps,
    log_every=50,
    width=100,
)

# 3. Update dans logging loop
if step % 50 == 0:
    live_display.update(
        step=step,
        loss=loss,
        ppl=ppl,
        lr=lr,
        grad_norm=grad_norm,
        tokens_per_sec=tokens_per_sec,
        gpu_memory_gb=gpu_mem_gb,
        gpu_memory_total_gb=24.0,
        # ... autres métriques ...
    )
```

**Option B**: Patching automatique (EXPÉRIMENTAL)

```bash
# Dry run (preview)
python scripts/patch_train_with_live_metrics.py --dry-run

# Apply patches
python scripts/patch_train_with_live_metrics.py
```

⚠️ **Note**: Le patching auto est expérimental. Vérifiez les changements avec `diff` avant utilisation.

### Étape 3: Test avec Training (2 minutes)

```bash
# Test rapide (1000 steps)
python scripts/train.py --config config.yaml --max-steps 1000
```

**Vérifiez**:
- ✅ Affichage coloré
- ✅ Métriques mises à jour
- ✅ Barres de progression
- ✅ Pas d'erreurs

---

## 📊 Métriques Affichées

### Core (toujours affichées)
| Métrique | Source | Couleur | Trend |
|----------|--------|---------|-------|
| **Loss** | CE loss | Rouge→Vert | ↑↓→ |
| **Perplexity** | exp(loss) | Rouge→Vert | ↑↓→ |
| **Learning Rate** | Scheduler | Cyan | - |
| **Grad Norm** | Before clip | Cyan | - |
| **Best Loss** | Historical | Green | - |
| **Best PPL** | Historical | Green | - |

### Validation (si disponible)
| Métrique | Source | Couleur | Notes |
|----------|--------|---------|-------|
| **Val Loss** | Validation pass | White | - |
| **Val PPL** | exp(val_loss) | White | - |
| **Train/Val Gap** | val_loss - loss | Rouge→Vert | Gap d'overfitting |
| **Best Val** | Historical | Green | Meilleur score |

### Performance
| Métrique | Calcul | Couleur | Notes |
|----------|--------|---------|-------|
| **Tokens/sec** | batch_size × seq_len / time | Green | Throughput |
| **Samples/sec** | tokens/sec / seq_len | Green | Vitesse |
| **Seq Length** | Curriculum | Cyan | Longueur actuelle |
| **GPU Memory** | torch.cuda.memory | Rouge→Vert | % utilisé |

### Landmarks (si learned_landmarks=true)
| Métrique | Source | Notes |
|----------|--------|-------|
| **Selected** | aux['landmark_indices'] | Nombre actif |
| **Global Weight** | Warmup schedule | 0→100% |
| **Spacing Loss** | Auxiliaire | Espacement |
| **Sparsity Loss** | Auxiliaire | Sélectivité |

### Progression
| Métrique | Calcul | Notes |
|----------|--------|-------|
| **Step** | Training step | Current/Total |
| **Progress** | step / max_steps | % et barre |
| **Elapsed** | time.time() - start | HH:MM:SS |
| **ETA** | (total_time / steps) × remaining | HH:MM:SS |

---

## 🎨 Customisation

### Ajuster la largeur
```python
live_display = LiveMetricsDisplay(
    max_steps=100000,
    width=120,  # Default: 100
)
```

### Mode compact (sans graphiques)
```python
live_display = LiveMetricsDisplay(
    max_steps=100000,
    compact=True,  # Pas de mini charts ASCII
)
```

### Changer fréquence logging
```python
live_display = LiveMetricsDisplay(
    max_steps=100000,
    log_every=100,  # Default: 50
)
```

### Désactiver couleurs (pour logs)
```python
# Dans src/live_metrics.py, début de classe:
class LiveMetricsDisplay:
    def __init__(self, ..., use_colors=False):
        if not use_colors:
            Colors.RESET = ""
            Colors.BOLD = ""
            # ... désactiver tous
```

### Ajouter métriques custom
```python
# Dans train.py:
live_display.update(
    step=step,
    # ... métriques standard ...
    my_custom_metric=value,
)

# Dans src/live_metrics.py, _render():
if 'my_custom_metric' in metrics:
    print(f"Custom: {metrics['my_custom_metric']:.4f}")
```

---

## 🔍 Détection d'Anomalies

Le système détecte automatiquement:

### 1. NaN Detection
```
⚠ WARNING
────────────────────────────────────────────────────────────────────────────────
  ⚠ NaN detected in loss (3x)
```

### 2. Loss Explosion
```
⚠ WARNING
────────────────────────────────────────────────────────────────────────────────
  ⚠ Loss explosion: 15.23
```

### 3. Gradient Explosion
```
⚠ WARNING
────────────────────────────────────────────────────────────────────────────────
  ⚠ Gradient explosion: 12.45
```

### 4. GPU Memory Critical
```
⚠ WARNING
────────────────────────────────────────────────────────────────────────────────
  ⚠ GPU memory critical: 97%
```

### 5. Throughput Drop
```
⚠ WARNING
────────────────────────────────────────────────────────────────────────────────
  ⚠ Throughput dropped 50%
```

---

## 📈 Tendances (↑↓→)

Le système calcule automatiquement les tendances sur fenêtre glissante:

- **↓** : Métrique en baisse (bon pour loss)
- **↑** : Métrique en hausse (mauvais pour loss)
- **→** : Métrique stable

**Fenêtre**: 10 dernières valeurs
**Seuil**: 5% de changement

---

## 🧪 Tests

### Test 1: Module seul
```bash
python -c "from src.live_metrics import test_display; test_display()"
```
**Durée**: 30 secondes
**Attendu**: Affichage animé avec métriques simulées

### Test 2: Intégration basic
```python
# Script de test minimal
from src.live_metrics import LiveMetricsDisplay
import time
import math

display = LiveMetricsDisplay(max_steps=100, log_every=1)

for step in range(0, 100, 10):
    display.update(
        step=step,
        loss=5.0 * math.exp(-step/50),
        ppl=math.exp(5.0 * math.exp(-step/50)),
        lr=2e-4,
        grad_norm=1.0,
        tokens_per_sec=5000,
        gpu_memory_gb=12.0,
        gpu_memory_total_gb=24.0,
    )
    time.sleep(0.1)
```

### Test 3: Training réel (1000 steps)
```bash
python scripts/train.py \
  --config config.yaml \
  --max-steps 1000 \
  --log-every 50
```

---

## 🆘 Troubleshooting

### Problème: Pas de couleurs

**Cause**: Terminal ne supporte pas ANSI

**Solution**:
```bash
# Vérifier support
echo -e "\033[31mTEST\033[0m"

# Windows: Utiliser Windows Terminal
# SSH: Vérifier TERM
export TERM=xterm-256color
```

### Problème: Symboles Unicode cassés

**Cause**: Encoding non-UTF8

**Solution**:
```python
# Dans live_metrics.py, ajouter fallback ASCII
class Symbols:
    ARROW_UP = "^"  # Au lieu de ↑
    ARROW_DOWN = "v"  # Au lieu de ↓
    # ...
```

### Problème: Flickering (clignotement)

**Cause**: Rafraîchissement trop rapide

**Solution**:
```python
# Dans _render(), commenter clear screen
# Ligne ~455:
# print("\033[2J\033[H", end="")  # DÉSACTIVER
```

### Problème: Performance dégradée

**Cause**: log_every trop petit

**Solution**:
```python
# Augmenter intervalle
live_display = LiveMetricsDisplay(
    max_steps=100000,
    log_every=100,  # Au lieu de 50
)
```

---

## 📊 Comparaison Avant/Après

### Avant (logging basique)
```
Step  15000 | Loss: 2.5213 | PPL:   12.45 | LR: 2.00e-04 | GradNorm:  1.38
           | SeqLen: 1918 | GW: 1.00 | LM: 48→24 | GPU:  1.4GB | Tok/s:  3422
```

**Limitations**:
- ❌ Pas de couleurs
- ❌ Difficile à lire
- ❌ Pas de contexte (best values, tendances)
- ❌ Pas d'ETA
- ❌ Pas de détection anomalies
- ❌ Pas de visualisation GPU%

### Après (LiveMetrics)
```
================================================================================
🚀 SLGA Training Live Metrics
Step 15,000 / 100,000 (15.0%) │ Elapsed: 11:37:45 │ ETA: 62:14:07
[████████████████████──────────────────] 15.0%

📊 Core Metrics
────────────────────────────────────────────────────────────────────────────────
  Loss:         2.5213 ↓  │  Best: 1.9679
  Perplexity:    12.45 ↓  │  Best: 7.16
  Learn Rate: 2.000e-04  │  Grad Norm: 1.3800

🚀 Performance
────────────────────────────────────────────────────────────────────────────────
  Throughput:    6,267 tok/s  │    3.3 samples/s
  GPU Memory:    12.45 / 24.0 GB  ( 51.9%)
```

**Avantages**:
- ✅ Couleurs selon valeurs
- ✅ Sections organisées
- ✅ Best values tracking
- ✅ Tendances ↑↓→
- ✅ ETA précis
- ✅ Graphiques ASCII
- ✅ Détection anomalies
- ✅ GPU% coloré

**Gain lisibilité**: 3/10 → 9/10

---

## 🎯 Prochaines Étapes

1. ✅ Module créé: `src/live_metrics.py`
2. ✅ Documentation: `docs/LIVE_METRICS_INTEGRATION.md`
3. ✅ Script patch: `scripts/patch_train_with_live_metrics.py`
4. ⏳ **Action requise**: Tester le module
5. ⏳ **Action requise**: Intégrer dans train.py
6. ⏳ Tester avec training réel

**Temps total estimé**: 15-20 minutes

---

## 📝 Checklist d'Intégration

- [ ] Tester module seul (`python -c "from src.live_metrics import test_display; test_display()"`)
- [ ] Backup train.py (`cp scripts/train.py scripts/train.py.backup`)
- [ ] Intégrer dans train.py (manuel ou patch)
- [ ] Test training 1000 steps
- [ ] Vérifier affichage coloré
- [ ] Vérifier métriques correctes
- [ ] Ajuster width/compact si besoin
- [ ] Déployer en production

---

## 💡 Conseils

1. **Testez d'abord** le module seul avant intégration
2. **Gardez un backup** de train.py
3. **Commencez simple** (mode compact) puis ajoutez features
4. **Ajustez la largeur** selon votre terminal
5. **Surveillez performance** (log_every >= 50 recommandé)

---

## 🚀 Résumé

**Vous avez maintenant**:
- ✅ Module complet de live metrics
- ✅ Documentation d'intégration
- ✅ Script de patching automatique
- ✅ Tests et troubleshooting

**Prochaine action**: Tester le module puis intégrer dans train.py

**Bénéfice**: Visualisation 3× meilleure du training avec détection d'anomalies

Bon training avec visualisation améliorée ! 🎉
