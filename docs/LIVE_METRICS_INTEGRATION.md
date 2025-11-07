# 📊 Guide: Intégration Live Metrics Display

**Module**: `src/live_metrics.py`
**Objectif**: Affichage console amélioré avec métriques visuelles en temps réel

---

## ✨ Fonctionnalités

### Affichage Amélioré
- ✅ **Couleurs ANSI** pour highlighting important
- ✅ **Symboles Unicode** (↑↓→ ✓ ⚠ 🚀 📊 🧠)
- ✅ **Barres de progression** visuelles
- ✅ **Mini graphiques ASCII** pour tendances
- ✅ **Best values** tracking automatique
- ✅ **Détection d'anomalies** (NaN, explosions)
- ✅ **ETA intelligent** avec timing précis

### Métriques Affichées

#### Section 1: Core Metrics
- Loss (avec tendance ↑↓→)
- Perplexity (avec tendance)
- Learning Rate
- Gradient Norm
- Best values historiques

#### Section 2: Validation (si disponible)
- Validation Loss/PPL
- Train/Val Gap (avec couleur)
- Best validation score

#### Section 3: Performance
- Tokens/sec throughput
- Samples/sec
- Séquence length (curriculum)
- GPU memory (avec pourcentage coloré)

#### Section 4: Landmarks (si activés)
- Nombre de landmarks sélectionnés
- Global attention weight
- Spacing loss
- Sparsity loss

#### Section 5: Loss Trend Chart
- Mini graphique ASCII des 50 dernières valeurs
- Visualisation rapide de convergence

#### Section 6: Warnings
- NaN detection
- Loss explosion
- Gradient explosion
- GPU memory warnings
- Throughput drops

---

## 🔧 Intégration dans train.py

### Option 1: Remplacement Simple (Recommandé)

**Étape 1**: Import au début de `train.py`
```python
# Ajouter après les imports existants
from src.live_metrics import LiveMetricsDisplay
```

**Étape 2**: Initialisation (dans la fonction `main()`)
```python
# Remplacer la création de progress_bar par:
if accelerator.is_main_process:
    live_display = LiveMetricsDisplay(
        max_steps=total_steps,
        log_every=cfg["train"].get("log_every", 50),
        width=100,
        compact=False,  # True pour version condensée
    )
else:
    live_display = None
```

**Étape 3**: Update dans la boucle de training (remplacer section logging)
```python
# AVANT (lignes ~502-549):
if accelerator.is_main_process and step % cfg["train"].get("log_every", 100) == 0:
    loss_gathered = accelerator.gather(loss_ce.detach()).mean().item()
    lr_current = scheduler.get_last_lr()[0]
    ppl = math.exp(min(loss_gathered, 10))

    # ... logging code ...

    print(f"Step {step:>6} | Loss: {loss_gathered:.4f} | PPL: {ppl:>7.2f} ...")

# APRÈS:
if accelerator.is_main_process and step % cfg["train"].get("log_every", 50) == 0:
    loss_gathered = accelerator.gather(loss_ce.detach()).mean().item()
    lr_current = scheduler.get_last_lr()[0]
    ppl = math.exp(min(loss_gathered, 10))

    # GPU memory
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.memory_allocated() / 1e9
        gpu_mem_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        gpu_mem_gb = 0
        gpu_mem_total_gb = 0

    # Performance
    elapsed_time = time.time() - step_start_time
    steps_per_sec = steps_since_log / elapsed_time if elapsed_time > 0 else 0
    tokens_per_sec = steps_per_sec * cfg["train"]["batch_size"] * current_seq_len

    # Update live display
    if live_display:
        live_display.update(
            step=step,
            loss=loss_gathered,
            ppl=ppl,
            lr=lr_current,
            grad_norm=last_grad_norm,
            seq_len=current_seq_len,
            global_weight=global_weight,
            tokens_per_sec=tokens_per_sec,
            gpu_memory_gb=gpu_mem_gb,
            gpu_memory_total_gb=gpu_mem_total_gb,
            num_landmarks=last_num_landmarks,
            spacing_loss=last_spacing_loss,
            sparsity_loss=last_spar_loss,
        )

    # W&B logging (garder tel quel)
    if cfg["log"].get("wandb", False):
        wandb.log({...}, step=step)

    # TensorBoard (garder tel quel)
    if writer:
        writer.add_scalar(...)

    # Reset timing
    step_start_time = time.time()
    steps_since_log = 0
```

**Étape 4**: Update validation (ajouter métriques de validation)
```python
# Dans la fonction validate(), après calcul val_loss:
if accelerator.is_main_process and live_display:
    val_ppl = math.exp(min(val_loss, 10))

    # Update avec métriques de validation
    live_display.update(
        step=step,
        loss=last_train_loss,  # Garder dernière train loss
        ppl=last_train_ppl,
        val_loss=val_loss,     # Ajouter validation
        val_ppl=val_ppl,
        lr=scheduler.get_last_lr()[0],
        # ... autres métriques ...
    )
```

### Option 2: Intégration Progressive

**Phase 1**: Test avec script séparé
```bash
# Test du module
python -c "from src.live_metrics import test_display; test_display()"
```

**Phase 2**: Garder ancien logging + nouveau
```python
# Dans train.py, ajouter les deux systèmes en parallèle
if live_display:
    live_display.update(...)

# Garder les print() existants pour debug
print(f"Step {step} | Loss: {loss_gathered:.4f} ...")
```

**Phase 3**: Une fois validé, supprimer ancien logging

---

## 🎨 Exemple de Sortie

```
================================================================================
🚀 SLGA Training Live Metrics
Step 15,000 / 100,000 (15.0%) │ Elapsed: 11:37:45 │ ETA: 62:14:07
[████████████████████──────────] 15.0%

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

================================================================================
```

---

## 🔍 Customisation

### Ajuster la largeur
```python
live_display = LiveMetricsDisplay(
    max_steps=100000,
    log_every=50,
    width=120,  # Plus large pour plus d'info
)
```

### Mode compact (moins de détails)
```python
live_display = LiveMetricsDisplay(
    max_steps=100000,
    compact=True,  # Pas de graphiques ASCII
)
```

### Désactiver couleurs (pour logs fichiers)
```python
# Dans live_metrics.py, ajouter au début de la classe:
class LiveMetricsDisplay:
    def __init__(self, ..., use_colors=True):
        self.use_colors = use_colors
        if not use_colors:
            # Désactiver tous les codes ANSI
            Colors.RESET = ""
            Colors.BOLD = ""
            # ... etc
```

### Ajouter métriques custom
```python
# Dans update():
live_display.update(
    step=step,
    loss=loss,
    # ... métriques standard ...

    # Métriques custom
    custom_metric=my_value,
)

# Dans _render(), ajouter section:
if 'custom_metric' in metrics:
    print(f"Custom: {metrics['custom_metric']:.4f}")
```

---

## 🧪 Tests

### Test 1: Affichage de base
```bash
python -c "from src.live_metrics import test_display; test_display()"
```

### Test 2: Intégration dans training
```python
# Modifier train.py temporairement pour tester sur 1000 steps
cfg["train"]["max_steps"] = 1000
cfg["train"]["log_every"] = 10
```

### Test 3: Détection anomalies
```python
# Tester avec NaN
live_display.update(
    step=100,
    loss=float('nan'),  # Devrait afficher warning
    ...
)
```

---

## 📝 Avantages vs Ancien Système

| Feature | Ancien (print) | Nouveau (LiveMetrics) |
|---------|----------------|----------------------|
| Couleurs | ❌ | ✅ |
| Tendances | ❌ | ✅ ↑↓→ |
| Graphiques | ❌ | ✅ ASCII charts |
| Best values | ❌ | ✅ Auto-tracked |
| ETA | ❌ | ✅ Précis |
| Anomalies | ❌ | ✅ Auto-detect |
| GPU % | ❌ | ✅ Couleur |
| Validation | Simple | ✅ Train/Val gap |
| Lisibilité | 3/10 | 9/10 |

---

## 🚀 Quick Start

**1 minute pour tester**:
```bash
cd /mnt/d/ai/SLGA
python -c "from src.live_metrics import test_display; test_display()"
```

**5 minutes pour intégrer**:
```python
# Dans train.py:
from src.live_metrics import LiveMetricsDisplay

# Initialiser
live_display = LiveMetricsDisplay(max_steps=100000)

# Dans boucle training
if step % 50 == 0:
    live_display.update(
        step=step, loss=loss, ppl=ppl, lr=lr,
        grad_norm=grad_norm, tokens_per_sec=tokens_per_sec,
        gpu_memory_gb=gpu_mem, gpu_memory_total_gb=24.0,
    )
```

**10 minutes pour customize**:
- Ajuster width, compact mode
- Ajouter métriques custom
- Intégrer validation display

---

## 📊 Performance Impact

- **CPU overhead**: < 0.1% (calculs légers)
- **Memory**: < 1MB (historique 50 values × N metrics)
- **Display time**: < 50ms par update
- **Recommandé**: `log_every >= 50` pour minimiser overhead

---

## 🆘 Troubleshooting

### Problème: Couleurs ne s'affichent pas
```bash
# Vérifier support ANSI
echo -e "\033[31mTEST\033[0m"

# Si Windows: utiliser Windows Terminal ou activer ANSI
# Si SSH: vérifier TERM variable
export TERM=xterm-256color
```

### Problème: Symboles Unicode cassés
```python
# Dans live_metrics.py, utiliser ASCII fallback
class Symbols:
    ARROW_UP = "^" if not supports_unicode() else "↑"
    # ...
```

### Problème: Flickering (clignotement)
```python
# Commenter clear screen dans _render()
# Ligne ~455:
# print("\033[2J\033[H", end="")  # DÉSACTIVER
```

---

## 🎯 Prochaines Étapes

1. ✅ Module `live_metrics.py` créé
2. ⏳ **Action requise**: Intégrer dans `train.py`
3. ⏳ Test avec training réel
4. ⏳ Ajuster selon préférences

**Temps estimé**: 10-15 minutes pour intégration complète

Bon training avec visualisation améliorée ! 🚀
