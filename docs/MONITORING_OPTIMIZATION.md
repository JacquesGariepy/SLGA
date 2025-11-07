# 📊 Monitoring & Optimisation pour RTX 3090

## État Actuel

### ✅ TensorBoard Configuré
- `tensorboard: true` dans config.yaml
- Logs sauvegardés dans `out_slga/tensorboard/`
- **Métriques actuellement loggées** :
  - `train/loss`
  - `train/perplexity`
  - `train/learning_rate`
  - `train/seq_len`
  - `train/global_weight`
  - `val/loss`
  - `val/perplexity`

### ⚠️ Métriques Manquantes (Importantes)

Les métriques suivantes aideraient grandement au debugging :
1. **Gradient norms** - Détecter vanishing/exploding gradients
2. **Loss components** - Diversity loss, sparsity loss séparément
3. **Landmark statistics** - Nombre de landmarks sélectionnés
4. **Memory usage** - Utilisation GPU
5. **Throughput** - Tokens/sec, steps/sec
6. **Activations stats** - Mean/std des activations par couche

---

## 🚀 Optimisation pour RTX 3090

### Configuration Actuelle (Conservative)

```yaml
batch_size: 4
accum_steps: 16
# Effective batch = 4 × 16 = 64
# VRAM utilisée: ~8-12 GB (sur 24 GB disponibles)
```

**Problème** : Vous n'utilisez que **40-50% de votre GPU** !

### Configuration Optimisée Recommandée

```yaml
batch_size: 8              # 4 → 8 (double)
accum_steps: 8             # 16 → 8 (réduit de moitié)
# Effective batch = 8 × 8 = 64 (identique)
# VRAM utilisée: ~16-20 GB (80-85% du GPU)
# Vitesse: ~2x plus rapide !
```

**Avantages** :
- ✅ Même effective batch size (64)
- ✅ 2x moins de steps d'accumulation → 2x plus rapide
- ✅ Meilleure utilisation du GPU (~80% au lieu de ~50%)
- ✅ Gradients plus fréquents → apprentissage plus stable

### Configuration Agressive (Si vous voulez aller encore plus vite)

```yaml
batch_size: 12             # Max avant OOM
accum_steps: 5-6           # Pour garder effective batch ~64
# VRAM utilisée: ~22-23 GB (95% du GPU)
# Vitesse: ~3x plus rapide qu'actuel !
```

**⚠️ Risque** : Très proche de la limite mémoire, peut crasher sur certaines séquences longues.

---

## 🔧 Améliorations à Implémenter

### 1. Métriques de Monitoring Avancées

Ajouter dans `scripts/train.py` (section logging) :

```python
# Gradient norms
if step % cfg["train"].get("log_every", 100) == 0:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    if writer is not None:
        writer.add_scalar("train/grad_norm", total_norm, step)

    # Log components de la loss
    if lambda_div > 0 and div_loss is not None:
        writer.add_scalar("train/loss_diversity", div_loss.item(), step)
    if lambda_spar > 0 and spar_loss is not None:
        writer.add_scalar("train/loss_sparsity", spar_loss.item(), step)
```

### 2. Statistics des Landmarks

```python
# Dans la boucle de training
if "landmark_gates" in aux and aux["landmark_gates"] is not None:
    gates = aux["landmark_gates"]
    num_selected = (gates > 0.5).sum().item()
    gates_mean = gates.mean().item()

    if writer is not None:
        writer.add_scalar("landmarks/num_selected", num_selected, step)
        writer.add_scalar("landmarks/gates_mean", gates_mean, step)
```

### 3. Throughput Metrics

```python
import time

# Au début de la boucle
step_start_time = time.time()

# À chaque log
if step % log_every == 0:
    step_time = time.time() - step_start_time
    tokens_per_sec = (batch_size * current_seq_len) / step_time

    if writer is not None:
        writer.add_scalar("perf/tokens_per_sec", tokens_per_sec, step)
        writer.add_scalar("perf/step_time", step_time, step)
```

### 4. Memory Usage

```python
if step % log_every == 0 and torch.cuda.is_available():
    mem_allocated = torch.cuda.memory_allocated() / 1e9  # GB
    mem_reserved = torch.cuda.memory_reserved() / 1e9    # GB

    if writer is not None:
        writer.add_scalar("perf/gpu_memory_allocated_gb", mem_allocated, step)
        writer.add_scalar("perf/gpu_memory_reserved_gb", mem_reserved, step)
```

---

## 📈 Dashboard TensorBoard Recommandé

### Vue d'Ensemble

Une fois lancé avec `tensorboard --logdir out_slga/tensorboard`, vous verrez :

**Scalars à surveiller** :

1. **train/loss** - Doit descendre régulièrement
2. **train/perplexity** - Cible : < 100 à 30K steps
3. **train/global_weight** - Doit rester à 0 jusqu'à 30K, puis monter
4. **train/learning_rate** - Warmup puis decay
5. **train/grad_norm** - Doit rester stable (~1-10)
6. **val/perplexity** - Ne doit pas diverger du train (overfitting)

**Signaux d'Alerte** :

- ❌ Loss qui stagne après 5K steps → Problème d'apprentissage
- ❌ Grad norm > 100 → Exploding gradients
- ❌ Grad norm < 0.001 → Vanishing gradients
- ❌ Val PPL >> Train PPL → Overfitting
- ❌ Memory usage qui augmente → Fuite mémoire

---

## 🎯 Configuration Finale Recommandée

### config.yaml Optimisé pour RTX 3090

```yaml
train:
  # Curriculum (inchangé)
  seq_len_start: 512
  seq_len_mid: 1024
  seq_len_final: 2048
  seq_len_warmup_steps: 15000

  # Batch optimisé pour 3090
  batch_size: 8              # ⬆️ 4 → 8
  accum_steps: 8             # ⬇️ 16 → 8

  # Learning rate (ajusté pour batch plus large)
  lr: 2.0e-4                 # OK
  warmup_steps: 2000         # ⬆️ 1000 → 2000 (plus de warmup = mieux)

  # Optimisations
  amp: true
  amp_dtype: "bf16"
  grad_checkpointing: true   # ✅ Garde ça même avec plus de batch

  # Logging plus fréquent pour debugging
  save_every: 2000
  eval_every: 500            # ⬇️ 1000 → 500 (plus fréquent)
  log_every: 50              # ⬇️ 100 → 50 (plus de feedback)

  # Global warmup
  global_warmup_start: 30000
  global_warmup_end: 50000
```

### Gains Attendus

| Metric | Avant | Après | Gain |
|--------|-------|-------|------|
| Steps/sec | ~0.5 | ~1.0 | **2x** |
| Temps pour 10K steps | ~10h | ~5h | **2x** |
| VRAM usage | 40-50% | 75-85% | Meilleur |
| Effective batch | 64 | 64 | Identique |

---

## 🛠️ Implémentation Rapide

Je peux créer un script qui :
1. ✅ Ajoute toutes les métriques avancées
2. ✅ Optimise la config pour RTX 3090
3. ✅ Ajoute un dashboard de monitoring en temps réel

Voulez-vous que je le fasse ?

---

## 📊 Exemple de Monitoring en Temps Réel

Avec les ajouts, voici ce que vous verrez dans la console :

```
Step 100 | Loss: 8.45 | PPL: 4623 | LR: 4.0e-5 | SeqLen: 512 | GW: 0.00
         | GradNorm: 2.3 | Landmarks: 23/24 | GPU: 16.2GB | Tok/s: 8192

Step 200 | Loss: 7.98 | PPL: 2931 | LR: 8.0e-5 | SeqLen: 512 | GW: 0.00
         | GradNorm: 1.9 | Landmarks: 24/24 | GPU: 16.4GB | Tok/s: 8543

Step 500 | Loss: 7.12 | PPL: 1243 | LR: 2.0e-4 | SeqLen: 512 | GW: 0.00
         | GradNorm: 1.5 | Landmarks: 22/24 | GPU: 16.1GB | Tok/s: 8421
         [VAL] Val Loss: 7.34 | Val PPL: 1542
```

Beaucoup plus d'informations pour diagnostiquer !

---

## ✅ Checklist Avant de Lancer

- [ ] Config TensorBoard activée (`tensorboard: true`)
- [ ] Batch size optimisé pour 3090 (8 recommandé)
- [ ] Métriques avancées ajoutées (optionnel mais recommandé)
- [ ] TensorBoard lancé dans un terminal séparé
- [ ] GPU monitoring actif (`watch -n 1 nvidia-smi`)

---

## 🚀 Commandes de Lancement

```bash
# Terminal 1: Training
python scripts/train.py

# Terminal 2: TensorBoard
tensorboard --logdir out_slga/tensorboard --port 6006

# Terminal 3: GPU Monitoring
watch -n 1 nvidia-smi

# Browser: http://localhost:6006
```

Avec ces optimisations, vous devriez atteindre 50K steps en **~25 heures** au lieu de 50h !
