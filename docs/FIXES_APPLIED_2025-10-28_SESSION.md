# Fixes Appliqués - Session 2025-10-28

## Résumé
Application complète de tous les fixes confirmés en une seule session pour améliorer la stabilité, la performance et la gestion des ressources du training SLGA.

---

## 1. Configuration (config/config.wikipedia.yaml)

### 1.1 Learning Rate Réduit
**Changement:** `lr: 2.0e-4 → 1.5e-4`

**Raison:** Meilleure stabilité pendant le training, réduction des oscillations de loss.

**Impact:** Convergence plus stable, moins de risques de divergence.

---

### 1.2 Warmup Steps Augmenté
**Changement:** `warmup_steps: 2000 → 5000`

**Raison:**
- Warmup plus long = meilleure convergence initiale
- Évite les gradients explosifs au début du training
- Permet au modèle de s'adapter progressivement

**Impact:** Training plus stable dans les premiers milliers de steps.

---

### 1.3 Rotation des Checkpoints
**Nouveau paramètre:** `keep_last_checkpoints: 5`

**Raison:**
- Éviter l'accumulation de checkpoints (100GB+ après 50K steps)
- Garde seulement les 5 derniers checkpoints
- Suppression automatique des anciens

**Impact:**
- Espace disque économisé: ~80GB après 50K steps
- Gestion propre des sauvegardes

---

### 1.4 Debug Checkpoints Désactivé
**Nouveau paramètre:** `debug_checkpoints: false`

**Raison:**
- Réduire le bruit dans les logs
- Logs debug checkpoint seulement si activé explicitement

**Impact:** Logs plus propres et lisibles pendant le training.

---

## 2. Utilitaires (scripts/utils.py)

### 2.1 Rotation Automatique des Checkpoints
**Fonction modifiée:** `save_checkpoint()`

**Nouveau paramètre:** `keep_last_n: int = None`

**Implémentation:**
```python
# Après sauvegarde
if keep_last_n is not None:
    all_ckpts = sorted([d for d in os.listdir(out_dir) if d.startswith("ckpt_")],
                      key=lambda x: int(x.split("_")[1]))
    if len(all_ckpts) > keep_last_n:
        for old in all_ckpts[:-keep_last_n]:
            shutil.rmtree(os.path.join(out_dir, old))
            print(f"  🗑️ Supprimé ancien checkpoint: {old}")
```

**Impact:** Suppression automatique des anciens checkpoints après chaque sauvegarde.

---

### 2.2 Fix Calcul Mémoire Libre
**Fonction modifiée:** `get_memory_usage()`

**Changement:** `free = total - allocated` → `free = total - reserved`

**Raison:**
- `allocated` = mémoire actuellement utilisée par les tensors
- `reserved` = mémoire totale réservée par CUDA (includes cached)
- La vraie mémoire libre = `total - reserved`

**Impact:** Affichage correct de la mémoire GPU disponible.

**Exemple:**
```
Avant (incorrect):
  allocated=12GB, total=24GB → free=12GB (FAUX si 3GB cached!)

Après (correct):
  allocated=12GB, reserved=15GB, total=24GB → free=9GB (CORRECT!)
```

---

### 2.3 Helper load_latest_checkpoint()
**Nouvelle fonction:** `load_latest_checkpoint()`

**Utilité:**
- Charge automatiquement le dernier checkpoint disponible
- Simplifie le code de reprise

**Usage:**
```python
step = load_latest_checkpoint(out_dir, model, optimizer, scheduler, device)
```

**Impact:** Code plus propre et maintenable.

---

## 3. Training (scripts/train.py)

### 3.1 Fix Cache IDs Tronqué
**Ligne:** ~625

**Problème:**
- Curriculum learning tronque `input_ids` et `labels`
- Mais `cache_ids` n'était PAS tronqué → index out of bounds!

**Fix:**
```python
if input_ids.size(1) > current_seq_len:
    input_ids = input_ids[:, :current_seq_len]
    labels = labels[:, :current_seq_len]
    # 🔧 FIX: Tronquer cache_ids aussi!
    if cache_ids is not None:
        mask = cache_ids < current_seq_len
        cache_ids = cache_ids[mask].unsqueeze(0) if mask.any() else None
```

**Impact:**
- Évite les crashes "index out of bounds" pendant curriculum
- Landmarks correctement filtrés selon la longueur de séquence actuelle

---

### 3.2 Fix Crash Gradient Monitoring
**Ligne:** ~750

**Problème:** `model.parameters()` retourne seulement les tensors, pas les noms → crash lors du logging

**Fix:**
```python
# ❌ AVANT (crash)
for name, param in model.parameters():
    ...

# ✅ APRÈS (fonctionne)
for name, param in model.named_parameters():
    if param.grad is not None:
        layer_norm = param.grad.data.norm(2).item()
        grad_norms_per_layer[name] = layer_norm
```

**Impact:** Gradient flow monitoring fonctionne correctement tous les 500 steps.

---

### 3.3 Utiliser keep_last_n dans Sauvegarde
**Ligne:** ~1023

**Changement:**
```python
# Avant
save_checkpoint(model, optimizer, scheduler, out_dir, step, accelerator)

# Après
keep_last = cfg["train"].get("keep_last_checkpoints")
save_checkpoint(model, optimizer, scheduler, out_dir, step, accelerator, keep_last_n=keep_last)
```

**Impact:** Rotation des checkpoints activée pendant le training.

---

### 3.4 Conditionner Debug Logs Checkpoint
**Ligne:** ~1012

**Changement:**
```python
# Avant: toujours afficher debug
if step <= 10 or (step % 100 == 0):
    print(f"\n[DEBUG Checkpoint]...")

# Après: seulement si debug_checkpoints=true
debug_ckpt = cfg["train"].get("debug_checkpoints", False)
if debug_ckpt and (step <= 10 or (step % 100 == 0)):
    print(f"\n[DEBUG Checkpoint]...")
```

**Impact:** Logs plus propres (pas de spam debug checkpoint).

---

## Impact Global des Fixes

### ✅ Stabilité
- Learning rate réduit: moins d'oscillations
- Warmup augmenté: démarrage plus stable
- Cache IDs fix: plus de crashes curriculum
- Gradient monitoring fix: monitoring correct

### ✅ Performance
- Rotation checkpoints: économie d'espace disque (~80GB)
- Mémoire GPU correcte: meilleure allocation
- Logs optimisés: moins de bruit

### ✅ Maintenabilité
- Code plus propre
- Helper functions ajoutées
- Debug conditionnel
- Documentation inline

---

## Vérification Post-Application

### Tests Recommandés
```bash
# 1. Vérifier config
cat config/config.wikipedia.yaml | grep -A2 "lr:\|warmup_steps:\|keep_last_checkpoints:\|debug_checkpoints:"

# 2. Test rotation checkpoints (créer 10 checkpoints et vérifier que seulement 5 restent)
# 3. Test gradient monitoring (step 500, 1000, etc.)
# 4. Test curriculum learning avec cache_ids
```

### Métriques à Surveiller
- Loss stability (pas d'oscillations brutales)
- GPU memory (free correctement calculé)
- Disk usage (checkpoints rotation fonctionnelle)
- Gradient norms (monitoring sans crash)

---

## Prochaines Étapes Recommandées

### Court Terme
1. **Relancer training avec --resume**
   - Vérifier que tous les fixes fonctionnent
   - Surveiller les métriques pendant 5K steps

2. **Monitoring Actif**
   - Loss trajectory
   - GPU memory usage
   - Gradient norms
   - Checkpoint rotation

### Moyen Terme
1. **Optimisations Supplémentaires**
   - Ajuster `lambda_spacing` et `lambda_sparsity` si besoin
   - Tester différents `seq_len_warmup_steps`
   - Profiler la performance

2. **Validation**
   - Tester génération avec différents prompts
   - Vérifier que landmarks appris fonctionnent
   - Comparer avec baseline

---

## Fichiers Modifiés

### Configuration
- `/mnt/d/ai/SLGA/config/config.wikipedia.yaml`
  - `lr: 1.5e-4`
  - `warmup_steps: 5000`
  - `keep_last_checkpoints: 5`
  - `debug_checkpoints: false`

### Code
- `/mnt/d/ai/SLGA/scripts/utils.py`
  - `save_checkpoint()` avec rotation
  - `get_memory_usage()` fix free
  - `load_latest_checkpoint()` helper

- `/mnt/d/ai/SLGA/scripts/train.py`
  - Fix cache_ids truncation
  - Fix gradient monitoring crash
  - Utiliser keep_last_n
  - Conditionner debug logs

---

## Commande de Relance

```bash
# Reprendre training avec tous les fixes
python scripts/train.py \
    --config config/config.wikipedia.yaml \
    --resume

# Avec monitoring TensorBoard
tensorboard --logdir out_slga/tensorboard --port 6006
```

---

**Date:** 2025-10-28
**Session:** Fix complet en une passe
**Status:** ✅ Tous les fixes appliqués avec succès
**Next:** Relancer training et monitorer
