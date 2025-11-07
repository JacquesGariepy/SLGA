# Quick Reference: Fixes Appliqués 2025-10-28

## 🚀 Commande de Relance Rapide

```bash
python scripts/train.py --config config/config.wikipedia.yaml --resume
```

---

## ✅ Fixes Appliqués (13 au Total)

### 📋 CONFIG (4 fixes)
1. ✅ `lr: 2.0e-4 → 1.5e-4` (stabilité)
2. ✅ `warmup_steps: 2000 → 5000` (convergence)
3. ✅ `keep_last_checkpoints: 5` (rotation, économie disque)
4. ✅ `debug_checkpoints: false` (logs propres)

### 🔧 UTILS (5 fixes)
1. ✅ `save_checkpoint()` + paramètre `keep_last_n`
2. ✅ Rotation automatique (supprime vieux checkpoints)
3. ✅ `get_memory_usage()` fix (`free = total - reserved`)
4. ✅ `load_latest_checkpoint()` helper ajouté
5. ✅ Import `shutil` pour rotation

### 🎯 TRAIN (4 fixes)
1. ✅ Fix `cache_ids` truncation (curriculum learning)
2. ✅ Fix gradient monitoring crash (`named_parameters()`)
3. ✅ Utiliser `keep_last_n` dans sauvegarde
4. ✅ Conditionner debug logs checkpoint

---

## 🧪 Vérification

```bash
# Vérifier que tous les fixes sont appliqués
python tests/verify_all_fixes_2025-10-28.py

# Devrait afficher: ✓ TOUS LES FIXES SONT CORRECTEMENT APPLIQUÉS
```

---

## 📊 Impact Attendu

### Stabilité
- Learning rate réduit → moins d'oscillations
- Warmup augmenté → démarrage stable
- Cache IDs fix → pas de crashes curriculum

### Performance
- Rotation checkpoints → économie ~80GB après 50K steps
- Mémoire GPU correcte → meilleure allocation
- Logs optimisés → moins de bruit

### Monitoring
- Gradient flow sans crash (step 500, 1000, ...)
- Mémoire GPU affichée correctement
- Checkpoints rotation visible dans logs

---

## 🔍 Métriques à Surveiller

### Training (tous les 10 steps)
- **Loss**: Devrait descendre progressivement sans oscillations brutales
- **PPL**: Devrait diminuer (100+ → 50 → 30 → ...)
- **LR**: Devrait augmenter pendant warmup (0 → 1.5e-4 sur 5000 steps)
- **SeqLen**: 384 → 1024 → 2048 (curriculum sur 15K steps)

### GPU (tous les 10 steps)
- **Memory**: Devrait rester stable (~15-18GB pour batch_size=8)
- **Free**: Devrait être cohérent (total - reserved)
- **Utilisation**: 75-85% cible

### Checkpoints (tous les 1000 steps)
- **Sauvegarde**: "✓ CHECKPOINT SAVED at step N"
- **Rotation**: "🗑️ Supprimé ancien checkpoint: ckpt_X" (si N > 5000)
- **Espace disque**: Max 5 checkpoints présents (~25GB au lieu de 100GB+)

### Validation (tous les 500 steps)
- **Val Loss**: Devrait suivre train loss
- **Val PPL**: Devrait diminuer aussi
- **Pas de crash**: Labels validation correctement masqués (-100)

---

## 🐛 Troubleshooting

### Si loss explose
```bash
# Vérifier learning rate
grep "lr:" config/config.wikipedia.yaml
# Devrait afficher: lr: 1.5e-4

# Vérifier warmup
grep "warmup_steps:" config/config.wikipedia.yaml
# Devrait afficher: warmup_steps: 5000
```

### Si crash "index out of bounds"
```bash
# Vérifier fix cache_ids
grep -A3 "cache_ids = cache_ids\[mask\]" scripts/train.py
# Devrait afficher la logique de filtrage
```

### Si gradient monitoring crash
```bash
# Vérifier named_parameters
grep "for name, param in model.named_parameters" scripts/train.py
# Devrait être présent (pas model.parameters())
```

### Si checkpoints s'accumulent
```bash
# Vérifier rotation
ls -lh out_slga/ | grep ckpt_
# Devrait montrer seulement 5 checkpoints max après step 5000

# Vérifier config
grep "keep_last_checkpoints:" config/config.wikipedia.yaml
# Devrait afficher: keep_last_checkpoints: 5
```

---

## 📈 Timeline Attendue (Batch_size=8, Accum=4)

```
Step 0-1000:    Warmup (LR 0 → 1.5e-4), SeqLen 384-512
Step 1000-5000: Warmup complet, SeqLen augmente progressivement
Step 5000:      Global warmup actif, première rotation checkpoint
Step 10000:     SeqLen ~1500, global weight ~50%
Step 15000:     SeqLen 2048, global weight 100%
Step 20000:     Training stable, tous les mécanismes actifs
```

---

## 📁 Fichiers Modifiés

```
config/config.wikipedia.yaml       (4 changements)
scripts/utils.py                   (5 changements)
scripts/train.py                   (4 changements)
docs/FIXES_APPLIED_2025-10-28_SESSION.md (documentation)
tests/verify_all_fixes_2025-10-28.py    (vérification)
```

---

## 🔄 Commandes Utiles

```bash
# Relancer training
python scripts/train.py --config config/config.wikipedia.yaml --resume

# Monitoring TensorBoard
tensorboard --logdir out_slga/tensorboard --port 6006

# Vérifier checkpoints
ls -lth out_slga/ | grep ckpt_ | head -10

# Suivre logs en temps réel
tail -f training.log  # Si vous redirigez la sortie

# Vérifier GPU
nvidia-smi -l 1

# Tester génération après training
python scripts/generate.py --checkpoint out_slga/ckpt_XXXXX --prompt "Once upon a time"
```

---

## ✨ Prochaines Étapes

1. **Relancer training** avec `--resume`
2. **Surveiller métriques** pendant 5K steps
3. **Vérifier rotation** des checkpoints
4. **Valider stabilité** (pas d'explosions de loss)
5. **Tester génération** après quelques K steps

---

**Date:** 2025-10-28
**Status:** ✅ Tous fixes appliqués et vérifiés
**Next:** Relancer training et monitorer
