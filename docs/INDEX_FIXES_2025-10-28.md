# Index des Fixes - Session 2025-10-28

## 📋 Navigation Rapide

### 🚀 Commencer Maintenant
- **[Quick Reference](QUICK_REFERENCE_FIXES_2025-10-28.md)** - Guide rapide (5 min)
- **[Commande de relance](#commande-de-relance)** - Lancer le training

### 📖 Documentation Détaillée
- **[Fixes Applied](FIXES_APPLIED_2025-10-28_SESSION.md)** - Documentation complète (20 min)
- **[Vérification](#scripts-de-vérification)** - Tester les fixes
- **[Troubleshooting](#troubleshooting)** - Résoudre les problèmes

---

## 🚀 Commande de Relance

```bash
# Relancer le training avec tous les fixes
python scripts/train.py --config config/config.wikipedia.yaml --resume

# Avec monitoring TensorBoard
tensorboard --logdir out_slga/tensorboard --port 6006 &
python scripts/train.py --config config/config.wikipedia.yaml --resume
```

---

## 📁 Structure de la Documentation

```
docs/
├── INDEX_FIXES_2025-10-28.md                    ← VOUS ÊTES ICI
├── QUICK_REFERENCE_FIXES_2025-10-28.md          ← Référence rapide
└── FIXES_APPLIED_2025-10-28_SESSION.md          ← Documentation complète

tests/
└── verify_all_fixes_2025-10-28.py               ← Script de vérification

scripts/
└── cleanup_old_checkpoints.py                   ← Nettoyage manuel
```

---

## 📝 Résumé des Fixes

### Configuration (4 fixes)
| Paramètre | Avant | Après | Impact |
|-----------|-------|-------|--------|
| `lr` | 2.0e-4 | 1.5e-4 | Meilleure stabilité |
| `warmup_steps` | 2000 | 5000 | Convergence améliorée |
| `keep_last_checkpoints` | N/A | 5 | Économie ~80GB |
| `debug_checkpoints` | N/A | false | Logs propres |

### Utilitaires (5 fixes)
- ✅ Rotation automatique des checkpoints
- ✅ Fix calcul mémoire GPU libre
- ✅ Helper `load_latest_checkpoint()`
- ✅ Suppression automatique vieux checkpoints
- ✅ Import `shutil` pour rotation

### Training (4 fixes)
- ✅ Fix truncation `cache_ids` (curriculum learning)
- ✅ Fix crash gradient monitoring
- ✅ Utilisation rotation dans sauvegarde
- ✅ Debug logs conditionnels

---

## 🧪 Scripts de Vérification

### Vérifier que Tous les Fixes Sont Appliqués
```bash
python tests/verify_all_fixes_2025-10-28.py
```

**Output attendu:**
```
✓ CONFIG: All checks passed (4/4)
✓ UTILS:  All checks passed (5/5)
✓ TRAIN:  All checks passed (4/4)
✓ TOUS LES FIXES SONT CORRECTEMENT APPLIQUÉS
```

### Nettoyer Vieux Checkpoints (Manuel)
```bash
# Dry-run (affiche ce qui serait supprimé)
python scripts/cleanup_old_checkpoints.py --out-dir out_slga --keep 5 --dry-run

# Vraie suppression
python scripts/cleanup_old_checkpoints.py --out-dir out_slga --keep 5

# Interactive (demande confirmation)
python scripts/cleanup_old_checkpoints.py --out-dir out_slga --keep 5 --interactive
```

---

## 📊 Métriques à Surveiller

### Training (log_every=10)
| Métrique | Attendu | Problème si |
|----------|---------|-------------|
| Loss | Descente progressive | Oscillations brutales |
| PPL | 100 → 50 → 30 → ... | Augmente ou stagne |
| LR | 0 → 1.5e-4 (5K steps) | Trop élevé trop vite |
| SeqLen | 384 → 1024 → 2048 | Crash "index out of bounds" |

### GPU (log_every=10)
| Métrique | Attendu | Problème si |
|----------|---------|-------------|
| Memory | 15-18GB stable | OOM ou trop bas (<10GB) |
| Free | 6-9GB | Incohérent (vérifier fix) |
| Utilization | 75-85% | <50% (sous-utilisé) |

### Checkpoints (save_every=1000)
| Événement | Attendu | Problème si |
|-----------|---------|-------------|
| Sauvegarde | "✓ CHECKPOINT SAVED" | Erreur ou silence |
| Rotation | "🗑️ Supprimé ancien" (step > 5000) | Checkpoints s'accumulent |
| Count | Max 5 checkpoints | Plus de 5 après step 5000 |

---

## 🐛 Troubleshooting

### Problème: Loss Explose
**Symptômes:** Loss passe de 5.0 à 50.0+ brutalement

**Solutions:**
1. Vérifier learning rate: `grep "lr:" config/config.wikipedia.yaml` → devrait être `1.5e-4`
2. Vérifier warmup: `grep "warmup_steps:" config/config.wikipedia.yaml` → devrait être `5000`
3. Réduire batch_size si problème persiste: `batch_size: 8 → 4`

**Fichiers:** `config/config.wikipedia.yaml` (lignes 32-36)

---

### Problème: Crash "Index Out of Bounds"
**Symptômes:** Crash pendant curriculum learning

**Solutions:**
1. Vérifier fix cache_ids: `grep -A3 "cache_ids = cache_ids\[mask\]" scripts/train.py`
2. Devrait afficher la logique de filtrage avec `mask = cache_ids < current_seq_len`

**Fichiers:** `scripts/train.py` (ligne ~625)

---

### Problème: Crash Gradient Monitoring
**Symptômes:** Crash tous les 500 steps avec "cannot unpack"

**Solutions:**
1. Vérifier named_parameters: `grep "for name, param in model.named_parameters" scripts/train.py`
2. Ne devrait PAS utiliser `model.parameters()` (sans `named_`)

**Fichiers:** `scripts/train.py` (ligne ~750)

---

### Problème: Checkpoints S'accumulent
**Symptômes:** Plus de 5 checkpoints après step 5000, disque plein

**Solutions:**
1. Vérifier config: `grep "keep_last_checkpoints:" config/config.wikipedia.yaml` → devrait être `5`
2. Vérifier utilisation: `grep "keep_last_n=keep_last" scripts/train.py` → devrait être présent
3. Nettoyer manuellement: `python scripts/cleanup_old_checkpoints.py --out-dir out_slga --keep 5`

**Fichiers:**
- `config/config.wikipedia.yaml` (ligne 64)
- `scripts/train.py` (ligne ~1023)
- `scripts/utils.py` (ligne ~88-97)

---

### Problème: Mémoire GPU Incohérente
**Symptômes:** Affichage de mémoire libre négatif ou incohérent

**Solutions:**
1. Vérifier fix: `grep "free = total - reserved" scripts/utils.py`
2. Ne devrait PAS être `free = total - allocated`

**Fichiers:** `scripts/utils.py` (ligne ~188)

---

## 📈 Timeline de Training Attendue

```
Step 0:         Initialisation, first forward pass
Step 0-1000:    Warmup phase (LR 0 → 0.3e-4), SeqLen 384-512
Step 1000:      First checkpoint saved
Step 1000-5000: Warmup continues (LR → 1.5e-4), SeqLen augmente
Step 5000:      Global warmup actif, première rotation checkpoint
                (Si checkpoints > 5, suppression des plus anciens)
Step 10000:     SeqLen ~1500, global weight ~50%
Step 15000:     SeqLen 2048, global weight 100%
Step 20000:     Training stable, tous mécanismes actifs
```

---

## 🔄 Workflow Recommandé

### 1. Vérification Pré-Training
```bash
# Vérifier tous les fixes
python tests/verify_all_fixes_2025-10-28.py

# Nettoyer vieux checkpoints si besoin
python scripts/cleanup_old_checkpoints.py --out-dir out_slga --keep 5 --dry-run
```

### 2. Lancer Training
```bash
# Relancer avec resume
python scripts/train.py --config config/config.wikipedia.yaml --resume
```

### 3. Monitoring (Terminal Séparé)
```bash
# TensorBoard
tensorboard --logdir out_slga/tensorboard --port 6006

# Ou watch GPU
watch -n 1 nvidia-smi
```

### 4. Validation Post-Training (Après 5K Steps)
```bash
# Vérifier checkpoints
ls -lth out_slga/ | grep ckpt_ | head -10

# Tester génération
python scripts/generate.py --checkpoint out_slga/ckpt_5000 --prompt "Once upon a time"
```

---

## 📚 Références Externes

### Documentation Originale
- [SLGA Paper](https://arxiv.org/abs/2XXX.XXXXX) - Si disponible
- [GitHub Issues](https://github.com/your-repo/SLGA/issues) - Si applicable

### Outils de Monitoring
- [TensorBoard Guide](https://www.tensorflow.org/tensorboard)
- [nvidia-smi Reference](https://developer.nvidia.com/nvidia-system-management-interface)

### Optimisations Futures
- Gradient accumulation tuning
- Mixed precision strategies
- Distributed training setup

---

## 🎯 Checklist Rapide

**Avant de relancer le training:**
- [ ] Vérifier fixes: `python tests/verify_all_fixes_2025-10-28.py`
- [ ] Nettoyer checkpoints si > 5: `ls out_slga/ | grep ckpt_ | wc -l`
- [ ] Vérifier espace disque: `df -h .`
- [ ] Vérifier GPU disponible: `nvidia-smi`

**Pendant le training:**
- [ ] Surveiller loss (devrait descendre)
- [ ] Vérifier GPU memory stable
- [ ] Confirmer rotation checkpoints (step > 5000)
- [ ] Pas de crashes curriculum

**Après 5K steps:**
- [ ] Valider métriques (loss < 4.0, ppl < 50)
- [ ] Tester génération
- [ ] Vérifier seulement 5 checkpoints max
- [ ] Confirmer espace disque économisé

---

## 📞 Support

### En Cas de Problème
1. Consulter [Troubleshooting](#troubleshooting) ci-dessus
2. Vérifier logs détaillés: `tail -n 100 training.log`
3. Relire [Documentation Complète](FIXES_APPLIED_2025-10-28_SESSION.md)

### Logs Utiles
```bash
# Derniers logs training
tail -f training.log  # Si redirigé

# Checkpoints info
ls -lth out_slga/ | grep ckpt_

# GPU usage
nvidia-smi -l 1

# TensorBoard
tensorboard --logdir out_slga/tensorboard --port 6006
```

---

**Dernière mise à jour:** 2025-10-28
**Status:** ✅ Tous fixes appliqués et vérifiés
**Prochaines étapes:** Relancer training et monitorer métriques
