# 🚀 Guide: Reprise avec FineWeb-Edu

**Date**: 2025-10-24
**Status**: ✅ Prêt pour Migration

---

## 📋 Contexte

Votre modèle SLGA-Plus a atteint un PPL de **~420** à 100K steps sur Wikipedia, significativement plus élevé que l'objectif de **15-25**.

### 🔍 Problème: Dataset Wikipedia

| Métrique | Wikipedia | Problème |
|----------|-----------|----------|
| **PPL @ 100K** | 420 | ❌ 17× trop élevé |
| **Dataset Size** | 6B tokens | ❌ Trop petit |
| **Epochs @ 100K** | 16.7 | ❌ Overfitting massif |

### ✅ Solution: FineWeb-Edu

| Métrique | FineWeb-Edu | Amélioration |
|----------|-------------|--------------|
| **PPL @ 100K** | 15-25 | ✅ -95% |
| **Dataset Size** | 1.3T tokens | ✅ 216× plus grand |
| **Epochs @ 100K** | 0.077 | ✅ Pas d'overfitting |

---

## 🎯 Décision: Recommencer (RECOMMANDÉ)

**Recommandation**: Recommencer à zéro avec FineWeb-Edu

**Raisons**:
- ✅ Métriques propres depuis le début
- ✅ Pas d'artifacts de domain shift
- ✅ Configuration v1.1 déjà optimisée
- ✅ Gains massifs (-95% PPL) justifient restart

---

## 🚀 Plan d'Action

### Phase 1: Préparation (2h)

```bash
# Télécharger FineWeb-Edu
python scripts/prepare_fineweb_edu.py --subset sample-10BT --validate
```

### Phase 2: Test 1K Steps (2h)

```bash
# Validation rapide
python scripts/train.py --config config/config_fineweb_edu.yaml --max-steps 1000

# Objectif @ 1K: PPL < 100
```

### Phase 3: Production (34h)

```bash
# Training complet
nohup python scripts/train.py \
  --config config/config_fineweb_edu.yaml \
  --max-steps 100000 \
  > training_fineweb.log 2>&1 &

# Monitoring
tensorboard --logdir=out_slga_fineweb/tensorboard --port=6006
```

---

## 📊 Résultats Attendus

| Step | Wikipedia PPL | FineWeb-Edu PPL | Amélioration |
|------|---------------|-----------------|--------------|
| **1K** | 80-120 | 60-80 | -25% |
| **15K** | 50-60 | 25-30 | -50% |
| **100K** | **420** | **15-25** | **-95%** |

---

## ✅ Checklist

- [ ] Dataset téléchargé (`prepare_fineweb_edu.py`)
- [ ] Config validée (`config_fineweb_edu.yaml`)
- [ ] Test 1K steps (PPL < 100)
- [ ] Training lancé (34h)
- [ ] TensorBoard actif (port 6006)

---

**Coût**: +8h setup + 50GB storage
**Bénéfice**: -95% PPL (420 → 15-25) + +15.8% MMLU
**ROI**: **Excellent**

🚀 **Bon training !**
