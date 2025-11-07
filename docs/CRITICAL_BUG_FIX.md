# 🚨 BUG CRITIQUE DÉCOUVERT ET CORRIGÉ

**Date**: 2025-10-23
**Sévérité**: CRITIQUE
**Impact**: Empêche le modèle d'apprendre correctement

---

## 🔍 Le Problème

Votre training à **step 2900** montre:
- **Perplexity: 4424** (devrait être ~1000-2000)
- **Génération: gibberish complet**
- Symptôme identique à l'ancien code bugué

### Cause Racine: Double-Shifting des Labels

Le collator ET la fonction de loss shiftaient tous les deux les labels:

**Collator** (src/data.py:110):
```python
labels[:, :-1] = input_ids[:, 1:]  # Shift une fois
```

**Loss function** (scripts/train.py:99):
```python
labels_shifted = labels[:, 1:]  # ❌ Shift ENCORE!
```

**Résultat**: Le modèle essaie de prédire le token **2 positions en avant** au lieu de **1 position**!

C'est comme demander:
- Entrée: "The cat sat"
- Au lieu de prédire: "cat sat on"
- Le modèle essayait de prédire: "sat on the" (décalé de 2!)

→ Impossible à apprendre correctement!

---

## ✅ La Correction

### Fichier modifié: `scripts/train.py`

**AVANT** (bugué):
```python
# Line 99
labels_shifted = labels[:, 1:]  # ❌ Double shift!
```

**APRÈS** (corrigé):
```python
# Line 102
labels_shifted = labels[:, :-1]  # ✅ Juste retirer dernière position
```

**Explication**:
- Le collator a DÉJÀ shifté: `labels[i]` contient le token suivant pour `input_ids[i]`
- On compare juste `logits[i]` avec `labels[i]` (pas `labels[i+1]`)
- On retire la dernière position de logits car elle n'a pas de target

---

## 🧪 Comment Vérifier Que Le Fix Fonctionne

### Test Automatique (2 minutes)

```bash
cd /mnt/d/ai/SLGA
conda activate slga
python scripts/test_fix.py
```

**Ce script teste**:
1. ✅ Alignement correct des labels
2. ✅ Forward/backward pass fonctionnent
3. ✅ Génération ne crash pas

**Output attendu**:
```
✅ PASS - Alignement des labels
✅ PASS - Step de training
✅ PASS - Génération

✅ TOUS LES TESTS RÉUSSIS!
```

### Test Manuel (optionnel)

Si vous voulez vérifier manuellement l'alignement:

```python
from transformers import GPT2Tokenizer
from src.data import CollatorLocal

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
collator = CollatorLocal(tokenizer, max_length=10)

batch = collator([{"text": "The cat sat on the mat"}])
input_ids = batch["input_ids"][0]
labels = batch["labels"][0]

# Vérifier: labels[i] devrait être input_ids[i+1]
for i in range(5):
    print(f"input[{i}]: {tokenizer.decode([input_ids[i]])} -> "
          f"label[{i}]: {tokenizer.decode([labels[i]])}")

# Attendu:
# input[0]: 'The' -> label[0]: ' cat'    ✅
# input[1]: ' cat' -> label[1]: ' sat'   ✅
# input[2]: ' sat' -> label[2]: ' on'    ✅
```

---

## 🚀 Relancer Le Training

Une fois les tests passés:

### 1. Arrêter le training actuel
```bash
# Dans le terminal de training: Ctrl+C
```

### 2. Nettoyer les anciens checkpoints (buggés)
```bash
bash scripts/clean_restart.sh

# Ou manuellement:
rm -rf out_slga/ckpt_*
rm -rf out_slga/tensorboard/*
```

### 3. Relancer le training
```bash
# Vérifier qu'on utilise config_3090
cp config_3090.yaml config.yaml

# Lancer
python scripts/train.py
```

### 4. Monitorer (autres terminaux)

**Terminal 2 - TensorBoard**:
```bash
tensorboard --logdir out_slga/tensorboard --port 6006
# Ouvrir: http://localhost:6006
```

**Terminal 3 - Monitor** (optionnel):
```bash
python scripts/monitor.py
```

---

## 📊 Résultats Attendus Après Fix

### Avec le fix (attendu):

| Step | Loss | Perplexity | Génération |
|------|------|------------|------------|
| 100 | 8.5 | ~5000 | Légèrement moins aléatoire |
| 500 | 7.5 | ~1800 | Mots reconnaissables |
| 2000 | 6.5-7.0 | **800-2000** | **Mots cohérents** |
| 10000 | 5.5-6.0 | 150-400 | Phrases partielles |

### Sans le fix (actuel):

| Step | Loss | Perplexity | Génération |
|------|------|------------|------------|
| 2900 | 8.4 | **4424** | **Gibberish total** ❌ |

Le fix devrait donner **~2000 de perplexité** au lieu de **4424** au step 2000!

---

## ⚠️ Points d'Attention

### 1. Les métriques TensorBoard N/A

Vous avez remarqué:
- Grad Norm: N/A
- Global Weight: N/A
- GPU Memory: 1.3GB (au lieu de ~16GB)

**Possible cause**: Le script `monitor.py` lit les logs TensorBoard qui peuvent avoir un délai de quelques steps.

**Vérification**:
```bash
# Vérifier les logs console de training.py directement
# Devraient afficher:
# Step 2950 | Loss: X.XXXX | PPL: XXXX.XX | LR: X.XXe-XX | GradNorm: X.XX
#           | SeqLen: XXX | GW: 0.00 | Landmarks: 24 | GPU: 16.XGB | Tok/s: XXXX
```

Si les logs console sont corrects mais monitor.py affiche N/A, c'est juste un problème de lecture des events TensorBoard (bénin).

### 2. Séquence Length: 709

C'est normal! Le curriculum learning augmente progressivement:
- Step 0-15000: 512 → 2048
- Step 2900: 709 (progression normale)

---

## 🎯 Checklist de Vérification

Avant de relancer le training:

- [ ] **Test script passé**: `python scripts/test_fix.py` → tous ✅
- [ ] **Training arrêté**: Ctrl+C dans le terminal de training
- [ ] **Checkpoints nettoyés**: `bash scripts/clean_restart.sh`
- [ ] **Config correcte**: `cp config_3090.yaml config.yaml`
- [ ] **Environnement activé**: `conda activate slga`

Après avoir relancé:

- [ ] **Step 100**: Loss descend déjà (~8.5)
- [ ] **Step 500**: PPL ~1500-2000
- [ ] **Step 2000**: PPL ~800-2000 (au lieu de 4424!)
- [ ] **Console logs**: GradNorm et GW s'affichent
- [ ] **TensorBoard**: Graphiques se remplissent

---

## 📈 Timing Attendu

Avec config_3090.yaml (RTX 3090):
- **Step 2000**: ~1h (checkpoint de validation)
- **Step 10000**: ~5h
- **Step 30000**: ~15h (global warmup commence)
- **Step 50000**: ~25h

**Test de validation** à step 2000:
```bash
# Après ~1h de training
python scripts/generate.py \
    --checkpoint out_slga/ckpt_2000 \
    --prompt "In the year 2024," \
    --max-tokens 50

# Attendu: Mots anglais reconnaissables (pas parfait mais lisible)
# Au lieu de: "bulbsDeliameterAntStreamerBot..." (gibberish actuel)
```

---

## 🔬 Analyse Technique

### Pourquoi ce bug n'a pas été détecté plus tôt?

1. **La loss descendait quand même**: Le modèle apprenait *quelque chose*, juste la mauvaise tâche
2. **Pas de crash**: Le code s'exécutait sans erreur
3. **Alignement semblait correct**: Les dimensions matchaient

### Comment ce bug a été découvert?

En analysant votre génération gibberish à step 2900, j'ai retracé:
1. Génération → Mauvaise prédiction
2. Mauvaise prédiction → Mauvaise loss
3. Mauvaise loss → Double vérification de l'alignement
4. Découverte du double-shift dans `cross_entropy_shifted`

---

## 📞 Support

Si après le fix, les résultats ne s'améliorent pas:

1. **Vérifier les tests**: `python scripts/test_fix.py` → tous doivent passer
2. **Vérifier la loss**: Devrait être ~8.5 au step 100, ~7.5 au step 500
3. **Vérifier les logs console**: Devraient afficher GradNorm et GW
4. **Partager les métriques**: Steps 100, 500, 2000

---

## ✅ Résumé

**Bug**: Double-shifting des labels (collator + loss)
**Fix**: Changer `labels[:, 1:]` en `labels[:, :-1]` dans train.py:102
**Test**: `python scripts/test_fix.py`
**Action**: Nettoyer checkpoints et relancer training
**Attendu**: PPL ~800-2000 au step 2000 (au lieu de 4424)

**Ce fix devrait permettre au modèle d'apprendre correctement!** 🎉
