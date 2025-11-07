# 🧪 Comment Vérifier Que Tout Fonctionne

## ⚡ Test Rapide (2-3 minutes)

```bash
cd /mnt/d/ai/SLGA
conda activate slga
python scripts/test_complete.py
```

### Ce que ce test vérifie:

1. **✅ Training** (20 steps réels)
   - Loss descend correctement
   - Gradients dans la bonne plage
   - Pas de NaN/Inf

2. **✅ Perplexity** (évaluation sur validation set)
   - Calcul fonctionne
   - Résultats cohérents
   - Pas d'erreurs

3. **✅ Génération** (3 prompts différents)
   - Génère du texte sans crash
   - Pas de répétition infinie
   - Tokens variés

### Output attendu:

```
================================================================================
TEST COMPLET: TRAINING + PERPLEXITY + GENERATION
================================================================================

TEST 1: TRAINING (20 steps réels)
...
Step   5 | Loss: 11.2345 | PPL: 76543.21 | GradNorm: 2.345
Step  10 | Loss: 10.8912 | PPL: 54321.12 | GradNorm: 2.123
Step  15 | Loss: 10.5423 | PPL: 37654.87 | GradNorm: 1.987
Step  20 | Loss: 10.2134 | PPL: 27123.45 | GradNorm: 1.876

Analyse des 20 steps:
  Loss initiale (steps 1-5):   11.2345
  Loss finale (steps 16-20):   10.2134
  Diminution: 1.0211 (9.1%)
  ✅ Loss descend correctement
  ✅ Gradients dans une plage correcte
  ✅ Pas de NaN dans les losses

✅ TEST 1 RÉUSSI: Training fonctionne!

TEST 2: ÉVALUATION PERPLEXITÉ
...
Résultats sur 20 batches:
  Loss moyenne: 10.3456
  Perplexité:   31234.56
  ✅ Perplexité calculée correctement

✅ TEST 2 RÉUSSI: Évaluation fonctionne!

TEST 3: GÉNÉRATION DE TEXTE
...
Prompt: 'The cat'
→ Généré: 'The cat is a very large and very large mammal that lives in...'
  ✅ Génération variée
  ✅ 18 tokens générés

✅ TEST 3 RÉUSSI: Génération fonctionne!

================================================================================
RÉSUMÉ
================================================================================
✅ TEST 1: Training (20 steps) - RÉUSSI
✅ TEST 2: Perplexity evaluation - RÉUSSI
✅ TEST 3: Génération - RÉUSSI

🎉 TOUS LES TESTS RÉUSSIS!
```

---

## 🚀 Si Les Tests Passent

### 1. Arrêter le training actuel

```bash
# Dans le terminal où tourne le training: Ctrl+C
```

### 2. Nettoyer les checkpoints buggés

```bash
bash scripts/clean_restart.sh
```

### 3. Relancer le training

```bash
# Utiliser la config optimisée pour RTX 3090
cp config_3090.yaml config.yaml

# Lancer
python scripts/train.py
```

### 4. Monitorer (terminaux séparés)

**Terminal 2 - TensorBoard**:
```bash
tensorboard --logdir out_slga/tensorboard --port 6006
# Ouvrir: http://localhost:6006
```

**Terminal 3 - Monitor en temps réel**:
```bash
python scripts/monitor.py
```

---

## 📊 Résultats Attendus Après Le Fix

### Au step 2000 (~1h de training):

**AVANT le fix** (votre run actuel):
```python
python scripts/generate.py --checkpoint out_slga/ckpt_2000 \
    --prompt "In the year 2024," --max-tokens 50

# Output:
# bulbsDeliameterAntStreamerBot Abortion its levels without...
# → GIBBERISH TOTAL ❌
```

**APRÈS le fix** (attendu):
```python
python scripts/generate.py --checkpoint out_slga/ckpt_2000 \
    --prompt "In the year 2024," --max-tokens 50

# Output possible:
# In the year 2024, the United States will be the first to...
# → MOTS COHÉRENTS ✅
```

**Perplexité**:
- Avant: **PPL ~4424** au step 2900 ❌
- Après: **PPL ~800-2000** au step 2000 ✅

---

## ⚠️ Si Un Test Échoue

### Test 1 (Training) échoue:

**Problème**: Loss ne descend pas
**Cause possible**: Le fix n'a pas été appliqué correctement
**Solution**: Vérifier que `scripts/train.py` ligne 102 a bien:
```python
labels_shifted = labels[:, :-1].contiguous()  # PAS labels[:, 1:]!
```

**Problème**: Gradients vanishing/exploding
**Cause possible**: Problème d'architecture ou de learning rate
**Solution**: Vérifier config, réduire LR si nécessaire

### Test 2 (Perplexity) échoue:

**Problème**: Erreur de calcul ou NaN
**Cause possible**: Bug dans la fonction de loss
**Solution**: Vérifier que `cross_entropy_shifted` utilise bien le fix

### Test 3 (Generation) échoue:

**Problème**: Répétition infinie
**Cause possible**: Temperature trop basse ou problème de sampling
**Solution**: Vérifier la fonction `generate` dans `src/model.py`

**Problème**: Crash durant génération
**Cause possible**: NaN dans logits
**Solution**: Vérifier la protection NaN dans `generate`

---

## 🔍 Vérification Détaillée (Optionnel)

Si vous voulez vérifier manuellement l'alignement des labels:

```bash
python scripts/test_fix.py
```

Ce script teste spécifiquement l'alignement:
```
TEST 1: Vérification de l'alignement des labels

Après collator:
input_ids: [464, 3797, 3332, ...]
labels:    [3797, 3332, 319, ...]

Vérification alignement:
  Position 0: input='The' -> label=' cat' (expected ' cat') ✅
  Position 1: input=' cat' -> label=' sat' (expected ' sat') ✅
  Position 2: input=' sat' -> label=' on' (expected ' on') ✅

✅ Alignement: 7/7 positions correctes
✅ TEST 1 RÉUSSI: Les labels sont correctement alignés!
```

---

## 📈 Timeline de Validation

Après avoir relancé le training, testez à ces checkpoints:

### Checkpoint 2000 (~1h):
```bash
# Perplexité
python scripts/eval_perplexity.py --checkpoint out_slga/ckpt_2000

# Attendu: PPL ~800-2000
# Si > 3000 → Problème persiste

# Génération
python scripts/generate.py --checkpoint out_slga/ckpt_2000 \
    --prompt "The cat sat on the" --max-tokens 30

# Attendu: Mots anglais reconnaissables
# Si gibberish → Problème persiste
```

### Checkpoint 10000 (~5h):
```bash
# Perplexité
python scripts/eval_perplexity.py --checkpoint out_slga/ckpt_10000

# Attendu: PPL ~150-400
# Génération devrait être plus cohérente
```

---

## 📞 Si Problème Persiste

Si après le fix et le restart, les résultats ne s'améliorent pas:

1. **Vérifier le fix est bien appliqué**:
```bash
grep -n "labels_shifted = labels\[:, :-1\]" scripts/train.py
# Devrait afficher: 102:    labels_shifted = labels[:, :-1].contiguous()
```

2. **Vérifier les tests passent**:
```bash
python scripts/test_complete.py
# Tous les tests doivent afficher ✅
```

3. **Partager les métriques**:
- Console output des steps 50, 100, 500
- Perplexité au ckpt_2000
- Génération au ckpt_2000

---

## ✅ Checklist Complète

Avant de relancer:
- [ ] Tests complets passés: `python scripts/test_complete.py`
- [ ] Training actuel arrêté (Ctrl+C)
- [ ] Checkpoints nettoyés: `bash scripts/clean_restart.sh`
- [ ] Config copiée: `cp config_3090.yaml config.yaml`
- [ ] Fix vérifié: `grep "labels\[:, :-1\]" scripts/train.py`

Après 1h de training (step 2000):
- [ ] PPL ~800-2000 (au lieu de ~4424)
- [ ] Génération produit des mots cohérents
- [ ] Loss descend régulièrement
- [ ] TensorBoard affiche les métriques

---

## 🎯 Résumé

**Test rapide**: `python scripts/test_complete.py` (2-3 min)

**Si réussi**: Nettoyer + relancer training

**Validation**: Tester génération et perplexité au step 2000

**Attendu**: PPL ~800-2000, génération cohérente (au lieu de gibberish)

**Ce fix devrait résoudre complètement le problème!** 🎉
