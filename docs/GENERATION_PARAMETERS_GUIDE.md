# Guide des Paramètres de Génération SLGA

## 📊 Progression de l'Entraînement (Step 4000)

**État actuel** : Step 4000/100,000 (4%)
- **Loss** : 5.73 (meilleur: 5.45)
- **Perplexité** : 308.5 (meilleur: 231.9)
- **Qualité** : Début d'apprentissage, toujours répétitif

---

## 🎯 Paramètres de Génération Recommandés par Niveau

### Step 2000-5000 (2-5%) - Apprentissage Précoce

**Problème** : Modèle apprend encore les patterns de base

✅ **RECOMMANDÉ** :
```bash
--temperature 0.8 à 1.2
--top-k 80 à 120
--top-p 0.90 à 0.95
```

❌ **À ÉVITER** :
```bash
--temperature < 0.5  # Trop restrictif, génère 0 tokens
--top-k < 50         # Trop limité
--top-p < 0.8        # Élimine trop de choix
```

**Exemple fonctionnel** :
```bash
python scripts/generate.py \
    --checkpoint out_slga_fineweb/ckpt_4000 \
    --prompt "Once upon a time" \
    --max-tokens 30 \
    --temperature 1.0 \
    --top-k 100
```

---

### Step 5000-10000 (5-10%) - Cohérence Émergente

**Attendu** : Début de phrases cohérentes

✅ **RECOMMANDÉ** :
```bash
--temperature 0.7 à 1.0
--top-k 60 à 100
--top-p 0.85 à 0.95
```

**Exemple** :
```bash
python scripts/generate.py \
    --checkpoint out_slga_fineweb/ckpt_8000 \
    --prompt "The capital of France is" \
    --max-tokens 20 \
    --temperature 0.8 \
    --top-p 0.9
```

---

### Step 10000-25000 (10-25%) - Qualité Acceptable

**Attendu** : Phrases cohérentes, début de logique

✅ **RECOMMANDÉ** :
```bash
--temperature 0.6 à 0.9
--top-k 40 à 80
--top-p 0.85 à 0.92
```

**Exemple** :
```bash
python scripts/generate.py \
    --checkpoint out_slga_fineweb/ckpt_15000 \
    --prompt "Explain quantum physics:" \
    --max-tokens 50 \
    --temperature 0.7 \
    --top-k 50
```

---

### Step 25000-100000 (25-100%) - Production

**Attendu** : Qualité production, raisonnement complexe

✅ **RECOMMANDÉ** :
```bash
--temperature 0.5 à 0.8
--top-k 40 à 60
--top-p 0.90 à 0.95
```

**Exemple** :
```bash
python scripts/generate.py \
    --checkpoint out_slga_fineweb/ckpt_50000 \
    --prompt "Write a story about" \
    --max-tokens 100 \
    --temperature 0.7 \
    --top-p 0.92
```

---

## 🐛 Problèmes Fréquents et Solutions

### Problème 1 : Génère 0 tokens

**Symptômes** :
```
GENERATED TEXT:
The capital of France is
```
(Aucun nouveau token)

**Causes** :
- ❌ Temperature < 0.3
- ❌ Top-K < 30 combiné avec temperature basse
- ❌ Top-P < 0.7

**Solution** :
```bash
# AVANT (ne marche pas)
--temperature 0.1 --top-k 40

# APRÈS (fonctionne)
--temperature 0.8 --top-k 80
```

---

### Problème 2 : Répétition excessive

**Symptômes** :
```
your your your your your your your your
```

**Causes** :
- Modèle sous-entraîné (< step 5000)
- Température trop haute
- Pas de répétition penalty

**Solution** :
1. **Court terme** : Réduire température
   ```bash
   --temperature 0.7  # au lieu de 1.2
   ```

2. **Moyen terme** : Attendre step 8000-10000

3. **Long terme** : Ajouter repetition penalty dans le code

---

### Problème 3 : Sortie incohérente/aléatoire

**Symptômes** :
```
Test oral healthy mental health Your your you
```

**Causes** :
- Temperature > 1.5
- Top-P trop permissif (> 0.98)
- Modèle encore en apprentissage

**Solution** :
```bash
# Réduire température
--temperature 0.8 --top-p 0.92
```

---

## 📈 Comparaison des Résultats (Step 4000)

### Test 1 : Temperature 0.1 (ÉCHEC)
```bash
--temperature 0.1 --top-k 40
Output: (rien - 0 tokens générés)
```
❌ Trop restrictif

### Test 2 : Temperature 0.8 (SUCCÈS)
```bash
--temperature 0.8 --top-k 80
Output: "and the the and the and and. and are and and..."
```
✅ Génère des tokens (répétitif mais normal à 4%)

### Test 3 : Temperature 1.0 (SUCCÈS)
```bash
--temperature 1.0 --top-p 0.95
Output: "you your your your. your your your..."
```
✅ Génère des tokens (plus de diversité)

### Test 4 : Temperature 2.0 (SUCCÈS mais chaotique)
```bash
--temperature 2.0
Output: "oral healthy mental health Your your you your stress"
```
✅ Génère mais trop aléatoire

---

## 🎓 Comprendre les Paramètres

### Temperature (température)
- **Rôle** : Contrôle le caractère aléatoire
- **Valeurs** :
  - `0.1-0.4` : Très déterministe (éviter avant step 10K)
  - `0.5-0.8` : Équilibré
  - `0.9-1.2` : Plus créatif
  - `>1.5` : Trop aléatoire

### Top-K
- **Rôle** : Limite aux K tokens les plus probables
- **Valeurs** :
  - `<40` : Trop restrictif (éviter avant step 5K)
  - `40-80` : Équilibré
  - `80-120` : Plus de diversité
  - `>150` : Peu d'effet

### Top-P (nucleus sampling)
- **Rôle** : Cumul de probabilité
- **Valeurs** :
  - `<0.8` : Trop restrictif
  - `0.85-0.92` : Équilibré
  - `0.93-0.98` : Plus de diversité
  - `>0.98` : Trop permissif

---

## 🚀 Commandes Rapides pour Step 4000

### Test Basique
```bash
python scripts/generate.py \
    --checkpoint out_slga_fineweb/ckpt_4000 \
    --prompt "Hello world" \
    --max-tokens 15 \
    --temperature 0.9
```

### Test Pattern Recognition
```bash
python scripts/generate.py \
    --checkpoint out_slga_fineweb/ckpt_4000 \
    --prompt "Paris is the capital of France. London is" \
    --max-tokens 10 \
    --temperature 0.8 \
    --top-k 100
```

### Test Créatif
```bash
python scripts/generate.py \
    --checkpoint out_slga_fineweb/ckpt_4000 \
    --prompt "Once upon a time" \
    --max-tokens 30 \
    --temperature 1.1 \
    --top-p 0.93
```

### Test Factuel
```bash
python scripts/generate.py \
    --checkpoint out_slga_fineweb/ckpt_4000 \
    --prompt "The capital of France is" \
    --max-tokens 20 \
    --temperature 0.7 \
    --top-k 80
```

---

## 📊 Qualité Attendue par Step

| Step | Perplexité | Qualité Génération | Paramètres Optimaux |
|------|------------|-------------------|-------------------|
| 2000 | >500 | Tokens aléatoires | temp=1.0, k=100 |
| **4000** | **~300** | **Répétition de mots** | **temp=0.8-1.0, k=80** |
| 5000 | ~200 | Début de phrases | temp=0.8, k=60 |
| 8000 | ~100 | Phrases simples | temp=0.7, k=50 |
| 10000 | ~80 | Cohérence acceptable | temp=0.7, k=40 |
| 25000 | ~35 | Bonne qualité | temp=0.6, k=40 |
| 100000 | 15-25 | Production | temp=0.6, p=0.92 |

---

## ⚠️ Notes Importantes

### À Step 4000 (où vous êtes maintenant)

✅ **NORMAL** :
- Répétition excessive ("your your your")
- Pas de structure grammaticale
- Tokens fréquents dominants ("the", "and", "your")
- Perplexité ~300

❌ **ANORMAL** :
- Génération de 0 tokens (vérifier paramètres)
- Crash pendant génération
- GPU OOM

### Quand Tester Sérieusement

- ⏰ **Step 10,000** : Premier test significatif
- ⏰ **Step 25,000** : Qualité évaluable
- ⏰ **Step 100,000** : Benchmarks finaux

---

## 🔧 Debug Rapide

Si la génération ne marche pas :

```bash
# 1. Vérifier checkpoint existe
ls -lh out_slga_fineweb/ckpt_4000/model.pt

# 2. Tester avec paramètres sûrs
python scripts/generate.py \
    --checkpoint out_slga_fineweb/ckpt_4000 \
    --prompt "Test" \
    --max-tokens 10 \
    --temperature 1.0

# 3. Si 0 tokens : augmenter température
--temperature 1.5

# 4. Si trop répétitif : réduire température
--temperature 0.6
```

---

## 📚 Références

- **Fix Plan** : `/docs/GENERATION_FIX_PLAN.md`
- **Fixes Applied** : `/docs/FIXES_APPLIED_2025-10-26.md`
- **Training Diagnosis** : `/docs/TRAINING_DIAGNOSIS_2025-10-26.md`

---

**Dernière Mise à Jour** : 2025-10-26 (Step 4000)
**Prochaine Révision** : Step 10,000
**Status** : ✅ Génération fonctionnelle avec bons paramètres
