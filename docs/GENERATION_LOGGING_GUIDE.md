# Guide du Système de Logging des Générations

## 📋 Vue d'Ensemble

Le script `generate.py` a été amélioré pour **logger toutes les générations** avec métadonnées complètes, sans jamais écraser les fichiers précédents.

### Nouveautés

✅ **Fichiers uniques** avec timestamp pour chaque génération
✅ **Métadonnées complètes** (config, checkpoint, paramètres)
✅ **Historique centralisé** en format JSONL
✅ **Aucune perte de données** - tous les appels sont conservés
✅ **Script de consultation** pour analyser l'historique

---

## 🚀 Utilisation

### Génération Simple

```bash
python scripts/generate.py \
    --checkpoint out_slga_fineweb/ckpt_4000 \
    --config config/config_fineweb_edu.yaml \
    --prompt "The capital of France is" \
    --max-tokens 20 \
    --temperature 0.8
```

### Sortie

Le script crée **3 fichiers** :

```
✓ Output saved to:
  - Full log: out_slga_fineweb/generation_20251027_165457_step4000.txt
  - History: out_slga_fineweb/generation_history.jsonl
  - Quick view: out_slga_fineweb/generated_sample.txt
```

---

## 📁 Structure des Fichiers

### 1. Fichier Unique avec Timestamp

**Format** : `generation_YYYYMMDD_HHMMSS_stepXXXX.txt`

**Contenu** : Log complet avec toutes les métadonnées

```
================================================================================
SLGA GENERATION LOG
================================================================================

Timestamp: 2025-10-27T16:54:57.303529
Generation time: 8.14s

--- CHECKPOINT INFO ---
Path: out_slga_fineweb/ckpt_4000
Step: 4000
Checkpoint saved: 2025-10-27T06:40:35.512949
Parameters: 153 tensors
First param mean: -0.000454

--- MODEL CONFIG ---
vocab_size: 50257
max_seq_len: 2048
embed_dim: 512
num_heads: 8
...

--- GENERATION PARAMS ---
Temperature: 0.8
Top-K: 40
Top-P: disabled
Max tokens: 10
Device: cuda

================================================================================
PROMPT:
The capital of France is
================================================================================

================================================================================
GENERATED TEXT:
Paris, the largest city in...
================================================================================
```

### 2. Historique Centralisé (JSONL)

**Format** : `generation_history.jsonl`

**Contenu** : Une ligne JSON par génération (append mode)

```jsonl
{"timestamp": "2025-10-27T16:54:57", "prompt": "...", "generated_text": "...", "checkpoint": {...}, ...}
{"timestamp": "2025-10-27T17:02:15", "prompt": "...", "generated_text": "...", "checkpoint": {...}, ...}
```

**Avantages** :
- ✅ Jamais écrasé (append only)
- ✅ Facile à parser avec jq ou Python
- ✅ Toutes les métadonnées disponibles
- ✅ Peut être importé dans des outils d'analyse

### 3. Fichier Legacy (Compatibilité)

**Format** : `generated_sample.txt`

**Contenu** : Juste le texte généré (écrasé à chaque fois)

---

## 🔍 Consulter l'Historique

### Script Interactif

```bash
# Voir tout l'historique
python scripts/view_generation_history.py out_slga_fineweb

# Voir les 5 dernières générations
python scripts/view_generation_history.py out_slga_fineweb --last 5

# Filtrer par step
python scripts/view_generation_history.py out_slga_fineweb --step 4000

# Rechercher un prompt
python scripts/view_generation_history.py out_slga_fineweb --search "capital"

# Voir détails d'une entrée
python scripts/view_generation_history.py out_slga_fineweb --detail 3
```

### Sortie Exemple

```
================================================================================
HISTORIQUE DES GÉNÉRATIONS (5 entrées)
================================================================================

[1] 2025-10-27 16:54:57
    Checkpoint: step 4000
    Prompt: Test logging system:
    Generated: Test logging system:  food your food your your essential...
    Params: temp=0.8, top_k=40, top_p=None
    Time: 8.14s

[2] 2025-10-27 17:02:15
    Checkpoint: step 4000
    Loss: 5.7318
    Prompt: The capital of France is
    Generated: The capital of France is and the the and the and...
    Params: temp=0.8, top_k=80, top_p=None
    Time: 7.52s

...

================================================================================
STATISTIQUES
================================================================================
Total générations: 5
Checkpoints utilisés: 2 (step 1000, step 4000)
Temps total: 42.3s
Temps moyen: 8.46s
```

---

## 📊 Métadonnées Disponibles

### Checkpoint Info

- `checkpoint_path` : Chemin du checkpoint
- `step` : Step d'entraînement
- `loss` : Loss au moment du checkpoint
- `timestamp` : Date de création du checkpoint
- `num_parameters` : Nombre de tensors
- `first_param_mean` : Moyenne du premier param (sanity check)

### Model Config

Tous les paramètres du modèle :
- `vocab_size`, `max_seq_len`, `embed_dim`, etc.
- Configuration SLGA (landmarks, attention, etc.)
- Architecture complète

### Generation Params

- `temperature` : Température de sampling
- `top_k` : Top-K filtering
- `top_p` : Nucleus sampling
- `max_tokens` : Tokens maximum générés
- `device` : Device utilisé

### Résultats

- `prompt` : Prompt utilisé
- `generated_text` : Texte généré
- `generation_time_seconds` : Temps de génération
- `model_params_millions` : Taille du modèle

---

## 💡 Cas d'Usage

### Comparer Différents Checkpoints

```bash
# Générer avec plusieurs checkpoints
for step in 1000 2000 4000 8000; do
    python scripts/generate.py \
        --checkpoint out_slga_fineweb/ckpt_${step} \
        --config config/config_fineweb_edu.yaml \
        --prompt "Once upon a time" \
        --max-tokens 30 \
        --temperature 0.8
done

# Consulter l'historique
python scripts/view_generation_history.py out_slga_fineweb
```

### Tester Différents Paramètres

```bash
# Tester différentes températures
for temp in 0.5 0.7 0.9 1.2; do
    python scripts/generate.py \
        --checkpoint out_slga_fineweb/ckpt_4000 \
        --config config/config_fineweb_edu.yaml \
        --prompt "The future of AI is" \
        --max-tokens 50 \
        --temperature ${temp}
done

# Filtrer par step
python scripts/view_generation_history.py out_slga_fineweb --step 4000
```

### Analyse de Qualité

```python
# Charger et analyser en Python
import json

history = []
with open("out_slga_fineweb/generation_history.jsonl") as f:
    for line in f:
        history.append(json.loads(line))

# Analyser par step
by_step = {}
for entry in history:
    step = entry["checkpoint"]["step"]
    if step not in by_step:
        by_step[step] = []
    by_step[step].append(entry)

# Calculer métrique de qualité (exemple simple)
for step, entries in by_step.items():
    avg_length = sum(len(e["generated_text"]) for e in entries) / len(entries)
    print(f"Step {step}: {len(entries)} générations, avg length {avg_length:.1f}")
```

---

## 🔧 Avancé : Parser le JSONL

### Avec Python

```python
import json

with open("out_slga_fineweb/generation_history.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        print(f"{entry['timestamp']}: {entry['prompt']}")
```

### Avec jq (si installé)

```bash
# Extraire tous les prompts
cat out_slga_fineweb/generation_history.jsonl | jq -r '.prompt'

# Filtrer par step
cat out_slga_fineweb/generation_history.jsonl | jq 'select(.checkpoint.step == 4000)'

# Statistiques
cat out_slga_fineweb/generation_history.jsonl | jq '.generation_time_seconds' | awk '{sum+=$1} END {print "Avg:", sum/NR}'
```

### Avec pandas

```python
import pandas as pd
import json

# Charger historique
data = []
with open("out_slga_fineweb/generation_history.jsonl") as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# Analyser
print(df.groupby("checkpoint.step")["generation_time_seconds"].mean())
print(df["generation_params"].apply(lambda x: x["temperature"]).value_counts())
```

---

## 📈 Bonnes Pratiques

### Organisation

1. **Séparer par dataset/config**
   ```bash
   # FineWeb
   python scripts/generate.py --checkpoint out_slga_fineweb/ckpt_4000 ...

   # Wikipedia
   python scripts/generate.py --checkpoint out_slga/ckpt_1000 ...
   ```

2. **Archiver périodiquement**
   ```bash
   # Sauvegarder l'historique
   cp out_slga_fineweb/generation_history.jsonl \
      archives/generation_history_$(date +%Y%m%d).jsonl
   ```

3. **Nettoyer les vieux fichiers**
   ```bash
   # Garder seulement les 100 derniers fichiers de génération
   ls -t out_slga_fineweb/generation_*.txt | tail -n +101 | xargs rm -f
   ```

### Tests Systématiques

Créer un script de test :

```bash
#!/bin/bash
# test_generations.sh

CHECKPOINT=$1
CONFIG=$2

PROMPTS=(
    "The capital of France is"
    "Once upon a time"
    "In the year 2050"
    "The meaning of life is"
)

for prompt in "${PROMPTS[@]}"; do
    python scripts/generate.py \
        --checkpoint "$CHECKPOINT" \
        --config "$CONFIG" \
        --prompt "$prompt" \
        --max-tokens 30 \
        --temperature 0.8
done

# Voir résultats
python scripts/view_generation_history.py
```

---

## 🐛 Dépannage

### Historique JSONL Corrompu

Si une ligne est invalide :

```python
import json

valid_lines = []
with open("generation_history.jsonl") as f:
    for i, line in enumerate(f, 1):
        try:
            entry = json.loads(line)
            valid_lines.append(entry)
        except json.JSONDecodeError:
            print(f"Ligne {i} invalide, ignorée")

# Réécrire fichier propre
with open("generation_history_clean.jsonl", "w") as f:
    for entry in valid_lines:
        f.write(json.dumps(entry) + "\n")
```

### Fichiers Trop Nombreux

```bash
# Compter les fichiers
ls out_slga_fineweb/generation_*.txt | wc -l

# Archiver les anciens
mkdir -p archives
mv out_slga_fineweb/generation_202510* archives/
```

---

## 📚 Références

- **Script principal** : `scripts/generate.py`
- **Script de consultation** : `scripts/view_generation_history.py`
- **Fichiers générés** :
  - `{output_dir}/generation_{timestamp}_step{step}.txt` (logs complets)
  - `{output_dir}/generation_history.jsonl` (historique centralisé)
  - `{output_dir}/generated_sample.txt` (fichier legacy)

---

## ✅ Résumé

**Avant** :
- ❌ Un seul fichier `generated_sample.txt`
- ❌ Écrasé à chaque appel
- ❌ Pas de métadonnées
- ❌ Impossible de comparer

**Maintenant** :
- ✅ Fichiers uniques avec timestamp
- ✅ Historique complet jamais effacé
- ✅ Toutes les métadonnées disponibles
- ✅ Script de consultation interactif
- ✅ Format JSONL pour analyse
- ✅ Traçabilité complète

**Chaque génération est maintenant un enregistrement permanent avec contexte complet !** 📝
