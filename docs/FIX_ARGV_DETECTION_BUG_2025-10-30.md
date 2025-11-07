# Fix: Argv Detection Bug for Equals-Sign Format (2025-10-30)

## 🐛 Problem Identified

**Bug #7**: La détection d'arguments explicites échouait avec le format `--top-k=value`.

### Symptôme
```bash
# ❌ Warning PAS déclenché (bug!)
python scripts/generate.py --checkpoint ckpt --top-k=50 --top-p=0.9

# ✅ Warning déclenché (fonctionnait)
python scripts/generate.py --checkpoint ckpt --top-k 50 --top-p 0.9
```

### Cause Racine

**Code bugué** (`scripts/generate.py:325-326`):
```python
# ❌ AVANT: Détection par égalité exacte
user_set_top_k = '--top-k' in sys.argv or '--top_k' in sys.argv
user_set_top_p = '--top-p' in sys.argv or '--top_p' in sys.argv
```

**Pourquoi ça ne fonctionnait pas?**

`sys.argv` avec format égal:
```python
sys.argv = ['generate.py', '--checkpoint', 'ckpt', '--top-k=50', '--top-p=0.9']
#                                                   ^^^^^^^^^^^  ^^^^^^^^^^^
#                                                   Un seul token chacun!

'--top-k' in sys.argv  # False! (cherche '--top-k' exact, pas '--top-k=50')
'--top-p' in sys.argv  # False! (cherche '--top-p' exact, pas '--top-p=0.9')
```

**Résultat**: Le warning n'était jamais déclenché avec `--top-k=value` format.

---

## ✅ Solution Appliquée

### Détection Robuste avec `startswith()`

**Nouveau code** (`scripts/generate.py:327-330`):
```python
# ✅ APRÈS: Détection par préfixe
user_set_top_k = any(arg.startswith('--top-k') or arg.startswith('--top_k')
                     for arg in sys.argv)
user_set_top_p = any(arg.startswith('--top-p') or arg.startswith('--top_p')
                     for arg in sys.argv)
```

**Pourquoi ça fonctionne?**

```python
sys.argv = ['generate.py', '--checkpoint', 'ckpt', '--top-k=50', '--top-p=0.9']

# Pour chaque argument:
'--top-k=50'.startswith('--top-k')  # ✅ True!
'--top-p=0.9'.startswith('--top-p')  # ✅ True!

# any() retourne True si au moins un match
user_set_top_k = True  # ✅ Détecté!
user_set_top_p = True  # ✅ Détecté!
```

---

## 🧪 Tests Complets

### Test Suite: Tous les Formats d'Arguments

**Fichier**: `tests/test_argv_detection_formats.py`

**Formats testés**:
1. ✅ `--top-k 40` (espace)
2. ✅ `--top-k=40` (égal) ← **Le bug corrigé**
3. ✅ `--top_k 40` (underscore + espace)
4. ✅ `--top_k=40` (underscore + égal)
5. ✅ Format mixte (`--top-k=40 --top_p 0.9`)
6. ✅ Un seul argument
7. ✅ Aucun argument

**Résultats**:
```bash
$ python tests/test_argv_detection_formats.py

✓ PASS: Space-separated (--top-k 40, --top-p 0.9)
✓ PASS: Equals-separated (--top-k=40, --top-p=0.9)  ← FIX VALIDÉ
✓ PASS: Underscore space (--top_k 40, --top_p 0.9)
✓ PASS: Underscore equals (--top_k=40, --top_p=0.9)
✓ PASS: Mixed format (--top-k=40 --top_p 0.9)
✓ PASS: Only --top-k=50
✓ PASS: Only --top-p=0.8
✓ PASS: No sampling arguments

✓ ALL TESTS PASSED - Detection handles all formats
```

---

## 📋 Comportement Avant/Après

### Scénario 1: Format avec égal (le bug)

**AVANT** (bugué):
```bash
$ python scripts/generate.py --checkpoint ckpt --top-k=50 --top-p=0.9
# ❌ Pas de warning (bug silencieux)
# Génération procède sans guidance utilisateur
```

**APRÈS** (corrigé):
```bash
$ python scripts/generate.py --checkpoint ckpt --top-k=50 --top-p=0.9
⚠️  PARAMETER WARNINGS
  Using both top_k=50 and top_p=0.9 simultaneously.
  Recommendation: Use only one for most use cases
# ✅ Warning approprié!
```

---

### Scénario 2: Format avec espace (fonctionnait déjà)

**AVANT ET APRÈS** (identique):
```bash
$ python scripts/generate.py --checkpoint ckpt --top-k 50 --top-p 0.9
⚠️  PARAMETER WARNINGS
  Using both top_k=50 and top_p=0.9 simultaneously...
# ✅ Toujours fonctionnel
```

---

### Scénario 3: Format mixte

**AVANT** (bugué):
```bash
$ python scripts/generate.py --checkpoint ckpt --top-k=50 --top-p 0.9
# ❌ Pas de warning (--top-k=50 non détecté)
```

**APRÈS** (corrigé):
```bash
$ python scripts/generate.py --checkpoint ckpt --top-k=50 --top-p 0.9
⚠️  PARAMETER WARNINGS
  Using both top_k=50 and top_p=0.9 simultaneously...
# ✅ Warning déclenché!
```

---

## 💡 Détails Techniques

### Méthode: `any()` + `startswith()`

**Avantages**:
- ✅ Détecte TOUS les formats (`--top-k`, `--top-k=40`, `--top_k`, `--top_k=40`)
- ✅ Simple et lisible
- ✅ Performant (O(n) avec short-circuit)
- ✅ Pas d'expressions régulières complexes
- ✅ Robuste aux variations

**Code complet**:
```python
user_set_top_k = any(arg.startswith('--top-k') or arg.startswith('--top_k')
                     for arg in sys.argv)
user_set_top_p = any(arg.startswith('--top-p') or arg.startswith('--top_p')
                     for arg in sys.argv)

if (user_set_top_k and user_set_top_p and
    args.top_k is not None and args.top_k > 0 and
    args.top_p is not None and args.top_p < 1.0):
    warnings.append(...)
```

---

### Alternative Non Retenue: Regex

```python
import re

# Plus complexe, moins maintenable
user_set_top_k = any(re.match(r'--top[-_]k(=.*)?$', arg) for arg in sys.argv)
```

**Rejeté car**:
- Plus difficile à lire
- Surcharge inutile (regex overkill)
- `startswith()` suffit et est plus clair

---

### Alternative Non Retenue: Parser Introspection

```python
# Comparer avec defaults du parser
user_set_top_k = (args.top_k != parser.get_default('top_k'))
```

**Rejeté car**:
- Ne détecte pas si utilisateur passe la valeur par défaut explicitement
- Exemple: `--top-k=80` (default) → pas détecté
- Moins précis que `sys.argv` inspection

---

## 🔧 Fichiers Modifiés

- `scripts/generate.py`:
  - Ligne 327-330: Validation (détection `startswith`)
  - Ligne 529-532: Logging runtime (détection `startswith`)

---

## 📚 Tests Créés

- `tests/test_argv_detection_formats.py` - Test de tous les formats d'arguments

---

## ✅ Checklist de Vérification

- [x] Bug identifié et compris
- [x] Fix appliqué avec `startswith()`
- [x] Tests créés (8 cas de test)
- [x] Tous formats testés (espace, égal, underscore, mixte)
- [x] Tests existants toujours passent
- [x] Pas de régression
- [x] Pas de breaking changes
- [x] Documentation complète

---

## 📊 Impact

### Robustesse

| Format | Avant | Après |
|--------|-------|-------|
| `--top-k 40` | ✅ | ✅ |
| `--top-k=40` | ❌ Bug | ✅ Fixed |
| `--top_k 40` | ✅ | ✅ |
| `--top_k=40` | ❌ Bug | ✅ Fixed |
| Mixte | ❌ Bug | ✅ Fixed |

### Expérience Utilisateur

- ✅ Warning déclenché correctement dans TOUS les cas
- ✅ Pas de comportement silencieux inattendu
- ✅ Guidance cohérente peu importe le format
- ✅ Utilisateurs protégés contre configuration sous-optimale

---

## 🙏 Remerciements

**Merci à l'utilisateur pour**:
- ✅ Avoir identifié ce bug subtil avec le format `--top-k=value`
- ✅ Avoir fourni un diagnostic précis
- ✅ Avoir suggéré `startswith()` comme solution

**Ce rapport montre l'importance de tester TOUS les formats d'arguments!**

---

## 🎯 Leçons Apprises

### Best Practice: Détection d'Arguments CLI

**DO**:
- ✅ Utiliser `arg.startswith()` pour détecter options
- ✅ Tester TOUS les formats (espace, égal, underscore)
- ✅ Créer des tests exhaustifs
- ✅ Documenter les cas d'edge

**DON'T**:
- ❌ Utiliser égalité exacte (`in sys.argv`)
- ❌ Assumer un seul format d'argument
- ❌ Oublier les variations (tiret vs underscore)
- ❌ Tester seulement le "happy path"

---

## 📖 Références

- `docs/FIX_VALIDATION_BUGS_2025-10-30.md` - Bugs de validation précédents
- `docs/CHANGELOG_HIVE_MIND_2025-10-30.md` - Historique complet
- `tests/test_argv_detection_formats.py` - Suite de tests

---

**Status**: ✅ Fix appliqué et validé
**Date**: 2025-10-30
**Version**: 1.3 (argv detection robustness)
**Bug #**: 7
