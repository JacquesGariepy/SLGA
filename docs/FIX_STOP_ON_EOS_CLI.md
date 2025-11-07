# Fix: --stop-on-eos CLI Argument Bug

## 🐛 Problem Identified

**Original Issue**: L'argument `--stop-on-eos` était déclaré avec:
```python
parser.add_argument(
    "--stop-on-eos",
    action="store_true",
    default=True,  # ❌ PROBLÈME
    help="Stop generation when EOS token is encountered",
)
```

**Conséquence**:
- `action="store_true"` + `default=True` → Toujours `True`
- Impossible de désactiver l'arrêt sur EOS
- `--stop-on-eos False` lèverait une erreur argparse

## ✅ Solution Appliquée

Utilisation d'un **groupe mutuellement exclusif** avec deux options complémentaires:

```python
# ✅ CORRECT: Groupe mutuellement exclusif
eos_group = parser.add_mutually_exclusive_group()

eos_group.add_argument(
    "--stop-on-eos",
    dest="stop_on_eos",
    action="store_true",
    default=True,
    help="Stop generation when EOS token is encountered (default)",
)

eos_group.add_argument(
    "--no-stop-on-eos",
    dest="stop_on_eos",
    action="store_false",
    help="Continue generation even after EOS token",
)
```

## 📋 Comportement

| Commande | Valeur `stop_on_eos` | Description |
|----------|---------------------|-------------|
| *(aucun arg)* | `True` | Par défaut: arrêt sur EOS activé |
| `--stop-on-eos` | `True` | Explicite: arrêt sur EOS |
| `--no-stop-on-eos` | `False` | Désactiver l'arrêt sur EOS |
| `--stop-on-eos --no-stop-on-eos` | ❌ Erreur | Arguments conflictuels |

## 🧪 Exemples d'Utilisation

### Par défaut (arrêt sur EOS)
```bash
python scripts/generate.py \
    --checkpoint out_slga/ckpt_11000 \
    --config config.yaml \
    --prompt "The future of AI"
# stop_on_eos = True (défaut)
```

### Arrêt explicite sur EOS
```bash
python scripts/generate.py \
    --checkpoint out_slga/ckpt_11000 \
    --config config.yaml \
    --prompt "The future of AI" \
    --stop-on-eos
# stop_on_eos = True (explicite)
```

### Continuer après EOS (nouveau!)
```bash
python scripts/generate.py \
    --checkpoint out_slga/ckpt_11000 \
    --config config.yaml \
    --prompt "The future of AI" \
    --no-stop-on-eos
# stop_on_eos = False
# La génération continue même après avoir généré EOS
```

## ✅ Validation

Test automatisé: `tests/test_stop_on_eos_cli.py`

```bash
python tests/test_stop_on_eos_cli.py
```

Résultats:
```
✓ PASS: Default (no args) → stop_on_eos = True
✓ PASS: Explicit --stop-on-eos → stop_on_eos = True
✓ PASS: Explicit --no-stop-on-eos → stop_on_eos = False
✓ PASS: Correctly rejects conflicting args
```

## 🎯 Avantages de cette Solution

1. **Comportement par défaut préservé**: Arrêt sur EOS activé par défaut (comportement souhaitable)
2. **Contrôle utilisateur**: Option claire `--no-stop-on-eos` pour désactiver
3. **Protection contre erreurs**: Arguments mutuellement exclusifs empêchent conflits
4. **Auto-documenté**: Les noms d'arguments sont clairs et symétriques
5. **Idiomatique argparse**: Pattern standard pour options booléennes avec défaut True

## 📝 Alternative Non Retenue

Une autre approche aurait été:
```python
parser.add_argument("--stop-on-eos", action="store_true", default=False)
```

**Rejetée car**:
- Change le comportement par défaut (pas d'arrêt sur EOS)
- Moins intuitif (il faudrait toujours passer --stop-on-eos)
- La solution avec `--no-stop-on-eos` est plus élégante

## 🔧 Fichiers Modifiés

- `scripts/generate.py` (lignes 395-409)
- `tests/test_stop_on_eos_cli.py` (nouveau fichier de test)

## 📚 Documentation Mise à Jour

- `docs/HIVE_MIND_FIXES_2025-10-30.md` (à mettre à jour)
- `docs/QUICK_REFERENCE_HIVE_MIND_FIXES.md` (à mettre à jour)

---

**Merci à l'utilisateur pour avoir repéré ce bug !** 🎉

Cette correction améliore l'ergonomie de l'interface CLI et suit les bonnes pratiques argparse.
