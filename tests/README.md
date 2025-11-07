# SLGA Test Suite

Suite de tests complète pour valider les correctifs du projet SLGA-Plus.

## 📁 Structure

```
tests/
├── __init__.py              # Package initialization
├── test_slga.py             # Tests module SLGA (15 tests)
├── test_landmarks.py        # Tests sélection landmarks (17 tests)
├── test_model.py            # Tests architecture modèle (19 tests)
├── test_training.py         # Tests pipeline d'entraînement
├── TEST_REPORT.md           # Rapport détaillé des résultats
└── README.md                # Ce fichier
```

## 🚀 Exécution Rapide

```bash
# Tous les tests
pytest tests/ -v

# Avec coverage
pytest tests/ --cov=src --cov-report=html

# Test spécifique
pytest tests/test_slga.py::TestSLGAModule::test_mask_caching -v

# En parallèle (rapide)
pytest tests/ -n auto
```

## 📊 Résultats

**Status**: ✅ **51/51 tests passés (100%)**

**Coverage**:
- SLGA Module: 84%
- Model: 67%
- Landmarks: 43%

## 🧪 Tests Principaux

### test_slga.py
Valide les 3 bugs critiques:
1. ✅ Validation de paramètres (BUG #1)
2. ✅ Cache de masques (BUG #2)
3. ✅ Top-K déterministe (BUG #3)

```bash
pytest tests/test_slga.py -v
```

### test_landmarks.py
Teste la sélection de landmarks:
- Temperature decay
- Loss functions (spacing, sparsity)
- Optimisation et convergence

```bash
pytest tests/test_landmarks.py -v
```

### test_model.py
Teste l'architecture complète:
- Configuration
- Forward/backward pass
- Génération de texte
- Persistence

```bash
pytest tests/test_model.py -v
```

## 📈 Rapport de Coverage

Le rapport HTML est généré dans `htmlcov/`:

```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## 🐛 Tests de Régression

Pour valider que les bugs ne reviennent pas:

```bash
# Bug #1: Validation
pytest tests/test_slga.py::TestSLGAModule::test_parameter_validation -v

# Bug #2: Cache
pytest tests/test_slga.py::TestSLGAModule::test_mask_caching -v

# Bug #3: Determinisme
pytest tests/test_slga.py::TestSLGAModule::test_deterministic_topk -v
```

## 🔧 Dépendances

```bash
pip install pytest pytest-cov
```

## 📝 Ajouter des Tests

1. Créer un fichier `test_*.py` dans `tests/`
2. Utiliser les classes de test existantes comme exemples
3. Exécuter `pytest tests/ -v` pour vérifier

### Template

```python
import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slga import SLGAModule

class TestNewFeature:
    def test_something(self):
        module = SLGAModule(embed_dim=384, num_heads=6, local_window=128, global_k=24)
        x = torch.randn(2, 256, 384)
        out = module(x)
        assert out.shape == (2, 256, 384)
```

## 📖 Documentation

Voir [TEST_REPORT.md](TEST_REPORT.md) pour:
- Rapport détaillé des tests
- Analyse de coverage
- Recommandations d'amélioration

## ✨ Commandes Utiles

```bash
# Tests en mode verbose
pytest tests/ -v

# Tests avec sortie détaillée
pytest tests/ -vv

# Arrêter au premier échec
pytest tests/ -x

# Tests marqués comme slow
pytest tests/ -m "not slow"

# Coverage par fichier
pytest tests/ --cov=src --cov-report=term-missing

# Rapport XML (pour CI)
pytest tests/ --cov=src --cov-report=xml
```
