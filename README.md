# SLGA: Sparse Local-Global Attention Transformer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SLGA** - Implémentation complète d'un Transformer LLM avec attention sparse locale-globale pour séquences longues (optimisé RTX 3090 24GB).

## 🎯 Caractéristiques

- **Attention Locale**: Fenêtre glissante O(L·W) avec masque causal
- **Attention Globale**: Cache sélectif O(L·G) avec top-K
- **Fusion Apprise**: Gating dynamique local/global
- **Landmarks Appris**: Sélection différentiable via Gumbel-Softmax
- **Optimisations**: AMP, gradient checkpointing, curriculum learning

## 📦 Installation

```bash
# Cloner le projet
git clone https://github.com/JacquesGariepy/SLGA.git
cd slga

# Créer environnement virtuel
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Installer
pip install -r requirements.txt

# OU installer en mode développement
pip install -e .
```

## 🚀 Démarrage Rapide

```bash
# 1. Télécharger le dataset (optionnel, se fait automatiquement)
python scripts/download_dataset.py

# 2. Entraîner le modèle
python scripts/train.py --config ./config.yaml

# 3. Évaluer
python scripts/eval_perplexity.py --checkpoint out_slga/ckpt_50000

# 4. Générer du texte
python scripts/generate.py --checkpoint out_slga/ckpt_50000 --prompt "Hello"
```

## 📁 Structure du Projet

```
slga_project/
├── src/                    # Code source principal
│   ├── slga.py            # Module d'attention SLGA
│   ├── landmarks.py       # Sélection de landmarks
│   ├── model.py           # Modèle Transformer
│   ├── data.py            # Chargement de données
│   └── utils.py           # Utilitaires
├── scripts/               # Scripts d'exécution
│   ├── train.py          # Entraînement
│   ├── eval_perplexity.py # Évaluation
│   ├── generate.py       # Génération
│   └── download_dataset.py # Téléchargement données
├── configs/               # Fichiers de configuration
│   ├── config_default.yaml # Config par défaut
│   ├── config_small.yaml   # Config légère (test)
│   └── config_large.yaml   # Config haute capacité
├── tests/                 # Tests unitaires
├── docs/                  # Documentation
└── data/                  # Données (téléchargées automatiquement)
```

## 📊 Performance Attendue (RTX 3090)

- **Throughput**: >4000 tokens/sec (training)
- **Perplexité**: <12 sur Wikipedia (100K steps)
- **Memory**: ~18GB (batch_size=4, seq_len=2048)
- **Long-QA F1**: >72% (SCROLLS benchmark)

## 🔧 Configuration

Trois configurations pré-définies :

```bash
# Test rapide (léger, 30 min)
python scripts/train.py --config configs/config_small.yaml

# Production (par défaut, 2-3 jours)
python scripts/train.py --config configs/config_default.yaml

# Haute capacité (si >24GB VRAM)
python scripts/train.py --config configs/config_large.yaml
```

## 📚 Documentation

- [Démarrage Rapide](docs/QUICKSTART.md) - Guide étape par étape
- [Architecture](docs/ARCHITECTURE.md) - Détails techniques
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Résolution de problèmes

## 🧪 Tests

```bash
# Tous les tests
python -m pytest tests/

# Test spécifique
python -m pytest tests/test_slga.py
```

## 🤝 Contribution

Les contributions sont bienvenues ! Voir [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 Licence

MIT License - voir [LICENSE](LICENSE)

## 📞 Contact

- Issues: [GitHub Issues](https://github.com/yourusername/slga/issues)
- Email: your.email@example.com

## 🙏 Citation

```bibtex
@software{slga_plus_2024,
  title={SLGA: Efficient Sparse Local-Global Attention},
  author={Your Name},
  year={2024},
  url={https://github.com/JacquesGariepy/SLGA}
}
```
