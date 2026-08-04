SLGA: Sparse Local-Global Attention Transformer

[![Python 3.8+] (https://img.shields.io/badge/python-3.8+-blue.svg) ] (https://www.python.org/downloads/)
[![PyTorch] (https://img.shields.io/badge/PyTorch-2.0+-orange.svg) ] (https://pytorch.org/)
[![License: MIT] (https://img.shields.io/badge/License-MIT-yellow.svg) ] (https://opensource.org/licenses/MIT)

**SLGA** - Complete implementation of a Transformer LLM with local-global sparse attention for long sequences (optimized RTX 3090 24GB).

## 🎯 Features

Local Attention**: Sliding Window O (L·W) with causal mask
**Global attention**: Selective cache O (L·G) with top-K
- **Fusion Apprise**: Local/global dynamic ging
- **Landmarks Learned**: Differentiable selection via Gumbel-Softmax
- **Optimizations**: AMP, gradient checkpointing, curriculum learning

## 📦 Installation

"bash
Cloning the Project
Git Clone https://github.com/JacquesGariepy/SLGA.git
CD Slga

# Create Virtual Environment
Python3 - m venv.venv
source .venv/bin/activate # Windows: .venv\Scripts\activate

# Install
pip install -r requirements.txt

# OR install in development mode
pip install -e .
"'

## 🚀 Quick Start

"bash
#1. Download the dataset (optional, done automatically)
python scripts/download_dataset.py

#2. Train the model
python scripts/train.py --config ./config.yaml

#3. Evaluate
python scripts/eval_perplexity.py --checkpoint out_slga/ckpt_50000

#4. Generate text
python scripts/generate.py --checkpoint out_slga/ckpt_50000 --prompt "Hello"
"'

## 📁 Project Structure

"'
slga_project/
src/# Main source code
slga.py # SLGA Attention Module
Landmarks.py # Selection of landmarks
Model.py # Transformer Model
Data.py # Data loading
utilities.py # Utilities
Scripts/# Execution scripts
Train.py # Training
eval_perplexity.py # Evaluation
generate.py # Generation
Download_dataset.py # Download Data
configs/# Configuration files
config_default.yaml # Default Config # Default
config_small.yaml # Light Config (test)
config_large.yaml # High capacity configuration
Tests/# Unit Tests
Docs/# Documentation
Data/# Data (uploaded automatically)
"'

## 📊 Expected Performance (RTX 3090)

- **Throughput**: > 4000 tokens/sec (training)
- **Perplexity**: <12 on Wikipedia (100K steps)
**Memory**: ~18GB (batch_size=4, seq_len=2048)
- **Long-QA F1**: > 72% (SCROLLS benchmark)

## 🔧 Configuration

Three pre-defined configurations:

"bash
# Quick test (light, 30 min)
python scripts/train.py --config configs/config_small.yaml

# Production (default, 2-3 days)
python scripts/train.py --config configs/config_default.yaml

# High capacity (if > 24GB VRAM)
python scripts/train.py --config configs/config_large.yaml
"'

## 📚 Documentation

- [Quick Start] (docs/quickstart.md) - Step by step guide
- [Architecture] (docs/architecture.md) - Technical details
- [Troubleshooting] (docs/troubleshooting.md) - Problem solving

## 🧪 Tests

"bash
# All Tests
Python - m pytest tests/

# Specific Test
python -m pytest tests/test_slga.py
"'

## 🤝 Contribution

Contributions are welcome! See [CONTRIBUTING.md] (CONTRIBUTING.md).

## 📄 License

MIT License - see [LICENSE] (LICENSE)

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
