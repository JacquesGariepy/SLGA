# SLGA: Sparse Local-Global Attention Transformer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SLGA** (Sparse Local-Global Attention) is a complete implementation of a causal Transformer Language Model designed specifically to handle extremely long sequences efficiently. By sparsifying the attention matrix, SLGA maintains high performance while significantly reducing VRAM requirements, making it fully optimized for hardware like the RTX 3090 (24GB).

## 🎯 Key Features

*   **Local Attention:** Sliding window attention $O(L \times W)$ with causal masking to capture immediate syntactic and contextual relationships.
*   **Global Attention:** Selective global cache $O(L \times G)$ utilizing a Top-K routing mechanism to track long-range dependencies.
*   **Learned Fusion:** Dynamic, learned gating mechanism that optimally balances local and global attention representations.
*   **Differentiable Landmarks:** End-to-end learned landmark selection using a Gumbel-Softmax mechanism, preventing memory collapse and ensuring diverse context sampling.
*   **Modern Optimizations:** Native support for Automatic Mixed Precision (AMP), gradient checkpointing, and curriculum learning for stable convergence on long contexts.

## 🧠 Architecture & Code Analysis

The repository is modularly designed to separate the complex routing logic from the standard Transformer backbone. Here is how the core components interact:

*   **`src/landmarks.py`:** Handles the differentiable selection of global tokens. Instead of using fixed or random landmarks, this module uses a Gumbel-Softmax distribution to dynamically learn which tokens carry the most semantic weight for the global context. It also includes auxiliary loss functions to enforce landmark diversity.
*   **`src/slga.py`:** The heart of the attention mechanism. It computes the standard sliding window attention, retrieves the global landmarks, and applies the **Learned Fusion (Gated Fusion)** to merge the two representations before passing them to the feed-forward network. 
*   **`src/model.py`:** The main Transformer architecture (defaulting to 12 stacked layers). It integrates the SLGA attention blocks and utilizes Rotary Position Embeddings (RoPE) to maintain robust relative positional awareness across massive context windows.

## 📦 Installation

```bash
# Clone the Project
git clone https://github.com/JacquesGariepy/SLGA.git
cd SLGA

# Create Virtual Environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# OR install in development mode
pip install -e .

```

## 🚀 Quick Start

```bash
# 1. Download the dataset (optional, done automatically)
python scripts/download_dataset.py

# 2. Train the model
python scripts/train.py --config ./configs/config_default.yaml

# 3. Evaluate perplexity
python scripts/eval_perplexity.py --checkpoint out_slga/ckpt_50000

# 4. Generate text
python scripts/generate.py --checkpoint out_slga/ckpt_50000 --prompt "Hello, the future of AI is"

```

## 📁 Project Structure

```text
slga_project/
├── src/                    # Main source code
│   ├── slga.py             # SLGA Attention Module & Gated Fusion
│   ├── landmarks.py        # Gumbel-Softmax Landmark Selection
│   ├── model.py            # Transformer Model & RoPE
│   ├── data.py             # Data loading and preprocessing
│   └── utils.py            # Utilities (MFU calculation, logging)
├── scripts/                # Execution scripts
│   ├── train.py            # Training loop
│   ├── eval_perplexity.py  # Evaluation pipeline
│   ├── generate.py         # Inference and text generation
│   └── download_dataset.py # Dataset fetching
├── configs/                # Configuration files
│   ├── config_default.yaml # Default config (Production)
│   ├── config_small.yaml   # Light config (Testing/Debugging)
│   └── config_large.yaml   # High capacity config (>24GB VRAM)
├── tests/                  # Unit tests (PyTest)
├── docs/                   # Extended Documentation
└── data/                   # Data directory (ignored in git)

```

## 📊 Expected Performance (RTX 3090 24GB)

* **Throughput:** > 4000 tokens/sec (during training)
* **Perplexity:** < 12 on Wikipedia (at 100K steps)
* **Memory Footprint:** ~18GB VRAM (batch_size=4, seq_len=2048)
* **Long-QA F1 Score:** > 72% (SCROLLS benchmark)

## 🔧 Configuration Profiles

The project includes three pre-defined configurations mapped to different hardware constraints and use cases:

```bash
# Quick test (lightweight, ~30 min run for debugging)
python scripts/train.py --config configs/config_small.yaml

# Production (default, 2-3 days for convergence)
python scripts/train.py --config configs/config_default.yaml

# High capacity (Requires multi-GPU or >24GB VRAM)
python scripts/train.py --config configs/config_large.yaml

```

## 📚 Documentation

* [Quick Start](https://www.google.com/search?q=docs/quickstart.md) - Step-by-step guide
* [Architecture](https://www.google.com/search?q=docs/architecture.md) - Deep dive into SLGA math and technical details
* [Troubleshooting](https://www.google.com/search?q=docs/troubleshooting.md) - Common issues and problem-solving

## 🧪 Tests

Ensure the environment is working correctly by running the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run a specific test for the attention module
python -m pytest tests/test_slga.py

```

## 🤝 Contribution

Contributions are welcome! Whether it's optimizing the CUDA kernels, adding new benchmarks, or fixing bugs, please see [CONTRIBUTING.md](https://www.google.com/search?q=CONTRIBUTING.md) for guidelines on how to open a PR.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## 📞 Contact

* **Issues:** [GitHub Issues](https://github.com/JacquesGariepy/SLGA/issues)
* **Author:** Jacques Gariépy

## 🙏 Citation

If you use SLGA in your research, please cite:

```bibtex
@software{slga_gariepy_2026,
  title={SLGA: Efficient Sparse Local-Global Attention},
  author={Jacques Gariépy},
  year={2026},
  url={[https://github.com/JacquesGariepy/SLGA](https://github.com/JacquesGariepy/SLGA)}
}

```
