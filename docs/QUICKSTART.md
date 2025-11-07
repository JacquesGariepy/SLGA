# 🚀 Guide de Démarrage Rapide SLGA-Plus

## Installation Express (5 minutes)

```bash
# 1. Créer environnement
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Installer PyTorch + CUDA
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Installer dépendances
pip install transformers datasets accelerate einops pyyaml tqdm scikit-learn

# 4. Vérifier GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

## Premier Entraînement (10 minutes)

```bash
# Éditer config.yaml pour test rapide
# Changer: max_steps: 1000, seq_len_start: 256

# Lancer
python train.py

# Vous devriez voir:
# Step 100 | Loss: 8.2341 | PPL: 3750.23 | LR: 4.00e-05 | SeqLen: 256
# Step 200 | Loss: 7.1234 | PPL: 1245.67 | ...
```

## Test des Modules (2 minutes)

```bash
# Tester chaque composant
python slga.py        # ✓ Test SLGA Module
python landmarks.py   # ✓ Test Landmark Selector
python model.py       # ✓ Test LLM Transformer
python data.py        # ✓ Test Collators
python utils.py       # ✓ Test Utils
```

## Configuration Minimale (RTX 3090)

```yaml
# config.yaml - Version light pour test
model:
  embed_dim: 256        # Réduit pour test rapide
  num_heads: 4
  n_layers: 6
  local_window: 64
  global_k: 16

train:
  batch_size: 4
  accum_steps: 8
  max_steps: 10000      # Test rapide
  seq_len_start: 256
  seq_len_final: 1024   # Plus court pour test
```

Bon entraînement ! 🚀
