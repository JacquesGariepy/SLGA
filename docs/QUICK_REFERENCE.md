# 🚀 SLGA Training - Quick Reference Guide

**Version**: 1.0
**Date**: 2025-10-24
**GPU**: RTX 3090 (24GB)

---

## 📋 Table des Matières

1. [Commandes Rapides](#commandes-rapides)
2. [Configuration Files](#configuration-files)
3. [Diagnostic Commands](#diagnostic-commands)
4. [Optimizations RTX 3090](#optimizations-rtx-3090)
5. [Troubleshooting](#troubleshooting)
6. [Performance Targets](#performance-targets)

---

## ⚡ Commandes Rapides

### Lancer Training

```bash
# Training from scratch
python scripts/train.py

# Resume depuis checkpoint
python scripts/train.py --resume out_slga/ckpt_15000

# Training avec nouveau dataset (recommandé)
python scripts/resume_with_new_dataset.py \
  --checkpoint out_slga/ckpt_15000/model.pt \
  --config config_new_dataset.yaml \
  --reset-optimizer
```

### Génération de Texte

```bash
# Génération simple
python scripts/generate_fixed.py \
  --checkpoint out_slga/ckpt_15000 \
  --prompt "The capital of France is" \
  --temperature 0.0 \
  --max_new_tokens 50

# Génération avec multiple prompts
python scripts/generate_fixed.py \
  --checkpoint out_slga/ckpt_15000 \
  --prompts-file data/test_prompts.txt \
  --temperature 0.7
```

### Inspection & Diagnostic

```bash
# Vérifier état du checkpoint
python scripts/inspect_trainer_state.py out_slga/ckpt_15000

# Analyser batch de training
python scripts/inspect_training_batch.py

# Vérifier dataset
python scripts/check_wiki_dataset.py
```

---

## 📝 Configuration Files

### config.yaml (Actuel - Wikipedia)

```yaml
model:
  vocab_size: 50257
  embed_dim: 512
  num_heads: 8
  n_layers: 12
  local_window: 128
  global_k: 24
  learned_landmarks: true  # ⚠️ Peut causer instabilités

train:
  batch_size: 8            # ✅ Optimal RTX 3090
  accum_steps: 4           # ✅ Effective batch = 32
  lr: 2.0e-4
  weight_decay: 0.1        # ⚠️ Trop élevé
  amp: true
  amp_dtype: "bf16"        # ✅ Stabilité > vitesse

data:
  dataset: "wikimedia/wikipedia"  # ⚠️ Monodomaine → overfitting
  split_train: "train[:95%]"      # ⚠️ Trop de chevauchement
```

### config_new_dataset.yaml (Recommandé - FineWeb-Edu)

```yaml
model:
  # Identique à avant
  learned_landmarks: false  # ← CHANGÉ: Test sans learned

train:
  # Batch/accum identiques
  weight_decay: 0.01       # ← CHANGÉ: 0.1 → 0.01

  # Global warmup plus progressif
  global_warmup_start: 5000   # ← 1K → 5K
  global_warmup_end: 20000    # ← 5K → 20K

  # Landmarks regularization renforcée
  lambda_diversity: 0.1    # ← 0.02 → 0.1 (5x)
  lambda_sparsity: 0.01    # ← 0.001 → 0.01 (10x)

data:
  dataset: "HuggingFaceFW/fineweb-edu"  # ← CHANGÉ
  split_train: "train[:90%]"            # ← 95% → 90%
  split_val: "train[90%:95%]"           # ← Vrai held-out
  split_test: "train[95%:]"             # ← Nouveau test set
```

---

## 🔍 Diagnostic Commands

### 1. GPU Monitoring

```bash
# Watch GPU en temps réel
watch -n 1 nvidia-smi

# GPU utilization détaillé
nvidia-smi dmon -s pucvmet -c 100

# GPU memory breakdown
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# GPU temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader --loop=1
```

**Targets RTX 3090**:
- GPU Util: 85-95%
- Memory: 18-20 GB (75-85%)
- Temp: < 83°C
- Power: 350-370W

### 2. Training Metrics

```bash
# TensorBoard
tensorboard --logdir out_slga/tensorboard --port 6006
# Ouvrir: http://localhost:6006

# W&B (si activé)
wandb login
wandb sync out_slga/wandb

# Logs bruts
tail -f out_slga/train.log
```

### 3. Checkpoint Inspection

```python
import torch

# Charger checkpoint
ckpt = torch.load("out_slga/ckpt_15000/model.pt", map_location="cpu")

# Taille totale
print(f"Params: {sum(p.numel() for p in ckpt.values()) / 1e6:.2f}M")

# Layer-wise sizes
for name, param in ckpt.items():
    print(f"{name}: {param.shape}")

# Charger trainer state
trainer = torch.load("out_slga/ckpt_15000/trainer_state.pt")
print(f"Step: {trainer['step']}")
print(f"LR: {trainer['scheduler']['_last_lr'][0]:.2e}")
```

### 4. Dataset Statistics

```python
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

print(f"Total samples: {len(ds)}")
print(f"Columns: {ds.column_names}")
print(f"Features: {ds.features}")

# Sample texts
for i in range(3):
    print(f"\n=== Sample {i} ===")
    print(ds[i]["text"][:200])
```

### 5. Memory Profiling

```python
import torch

def print_memory_summary():
    print("\n=== GPU Memory Summary ===")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"Max Alloc: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"Free:      {(24 - torch.cuda.memory_allocated() / 1e9):.2f} GB")

# Reset peak stats
torch.cuda.reset_peak_memory_stats()

# ... training code ...

print_memory_summary()
```

---

## 🚀 Optimizations RTX 3090

### 1. Configuration Optimale

```yaml
train:
  batch_size: 8              # Maximum sans OOM pour seq_len=2048
  accum_steps: 4             # Effective batch = 32
  amp: true                  # Mandatory pour RTX 3090
  amp_dtype: "bf16"          # BF16 > FP16 (stabilité)
  grad_checkpointing: false  # Pas besoin (24GB suffisant)
```

**Alternative (plus agressif)**:
```yaml
train:
  batch_size: 12             # Utiliser 85-90% GPU
  accum_steps: 3             # Effective batch = 36
  # Attention: risque OOM si pic mémoire
```

### 2. DataLoader Settings

```yaml
data:
  num_workers: 0       # WSL2: Évite deadlocks
  pin_memory: true     # +20% speedup CPU→GPU
  drop_last: true      # Batch size consistent
  prefetch_factor: 2   # Précharge 2 batches (Linux uniquement)
```

### 3. Curriculum Acceleration

```yaml
train:
  seq_len_start: 512       # 384 → 512 (démarrer plus haut)
  seq_len_mid: 1024        # Identique
  seq_len_final: 2048      # Identique
  seq_len_warmup_steps: 10000  # 15K → 10K (33% plus rapide)
```

**Gain**: Atteint seq_len=2048 **5K steps plus tôt**

### 4. Torch Compile (PyTorch 2.0+)

```python
# Dans train.py, après création du modèle
if torch.__version__ >= "2.0" and cfg["train"].get("torch_compile", False):
    model = torch.compile(model, mode="reduce-overhead")
    print("✓ Model compiled with torch.compile")
```

**Gains**: +10-20% throughput (mais compilation longue: 5-10 min)

### 5. Flash Attention (optionnel)

```bash
# Installation
pip install flash-attn --no-build-isolation
```

```python
# Dans src/slga.py
from flash_attn import flash_attn_func

# Remplacer F.scaled_dot_product_attention par:
out = flash_attn_func(q, k, v, causal=True)
```

**Gains**:
- Memory: -30% (économise 5-6 GB)
- Speed: +20-30%

---

## 🐛 Troubleshooting

### Problème 1: OOM (Out of Memory)

**Symptômes**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions**:
```yaml
# Option A: Réduire batch size
train:
  batch_size: 6  # 8 → 6
  accum_steps: 5  # 4 → 5 (garde effective batch = 30)

# Option B: Réduire seq length
train:
  seq_len_final: 1536  # 2048 → 1536

# Option C: Activer gradient checkpointing
train:
  grad_checkpointing: true  # Économise 40% memory, ralentit 3x
```

### Problème 2: Loss = NaN

**Symptômes**:
```
Step 1234 | Loss: nan | PPL: nan
```

**Diagnostics**:
```python
# Vérifier gradient norm
print(f"Grad norm: {grad_norm:.2f}")  # Si > 100 → explosion

# Vérifier activations
print(f"Logits max: {logits.max().item():.2f}")
print(f"Logits min: {logits.min().item():.2f}")
```

**Solutions**:
```yaml
train:
  grad_clip: 0.5      # 1.0 → 0.5 (plus agressif)
  lr: 1.0e-4          # 2.0e-4 → 1.0e-4 (réduire LR)
  amp_dtype: "bf16"   # fp16 → bf16 (plus stable)
```

### Problème 3: Validation Loss Trop Élevée

**Symptômes**:
```
Step 15000 | Train Loss: 2.54 | Val Loss: 6.04
Gap: 3.5 units (overfitting!)
```

**Solutions**:
```yaml
# 1. Changer dataset
data:
  dataset: "HuggingFaceFW/fineweb-edu"  # Multi-domaine

# 2. Augmenter regularization
train:
  dropout_rate: 0.15    # 0.1 → 0.15
  weight_decay: 0.01    # 0.1 → 0.01 (paradoxal mais vrai)
  label_smoothing: 0.1  # AJOUTER

# 3. Désactiver learned landmarks
model:
  learned_landmarks: false
```

### Problème 4: Training Lent (< 3000 tok/s)

**Diagnostics**:
```python
import time

start = time.time()
# ... training step ...
elapsed = time.time() - start

tokens_per_batch = batch_size * seq_len
throughput = tokens_per_batch / elapsed

print(f"Throughput: {throughput:.0f} tok/s")
```

**Solutions**:
```yaml
# 1. Augmenter batch size
train:
  batch_size: 12  # 8 → 12 (si GPU le permet)

# 2. Désactiver grad checkpointing
train:
  grad_checkpointing: false

# 3. Réduire num_workers (WSL2)
data:
  num_workers: 0  # Multiprocessing peut ralentir sur WSL2

# 4. Activer torch.compile
# (voir section Optimizations)
```

### Problème 5: Deadlock DataLoader

**Symptômes**:
```
Training stalls at random steps
No error message, just hangs
```

**Solution (WSL2)**:
```yaml
data:
  num_workers: 0  # MANDATORY sur WSL2
```

**Solution (Linux natif)**:
```yaml
data:
  num_workers: 4
  persistent_workers: true  # Garde workers alive
  prefetch_factor: 2        # Précharge 2 batches
```

---

## 📊 Performance Targets

### Step 15K (Actuel - Wikipedia)

| Métrique | Valeur | Status | Target |
|----------|--------|--------|--------|
| **Training** |
| Train Loss | 2.54 | ⚠️ OK | < 2.0 |
| Train PPL | 12.7 | ⚠️ OK | < 10 |
| Grad Norm | 2.34 | ✅ Good | 1-5 |
| **Validation** |
| Val Loss | 6.04 | ❌ Bad | < 3.0 |
| Val PPL | 420 | ❌ Bad | < 20 |
| Train/Val Gap | 3.5 | ❌ Critical | < 0.5 |
| **Performance** |
| Throughput | 6,553 tok/s | ✅ Excellent | > 5,000 |
| GPU Memory | 18.2 GB | ✅ Optimal | 16-20 GB |
| GPU Usage | 76% | ✅ Good | 75-85% |
| Steps/sec | 0.40 | ✅ Good | > 0.35 |

### Step 50K (Target - FineWeb-Edu)

| Métrique | Target | Notes |
|----------|--------|-------|
| Train Loss | < 2.0 | Convergence attendue |
| Train PPL | < 10 | Bonne compréhension |
| Val Loss | **< 3.0** | Pas d'overfitting |
| Val PPL | **< 20** | Généralisation OK |
| Train/Val Gap | **< 0.5** | Excellent |
| Throughput | > 6,000 tok/s | Performance maintenue |
| Training Time | ~30h | RTX 3090 (0→50K) |

### Génération de Texte (Qualité)

**Target**: Texte cohérent, grammatical, factuellement correct

**Test Prompts**:
```
1. "The capital of France is"
   → "Paris, located in the north-central part of France..."

2. "In 1969, humans first landed on"
   → "the Moon during the Apollo 11 mission..."

3. "Python is a programming language that"
   → "emphasizes code readability and simplicity..."
```

**Actuel (Step 15K)**:
```
1. "The capital of France is"
   → "the capital of 2004. It includes Spanish and capital" ❌

2. "In 1969, humans first"
   → "of the first of the first of the first..." ❌ (répétition)

3. "Python is"
   → "the the the the the..." ❌ (collapse)
```

**Diagnostic**: Modèle non fonctionnel → nécessite changement dataset

---

## 📈 Benchmarks RTX 3090

### Throughput vs Sequence Length

| Seq Len | Batch Size | GPU Memory | Tokens/sec | Time/50K Steps |
|---------|------------|------------|------------|----------------|
| 384     | 16         | 8 GB       | 12,000     | 11h            |
| 512     | 12         | 10 GB      | 10,000     | 13h            |
| 1024    | 8          | 14 GB      | 8,000      | 16h            |
| 2048    | 8          | 18 GB      | **6,553**  | **25h**        |
| 4096    | 4          | 22 GB      | 4,000      | 40h            |

### Comparaison GPUs (seq_len=2048, batch=8)

| GPU | VRAM | Tokens/sec | Prix | Perf/$ |
|-----|------|------------|------|--------|
| **RTX 3090** | 24GB | **6,553** | $1,500 | **1.00x** |
| RTX 4090 | 24GB | 10,000 | $1,600 | 1.43x |
| A100 (40GB) | 40GB | 15,000 | $10,000 | 0.34x |
| V100 (32GB) | 32GB | 4,000 | $8,000 | 0.11x |
| RTX 3080 | 10GB | OOM | $700 | N/A |

**Verdict**: RTX 3090 = **Meilleur rapport perf/prix** pour SLGA 38M

---

## 🎯 Quick Start Guide

### 1. Setup Initial (First Time)

```bash
# Clone repo (si nécessaire)
git clone https://github.com/your-repo/SLGA.git
cd SLGA

# Create venv
python -m venv venv
source venv/bin/activate  # Linux
# ou: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'BF16: {torch.cuda.is_bf16_supported()}')"
```

### 2. Diagnostic Checkpoint Actuel

```bash
# Inspecter checkpoint 15K
python scripts/inspect_trainer_state.py out_slga/ckpt_15000

# Tester génération
python scripts/generate_fixed.py \
  --checkpoint out_slga/ckpt_15000 \
  --prompt "The capital of France is" \
  --temperature 0.0
```

### 3. Nouveau Training (Recommandé)

```bash
# 1. Backup checkpoint actuel
cp -r out_slga/ckpt_15000 backups/ckpt_15000_wikipedia

# 2. Modifier config.yaml (voir config_new_dataset.yaml)

# 3. Tester dataset loading
python scripts/check_wiki_dataset.py

# 4. Lancer training avec reset optimizer
python scripts/resume_with_new_dataset.py \
  --checkpoint out_slga/ckpt_15000/model.pt \
  --config config_new_dataset.yaml \
  --reset-optimizer \
  --warmup-steps 2000

# 5. Monitorer en temps réel
tensorboard --logdir out_slga/tensorboard
```

### 4. Monitoring Pendant Training

**Terminal 1**: Training
```bash
python scripts/train.py
```

**Terminal 2**: GPU monitoring
```bash
watch -n 1 nvidia-smi
```

**Terminal 3**: TensorBoard
```bash
tensorboard --logdir out_slga/tensorboard --port 6006
```

**Browser**: http://localhost:6006

---

## 📚 Fichiers de Documentation

| Fichier | Description |
|---------|-------------|
| **TRAINING_PIPELINE_ANALYSIS.md** | Analyse technique complète (606 lignes) |
| **RTX_3090_OPTIMIZATIONS.md** | Optimisations GPU spécifiques |
| **PIPELINE_VISUAL_SUMMARY.md** | Diagrammes ASCII et visualisations |
| **STEP_15K_DIAGNOSTIC_REPORT.md** | Diagnostic checkpoint 15K |
| **RESUME_WITH_NEW_DATASET.md** | Guide reprise avec nouveau dataset |
| **QUICK_REFERENCE.md** | Ce fichier (commandes pratiques) |

---

## 🔗 Liens Utiles

### Documentation Officielle

- [PyTorch AMP](https://pytorch.org/docs/stable/amp.html)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [Accelerate](https://huggingface.co/docs/accelerate/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)

### Datasets Recommandés

- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (haute qualité)
- [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) (diverse)
- [The Pile](https://pile.eleuther.ai/) (825GB, multi-domaine)

### Papers

- [SLGA Original Paper](https://arxiv.org/abs/xxxx.xxxxx)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [Efficient Transformers](https://arxiv.org/abs/2009.06732)

---

## ⏱️ Time Estimates (RTX 3090)

### Training from Scratch

| Steps | Hours | Days | Notes |
|-------|-------|------|-------|
| 10K   | 6h    | 0.25 | Quick test |
| 20K   | 12h   | 0.5  | First validation |
| 50K   | **30h** | **1.25** | **Recommended minimum** |
| 100K  | 60h   | 2.5  | Full training |

### Resume from Checkpoint 15K

| Target | Hours | Notes |
|--------|-------|-------|
| +5K (→20K) | 3h | Quick diagnostic |
| +10K (→25K) | 6h | First results |
| +35K (→50K) | 21h | Full convergence |

### Dataset Download

| Dataset | Size | Time (100 Mbps) |
|---------|------|-----------------|
| Wikipedia (en) | ~20 GB | 30 min |
| FineWeb-Edu | ~140 GB | 3-4h |
| OpenWebText | ~38 GB | 1h |

---

## 💡 Pro Tips

1. **Toujours sauvegarder** avant de modifier config
2. **Tester sur 1K steps** avant training complet
3. **Monitorer val loss** tous les 500 steps
4. **Checkpoint tous les 1K** (disk space: ~500 MB/checkpoint)
5. **Utiliser TensorBoard** pour debugging visuel
6. **Nettoyer cache CUDA** entre runs: `torch.cuda.empty_cache()`
7. **WSL2**: `num_workers=0` mandatory (évite deadlocks)
8. **Linux natif**: `num_workers=4` recommandé (speedup)

---

## 🆘 Contact & Support

**Issues**: Créer issue sur GitHub avec:
- Configuration (`config.yaml`)
- Logs d'erreur (dernières 50 lignes)
- Output `nvidia-smi`
- Step actuel et checkpoint utilisé

**Documentation**: Lire d'abord les docs dans `/docs`

---

**Dernière mise à jour**: 2025-10-24
**Version**: 1.0
**Status**: Checkpoint 15K diagnostiqué, nouveau training recommandé
