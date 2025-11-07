# 📋 Pre-Training Checklist - SLGA-Plus

**Version**: 1.1
**Target Hardware**: RTX 3090 (24GB VRAM)
**Dataset**: OpenWebText (9B tokens)
**Estimated Duration**: ~28 hours for 100K steps

---

## 🎯 Quick Validation Command

```bash
# Run all checks at once
python scripts/validate_pretraining.py --full-check
```

---

## 1️⃣ Validation de l'Environnement

### GPU & CUDA
- [ ] **CUDA disponible et version ≥11.8**
  ```bash
  python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
  python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
  ```
  ✅ **Attendu**: `CUDA Available: True`, `CUDA Version: 11.8+`

- [ ] **GPU détectée (RTX 3090)**
  ```bash
  python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
  nvidia-smi --query-gpu=name,memory.total --format=csv
  ```
  ✅ **Attendu**: `NVIDIA GeForce RTX 3090`, `24GB`

- [ ] **VRAM libre suffisant (>20GB)**
  ```bash
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
  ```
  ✅ **Attendu**: `>20000` MB libre

- [ ] **Drivers NVIDIA à jour**
  ```bash
  nvidia-smi --query-gpu=driver_version --format=csv,noheader
  ```
  ✅ **Attendu**: `≥525.xx` (pour CUDA 11.8+)

### PyTorch & Dependencies
- [ ] **PyTorch version ≥2.0**
  ```bash
  python -c "import torch; print(f'PyTorch: {torch.__version__}')"
  ```
  ✅ **Attendu**: `2.0.0+` ou supérieur

- [ ] **Toutes les dépendances installées**
  ```bash
  pip list | grep -E "torch|transformers|datasets|tensorboard|pyyaml|tqdm"
  ```
  ✅ **Attendu**: Toutes présentes

- [ ] **Aucun conflit de dépendances**
  ```bash
  pip check
  ```
  ✅ **Attendu**: `No broken requirements found.`

---

## 2️⃣ Configuration Validée

### Fichier de Configuration
- [ ] **`config_3090_v1.1.yaml` existe**
  ```bash
  ls -lh config_3090_v1.1.yaml
  ```
  ✅ **Attendu**: Fichier présent (~2KB)

- [ ] **Tous les nouveaux paramètres v1.1 présents**
  ```bash
  grep -E "global_warmup_steps|landmark_selection_strategy|attention_dropout|residual_dropout|label_smoothing|max_grad_norm|dynamic_landmarks|batch_accumulation" config_3090_v1.1.yaml
  ```
  ✅ **Attendu**: 8 paramètres trouvés

- [ ] **Validation automatique passe**
  ```bash
  python scripts/test_validation.py --config config_3090_v1.1.yaml
  ```
  ✅ **Attendu**: `✅ Configuration validation PASSED`

- [ ] **Batch size adapté (16 pour RTX 3090)**
  ```bash
  grep "batch_size:" config_3090_v1.1.yaml
  ```
  ✅ **Attendu**: `batch_size: 16`

- [ ] **Curriculum learning configuré (384→2048)**
  ```bash
  grep -A 10 "curriculum_learning:" config_3090_v1.1.yaml
  ```
  ✅ **Attendu**: `start_seq_len: 384`, `end_seq_len: 2048`

### Hyperparamètres Critiques
- [ ] **Learning rate: 6e-4 (v1.1)**
  ```bash
  grep "learning_rate:" config_3090_v1.1.yaml
  ```
  ✅ **Attendu**: `learning_rate: 0.0006`

- [ ] **Warmup steps: 2000**
  ```bash
  grep "warmup_steps:" config_3090_v1.1.yaml
  ```
  ✅ **Attendu**: `warmup_steps: 2000`

- [ ] **Max grad norm: 1.0**
  ```bash
  grep "max_grad_norm:" config_3090_v1.1.yaml
  ```
  ✅ **Attendu**: `max_grad_norm: 1.0`

---

## 3️⃣ Dataset Prêt

### Téléchargement et Accès
- [ ] **Dataset téléchargé et accessible**
  ```bash
  python scripts/check_dataset.py --dataset openwebtext
  ```
  ✅ **Attendu**: `✅ Dataset loaded successfully`

- [ ] **Nombre de samples vérifié**
  ```bash
  python -c "from datasets import load_dataset; ds = load_dataset('openwebtext', split='train'); print(f'Total samples: {len(ds):,}')"
  ```
  ✅ **Attendu**: `~8,013,769` samples

- [ ] **Splits train/val créés correctement**
  ```bash
  python scripts/check_dataset.py --check-splits
  ```
  ✅ **Attendu**: `Train: 90%`, `Val: 10%`

### Tokenizer
- [ ] **Tokenizer chargé correctement**
  ```bash
  python -c "from transformers import GPT2Tokenizer; tok = GPT2Tokenizer.from_pretrained('gpt2'); print(f'Vocab size: {len(tok)}')"
  ```
  ✅ **Attendu**: `Vocab size: 50257`

- [ ] **Test de tokenization**
  ```bash
  python scripts/test_tokenizer.py --text "The quick brown fox jumps over the lazy dog."
  ```
  ✅ **Attendu**: Tokens générés sans erreur

### Premier Batch
- [ ] **Premier batch chargé sans erreur**
  ```bash
  python scripts/inspect_training_batch.py --num-batches 1
  ```
  ✅ **Attendu**: Batch shape `(16, 384)` pour curriculum start

---

## 4️⃣ Modèle Initialisé

### Architecture
- [ ] **Architecture SLGA créée (65.3M params)**
  ```bash
  python -c "from src.models.slga_plus import SLGAPlus; from src.utils.config import load_config; cfg = load_config('config_3090_v1.1.yaml'); model = SLGAPlus(cfg); print(f'Total params: {sum(p.numel() for p in model.parameters()):,}')"
  ```
  ✅ **Attendu**: `~65,300,000` parameters

- [ ] **Poids initialisés correctement**
  ```bash
  python scripts/test_model_init.py --config config_3090_v1.1.yaml
  ```
  ✅ **Attendu**: `✅ Model initialized successfully`

### Tests Fonctionnels
- [ ] **Forward pass test réussi**
  ```bash
  python scripts/test_forward_pass.py --batch-size 4 --seq-len 384
  ```
  ✅ **Attendu**: `Output shape: (4, 384, 50257)` sans NaN

- [ ] **Génération test produit du texte**
  ```bash
  python scripts/test_generation.py --prompt "Once upon a time" --max-length 50
  ```
  ✅ **Attendu**: Texte généré (même si peu cohérent initialement)

- [ ] **Landmarks sélectionnés sans erreur**
  ```bash
  python scripts/test_landmarks.py --strategy "random" --num-landmarks 32
  ```
  ✅ **Attendu**: `✅ Landmarks selected: 32/384 positions`

### Gradient Flow
- [ ] **Gradients calculés correctement**
  ```bash
  python scripts/test_gradients.py --check-flow
  ```
  ✅ **Attendu**: `✅ All gradients computed, no NaN/Inf`

---

## 5️⃣ Monitoring Configuré

### TensorBoard & Logging
- [ ] **TensorBoard logging actif**
  ```bash
  ls -d runs/
  tensorboard --logdir=runs/ --port=6006 &
  ```
  ✅ **Attendu**: Dossier `runs/` créé, TensorBoard accessible à `http://localhost:6006`

- [ ] **Output directory créé**
  ```bash
  mkdir -p out_slga/checkpoints out_slga/logs
  ls -ld out_slga/
  ```
  ✅ **Attendu**: Dossiers créés

- [ ] **Checkpoint saving configuré**
  ```bash
  grep "checkpoint_every_n_steps:" config_3090_v1.1.yaml
  ```
  ✅ **Attendu**: `checkpoint_every_n_steps: 1000`

- [ ] **Logging interval testé (50 steps)**
  ```bash
  grep "log_interval:" config_3090_v1.1.yaml
  ```
  ✅ **Attendu**: `log_interval: 50`

### Nouvelles Métriques v1.1
- [ ] **7 nouvelles métriques vérifiées**
  ```bash
  python scripts/test_metrics.py --all
  ```
  ✅ **Attendu**:
  - `landmark_attention_entropy`
  - `landmark_attention_concentration`
  - `landmark_usage_variance`
  - `layer_activation_mean`
  - `layer_activation_std`
  - `effective_batch_size`
  - `tokens_per_second`

---

## 6️⃣ Tests Passent

### Suite de Tests Complète
- [ ] **Tous les tests unitaires passent**
  ```bash
  pytest tests/ -v --tb=short
  ```
  ✅ **Attendu**: `51/51 PASSED` (0 failed)

- [ ] **Coverage report généré (>75%)**
  ```bash
  pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
  ```
  ✅ **Attendu**: `Coverage: >75%`, rapport dans `htmlcov/`

- [ ] **Validation module testé**
  ```bash
  pytest tests/test_validation.py -v
  ```
  ✅ **Attendu**: Tous les tests passent

- [ ] **Integration example exécuté**
  ```bash
  python examples/integration_example.py --steps 10
  ```
  ✅ **Attendu**: 10 steps complétés sans erreur

### Tests Spécifiques
- [ ] **Test de curriculum learning**
  ```bash
  pytest tests/test_curriculum.py -v
  ```
  ✅ **Attendu**: Sequence length augmente progressivement

- [ ] **Test de landmark selection**
  ```bash
  pytest tests/test_landmarks.py -v
  ```
  ✅ **Attendu**: Toutes les stratégies fonctionnent

---

## 7️⃣ Sauvegarde et Reprise

### Checkpoints
- [ ] **Checkpoint directory créé**
  ```bash
  mkdir -p out_slga/checkpoints
  ls -ld out_slga/checkpoints/
  ```
  ✅ **Attendu**: Dossier créé avec permissions d'écriture

- [ ] **Saving every 1000 steps configuré**
  ```bash
  grep "checkpoint_every_n_steps:" config_3090_v1.1.yaml
  ```
  ✅ **Attendu**: `checkpoint_every_n_steps: 1000`

- [ ] **Resume from checkpoint testé**
  ```bash
  python scripts/test_resume.py --checkpoint out_slga/checkpoints/step_1000.pt
  ```
  ✅ **Attendu**: `✅ Successfully resumed from checkpoint`

- [ ] **Optimizer state sauvegardé**
  ```bash
  python scripts/check_checkpoint.py --checkpoint out_slga/checkpoints/step_1000.pt --check-optimizer
  ```
  ✅ **Attendu**: `✅ Optimizer state present and valid`

### Recovery
- [ ] **Test de recovery après interruption**
  ```bash
  python scripts/test_crash_recovery.py --simulate-crash
  ```
  ✅ **Attendu**: Training reprend au bon step

---

## 8️⃣ Dernières Vérifications

### Git & Documentation
- [ ] **Config commité dans git**
  ```bash
  git status config_3090_v1.1.yaml
  ```
  ✅ **Attendu**: `Changes to be committed` ou `committed`

- [ ] **README à jour avec v1.1**
  ```bash
  grep "v1.1" README.md
  ```
  ✅ **Attendu**: Version 1.1 mentionnée

- [ ] **Training command documentée**
  ```bash
  grep "python scripts/train.py" docs/*.md
  ```
  ✅ **Attendu**: Commande présente dans docs

### Monitoring Dashboard
- [ ] **TensorBoard accessible**
  ```bash
  curl -I http://localhost:6006 2>/dev/null | head -n 1
  ```
  ✅ **Attendu**: `HTTP/1.1 200 OK`

- [ ] **Métriques visibles dans dashboard**
  - Ouvrir `http://localhost:6006`
  - Vérifier présence des graphes: Loss, PPL, LR, Grad Norm

### Estimation de Temps
- [ ] **Temps estimé confirmé (~28h)**
  ```bash
  python scripts/estimate_training_time.py --steps 100000 --batch-size 16
  ```
  ✅ **Attendu**: `Estimated time: ~28 hours`

---

## 🚀 Commande de Lancement Finale

Une fois toutes les cases cochées, lancer l'entraînement:

```bash
# Lancement avec monitoring
python scripts/train.py \
  --config config_3090_v1.1.yaml \
  --output-dir out_slga \
  --tensorboard-dir runs/slga_v1.1 \
  --max-steps 100000 \
  --resume-from-checkpoint auto \
  2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log
```

### Monitoring en Temps Réel

Terminal 1 (Training):
```bash
python scripts/train.py --config config_3090_v1.1.yaml
```

Terminal 2 (GPU monitoring):
```bash
watch -n 1 nvidia-smi
```

Terminal 3 (TensorBoard):
```bash
tensorboard --logdir=runs/ --port=6006 --bind_all
```

Terminal 4 (Logs):
```bash
tail -f logs/training_*.log
```

---

## 📊 Métriques de Succès Attendues

| Step | Loss | PPL | LR | Temp | Spacing Quality | Landmark Entropy |
|------|------|-----|----|----|----------------|------------------|
| **100** | 8-10 | 3000-22000 | 3e-5 | 1.0 | Random | Low (2-3) |
| **1K** | 6-8 | 400-3000 | 6e-4 | 1.0 | Improving | Medium (3-4) |
| **5K** | 4-5 | 55-150 | 6e-4 | 0.5 | Good | High (4-5) |
| **15K** | 3-4 | 20-55 | 5e-4 | 0.3 | Very Good | High (4.5-5) |
| **50K** | 2.5-3 | 12-20 | 3e-4 | 0.3 | Excellent | Very High (5+) |
| **100K** | 2-2.5 | 7-12 | 1e-4 | 0.3 | Optimal | Very High (5+) |

### Métriques Détaillées par Phase

#### Phase 1: Warmup (0-2K steps)
- **Loss**: Descend rapidement de 10→6
- **LR**: Monte progressivement 0→6e-4
- **Landmarks**: Apprentissage de la structure
- **Spacing**: Encore aléatoire mais s'améliore

#### Phase 2: Apprentissage Initial (2K-15K)
- **Loss**: Descend régulièrement 6→3
- **PPL**: Amélioration significative
- **Landmarks**: Commencent à se spécialiser
- **Spacing**: Patterns clairs émergent

#### Phase 3: Convergence (15K-50K)
- **Loss**: Stabilisation progressive
- **Attention**: Landmarks bien définis
- **Spacing**: Qualité proche de l'optimal
- **Génération**: Texte cohérent sur 50-100 tokens

#### Phase 4: Fine-tuning (50K-100K)
- **Loss**: Amélioration lente mais continue
- **Génération**: Texte cohérent sur 200+ tokens
- **Landmarks**: Usage optimal et stable
- **Spacing**: Qualité production-ready

---

## 🔧 Troubleshooting

### ❌ CUDA Out of Memory
**Symptôme**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# 1. Réduire batch size
sed -i 's/batch_size: 16/batch_size: 12/' config_3090_v1.1.yaml

# 2. Activer gradient accumulation
sed -i 's/batch_accumulation: 1/batch_accumulation: 2/' config_3090_v1.1.yaml

# 3. Réduire sequence length initiale
sed -i 's/start_seq_len: 384/start_seq_len: 256/' config_3090_v1.1.yaml

# 4. Vérifier VRAM disponible
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

### ❌ Loss devient NaN
**Symptôme**: `Loss = nan` après quelques steps

**Solutions**:
```bash
# 1. Réduire learning rate
sed -i 's/learning_rate: 0.0006/learning_rate: 0.0003/' config_3090_v1.1.yaml

# 2. Augmenter warmup
sed -i 's/warmup_steps: 2000/warmup_steps: 5000/' config_3090_v1.1.yaml

# 3. Réduire max_grad_norm
sed -i 's/max_grad_norm: 1.0/max_grad_norm: 0.5/' config_3090_v1.1.yaml

# 4. Activer AMP (mixed precision)
python scripts/train.py --config config_3090_v1.1.yaml --amp
```

### ❌ Dataset Loading Slow
**Symptôme**: `Loading dataset...` prend >10 minutes

**Solutions**:
```bash
# 1. Télécharger en avance
python scripts/download_dataset.py --dataset openwebtext --cache-dir ~/.cache/huggingface

# 2. Utiliser cache local
export HF_DATASETS_CACHE="/path/to/fast/ssd/cache"

# 3. Réduire num_workers si I/O bottleneck
sed -i 's/num_workers: 4/num_workers: 2/' config_3090_v1.1.yaml
```

### ❌ Checkpoint Saving Fails
**Symptôme**: `PermissionError` ou `OSError` lors du save

**Solutions**:
```bash
# 1. Vérifier permissions
chmod -R 755 out_slga/

# 2. Vérifier espace disque
df -h out_slga/

# 3. Tester écriture
touch out_slga/checkpoints/test.txt && rm out_slga/checkpoints/test.txt
```

### ❌ Training Too Slow (<500 tokens/sec)
**Symptôme**: `tokens_per_second` < 500

**Solutions**:
```bash
# 1. Vérifier GPU utilization
nvidia-smi dmon -s u -c 10

# 2. Profiler le code
python scripts/train.py --profile --max-steps 100

# 3. Augmenter batch_size si VRAM disponible
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
```

### ❌ TensorBoard Empty
**Symptôme**: Graphes ne s'affichent pas

**Solutions**:
```bash
# 1. Vérifier logs créés
ls -lh runs/*/events.out.tfevents.*

# 2. Relancer TensorBoard avec bon logdir
tensorboard --logdir=runs/ --reload_interval 5

# 3. Vérifier log_interval pas trop élevé
grep "log_interval:" config_3090_v1.1.yaml
```

---

## 📈 Monitoring Dashboard URLs

Une fois l'entraînement lancé:

- **TensorBoard**: http://localhost:6006
- **Logs**: `tail -f logs/training_*.log`
- **GPU Stats**: `watch -n 1 nvidia-smi`
- **Checkpoint Status**: `ls -lht out_slga/checkpoints/ | head`

---

## ✅ Validation Finale

Avant de lancer l'entraînement complet, exécuter:

```bash
python scripts/validate_pretraining.py \
  --full-check \
  --config config_3090_v1.1.yaml \
  --dry-run-steps 10 \
  --check-resume
```

**Attendu**: `✅ ALL CHECKS PASSED - Ready for training!`

---

## 🎯 Next Steps After All Checks Pass

1. **Commit final state**:
   ```bash
   git add .
   git commit -m "Pre-training validation complete - ready for 100K steps"
   git push
   ```

2. **Start training**:
   ```bash
   python scripts/train.py --config config_3090_v1.1.yaml
   ```

3. **Monitor first 1000 steps closely**:
   - Loss should decrease rapidly: 10→6
   - No NaN/Inf values
   - GPU utilization >90%
   - Memory usage stable

4. **Set up alerts** (optional):
   ```bash
   python scripts/setup_alerts.py --email your@email.com --loss-threshold 15
   ```

---

**Last Updated**: 2025-10-24
**Config Version**: v1.1
**Expected Completion**: Step 100K (~28 hours)

🚀 **Good luck with training!**
