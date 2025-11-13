# SLGA2 Scripts Directory

This directory contains production-ready scripts for training, evaluation, and dataset management for the SLGA (Sparse Landmark Global Attention) model.

**Last Updated:** 2025-11-12
**Total Scripts:** 16 production scripts

---

## Core Training & Evaluation

### `train.py`
Main training script for SLGA models.

**Usage:**
```bash
python scripts/train.py --config config/config_3090.yaml
python scripts/train.py --config config/config_quick_test.yaml --max-steps 5000
python scripts/train.py --config config.yaml --resume
```

**Arguments:**
- `--config CONFIG` - Path to YAML configuration file (default: config.yaml)
- `--max-steps MAX_STEPS` - Override max training steps from config
- `--resume` - Resume training from latest checkpoint

**Key Features:**
- Automatic mixed precision (AMP) training
- Gradient accumulation support
- Sequence length curriculum learning
- Real-time metrics display
- Validation during training
- Checkpoint management
- TensorBoard logging (optional)
- System monitoring

---

### `generate.py`
Text generation CLI for inference and testing.

**Usage:**
```bash
python scripts/generate.py --checkpoint checkpoints/step_5000 --prompt "Once upon a time"
python scripts/generate.py --checkpoint checkpoints/latest --max-tokens 100 --temperature 0.8
```

**Key Features:**
- Interactive and batch generation
- Multiple decoding strategies (greedy, sampling, top-p, top-k)
- Temperature control
- Generation logging and history

---

### `eval_perplexity.py`
Evaluate model perplexity on validation/test sets.

**Usage:**
```bash
python scripts/eval_perplexity.py --checkpoint checkpoints/step_5000 --config config.yaml
```

**Outputs:**
- Perplexity scores
- Loss metrics
- Token-level statistics

---

## Dataset Management

### `download_dataset.py`
Download and cache datasets from HuggingFace.

**Usage:**
```bash
python scripts/download_dataset.py --dataset wikipedia --config 20220301.en
python scripts/download_dataset.py --dataset wikicorpus --streaming
```

**Supported Datasets:**
- Wikipedia (20220301.en)
- FineWeb-Edu
- Wikicorpus
- Custom HuggingFace datasets

---

### `prepare_fineweb_edu.py`
Prepare and process FineWeb-Edu dataset for training.

**Usage:**
```bash
python scripts/prepare_fineweb_edu.py --output-dir data/fineweb_edu
```

**Features:**
- Dataset cleaning and filtering
- Quality scoring
- Token counting
- Efficient streaming processing

---

### `create_mixed_dataset.py`
Create mixed datasets from multiple sources.

**Usage:**
```bash
python scripts/create_mixed_dataset.py --sources wikipedia,fineweb --weights 0.5,0.5
```

**Use Cases:**
- Domain mixing for better generalization
- Curriculum learning with progressive sources
- Custom dataset composition

---

### `resume_with_new_dataset.py`
Resume training checkpoint with a different dataset.

**Usage:**
```bash
python scripts/resume_with_new_dataset.py \
  --checkpoint checkpoints/step_10000 \
  --new-config config_fineweb.yaml
```

**Use Cases:**
- Domain adaptation
- Continue training on new data
- Fine-tuning experiments

---

## Model Analysis & Inspection

### `inspect_trainer_state.py`
Inspect and analyze trainer state from checkpoints.

**Usage:**
```bash
python scripts/inspect_trainer_state.py --checkpoint checkpoints/step_5000
```

**Outputs:**
- Training step information
- Optimizer state
- Scheduler state
- Loss history
- Configuration snapshot

---

### `inspect_training_batch.py`
Inspect individual training batches and data pipeline.

**Usage:**
```bash
python scripts/inspect_training_batch.py --config config.yaml --num-batches 5
```

**Useful For:**
- Debugging data pipeline issues
- Verifying batch composition
- Checking landmark positions
- Token distribution analysis

---

### `find_best_checkpoint.py`
Find the best checkpoint based on validation metrics.

**Usage:**
```bash
python scripts/find_best_checkpoint.py --checkpoint-dir checkpoints/
```

**Criteria:**
- Lowest validation loss
- Best perplexity
- Custom metric thresholds

---

## Generation & Sampling

### `grid_search_generation.py`
Grid search over generation hyperparameters.

**Usage:**
```bash
python scripts/grid_search_generation.py \
  --checkpoint checkpoints/step_5000 \
  --temperatures 0.7,0.8,0.9 \
  --top-p 0.9,0.95 \
  --top-k 40,50
```

**Outputs:**
- Quality scores per configuration
- Diversity metrics
- Recommended hyperparameters

---

### `verify_generation.py`
Verify generation quality and coherence.

**Usage:**
```bash
python scripts/verify_generation.py --checkpoint checkpoints/latest --num-samples 10
```

**Checks:**
- Text coherence
- Repetition detection
- EOS token handling
- Generation quality scores

---

### `view_generation_history.py`
View and analyze generation history logs.

**Usage:**
```bash
python scripts/view_generation_history.py --log-file logs/generation_history.jsonl
```

**Features:**
- Browse past generations
- Filter by prompt/timestamp
- Export to different formats

---

## Utilities

### `utils.py`
Shared utility functions for all scripts.

**Exports:**
- `set_seed()` - Reproducible random seed setting
- `get_device()` - Device detection and setup
- `format_metrics()` - Metrics formatting for display
- `load_checkpoint()` - Safe checkpoint loading
- `save_checkpoint()` - Checkpoint saving with atomic writes

---

### `cleanup_old_checkpoints.py`
Disk space management for checkpoints.

**Usage:**
```bash
python scripts/cleanup_old_checkpoints.py --checkpoint-dir checkpoints/ --keep-last 5
python scripts/cleanup_old_checkpoints.py --checkpoint-dir checkpoints/ --min-steps 1000
```

**Strategies:**
- Keep last N checkpoints
- Keep checkpoints above minimum steps
- Keep best validation checkpoints
- Dry-run mode for safety

---

### `visualize_training_metrics.py`
Visualize training metrics and progress.

**Usage:**
```bash
python scripts/visualize_training_metrics.py --checkpoint-dir checkpoints/
python scripts/visualize_training_metrics.py --tensorboard-dir runs/
```

**Outputs:**
- Loss curves
- Learning rate schedules
- Gradient norm plots
- Throughput graphs
- GPU memory usage

---

## Configuration Files

All scripts use YAML configuration files from `/config/`:

```yaml
# Example: config/config_3090.yaml
model:
  embed_dim: 512
  n_layers: 12
  num_heads: 8

train:
  batch_size: 32
  max_steps: 100000
  lr: 2.0e-4

data:
  dataset_name: "wikipedia"
  streaming: true
```

**Available Configs:**
- `config_quick_test.yaml` - Quick testing (256-dim, 6 layers, 5K steps)
- `config_3090.yaml` - RTX 3090 optimized (512-dim, 12 layers)
- `config_2x3090_7B.yaml` - Multi-GPU 7B model
- `config_H100_13B.yaml` - H100 13B model

---

## Common Workflows

### Start New Training Run

```bash
# 1. Download dataset
python scripts/download_dataset.py --dataset wikipedia

# 2. Start training
python scripts/train.py --config config/config_3090.yaml

# 3. Monitor progress (automatic in train.py)
# Or use TensorBoard if enabled in config
```

### Resume Training

```bash
python scripts/train.py --config config/config_3090.yaml --resume
```

### Evaluate Checkpoint

```bash
# Calculate perplexity
python scripts/eval_perplexity.py --checkpoint checkpoints/step_10000

# Test generation
python scripts/generate.py --checkpoint checkpoints/step_10000 \
  --prompt "The future of AI is"

# Verify generation quality
python scripts/verify_generation.py --checkpoint checkpoints/step_10000
```

### Find Best Model

```bash
# Find best checkpoint
python scripts/find_best_checkpoint.py --checkpoint-dir checkpoints/

# Use best checkpoint for inference
python scripts/generate.py --checkpoint checkpoints/best_model
```

### Cleanup Disk Space

```bash
# Keep only last 5 checkpoints
python scripts/cleanup_old_checkpoints.py --checkpoint-dir checkpoints/ --keep-last 5

# Visualize training before cleanup
python scripts/visualize_training_metrics.py --checkpoint-dir checkpoints/
```

---

## Archived Scripts

Historical and debugging scripts have been moved to `/archive/scripts/`:

- `monitoring_v2.py` - Old monitoring system (replaced by src/monitoring/)
- `profile_bottleneck.py` - Performance profiling (use PyTorch Profiler instead)
- `check_training_losses.py` - Loss verification (built into train.py)
- `check_wiki_dataset.py` - Dataset verification (use inspect_training_batch.py)
- `compare_datasets.py` - Dataset comparison (use create_mixed_dataset.py)
- `compare_sampling.py` - Sampling comparison (use grid_search_generation.py)
- `integration_example.py` - Integration example (see docs/)
- `test_validation.py` - Validation tests (moved to tests/)

---

## Development Guidelines

### Adding New Scripts

1. Follow naming convention: `verb_noun.py` (e.g., `train.py`, `generate.py`)
2. Add comprehensive docstring with usage examples
3. Use `argparse` for CLI arguments with `--help` support
4. Import from `src/` modules (not legacy imports)
5. Use `scripts/utils.py` for common functionality
6. Document in this README.md

### Script Requirements

- Must work with config files (YAML)
- Must handle errors gracefully
- Must support `--help` argument
- Must log progress for long-running operations
- Must be testable (unit tests in `/tests/`)

### Import Guidelines

**✅ Correct:**
```python
from src.models import LLMTransformer, Config
from src.training import SLGATrainer
from src.data import build_loaders
from src.monitoring import RealtimeTrainingDisplay
```

**❌ Incorrect:**
```python
from src.legacy.model import ...  # Use src.models instead
from src.realtime_display import ...  # Use src.monitoring.realtime_display
from src.validation import ...  # Deleted module
```

---

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, verify:
1. You're running from project root (`/mnt/d/ai/SLGA2`)
2. Python path includes project root
3. Module hasn't been moved/deleted during refactoring

### CUDA Out of Memory

```bash
# Reduce batch size in config
batch_size: 16  # Instead of 32

# Enable gradient checkpointing
grad_checkpointing: true

# Use smaller sequence length
seq_len_start: 256  # Instead of 512
```

### Slow Training

```bash
# Enable AMP
amp: true
amp_dtype: "fp16"  # For RTX 3090

# Disable torch.compile if unstable
torch_compile: false

# Increase batch size if GPU underutilized
batch_size: 32  # Up from 16
```

---

## Support

- **Documentation:** `/docs/`
- **Issues:** Check `/docs/FINAL_CLEANUP_PLAN.md` for migration guides
- **Config Examples:** `/config/`
- **Tests:** `/tests/`

---

**Production Status:** ✅ All scripts verified functional as of 2025-11-12
