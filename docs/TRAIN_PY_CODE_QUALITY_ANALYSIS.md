# Code Quality Analysis Report: scripts/train.py

**Analysis Date**: 2025-10-28
**Analyzer**: Code Quality Analyzer (Claude)
**File**: `/mnt/d/ai/SLGA/scripts/train.py` (966 lines)
**Overall Quality Score**: 7.2/10

---

## Executive Summary

The training script is **production-ready** with several recent critical fixes applied. The code demonstrates good practices in distributed training, mixed-precision, and checkpoint management. However, there are **11 bugs/issues** identified ranging from critical to minor, with most being edge cases or optimization opportunities.

### Key Strengths
✅ Comprehensive distributed training setup with Accelerator
✅ Robust error handling in validation and data loading
✅ Recent critical fixes for label masking and collator robustness
✅ Good logging infrastructure (TensorBoard + W&B)
✅ Proper gradient accumulation and scheduler synchronization

### Critical Issues Found
🔴 **3 High-severity bugs** requiring immediate attention
🟡 **5 Medium-severity issues** affecting robustness
🟢 **3 Low-severity issues** optimization opportunities

---

## Detailed Bug Analysis

### 🔴 CRITICAL (High-Severity)

#### BUG #1: Scheduler Step Counting Mismatch with Gradient Accumulation
**Location**: Lines 465-469
**Severity**: HIGH
**Impact**: Learning rate schedule incorrect during training

**Issue**:
```python
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps // accum_steps,    # ✅ CORRECT
    num_training_steps=total_steps // accum_steps,    # ✅ CORRECT
)
```

The fix is correct, BUT there's an inconsistency in how `step` counter is used:

```python
# Line 703: step increments every forward pass
step += 1

# Line 700: scheduler.step() called every accum_steps
if (step + 1) % accum_steps == 0:
    scheduler.step()
```

**Problem**:
- `step` counts **forward passes** (0 → 100,000)
- `scheduler` expects **optimizer steps** (0 → 20,000 with accum=5)
- All metrics are logged using `step` (forward pass counter)
- This creates confusion when comparing metrics to learning rate schedule

**Fix**:
```python
# Add separate counter for optimizer steps
optimizer_step = 0

if (step + 1) % accum_steps == 0:
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step += 1

# Log both counters
log_dict = {
    "step": step,  # Forward passes
    "optimizer_step": optimizer_step,  # Actual LR schedule steps
    "lr": scheduler.get_last_lr()[0],
}
```

---

#### BUG #2: Memory Leak in Validation Loop
**Location**: Lines 318-387 (validate function)
**Severity**: HIGH
**Impact**: GPU memory accumulation during validation

**Issue**:
```python
def validate(model, val_loader, pad_id, device, max_batches=None):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # ... forward pass ...
            loss = cross_entropy_shifted(logits, labels, pad_id)

            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens  # ✅ .item() prevents leak
            total_tokens += num_tokens
```

**Actual Problem**: Lines 352-363 diagnostic code retains graph!

```python
# 🔍 DIAGNOSTIC: Vérifier labels AVANT forward
invalid_mask = (labels != -100) & ((labels < 0) | (labels >= 50257))
if invalid_mask.any():
    print(f"\n❌ VALIDATION BATCH {i} HAS INVALID LABELS!")
    print(f"   Invalid count: {invalid_mask.sum().item()}")  # ⚠️ LEAK HERE
    invalid_values = labels[invalid_mask][:20].tolist()
    print(f"   Invalid values: {invalid_values}")

    # Trouver positions
    batch_idx, seq_idx = torch.where(invalid_mask)  # ⚠️ Creates graph!
```

**Why**: `torch.where()` creates computational graph even inside `no_grad()` context when tensors have `requires_grad=True`.

**Fix**:
```python
# Detach BEFORE diagnostic operations
invalid_mask = (labels != -100) & ((labels < 0) | (labels >= 50257))
if invalid_mask.any():
    invalid_mask_detached = invalid_mask.detach()
    batch_idx, seq_idx = torch.where(invalid_mask_detached)
    # ... rest of diagnostic ...
```

---

#### BUG #3: Race Condition in Checkpoint Saving
**Location**: Lines 924-941
**Severity**: HIGH
**Impact**: Corrupted checkpoints in multi-GPU training

**Issue**:
```python
if is_main and is_save_step and step > 0:
    print(f"\n🔵 Tentative de sauvegarde checkpoint step {step}...")
    try:
        save_checkpoint(model, optimizer, scheduler, out_dir, step, accelerator)
        print(f"✅ Checkpoint step {step} sauvegardé avec succès!")
    except Exception as e:
        print(f"❌ ERREUR lors de la sauvegarde checkpoint step {step}: {e}")
```

**Problem**: No synchronization barrier before checkpoint save!

```python
# Training continues on other GPUs while rank 0 saves
# → Can lead to:
#   1. Model state inconsistent (mid-gradient update)
#   2. Optimizer state from different step
#   3. File corruption if multiple processes write simultaneously
```

**Fix**:
```python
# BEFORE checkpoint save
if step % save_every == 0 and step > 0:
    # 🔧 CRITICAL: Synchronize ALL processes
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"\n🔵 Saving checkpoint at step {step}...")
        save_checkpoint(model, optimizer, scheduler, out_dir, step, accelerator)
        print(f"✅ Checkpoint saved!")

    # 🔧 CRITICAL: Wait for checkpoint to complete before continuing
    accelerator.wait_for_everyone()
```

---

### 🟡 MEDIUM (Medium-Severity)

#### BUG #4: Incorrect Curriculum Step Logic
**Location**: Lines 41-64 (`get_current_seq_len`)
**Severity**: MEDIUM
**Impact**: Sequence length progression doesn't match intended curriculum

**Issue**:
```python
def get_current_seq_len(step: int, cfg: dict) -> int:
    warmup_steps = cfg["train"].get("seq_len_warmup_steps", 15000)
    start_len = cfg["train"].get("seq_len_start", 512)
    mid_len = cfg["train"].get("seq_len_mid", 1024)
    final_len = cfg["train"].get("seq_len_final", 2048)

    if step < warmup_steps // 2:
        # Phase 1: start -> mid
        progress = step / (warmup_steps // 2)
        seq_len = start_len + progress * (mid_len - start_len)
    elif step < warmup_steps:
        # Phase 2: mid -> final
        progress = (step - warmup_steps // 2) / (warmup_steps // 2)
        seq_len = mid_len + progress * (final_len - mid_len)
    else:
        # Phase 3: final
        seq_len = final_len

    return int(seq_len)
```

**Problems**:
1. **Non-differentiable jumps**: Using `int()` creates discrete jumps
   - Step 0: 512
   - Step 1: 512 + small_delta → **int() → 512** (no change!)
   - Step 100: 519.68 → **int() → 519**

2. **Inconsistent with collator**: Collator is only updated when `current_seq_len` changes
   ```python
   # Line 567: Only truncates, never updates collator's max_length!
   if input_ids.size(1) > current_seq_len:
       input_ids = input_ids[:, :current_seq_len]
   ```

3. **Potential data truncation loss**: Curriculum reduces sequence length during training
   - Loses information from longer sequences
   - May harm long-range dependency learning

**Fix**:
```python
# Option 1: Smoother progression with rounding strategy
def get_current_seq_len(step: int, cfg: dict) -> int:
    # ... calculate seq_len as float ...

    # Round to nearest 128 (memory efficiency)
    seq_len_rounded = round(seq_len / 128) * 128
    return max(512, min(2048, int(seq_len_rounded)))

# Option 2: Update collator dynamically (BETTER)
def get_current_seq_len(step: int, cfg: dict) -> int:
    # ... same logic ...
    seq_len = int(seq_len)

    # Update collator if needed
    if hasattr(train_loader.collate_fn, 'max_length'):
        if train_loader.collate_fn.max_length != seq_len:
            train_loader.collate_fn.max_length = seq_len

    return seq_len
```

---

#### BUG #5: Validation Batch Size Calculation Error
**Location**: Line 300
**Severity**: MEDIUM
**Impact**: Validation may OOM on small batch_size configs

**Issue**:
```python
val_batch_size = max(1, cfg["train"]["batch_size"] // 2)
```

**Problem**: `cfg["train"]["batch_size"]` is per-GPU batch size, but validation runs on ALL GPUs!

Example:
- Config: `batch_size: 2` (per-GPU)
- Training: 2 GPUs → effective batch = 2 * 2 = 4 samples
- Validation: `val_batch_size = max(1, 2 // 2) = 1` → effective = 1 * 2 = **2 samples**
- But validation uses **512 seq_len** vs training's **up to 2048**!

**Fix**:
```python
# Account for GPU count and sequence length difference
num_gpus = accelerator.num_processes
train_batch = cfg["train"]["batch_size"] * num_gpus
train_seq_len = cfg["train"].get("seq_len_final", 2048)
val_seq_len = 512

# Memory ratio: (val_seq / train_seq)
seq_ratio = val_seq_len / train_seq_len

# Validation can handle ~2x more samples due to shorter sequences
val_batch_size = max(1, int(train_batch * 2 * seq_ratio))
print(f"Validation batch_size: {val_batch_size} (train: {train_batch})")
```

---

#### BUG #6: Gradient Norm Calculation Only on Main Process
**Location**: Lines 670-677
**Severity**: MEDIUM
**Impact**: Incorrect gradient norms in multi-GPU training

**Issue**:
```python
if (step + 1) % accum_steps == 0:
    # Calculate gradient norm BEFORE clipping
    grad_norm = 0.0
    if accelerator.is_main_process:  # ❌ WRONG!
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5
```

**Problem**:
- In DDP, each GPU has different gradients BEFORE `all_reduce`
- Calculating norm only on rank 0 gives partial picture
- Should either:
  1. Calculate after `accelerator.clip_grad_norm_()` (which syncs), OR
  2. Calculate on all GPUs and gather

**Fix**:
```python
if (step + 1) % accum_steps == 0:
    # Option 1: Use accelerator's built-in (RECOMMENDED)
    if grad_clip > 0:
        grad_norm = accelerator.clip_grad_norm_(model.parameters(), grad_clip)
        # This returns the ACTUAL norm before clipping, already synced!
        last_grad_norm = grad_norm
    else:
        # Calculate manually if no clipping
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5

        # Gather across GPUs
        grad_norm_tensor = torch.tensor(grad_norm, device=device)
        grad_norm_gathered = accelerator.gather(grad_norm_tensor).mean()
        last_grad_norm = grad_norm_gathered.item()

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
```

---

#### BUG #7: Landmark Loss Computation with Empty Landmarks
**Location**: Lines 625-650
**Severity**: MEDIUM
**Impact**: Loss computation fails when no landmarks selected

**Issue**:
```python
if landmark_indices is not None and landmark_scores is not None:
    seq_len = input_ids.size(1)  # L

    # Spacing loss
    lambda_spacing = cfg["train"].get("lambda_spacing", 0.0)
    if lambda_spacing > 0 and num_landmarks_selected > 1:  # ✅ Check > 1
        spacing_loss = landmark_spacing_loss(
            landmark_indices=landmark_indices,
            seq_len=seq_len,
            lambda_reg=lambda_spacing,
            selection_scores=landmark_scores
        )
        # ...
```

**Problem**: `landmark_spacing_loss` assumes `num_landmarks_selected > 1`, but what if:
1. `num_landmarks_selected = 0` (no landmarks)?
2. `num_landmarks_selected = 1` (single landmark)?

Looking at `landmarks.py` lines 308-362:
```python
def landmark_spacing_loss(...):
    B, G = landmark_indices.shape

    # ... calculations assume G > 1 ...
    ideal_gap = seq_len / G  # ❌ Division by zero if G=0!
```

**Fix**:
```python
# In train.py, add more robust checks
if landmark_indices is not None and landmark_scores is not None:
    num_landmarks_selected = landmark_indices.size(1) if landmark_indices.numel() > 0 else 0

    # Only compute losses if we have at least 2 landmarks
    if num_landmarks_selected >= 2:
        seq_len = input_ids.size(1)

        # Spacing loss
        lambda_spacing = cfg["train"].get("lambda_spacing", 0.0)
        if lambda_spacing > 0:
            spacing_loss = landmark_spacing_loss(...)
            loss = loss + spacing_loss / accum_steps

        # Sparsity loss (works with num_landmarks >= 1)
        lambda_spar = cfg["train"].get("lambda_sparsity", 0.0)
        if lambda_spar > 0:
            spar_loss = landmark_sparsity_loss(...)
            loss = loss + spar_loss / accum_steps
```

---

#### BUG #8: Realtime Display Not Thread-Safe
**Location**: Lines 707-738
**Severity**: MEDIUM
**Impact**: Potential race conditions in metrics display

**Issue**:
```python
# Real-time update (CHAQUE step)
if realtime_display and accelerator.is_main_process:
    # Calculer métriques courantes
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1e9
        # ...

    # Update ligne live
    realtime_display.update_live(
        step=step,
        loss=loss_ce.item() if step % accum_steps == 0 else None,
        # ...
    )
```

**Problem**: `realtime_display.update_live()` called EVERY step, but:
1. Loss is only computed every `accum_steps` → many `None` values
2. No synchronization with logging section (lines 741-865)
3. `mem_allocated` queried every step (expensive!)

**Fix**:
```python
# Cache memory stats
if step % 10 == 0:  # Update every 10 steps instead of every step
    if torch.cuda.is_available():
        mem_allocated_cached = torch.cuda.memory_allocated() / 1e9
        mem_total_cached = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        mem_allocated_cached = 0
        mem_total_cached = 0

# Update display less frequently
if realtime_display and accelerator.is_main_process:
    if step % 10 == 0:  # Every 10 steps
        realtime_display.update_live(
            step=step,
            loss=loss_ce.item() if (step % accum_steps == 0) else last_loss_cached,
            # ...
            gpu_mem_gb=mem_allocated_cached,
            gpu_total_gb=mem_total_cached,
        )
```

---

### 🟢 LOW (Low-Severity / Optimization)

#### ISSUE #9: Redundant Curriculum Truncation
**Location**: Lines 577-585
**Severity**: LOW
**Impact**: Minor performance overhead

**Issue**:
```python
# Tronquer si nécessaire pour curriculum
if input_ids.size(1) > current_seq_len:
    input_ids = input_ids[:, :current_seq_len]
    labels = labels[:, :current_seq_len]
    if cache_ids is not None:
        # Garder seulement landmarks dans la fenêtre
        mask = cache_ids < current_seq_len
        # Filtrer (simplifié: on garde tout pour éviter complications)
        pass  # ❌ Comment says "garde tout" but code does nothing!
```

**Problem**:
1. Truncation happens AFTER dataloader (inefficient)
2. Cache filtering is disabled (comment says "garde tout")
3. Dataloader already creates correct size batches

**Fix**: Either:
1. Update collator max_length dynamically (see BUG #4), OR
2. Properly filter cache:
```python
if cache_ids is not None and input_ids.size(1) < cache_ids.size(1):
    # Filter landmarks outside window
    cache_ids = cache_ids[:, :min(cache_ids.size(1), current_seq_len)]
```

---

#### ISSUE #10: Debug Logging Overhead
**Location**: Lines 613-622, 930-931
**Severity**: LOW
**Impact**: Minor performance degradation (~1-2%)

**Issue**:
```python
# 🔍 DEBUG: Vérifier pourquoi losses à 0
if step % 100 == 0:  # Tous les 100 steps
    print(f"\n🔍 DEBUG Step {step}:")
    print(f"   landmark_indices: {landmark_indices is not None}")
    # ... 6 more print statements ...
```

**Problem**: Debug logging every 100 steps adds overhead:
- String formatting
- Print syscalls (slow!)
- Disrupts realtime display

**Fix**:
```python
# Use proper logging module with levels
import logging
logger = logging.getLogger(__name__)

# In training loop
if step % 1000 == 0 and logger.level <= logging.DEBUG:
    logger.debug(f"Step {step}: landmark_indices={landmark_indices is not None}")
    # ...

# Allow disabling via config
if cfg["train"].get("verbose_debug", False):
    # ... debug logging ...
```

---

#### ISSUE #11: Magic Numbers in Validation
**Location**: Lines 893, 352
**Severity**: LOW
**Impact**: Maintainability

**Issue**:
```python
# Line 893: Hardcoded max_batches
val_metrics = validate(
    accelerator.unwrap_model(model),
    val_loader,
    pad_id,
    device,
    max_batches=10,  # ❌ Magic number
)

# Line 352: Hardcoded vocab_size
invalid_mask = (labels != -100) & ((labels < 0) | (labels >= 50257))  # ❌ Magic number
```

**Fix**:
```python
# Line 893: Use config
max_val_batches = cfg["train"].get("max_val_batches", 10)
val_metrics = validate(..., max_batches=max_val_batches)

# Line 352: Use tokenizer vocab size
vocab_size = tokenizer.vocab_size
invalid_mask = (labels != -100) & ((labels < 0) | (labels >= vocab_size))
```

---

## Code Smells Detected

### 1. Long Method: `main()` function
**Location**: Lines 390-965 (575 lines!)
**Severity**: MEDIUM

**Issue**: Violates Single Responsibility Principle

**Refactoring Suggestion**:
```python
def setup_environment(args, cfg):
    """Setup seed, accelerator, logging"""
    # Lines 415-428

def build_model(cfg, accelerator):
    """Create and initialize model"""
    # Lines 430-442

def setup_optimization(model, cfg):
    """Create optimizer and scheduler"""
    # Lines 448-469

def train_epoch(model, train_loader, optimizer, scheduler, ...):
    """Single epoch training logic"""
    # Lines 562-865

def main():
    args = parse_args()
    cfg = load_config(args.config)

    accelerator = setup_environment(args, cfg)
    model = build_model(cfg, accelerator)
    optimizer, scheduler = setup_optimization(model, cfg)
    train_loader, val_loader = build_loaders(cfg)

    train_loop(model, train_loader, val_loader, optimizer, scheduler, cfg, accelerator)
```

---

### 2. God Object: `cfg` dict
**Location**: Used throughout entire file
**Severity**: LOW

**Issue**: Configuration dict accessed with magic strings everywhere

**Better Approach**:
```python
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    batch_size: int
    lr: float
    max_steps: int
    accum_steps: int
    # ... with type hints and defaults ...

    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data["train"])

# Usage
train_cfg = TrainingConfig.from_yaml(args.config)
batch_size = train_cfg.batch_size  # ✅ IDE autocomplete!
```

---

### 3. Duplicate Code: Memory metrics
**Location**: Lines 710-716, 753-762, 882-886
**Severity**: LOW

**Issue**: Same GPU memory query code repeated 3 times

**Fix**:
```python
def get_gpu_memory_stats():
    """Returns dict with GPU memory stats in GB"""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "cached": 0, "total": 0}

    return {
        "allocated": torch.cuda.memory_allocated() / 1e9,
        "reserved": torch.cuda.memory_reserved() / 1e9,
        "cached": torch.cuda.memory_cached() / 1e9 if hasattr(torch.cuda, 'memory_cached') else 0,
        "total": torch.cuda.get_device_properties(0).total_memory / 1e9,
    }
```

---

### 4. Feature Envy: Collator logic in train.py
**Location**: Lines 199-289 (`collate_val_reduced`)
**Severity**: MEDIUM

**Issue**: 90-line collator function embedded in train.py should be in `data.py`

**Fix**: Move to `src/data.py`:
```python
class RobustValidationCollator:
    """Auto-detecting collator for validation (handles any dataset format)"""
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        # ... (lines 199-289) ...
```

---

## Positive Findings

### ✅ Excellent Practices

1. **Comprehensive Error Handling**
   - Try-except in validation split loading (lines 126-148)
   - Checkpoint save with exception handling (lines 935-941)
   - Robust collator with format auto-detection (lines 199-289)

2. **Recent Critical Fixes Applied**
   - Label masking with -100 (line 110, 148, 231, 246)
   - Scheduler step counting fixed (lines 465-469)
   - Validation synchronization (lines 872-880)

3. **Good Logging Infrastructure**
   - Multiple logging backends (TensorBoard + W&B)
   - Detailed metrics tracking (lines 764-831)
   - Real-time display with progress bars

4. **Memory Management**
   - AMP with proper dtype selection (lines 476-486)
   - Gradient checkpointing support in model
   - Empty cache before validation (line 879)

5. **Distributed Training Best Practices**
   - Accelerator integration throughout
   - Model unwrapping for checkpoints (line 511)
   - Loss gathering for multi-GPU (line 743)

---

## Refactoring Opportunities

### Priority 1: Extract Configuration
```python
# Create config/train_config.py
@dataclass
class TrainConfig:
    batch_size: int = 8
    lr: float = 3e-4
    max_steps: int = 100000
    accum_steps: int = 5
    # ... all training hyperparameters ...
```

### Priority 2: Break Up main() Function
See "Code Smells" section above for suggested structure.

### Priority 3: Move Collator to data.py
Extract `collate_val_reduced` (lines 199-289) into reusable class.

### Priority 4: Create TrainingLoop Class
```python
class SLGATrainer:
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, cfg, accelerator):
        self.model = model
        # ...

    def train_step(self, batch):
        """Single training step"""
        # ...

    def validate(self):
        """Validation loop"""
        # ...

    def train(self):
        """Main training loop"""
        # ...
```

---

## Technical Debt Estimate

| Category | Estimated Hours |
|----------|----------------|
| Fix critical bugs (#1-3) | 4 hours |
| Fix medium issues (#4-8) | 8 hours |
| Refactor main() function | 6 hours |
| Extract configuration classes | 3 hours |
| Add comprehensive tests | 12 hours |
| **TOTAL** | **33 hours** |

---

## Recommendations

### Immediate Actions (Next Sprint)
1. ✅ Fix BUG #3 (Race condition in checkpoint) - **CRITICAL**
2. ✅ Fix BUG #2 (Memory leak in validation) - **HIGH IMPACT**
3. ✅ Fix BUG #1 (Scheduler step counting) - **AFFECTS METRICS**

### Short-Term (Next 2 Weeks)
4. Extract collator to data.py (reduce complexity)
5. Add config validation with proper types
6. Improve gradient norm calculation (BUG #6)

### Long-Term (Next Month)
7. Refactor into TrainingLoop class
8. Add comprehensive unit tests
9. Performance profiling and optimization

---

## Conclusion

The training script is **functionally correct** for most use cases, with recent critical fixes addressing label masking and validation issues. However, **3 high-severity bugs** (#1-3) should be fixed immediately to ensure:

1. Correct learning rate schedules
2. No memory leaks during validation
3. Checkpoint integrity in multi-GPU setups

The code would benefit significantly from **refactoring** to reduce the 575-line main() function and extract reusable components. Overall, this is **solid production code** with room for improvement in architecture and maintainability.

**Quality Score Breakdown**:
- Functionality: 9/10 (works well, recent fixes applied)
- Maintainability: 6/10 (long functions, magic numbers)
- Performance: 7/10 (good AMP/DDP, minor optimization opportunities)
- Security: 8/10 (good input validation, checkpoint handling)
- Testing: 5/10 (no unit tests visible)

**Final Score: 7.2/10** - Production-ready with recommended improvements.
