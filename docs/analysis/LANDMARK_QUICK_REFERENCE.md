# Landmark Selection - Quick Reference Card

**File**: `src/landmarks.py`
**Purpose**: Differentiable selection of important tokens for global attention

---

## 📊 At a Glance

| Metric | Value | Notes |
|--------|-------|-------|
| **Selectors** | 3 types | Learnable, Positional, Hybrid |
| **Methods** | 2 algorithms | Gumbel-Softmax, Straight-Through |
| **Loss functions** | 3 types | Spacing, Sparsity, Diversity |
| **Complexity** | O(L·D²) | Dominated by scorer MLP |
| **Parameters** | ~(D²/2 + D) | 2-layer scorer |

---

## 🎯 Three Selectors

### 1. LearnableLandmarkSelector (Content-Based)
```python
selector = LearnableLandmarkSelector(
    embed_dim=512,
    num_landmarks=24,
    temperature=1.0,
    temperature_decay=0.999,  # v1.1: was 0.9999
    min_temperature=0.3       # v1.1: was 0.5
)
```

**How it works**:
1. Score each position with 2-layer MLP
2. Select top-K via Gumbel-Softmax or STE
3. Gather selected token embeddings

**Use case**: Adaptive to content, general purpose

### 2. PositionalLandmarkSelector (Position-Based)
```python
selector = PositionalLandmarkSelector(
    max_seq_len=2048,
    num_landmarks=24,
    embed_dim=512
)
```

**How it works**:
1. Learn importance of each position (fixed pattern)
2. Select top-K positions (same for all sequences)
3. Gather tokens at those positions

**Use case**: Structured text (code, markup, formal documents)

### 3. HybridLandmarkSelector (Fusion)
```python
selector = HybridLandmarkSelector(
    embed_dim=512,
    max_seq_len=2048,
    num_landmarks=24
)
```

**How it works**:
1. Run both content & positional selectors
2. Learned gate combines their scores
3. Re-select top-K from combined scores

**Use case**: Heterogeneous data, maximize robustness

---

## 🔧 Two Selection Methods

### Gumbel-Softmax (Exploration)
```python
indices, states, scores = selector(x, use_gumbel=True)
```

**Properties**:
- ✅ Stochastic exploration
- ✅ Theoretically sound gradients
- ❌ High variance
- ❌ Slower (softmax over L positions)

**When to use**: Early training, scratch training, RL-like tasks

### Straight-Through Estimator (Exploitation)
```python
indices, states, scores = selector(x, use_gumbel=False)
```

**Properties**:
- ✅ Deterministic
- ✅ Fast (no softmax)
- ✅ Low variance
- ❌ Biased gradients

**When to use**: Fine-tuning, production, stable training

**Config v1.1**: Uses STE by default (`use_gumbel: false`)

---

## 📉 Three Loss Functions

### 1. Spacing Loss (NEW in v1.1) ✅ RECOMMENDED

```python
loss = landmark_spacing_loss(
    landmark_indices=indices,  # (B, G)
    seq_len=L,
    lambda_reg=0.01
)
```

**Formula**:
```
gaps = [g₁, g₂, ..., g_{G-1}]  # Consecutive landmark distances
ideal = L / G
loss = λ × mean((gaps - ideal)²)
```

**Prevents**: Landmark clustering
**Config**: `lambda_spacing: 0.01`

### 2. Sparsity Loss (Adaptive in v1.1) ✅

```python
loss = landmark_sparsity_loss(
    selection_scores=scores,   # (B, L) softmax probs
    num_landmarks=G,
    lambda_reg=0.001
)
```

**Formula**:
```
target = (G / L) × 1.2  # 20% margin
active = count(scores > 0.01) / L
loss = λ × relu(active - target)
```

**Prevents**: Too many positions active
**Config**: `lambda_sparsity: 0.001`

### 3. Diversity Loss (DEPRECATED in v1.1) ❌

```python
loss = landmark_diversity_loss(
    selection_scores=scores,
    lambda_reg=0.0  # Disabled in v1.1
)
```

**Formula**:
```
H = -Σ(p × log(p))  # Entropy
loss = λ × (1 - H / log(L))
```

**Problem**: Cannot detect spatial clustering
**Config**: `lambda_diversity: 0.0` (disabled)

---

## 🌡️ Temperature Annealing

**Schedule**: Exponential decay
```
τ(t) = max(τ₀ × decay^t, τ_min)
```

**Parameters (v1.1)**:
```yaml
temperature: 1.0            # Starting temperature
temperature_decay: 0.999    # Decay factor (was 0.9999)
min_temperature: 0.3        # Floor (was 0.5)
```

**Convergence time**:
```
t* = log(τ_min / τ₀) / log(decay)
   = log(0.3 / 1.0) / log(0.999)
   ≈ 1,204 steps  (reaches minimum)
```

**Old values (v1.0)**: ~6,931 steps (5.7× slower!)

**Effect**:
- **High τ** (1.0): Soft, exploratory selection
- **Low τ** (0.3): Hard, discriminative selection

---

## 🔄 Forward Pass Flow

```
Input: x (B, L, D)
  ↓
1. Score each position
   scores = scorer(x)  # (B, L)
  ↓
2. Select top-K
   if use_gumbel:
       add Gumbel noise + temperature scaling
       soft_probs = softmax((scores + G) / τ)
       indices = topk(scores + G)  # Stochastic
   else:
       indices = topk(scores)      # Deterministic
  ↓
3. Gather landmarks
   landmark_states = gather(x, indices)  # (B, G, D)
  ↓
Output: indices (B, G), states (B, G, D), scores (B, L)
```

---

## 🐛 Common Issues & Fixes

### Issue 1: Landmark Clustering
```
Symptom: Landmarks at [147, 148, 149, ..., 163] (clustered)
Cause:   Diversity loss ineffective
Fix:     Enable spacing loss (lambda_spacing: 0.01)
```

### Issue 2: Oscillating Selection
```
Symptom: Landmarks change 20%+ per step
Cause:   STE bias + competitive scores + low temperature
Fix:     Add stability loss OR reduce min_temperature: 0.2
```

### Issue 3: Temperature Resets
```
Symptom: Temperature jumps to 1.0 after loading checkpoint
Cause:   Buffer not persistent (persistent=False)
Fix:     src/landmarks.py line 62: persistent=True
```

### Issue 4: Gradient Explosion
```
Symptom: grad_norm > 10.0
Cause:   Landmark competition + attention feedback
Fix:     Reduce grad_clip: 0.5, lr: 1.0e-4
```

### Issue 5: Sparsity Always Penalized
```
Symptom: Sparsity loss never reaches 0
Cause:   Fixed target (5%) < G/L (9.4%)
Fix:     v1.1 has adaptive target (already fixed)
```

---

## 📈 Monitoring Metrics

**TensorBoard logging**:
```python
# Position statistics
landmarks/mean_position     # Should ≈ L/2
landmarks/std_position      # Should ≈ L/4

# Spacing statistics
landmarks/mean_gap          # Should ≈ L/G (ideal: 16)
landmarks/std_gap           # Should < L/(2G) (target: <8)
landmarks/min_gap           # Should > L/(3G) (avoid clustering)
landmarks/max_gap           # Should < L/(G/2) (avoid large holes)

# Score statistics
landmarks/score_mean        # Typical: ~1/L after softmax
landmarks/score_std         # Should decrease during training
landmarks/temperature       # Should decay to min

# Stability
landmarks/changes_per_step  # Should < 5% (low churn)
```

**Healthy ranges**:
```
mean_gap:    14-18 (ideal: 16 for L=384, G=24)
std_gap:     2-8 (uniform spacing)
temperature: 0.3-1.0 (decaying)
changes:     0-5% (stable selection)
```

---

## 🎛️ Configuration Cheat Sheet

### Recommended Settings (v1.1)

```yaml
model:
  learned_landmarks: true
  global_k: 24
  landmark_selector:
    temperature_decay: 0.999    # 10× faster than v1.0
    min_temperature: 0.3        # More discriminative
    use_gumbel: false           # Use STE (stable)

train:
  # Loss weights
  lambda_spacing: 0.01          # ✅ PRIMARY FIX
  lambda_sparsity: 0.001        # ✅ With adaptive target
  lambda_diversity: 0.0         # ❌ DEPRECATED

  # Gradient control
  grad_clip: 1.0                # Prevents explosion

  # Warmup
  global_warmup_start: 1000     # Start early
  global_warmup_end: 5000       # Ramp up over 4K steps
```

### For Retraining from Scratch

```yaml
model:
  landmark_selector:
    temperature_decay: 0.999
    min_temperature: 0.2        # Even harder
    use_gumbel: false

train:
  lambda_spacing: 0.02          # 2× stronger (clean start)
  lambda_sparsity: 0.001
  lambda_diversity: 0.0
  global_warmup_start: 5000     # Later start
  global_warmup_end: 15000      # Longer ramp
```

### For Debugging Instability

```yaml
train:
  lambda_spacing: 0.03          # 3× stronger (aggressive)
  lambda_sparsity: 0.0005       # Weaker (less conflict)
  grad_clip: 0.5                # Stricter clipping
  lr: 1.0e-4                    # Lower LR
```

---

## 🔬 Diagnostic Commands

### Analyze Landmark Distribution
```bash
python scripts/diagnose_landmarks.py out_slga/ckpt_15000/model.pt
```

**Output**:
```
Position Statistics:
  Mean position: 192.3
  Std position:  110.5
  Min position:  8
  Max position:  376

Spacing Statistics:
  Ideal gap:     16.0
  Mean gap:      16.8
  Std gap:       24.3  ← HIGH = CLUSTERING!
  Min gap:       3     ← TOO SMALL!
  Max gap:       104   ← TOO LARGE!

Clustering Detection:
  Gaps < 8:   15 (65%)  ← PROBLEM!
  Gaps > 32:  5 (22%)   ← PROBLEM!
```

### Visualize Temperature Decay
```python
import torch
import matplotlib.pyplot as plt

tau_0 = 1.0
decay = 0.999
tau_min = 0.3

steps = torch.arange(0, 10000)
temps = torch.clamp(tau_0 * (decay ** steps), min=tau_min)

plt.plot(steps, temps)
plt.axhline(tau_min, color='r', linestyle='--', label=f'Min = {tau_min}')
plt.xlabel("Step")
plt.ylabel("Temperature")
plt.title("Temperature Decay Schedule")
plt.legend()
plt.savefig("temperature_schedule.png")
```

### Check Gradient Flow
```bash
# In train.py, add before optimizer.step():
for name, param in model.named_parameters():
    if "landmark_selector" in name and param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
```

**Healthy output**:
```
landmark_selector.scorer.0.weight: grad_norm=0.234
landmark_selector.scorer.0.bias: grad_norm=0.156
landmark_selector.scorer.3.weight: grad_norm=0.412
landmark_selector.scorer.3.bias: grad_norm=0.089
```

**Unhealthy output** (oscillation):
```
landmark_selector.scorer.0.weight: grad_norm=5.234  ← TOO HIGH!
landmark_selector.scorer.3.weight: grad_norm=12.41  ← EXPLODING!
```

---

## 🚨 Emergency Troubleshooting

### Landmarks All Zero
```bash
# Check if landmark_indices are actually returned
# See /docs/FIX_LANDMARKS_ZERO.md

# Was a monitoring bug (logs showed 0, but working)
# Fixed in v1.1: return landmark_indices in aux dict
```

### NaN/Inf in Training
```bash
# Check gradient norms
python -c "
import torch
ckpt = torch.load('out_slga/ckpt_15000/model.pt')
for k, v in ckpt['optimizer'].items():
    if 'grad' in k and v is not None:
        print(f'{k}: {v.norm():.4f}')
"

# If any grad_norm > 100: Gradient explosion
# Fix: Reduce LR, increase grad_clip
```

### Throughput Collapse
```bash
# Usually caused by unstable landmark selection
# Check if landmarks oscillate:

python scripts/diagnose_landmarks.py ckpt_1/model.pt > stats_1.txt
python scripts/diagnose_landmarks.py ckpt_2/model.pt > stats_2.txt
diff stats_1.txt stats_2.txt

# If positions change >20%: INSTABILITY!
# Fix: Enable spacing loss, add stability loss
```

---

## 📚 References

**Full documentation**:
- `/docs/analysis/LANDMARK_MECHANISM_ANALYSIS.md` (50 pages, line-by-line)
- `/docs/analysis/LANDMARK_ANALYSIS_SUMMARY.md` (Executive summary)
- `/docs/LANDMARKS_OPTIMIZATIONS.md` (v1.1 changelog)
- `/docs/STEP_15K_DIAGNOSTIC_REPORT.md` (Instability analysis)

**Code files**:
- `/src/landmarks.py` (Implementation)
- `/src/model.py` (Integration with SLGA)
- `/scripts/train.py` (Loss calculation, lines 450-476)
- `/config/config_3090_v1.1.yaml` (Configuration)

**Papers**:
- Jang et al. (2017): "Categorical Reparameterization with Gumbel-Softmax"
- Bengio et al. (2013): "Estimating or Propagating Gradients Through Stochastic Neurons"

---

**Last updated**: 2025-10-24
**Version**: 1.1 (with spacing loss)
**Status**: ✅ Ready for production
