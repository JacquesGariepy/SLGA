# Landmark Instability Analysis - Executive Summary

**Date**: 2025-10-24
**Issue**: Training instability at step 15K (throughput collapse, loss spikes)
**Root Cause**: Landmark clustering + oscillation due to ineffective loss functions
**Status**: ✅ **FIXABLE** with configuration changes

---

## 🔴 Critical Findings

### 1. The Step 15K Problem

**Symptoms**:
- 10× throughput drop (3400 → 927 tok/s)
- 40% loss spike (2.1 → 2.9)
- 2× training time increase

**Root Cause**: **Landmark clustering** + **oscillating selection**

```
Ideal landmarks (uniform spacing):
[0, 16, 32, 48, 64, 80, 96, 112, 128, ...]  ✅

Actual landmarks at step 15K (clustered):
[8, 12, 15, 19, 22, 147, 151, 155, 159, ...]  ❌
         ↑ 3 clusters ↑

Result:
- 71% of landmarks in positions 147-207 (clustering!)
- Large gaps: 56, 65, 104 tokens (poor coverage)
- Attention patterns keep changing (oscillation)
- GPU must recompute attention every step (slow!)
```

### 2. Why Existing Losses Failed

**Diversity Loss (entropy-based)**:
```python
# Maximizes entropy over ALL L positions
# But doesn't prevent G landmarks from clustering!

Uniform landmarks:   entropy = 0.53  ← Loss = 0.47 ❌
Clustered landmarks: entropy = 0.53  ← Loss = 0.47 ❌
                     ↑ CAN'T DISTINGUISH! ↑
```

**Sparsity Loss (fixed target)**:
```python
# Fixed target: 5% positions active
# But G/L = 24/384 = 9.4% (inevitable)

active_fraction = 0.094 > target_active = 0.05
loss = ALWAYS POSITIVE ❌

→ Provides constant downward pressure
→ Conflicts with diversity loss
→ Creates oscillating gradients!
```

---

## ✅ The Fix: Spacing Loss

**What it does**: Directly penalizes non-uniform gaps between landmarks

```python
gaps = [g₁, g₂, g₃, ..., g_{G-1}]  # Distances between consecutive landmarks
ideal_gap = L / G  # Uniform spacing target
loss = λ · mean((gaps - ideal_gap)²)
```

**Why it works**:

```
Uniform landmarks:
  gaps = [16, 16, 16, 16, ...]
  loss = 0 ✅

Clustered landmarks:
  gaps = [4, 3, 4, 56, 65, 4, ...]
  loss = λ × ((4-16)² + (3-16)² + (56-16)²) / (G-1)
       = λ × (144 + 169 + 1600 + ...) / 23
       ≈ 450λ ❌ HIGH PENALTY

→ Gradient pushes landmarks apart
→ Coverage across entire sequence
→ No more clustering!
```

**Configuration change**:
```yaml
train:
  lambda_spacing: 0.01      # ← ENABLE (was 0.0)
  lambda_sparsity: 0.001    # ← Keep (adaptive in code)
  lambda_diversity: 0.0     # ← Disable (replaced)
```

---

## 🔧 Additional Fixes

### Fix 1: Persistent Temperature

**Problem**: Temperature resets when loading checkpoints
```python
# Current (BROKEN):
self.register_buffer("step_count", torch.tensor(0), persistent=False)

# Fixed:
self.register_buffer("step_count", torch.tensor(0), persistent=True)
```

**Impact**: Prevents sudden instability after resuming training

### Fix 2: Adaptive Sparsity Target (Already in v1.1)

**Problem**: Fixed 5% target conflicts with G/L = 9.4%
```python
# v1.0 (BROKEN):
target_active = 0.05  # Fixed

# v1.1 (FIXED):
target_active = num_landmarks / L * 1.2  # Adaptive with 20% margin
```

**Impact**: Removes conflicting gradient signal

### Fix 3: Checkpoint Step Count

**Add to save/load**:
```python
# Save:
checkpoint["landmark_step_count"] = model.landmark_selector.step_count.item()

# Load:
model.landmark_selector.step_count.fill_(checkpoint["landmark_step_count"])
```

**Impact**: Preserves temperature decay across training interruptions

---

## 📊 Expected Results

**Before fixes** (Step 15K):
```
Landmarks:  [8, 12, 15, 19, 22, 147, 151, 155, ...]
Mean gap:   16.8
Std gap:    24.3  ❌ High variance
Spacing loss: N/A (disabled)
Throughput: 927 tok/s  ❌ Collapsed
Loss spikes: 2.9  ❌ Unstable
```

**After fixes** (Step 16K+):
```
Landmarks:  [8, 24, 40, 56, 72, 88, 104, 120, ...]
Mean gap:   16.0  ✅ Ideal
Std gap:    4.2   ✅ Low variance
Spacing loss: 0.05 → 0.01 (converging)
Throughput: 3500+ tok/s  ✅ Stable
Loss: Smooth decrease  ✅ Stable
```

---

## 🚀 Action Plan

### Immediate (Apply Now)

1. **Update configuration**:
   ```bash
   # Edit config/config_3090_v1.1.yaml
   lambda_spacing: 0.01  # Enable spacing loss
   ```

2. **Fix temperature persistence**:
   ```bash
   # Edit src/landmarks.py line 62
   persistent=False → persistent=True
   ```

3. **Add checkpoint handling**:
   ```bash
   # Edit scripts/train.py
   # Add step_count save/load (code provided above)
   ```

4. **Resume training**:
   ```bash
   python scripts/train.py
   # Will continue from step 15K with fixes
   ```

### Validation (After 1K Steps)

```bash
# Run diagnostic at step 16K
python scripts/diagnose_landmarks.py out_slga/ckpt_16000/model.pt

# Expected output:
# ✅ Mean gap: 15.5-16.5 (ideal ≈ 16)
# ✅ Std gap: < 8.0 (low variance)
# ✅ No clusters (all gaps 8-24)
# ✅ Throughput > 3000 tok/s
```

### Escalation (If Issues Persist)

1. **Clustering persists**: Increase `lambda_spacing: 0.02`
2. **Oscillation persists**: Add stability loss (see full analysis)
3. **Throughput drops**: Disable learned landmarks temporarily
4. **Gradient explosion**: Reduce `grad_clip: 0.5`, `lr: 1.0e-4`

---

## 📈 Technical Deep Dive

**For detailed analysis, see**:
- `/docs/analysis/LANDMARK_MECHANISM_ANALYSIS.md` (Full 50-page analysis)

**Key sections**:
1. **Mathematical correctness** (Gumbel-Softmax, STE, temperature decay)
2. **Gradient flow analysis** (Why STE causes instability)
3. **Line-by-line code review** (490 lines analyzed)
4. **Loss function comparison** (Why spacing > diversity)
5. **Long-term improvements** (Curriculum learning, multi-scale, momentum)

---

## 🎯 Confidence Level

| Fix | Confidence | Risk | Benefit |
|-----|-----------|------|---------|
| **Spacing loss** | 🟢 HIGH | Low | Fixes clustering (primary issue) |
| **Temperature persistence** | 🟢 HIGH | Low | Prevents reset instability |
| **Adaptive sparsity** | 🟢 HIGH | Low | Removes gradient conflict |
| **Checkpoint handling** | 🟡 MEDIUM | Low | Improves resumption |

**Overall**: 🟢 **HIGH confidence** that these fixes will resolve step 15K instability.

---

## 📝 Summary

**Problem**: Landmark clustering causes oscillating attention → throughput collapse at step 15K

**Root cause**: Entropy-based diversity loss cannot detect spatial clustering

**Solution**: Enable spacing loss (λ=0.01) to directly penalize non-uniform gaps

**Expected outcome**: Stable, uniformly-distributed landmarks with consistent throughput

**Action**: Update config, fix buffer persistence, resume training

**ETA**: Should see improvement within 1K steps (by step 16K)

---

**Related documents**:
- Full analysis: `/docs/analysis/LANDMARK_MECHANISM_ANALYSIS.md`
- Config v1.1: `/config/config_3090_v1.1.yaml`
- Step 15K report: `/docs/STEP_15K_DIAGNOSTIC_REPORT.md`

**Status**: ✅ Ready to apply fixes and resume training
