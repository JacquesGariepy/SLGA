# 📊 Pipeline d'Entraînement SLGA - Résumé Visuel

**Version**: 1.0
**Date**: 2025-10-24
**GPU**: RTX 3090 (24GB)

---

## 🔄 Architecture Globale du Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    SLGA TRAINING PIPELINE                       │
│                    scripts/train.py (606 lines)                 │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────┐
    │  1. CONFIGURATION LOADING (config.yaml)          │
    │     - Model config (38M params)                  │
    │     - Training hyperparams (LR, batch, etc.)     │
    │     - Dataset config (Wikipedia → FineWeb)       │
    └──────────────────────────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────┐
    │  2. DATA LOADING (src/data.py)                   │
    │     ┌────────────────────────────────┐           │
    │     │ CollatorLocal (learned=true)   │           │
    │     │   → No heuristic landmarks     │           │
    │     │   → Model selects dynamically  │           │
    │     └────────────────────────────────┘           │
    │     ┌────────────────────────────────┐           │
    │     │ CollatorLocalGlobal (false)    │           │
    │     │   → Regular spacing (0,128,...)│           │
    │     │   → Fixed landmarks            │           │
    │     └────────────────────────────────┘           │
    └──────────────────────────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────┐
    │  3. CURRICULUM LEARNING (lines 39-81)            │
    │                                                  │
    │  Seq Length Progression:                         │
    │   Step 0      → 384 tokens  (6 GB GPU)          │
    │   Step 7,500  → 1024 tokens (12 GB GPU)         │
    │   Step 15,000 → 2048 tokens (18 GB GPU)         │
    │                                                  │
    │  Global Attention Warmup:                        │
    │   Step 0-1K   → weight = 0.0 (local only)       │
    │   Step 1K-5K  → weight = 0.0→1.0 (progressive)  │
    │   Step 5K+    → weight = 1.0 (full global)      │
    └──────────────────────────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────┐
    │  4. TRAINING LOOP (lines 266-602)                │
    │                                                  │
    │  FOR step in 0..100,000:                         │
    │    ┌─────────────────────────────────┐           │
    │    │ A. FORWARD PASS (AMP BF16)      │           │
    │    │    - Model(input_ids)           │           │
    │    │    - Landmark selection         │           │
    │    │    - Local + Global attention   │           │
    │    └─────────────────────────────────┘           │
    │                    │                             │
    │    ┌─────────────────────────────────┐           │
    │    │ B. LOSS COMPUTATION             │           │
    │    │    - Cross-entropy (shifted)    │           │
    │    │    - Diversity loss (0.02)      │           │
    │    │    - Sparsity loss (0.001)      │           │
    │    └─────────────────────────────────┘           │
    │                    │                             │
    │    ┌─────────────────────────────────┐           │
    │    │ C. BACKWARD PASS                │           │
    │    │    - Gradient computation       │           │
    │    │    - Gradient accumulation (4x) │           │
    │    └─────────────────────────────────┘           │
    │                    │                             │
    │    ┌─────────────────────────────────┐           │
    │    │ D. OPTIMIZER STEP (every 4)     │           │
    │    │    - Gradient clipping (1.0)    │           │
    │    │    - AdamW update               │           │
    │    │    - LR schedule (cosine)       │           │
    │    └─────────────────────────────────┘           │
    │                    │                             │
    │    ┌─────────────────────────────────┐           │
    │    │ E. VALIDATION (every 500)       │           │
    │    │    - 10 batches (~80 samples)   │           │
    │    │    - Compute loss & perplexity  │           │
    │    └─────────────────────────────────┘           │
    │                    │                             │
    │    ┌─────────────────────────────────┐           │
    │    │ F. LOGGING (every 50)           │           │
    │    │    - TensorBoard metrics        │           │
    │    │    - W&B (optional)             │           │
    │    │    - Console output             │           │
    │    └─────────────────────────────────┘           │
    │                    │                             │
    │    ┌─────────────────────────────────┐           │
    │    │ G. CHECKPOINTING (every 1000)   │           │
    │    │    - model.pt                   │           │
    │    │    - trainer_state.pt           │           │
    │    └─────────────────────────────────┘           │
    │  END FOR                                         │
    └──────────────────────────────────────────────────┘
```

---

## 📈 Progression Curriculum Learning

### Sequence Length (15,000 steps)

```
Seq Len
2048 ┤                            ╭──────────────────────
     ┤                           ╱
     ┤                          ╱
1536 ┤                         ╱
     ┤                        ╱
1024 ┤            ╭──────────╯
     ┤           ╱
 768 ┤          ╱
     ┤         ╱
 512 ┤        ╱
     ┤       ╱
 384 ┤──────╯
     └──────┴──────┴──────┴──────┴──────┴──────┴──────> Step
     0     2.5K    5K    7.5K   10K   12.5K   15K

Phase 1 (0→7.5K):    384 → 1024 (linear)
Phase 2 (7.5K→15K):  1024 → 2048 (linear)
Phase 3 (15K+):      2048 (constant)
```

### Global Attention Weight (5,000 steps)

```
Weight
1.0 ┤         ╭────────────────────────────────────────
    ┤        ╱
0.8 ┤       ╱
    ┤      ╱
0.6 ┤     ╱
    ┤    ╱
0.4 ┤   ╱
    ┤  ╱
0.2 ┤ ╱
    ┤╱
0.0 ┤────────╯
    └────────┴────────┴────────┴────────┴────────> Step
    0       1K       2K       3K       4K       5K

Step 0-1K:  weight = 0.0 (local only)
Step 1K-5K: weight = 0.0 → 1.0 (linear ramp)
Step 5K+:   weight = 1.0 (full global attention)
```

### GPU Memory Usage

```
Memory (GB)
24 ┤                                          ─────── (limit)
   ┤
20 ┤                              ╭──────────────────
   ┤                             ╱
16 ┤                       ╭────╯
   ┤                      ╱
12 ┤               ╭─────╯
   ┤              ╱
 8 ┤        ╭────╯
   ┤       ╱
 4 ┤  ╭───╯
   ┤ ╱
 0 ┤─╯
   └──────┴──────┴──────┴──────┴──────┴──────┴──────> Step
   0     2.5K    5K    7.5K   10K   12.5K   15K

Step 0:     6 GB  (seq=384)
Step 5K:    9 GB  (seq=683)
Step 10K:   14 GB (seq=1365)
Step 15K:   18 GB (seq=2048) ← Actuel
```

---

## 🎯 Data Flow dans une Itération

```
┌─────────────────────────────────────────────────────────────┐
│                    SINGLE TRAINING STEP                     │
└─────────────────────────────────────────────────────────────┘

INPUT (Batch)
  ├─ input_ids: (8, 2048)      ← Tokenized text
  ├─ labels: (8, 2048)         ← Shifted labels (target)
  └─ cache_global_ids: (8, 64) ← Landmark positions (optional)
                │
                ▼
        ┌──────────────────┐
        │  SLGA FORWARD    │
        │  (src/slga.py)   │
        └──────────────────┘
                │
                ├─── Local Attention (window=128)
                │     ├─ Query:  (8, 8, 2048, 64)
                │     ├─ Key:    (8, 8, 2048, 64)
                │     ├─ Value:  (8, 8, 2048, 64)
                │     └─ Output: (8, 8, 2048, 64)
                │
                ├─── Landmark Selection (if learned=true)
                │     ├─ Scores:  (8, 2048)       ← Softmax probs
                │     ├─ Top-K:   (8, 147)        ← Select candidates
                │     └─ Diverse: (8, 24)         ← Per-head diversity
                │
                └─── Global Attention (k=24)
                      ├─ Query:  (8, 8, 2048, 64)
                      ├─ Key:    (8, 8, 24, 64)   ← Landmarks only
                      ├─ Value:  (8, 8, 24, 64)
                      └─ Output: (8, 8, 2048, 64)
                │
                ├─── Gated Fusion
                │     ├─ gate = sigmoid(W·x)      ← Learned weights
                │     └─ out = gate·local + (1-gate)·global
                │
                ▼
        ┌──────────────────┐
        │  LOGITS          │
        │  (8, 2048, 50257)│  ← Vocabulary predictions
        └──────────────────┘
                │
                ▼
        ┌──────────────────┐
        │  LOSS COMPUTE    │
        │  (lines 83-111)  │
        └──────────────────┘
                │
                ├─── Cross-Entropy Loss
                │     ├─ Shift logits[:, :-1]
                │     ├─ Shift labels[:, :-1]
                │     └─ loss_ce = F.cross_entropy(...)
                │
                ├─── Diversity Loss (λ=0.02)
                │     └─ Penalize landmark clustering
                │
                └─── Sparsity Loss (λ=0.001)
                      └─ Penalize too many landmarks
                │
                ▼
        ┌──────────────────┐
        │  TOTAL LOSS      │
        │  / accum_steps   │  ← Divide by 4 for accumulation
        └──────────────────┘
                │
                ▼
        ┌──────────────────┐
        │  BACKWARD PASS   │
        │  (AMP BF16)      │
        └──────────────────┘
                │
                ▼
        ┌──────────────────┐
        │  GRADIENT ACCUM  │
        │  (4 steps)       │
        └──────────────────┘
                │
                ▼ (every 4 steps)
        ┌──────────────────┐
        │  OPTIMIZER STEP  │
        │  - Clip grad (1.0)│
        │  - AdamW update  │
        │  - LR schedule   │
        └──────────────────┘
```

---

## 🔥 Métriques en Temps Réel (Step 15K)

```
┌────────────────────────────────────────────────────────────┐
│                    TRAINING METRICS                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Step:          15,000 / 100,000 (15%)                     │
│  Epoch:         ~12 (multiple passes over dataset)         │
│                                                            │
│  ┌──────────────────────────────────────────────────┐     │
│  │ LOSS & PERPLEXITY                                │     │
│  │  Train Loss:      2.54                           │     │
│  │  Train PPL:       12.7                           │     │
│  │  Val Loss:        6.04   ⚠️ TOO HIGH             │     │
│  │  Val PPL:         420    ⚠️ OVERFITTING          │     │
│  │  Train/Val Gap:   3.50   ⚠️ CRITICAL             │     │
│  └──────────────────────────────────────────────────┘     │
│                                                            │
│  ┌──────────────────────────────────────────────────┐     │
│  │ LEARNING RATE                                    │     │
│  │  Current LR:      1.998e-4                       │     │
│  │  Max LR:          2.000e-4                       │     │
│  │  Progress:        99.9% of max (plateau)         │     │
│  │  Scheduler:       Cosine decay with warmup       │     │
│  └──────────────────────────────────────────────────┘     │
│                                                            │
│  ┌──────────────────────────────────────────────────┐     │
│  │ CURRICULUM                                       │     │
│  │  Seq Length:      2048 (final, since step 15K)  │     │
│  │  Global Weight:   1.00 (active since step 5K)   │     │
│  │  Landmarks:       147 → 24 (top-K per head)     │     │
│  └──────────────────────────────────────────────────┘     │
│                                                            │
│  ┌──────────────────────────────────────────────────┐     │
│  │ PERFORMANCE (RTX 3090)                           │     │
│  │  Steps/sec:       0.40 step/s (2.5s/step)       │     │
│  │  Tokens/sec:      6,553 tok/s ✅ EXCELLENT       │     │
│  │  GPU Memory:      18.2 GB / 24 GB (76%)         │     │
│  │  GPU Reserved:    19.1 GB (PyTorch overhead)    │     │
│  │  Grad Norm:       2.34 ✅ STABLE                 │     │
│  └──────────────────────────────────────────────────┘     │
│                                                            │
│  ┌──────────────────────────────────────────────────┐     │
│  │ ESTIMATED TIME                                   │     │
│  │  Elapsed:         ~12 hours (0 → 15K)           │     │
│  │  Remaining (50K): ~25 hours                      │     │
│  │  Remaining (100K):~50 hours                      │     │
│  └──────────────────────────────────────────────────┘     │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 🚨 Problèmes Identifiés (Step 15K)

```
┌────────────────────────────────────────────────────────────┐
│                    DIAGNOSTIC REPORT                       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  🔴 CRITIQUE #1: Overfitting Massif                       │
│  ├─ Val PPL: 420 (vs Train PPL: 12.7)                     │
│  ├─ Gap: 3.5 loss units (inacceptable)                    │
│  └─ Cause: Wikipedia seul, 95/5 split                     │
│     Solution: → FineWeb-Edu, 90/5/5 split                 │
│                                                            │
│  🔴 CRITIQUE #2: Landmarks Sous-Optimaux                  │
│  ├─ Diversity λ=0.02 (trop faible)                        │
│  ├─ Sparsity λ=0.001 (trop faible)                        │
│  └─ Résultat: Landmarks clustered, inefficaces            │
│     Solution: → Diversity 0.1, Sparsity 0.01              │
│                                                            │
│  🟡 ATTENTION #3: Global Warmup Rapide                    │
│  ├─ 1K → 5K = 4K steps seulement                          │
│  ├─ Landmarks pas bien appris à step 1K                   │
│  └─ Solution: → 5K → 20K (15K steps warmup)               │
│                                                            │
│  🟡 ATTENTION #4: Weight Decay Élevé                      │
│  ├─ 0.1 = Très fort (pour GPT-3 175B)                     │
│  ├─ SLGA 38M = Sur-régularisation                         │
│  └─ Solution: → 0.01 (standard)                           │
│                                                            │
│  ✅ OK: GPU Performance                                   │
│  ├─ Throughput: 6,553 tok/s (excellent)                   │
│  ├─ Memory: 76% utilisé (optimal)                         │
│  └─ Grad norm: 2.34 (stable, pas d'explosion)             │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 📊 Loss Auxiliaires (Landmarks)

```
┌────────────────────────────────────────────────────────────┐
│              LANDMARK REGULARIZATION LOSSES                │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Total Loss = CE Loss + Diversity Loss + Sparsity Loss    │
│                                                            │
│  ┌──────────────────────────────────────────────────┐     │
│  │ 1. CROSS-ENTROPY LOSS (lignes 83-111)           │     │
│  │                                                  │     │
│  │    loss_ce = F.cross_entropy(                   │     │
│  │        logits[:, :-1, :],    # (B, L-1, V)      │     │
│  │        labels[:, :-1],       # (B, L-1)         │     │
│  │        ignore_index=pad_id                      │     │
│  │    )                                            │     │
│  │                                                  │     │
│  │    Actuel: 2.54 (step 15K)                      │     │
│  └──────────────────────────────────────────────────┘     │
│                                                            │
│  ┌──────────────────────────────────────────────────┐     │
│  │ 2. DIVERSITY LOSS (λ=0.02)                      │     │
│  │                                                  │     │
│  │    Objectif: Landmarks spatially diverse        │     │
│  │                                                  │     │
│  │    loss_div = λ * (-entropy(landmark_scores))   │     │
│  │             = 0.02 * H(scores)                  │     │
│  │                                                  │     │
│  │    ⚠️ TROP FAIBLE → landmarks clustered          │     │
│  │    Recommandé: λ = 0.1 (5x increase)            │     │
│  └──────────────────────────────────────────────────┘     │
│                                                            │
│  ┌──────────────────────────────────────────────────┐     │
│  │ 3. SPARSITY LOSS (λ=0.001)                      │     │
│  │                                                  │     │
│  │    Objectif: Pénaliser trop de landmarks actifs │     │
│  │                                                  │     │
│  │    loss_spar = λ * sum(scores)                  │     │
│  │              = 0.001 * ||scores||_1             │     │
│  │                                                  │     │
│  │    ⚠️ TROP FAIBLE → tous landmarks actifs        │     │
│  │    Recommandé: λ = 0.01 (10x increase)          │     │
│  └──────────────────────────────────────────────────┘     │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 🎛️ Configuration Batch & Accumulation

```
┌────────────────────────────────────────────────────────────┐
│           GRADIENT ACCUMULATION STRATEGY                   │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Config:                                                   │
│    batch_size = 8                                          │
│    accum_steps = 4                                         │
│    Effective batch = 8 × 4 = 32                            │
│                                                            │
│  Timeline (4 micro-steps):                                 │
│                                                            │
│  Step N+0:                                                 │
│    ├─ Forward(batch_0) → logits                           │
│    ├─ Loss = CE / 4                                        │
│    └─ Backward → grad_0 (accumulate)                       │
│                                                            │
│  Step N+1:                                                 │
│    ├─ Forward(batch_1) → logits                           │
│    ├─ Loss = CE / 4                                        │
│    └─ Backward → grad_1 (accumulate)                       │
│                                                            │
│  Step N+2:                                                 │
│    ├─ Forward(batch_2) → logits                           │
│    ├─ Loss = CE / 4                                        │
│    └─ Backward → grad_2 (accumulate)                       │
│                                                            │
│  Step N+3:                                                 │
│    ├─ Forward(batch_3) → logits                           │
│    ├─ Loss = CE / 4                                        │
│    └─ Backward → grad_3 (accumulate)                       │
│                                                            │
│  Step N+4: OPTIMIZER STEP                                  │
│    ├─ Total grad = (grad_0 + grad_1 + grad_2 + grad_3)    │
│    ├─ Clip grad norm (max = 1.0)                          │
│    ├─ AdamW update: θ ← θ - lr * grad                     │
│    ├─ LR schedule step                                     │
│    └─ Zero gradients                                       │
│                                                            │
│  Result:                                                   │
│    ✅ Effective batch = 32 (memory efficient)             │
│    ✅ GPU 76% utilized (optimal for RTX 3090)             │
│    ✅ Updates every 4 steps (fast convergence)            │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 🧠 Attention Flow (SLGA)

```
┌────────────────────────────────────────────────────────────┐
│              SLGA ATTENTION MECHANISM                      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  INPUT: x = (Batch=8, SeqLen=2048, Embed=512)             │
│                                                            │
│  ┌──────────────────────────────────────────────────┐     │
│  │ BRANCH 1: LOCAL ATTENTION (window=128)          │     │
│  │                                                  │     │
│  │  For each position i in [0, 2048]:              │     │
│  │    Attend to [i-64, i+64] (128 tokens)          │     │
│  │                                                  │     │
│  │  Output: out_local = (8, 2048, 512)             │     │
│  │                                                  │     │
│  │  Complexity: O(L * W) = O(2048 * 128)           │     │
│  │            = 262K ops/token (linear!)            │     │
│  └──────────────────────────────────────────────────┘     │
│                                                            │
│  ┌──────────────────────────────────────────────────┐     │
│  │ BRANCH 2: GLOBAL ATTENTION (k=24)               │     │
│  │                                                  │     │
│  │  Step A: Landmark Selection                     │     │
│  │    - Learned scores: (8, 2048) softmax          │     │
│  │    - Top-K: Select 147 candidates               │     │
│  │    - Diverse: 24 landmarks per head             │     │
│  │                                                  │     │
│  │  Step B: Global Attention                       │     │
│  │    For each position i in [0, 2048]:            │     │
│  │      Attend to 24 landmark tokens               │     │
│  │                                                  │     │
│  │  Output: out_global = (8, 2048, 512)            │     │
│  │                                                  │     │
│  │  Complexity: O(L * K) = O(2048 * 24)            │     │
│  │            = 49K ops/token (very efficient!)     │     │
│  └──────────────────────────────────────────────────┘     │
│                                                            │
│  ┌──────────────────────────────────────────────────┐     │
│  │ FUSION: Gated Combination                       │     │
│  │                                                  │     │
│  │  gate = sigmoid(W_gate · x)  # Learned weights  │     │
│  │  out = gate * out_local + (1 - gate) * out_global│    │
│  │                                                  │     │
│  │  gate ≈ [0.6-0.8] → More local                  │     │
│  │  gate ≈ [0.2-0.4] → More global                 │     │
│  └──────────────────────────────────────────────────┘     │
│                                                            │
│  TOTAL COMPLEXITY: O(L * (W + K))                         │
│                  = O(2048 * 152)                          │
│                  = 311K ops/token                         │
│                                                            │
│  vs Standard Attention: O(L²) = O(2048²) = 4.2M ops      │
│  Speedup: 13.5x faster! 🚀                                │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 📝 Console Output Example (Step 15K)

```
======================= TRAINING LOGS =======================

Step  14050 | Loss: 2.5213 | PPL:   12.45 | LR: 1.99e-04 | GradNorm:  2.31
            | SeqLen: 2048 | GW: 1.00 | LM: 145→24 | GPU: 18.1GB | Tok/s:  3422

Step  14100 | Loss: 2.9447 | PPL:   19.00 | LR: 1.99e-04 | GradNorm:  4.12
            | SeqLen: 2048 | GW: 1.00 | LM: 152→24 | GPU: 18.2GB | Tok/s:  6143

Step  14300 | Loss: 2.4918 | PPL:   12.09 | LR: 1.99e-04 | GradNorm:  2.05
            | SeqLen: 2048 | GW: 1.00 | LM: 141→24 | GPU: 18.0GB | Tok/s:   927  ⚠️

=== Validation ===
  Validation: 10/10 batches...
Val Loss: 6.0310 | Val PPL: 416.12

Step  14500 | Loss: 2.1679 | PPL:    8.74 | LR: 1.99e-04 | GradNorm:  1.89
            | SeqLen: 2048 | GW: 1.00 | LM: 138→24 | GPU: 18.2GB | Tok/s:  6022

Step  14850 | Loss: 2.7103 | PPL:   15.03 | LR: 1.99e-04 | GradNorm:  3.24
            | SeqLen: 2048 | GW: 1.00 | LM: 149→24 | GPU: 18.1GB | Tok/s:  3229  ⚠️

=== Validation ===
  Validation: 10/10 batches...
Val Loss: 6.0425 | Val PPL: 420.94

Step  15000 | Loss: 2.5448 | PPL:   12.74 | LR: 1.99e-04 | GradNorm:  2.34
            | SeqLen: 2048 | GW: 1.00 | LM: 147→24 | GPU: 18.2GB | Tok/s:  6553

============================================================
✓ CHECKPOINT SAVED at step 15000
  Location: out_slga/ckpt_15000
  Files: model.pt, trainer_state.pt
============================================================

Legend:
  Loss    = Cross-entropy loss (lower is better)
  PPL     = Perplexity = exp(Loss) (lower is better)
  LR      = Learning rate (current)
  GradNorm= L2 norm of gradients (2-5 is normal)
  SeqLen  = Current sequence length (curriculum)
  GW      = Global attention weight (0.0-1.0)
  LM      = Landmarks: candidates→selected per head
  GPU     = GPU memory allocated
  Tok/s   = Throughput (tokens per second)
```

---

## ✅ Checklist de Santé du Pipeline

```
┌────────────────────────────────────────────────────────────┐
│                  PIPELINE HEALTH CHECK                     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ✅ Configuration                                          │
│  ├─ [✓] Config.yaml loaded correctly                      │
│  ├─ [✓] Model config valid (38M params)                   │
│  └─ [✓] Training hyperparams reasonable                   │
│                                                            │
│  ✅ Data Loading                                           │
│  ├─ [✓] Dataset loads without error                       │
│  ├─ [✓] Collator produces correct shapes                  │
│  ├─ [✓] Tokenizer handles special tokens                  │
│  └─ [⚠️] Dataset diversity (Wikipedia only)                │
│                                                            │
│  ✅ Model                                                  │
│  ├─ [✓] Model initializes on GPU                          │
│  ├─ [✓] Forward pass runs without OOM                     │
│  ├─ [✓] Landmarks selection works                         │
│  └─ [⚠️] Landmark diversity (need stronger reg)            │
│                                                            │
│  ✅ Training Loop                                          │
│  ├─ [✓] AMP (BF16) enabled and stable                     │
│  ├─ [✓] Gradient accumulation correct                     │
│  ├─ [✓] Gradient clipping prevents explosion              │
│  ├─ [✓] LR schedule progresses correctly                  │
│  └─ [✓] Checkpointing saves successfully                  │
│                                                            │
│  ⚠️ Performance                                            │
│  ├─ [✓] Throughput: 6,553 tok/s (excellent)               │
│  ├─ [✓] GPU usage: 76% (optimal)                          │
│  ├─ [✓] Grad norm: 2.34 (stable)                          │
│  ├─ [❌] Val loss: 6.04 (too high, overfitting)            │
│  └─ [❌] Train/Val gap: 3.5 (critical overfitting)         │
│                                                            │
│  📋 RECOMMENDATIONS                                        │
│  1. Change dataset → FineWeb-Edu (multi-domain)           │
│  2. Strengthen landmark regularization (5x-10x)           │
│  3. Reduce weight decay (0.1 → 0.01)                      │
│  4. Extend global warmup (5K → 20K steps)                 │
│  5. Test learned_landmarks=false (diagnostic)             │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

**Fichiers associés**:
- `docs/TRAINING_PIPELINE_ANALYSIS.md` - Analyse technique complète
- `docs/RTX_3090_OPTIMIZATIONS.md` - Optimisations GPU spécifiques
- `docs/STEP_15K_DIAGNOSTIC_REPORT.md` - Diagnostic checkpoint 15K
- `docs/RESUME_WITH_NEW_DATASET.md` - Guide de reprise avec nouveau dataset

**Dernière mise à jour**: 2025-10-24
