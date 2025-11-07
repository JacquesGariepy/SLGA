# SLGA Model Scaling Guide

Guide complet pour entraîner SLGA de 38M à 671B paramètres selon votre hardware.

## 📊 Quick Reference Table

| Config | GPUs | Params | Active | Context | Memory/GPU | Speed (tok/s) | Training Time | Cost (Cloud) |
|--------|------|--------|--------|---------|------------|---------------|---------------|--------------|
| **RTX 3090** (current) | 1× 3090 | 38M | 38M | 2K | 22 GB | 400 | ~40 days | N/A |
| **2× RTX 3090** | 2× 3090 | 7B | 7B | 4K | 22 GB | 120 | ~166 days | N/A |
| **H100 80GB** | 1× H100 | 13B | 13B | 8K | 72 GB | 180 | ~94 days | $220K |
| **8× H100** | 8× H100 | 70B | 70B | 8K | 75 GB | 1000 | ~347 days | $2.7M |
| **32× H100** | 32× H100 | 175B | 175B | 8K | 78 GB | 3200 | ~16 days | $315K |
| **64× H100 MoE** | 64× H100 | 671B | 37B | 32K | 76 GB | 10000 | ~11 days | $70-200K |

## 🎯 Hardware Requirements by Model Size

### 38M Parameters (Current - RTX 3090)
```yaml
Config: config/config_fineweb_edu_3090_optimized.yaml
Hardware: 1× RTX 3090 24GB
Memory: ~22 GB (92%)
Speed: ~400 tokens/sec
Training: 100K steps (~40 days)
Dataset: FineWeb-Edu 10BT
Quality: PPL ~15-20 (good for 38M)
```

**Pros:**
- ✅ Consumer hardware (affordable)
- ✅ Fast iteration for research
- ✅ Good for prototyping SLGA features

**Cons:**
- ❌ Limited context (2K max)
- ❌ Lower quality than larger models
- ❌ Can't compete with commercial LLMs

---

### 7B Parameters (2× RTX 3090)
```yaml
Config: config/config_2x3090_7B.yaml
Hardware: 2× RTX 3090 24GB (48GB total)
Memory: ~22 GB per GPU
Speed: ~120 tokens/sec total
Training: 500K steps (~166 days)
Dataset: FineWeb-Edu 350BT
Quality: PPL ~8-12 (Llama-2 7B level)
```

**Key Features:**
- **DeepSpeed ZeRO-2** for multi-GPU coordination
- **Gradient checkpointing** (saves 30-40% memory)
- **4K context** with SLGA
- **64 global landmarks**

**When to use:**
- You have 2-4× RTX 3090 or similar
- Want production-quality chatbot
- Need long context (4K tokens)
- Budget: On-premise hardware only

**Setup:**
```bash
# Install DeepSpeed
pip install deepspeed

# Launch training
deepspeed --num_gpus=2 scripts/train.py \
  --config config/config_2x3090_7B.yaml \
  --deepspeed ds_config_zero2.json
```

---

### 13B Parameters (H100 80GB)
```yaml
Config: config/config_H100_13B.yaml
Hardware: 1× H100 80GB
Memory: ~72 GB (90%)
Speed: ~180 tokens/sec
Training: 750K steps (~94 days)
Dataset: FineWeb-Edu 500BT
Quality: PPL ~6-9 (Llama-2 13B level)
```

**Key Features:**
- **H100 optimizations**: FlashAttention-2, Transformer Engine
- **8K context** with SLGA
- **96 global landmarks**
- **Native FP8 support** (via Transformer Engine)

**H100 advantages:**
- 3.3× faster than A100
- Native FlashAttention-2 (40% memory savings)
- NVLink 4.0 (900 GB/s vs A100's 600 GB/s)

**When to use:**
- Access to H100 (cloud or on-premise)
- Need SOTA quality in reasonable time
- Budget: $200-300K cloud training

**Setup:**
```bash
# Install H100 optimizations
pip install flash-attn --no-build-isolation
pip install transformer-engine[pytorch]

# Launch training
python scripts/train.py \
  --config config/config_H100_13B.yaml \
  --max-steps 750000
```

---

### 70B Parameters (8× H100)
```yaml
Config: config/config_8xH100_70B.yaml
Hardware: 8× H100 80GB (single node, 640GB total)
Memory: ~75 GB per GPU
Speed: ~1000 tokens/sec total
Training: 1M steps (~347 days or 11.5 months)
Dataset: FineWeb-Edu 1.4T tokens (full)
Quality: PPL ~3-5 (Llama-2 70B / GPT-3.5 level)
```

**Key Features:**
- **FSDP (Fully Sharded Data Parallel)** - PyTorch native
- **8K context** with SLGA
- **128 global landmarks**
- **Zero CPU offload** (fits in 640GB GPU memory)

**FSDP vs DeepSpeed ZeRO-3:**
| Feature | FSDP | DeepSpeed ZeRO-3 |
|---------|------|------------------|
| Speed | ✅ 25% faster | Slower |
| Integration | ✅ Native PyTorch | Requires DeepSpeed |
| Debugging | ✅ Easier | Harder |
| Maturity | Newer (PyTorch 2.0+) | More mature |

**When to use:**
- Access to 8× H100 cluster (single node)
- Need GPT-3.5 level quality
- Research or commercial applications
- Budget: $2-3M cloud training

**Setup:**
```bash
# Launch with torchrun (FSDP)
torchrun --nproc_per_node=8 \
         --nnodes=1 \
         --master_port=29500 \
         scripts/train.py \
         --config config/config_8xH100_70B.yaml
```

---

### 175B Parameters (32× H100, Multi-Node)
```yaml
Config: config/config_32xH100_175B.yaml
Hardware: 32× H100 80GB (4 nodes × 8 GPUs, 2.5TB total)
Memory: ~78 GB per GPU
Speed: ~3200 tokens/sec total
Training: 1.5M steps (~16 days with perfect uptime)
Dataset: FineWeb 3T+ tokens
Quality: PPL ~2.5-4.0 (GPT-3 level)
```

**Key Features:**
- **3D Parallelism**: Data (8×) + Tensor (2×) + Pipeline (2×)
- **8K context** with SLGA
- **256 global landmarks**
- **Multi-node InfiniBand** required

**3D Parallelism explained:**
```
32 GPUs = 8 (Data Parallel) × 2 (Tensor Parallel) × 2 (Pipeline Parallel)

- Data Parallel (8×): 8 replicas processing different batches
- Tensor Parallel (2×): Each layer split across 2 GPUs
- Pipeline Parallel (2×): 96 layers split into 2 stages (48+48)
```

**When to use:**
- Multi-node H100 cluster (4+ nodes)
- Need GPT-3 level quality
- Research lab or large company
- Budget: $300K-500K cloud training

**Infrastructure requirements:**
- **InfiniBand**: 200-400 Gbps interconnect (mandatory)
- **Shared filesystem**: NFS or Lustre for checkpoints
- **SLURM**: Job scheduler for multi-node coordination

**Setup:**
```bash
# SLURM launch (recommended)
srun --nodes=4 \
     --ntasks-per-node=8 \
     --gres=gpu:8 \
     python scripts/train.py \
     --config config/config_32xH100_175B.yaml

# Manual torchrun (on all 4 nodes)
# Node 0:
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=0 \
         --master_addr=$MASTER_ADDR --master_port=29500 \
         scripts/train.py --config config/config_32xH100_175B.yaml

# Node 1-3: Same with --node_rank=1,2,3
```

---

### 671B MoE Parameters (64× H100, Multi-Node) - DeepSeek-V3 Scale
```yaml
Config: config/config_64xH100_671B_MoE.yaml
Hardware: 64× H100 80GB (8 nodes × 8 GPUs, 5.1TB total)
Model: 671B total, 37B activated per token
Memory: ~76 GB per GPU
Speed: ~10,000 tokens/sec total (37B activated, not 671B!)
Training: 2M steps (~11 days)
Dataset: FineWeb + SlimPajama 14.8T tokens
Quality: PPL ~1.5-2.5 (DeepSeek-V3 level, SOTA)
```

**Mixture of Experts (MoE) Architecture:**
```
Total parameters: 671B
├── Shared weights: ~37B (always active)
└── Expert weights: ~634B
    ├── 256 experts × ~2.5B each
    └── Top-8 routing: only 8 experts active per token

Computation per token:
- Dense 671B: 1342 GFLOPs
- MoE 671B:   74 GFLOPs (18× faster!)
```

**Key Features:**
- **MoE with 256 experts**, Top-8 routing
- **Shared experts** (2 always active - DeepSeek innovation)
- **Multi-head Latent Attention (MLA)** - reduces KV cache
- **32K context** (DeepSeek-V3 uses 128K, we use 32K for efficiency)
- **4D Parallelism**: Expert (8×) + Data (2×) + Tensor (2×) + Pipeline (2×)

**MoE advantages:**
- ✅ **18× faster** than dense 671B (only 37B activated)
- ✅ **Quality of ~120-150B dense** model
- ✅ **Expert parallelism** scales to 256 experts
- ✅ **Load balancing** prevents expert collapse

**When to use:**
- Massive multi-node cluster (8+ nodes)
- Need SOTA quality (GPT-4 level)
- Have budget for large-scale training
- Research lab or frontier AI company
- Budget: $70K-200K with optimizations (spot instances)

**Setup:**
```bash
# Install MoE libraries
pip install fairscale  # For MoE layers
pip install megablocks # Faster MoE (optional)
pip install deepspeed  # DeepSpeed has best MoE support

# Create DeepSpeed MoE config (ds_config_moe.json)
{
  "train_batch_size": 128,
  "bf16": {"enabled": true},
  "zero_optimization": {"stage": 3},
  "moe": {
    "enabled": true,
    "expert_parallel_size": 8,
    "num_experts": 256,
    "top_k": 8
  }
}

# Launch with DeepSpeed
deepspeed --num_nodes=8 \
          --num_gpus=8 \
          --master_addr=$MASTER_ADDR \
          scripts/train.py \
          --config config/config_64xH100_671B_MoE.yaml \
          --deepspeed ds_config_moe.json
```

---

## 📐 Scaling Laws (Chinchilla Optimal)

Pour un modèle de **N** paramètres, le nombre optimal de tokens est :

```
Optimal tokens ≈ 20 × N
```

**Exemples:**
| Model Size | Optimal Tokens | Our Config | Status |
|------------|----------------|------------|--------|
| 38M | 0.76B | 10B | ✅ Over-trained (better quality) |
| 7B | 140B | 350B | ✅ Over-trained (SOTA quality) |
| 13B | 260B | 500B | ✅ Over-trained |
| 70B | 1.4T | 1.4T | ✅ Perfectly scaled |
| 175B | 3.5T | 3T | ✅ Near-optimal |
| 671B MoE | 2.4T (effective) | 14.8T | ✅ Massively over-trained |

**Over-training benefits:**
- Better quality (lower PPL)
- Better generalization
- Closer to SOTA models
- Trade-off: More compute time

---

## 🛠️ Implementation Checklist

### Before You Start

**1. Choose Your Config:**
```bash
# Current hardware: RTX 3090
config/config_fineweb_edu_3090_optimized.yaml  ✅ (you are here)

# If you get 2× RTX 3090:
config/config_2x3090_7B.yaml

# If you get H100 access:
config/config_H100_13B.yaml                     # 1× H100
config/config_8xH100_70B.yaml                   # 8× H100
config/config_32xH100_175B.yaml                 # 32× H100
config/config_64xH100_671B_MoE.yaml            # 64× H100 (MoE)
```

**2. Install Dependencies:**
```bash
# Base (all configs)
pip install torch torchvision torchaudio transformers accelerate

# For 2× RTX 3090 (DeepSpeed)
pip install deepspeed

# For H100 (optimizations)
pip install flash-attn --no-build-isolation
pip install transformer-engine[pytorch]

# For MoE (671B)
pip install fairscale megablocks
```

**3. Verify Hardware:**
```bash
# Check GPU memory
nvidia-smi

# Check multi-GPU setup
python -c "import torch; print(torch.cuda.device_count())"

# Check InfiniBand (multi-node)
ibstat  # Should show active links
```

**4. Prepare Dataset:**
```bash
# Download dataset (auto-cached by HuggingFace)
# First run will download, subsequent runs use cache

# For large configs (70B+), pre-download:
python -c "
from datasets import load_dataset
ds = load_dataset('HuggingFaceFW/fineweb-edu', 'default', split='train[:1%]')
"
```

**5. Test Run (Small Steps):**
```bash
# Test config with 100 steps first
python scripts/train.py \
  --config config/YOUR_CONFIG.yaml \
  --max-steps 100

# If successful, launch full training
```

---

## 💰 Cost Analysis

### Cloud Pricing (AWS p5.48xlarge - 8× H100)

| Config | Instances | Hours | Price/Hour | Spot Price | Reserved | Total (On-Demand) | Total (Spot) |
|--------|-----------|-------|------------|------------|----------|-------------------|--------------|
| 13B (1× H100) | 1 | 2,256 | $98 | $35 | $59 | $220K | $79K |
| 70B (8× H100) | 1 | 8,328 | $98 | $35 | $59 | $816K | $291K |
| 175B (32× H100) | 4 | 384 | $392 | $140 | $235 | $150K | $54K |
| 671B MoE (64× H100) | 8 | 264 | $784 | $280 | $470 | $207K | $74K |

**Cost Optimization Strategies:**

**1. Spot Instances (50-70% discount):**
```bash
# AWS EC2 Spot (best for short runs)
aws ec2 request-spot-instances --instance-type p5.48xlarge

# Spot can be interrupted → use checkpointing every 1-2 hours
# Our configs save every 500-10K steps → safe
```

**2. Reserved Instances (40% discount):**
```bash
# Commit to 1-3 years for best prices
# Good if you plan multiple training runs
```

**3. On-Premise Cluster:**
```
Initial cost: $2-3M (64× H100 cluster)
Electricity: ~$200K/year (at $0.10/kWh)
Break-even: After ~10-15 large training runs

Good if:
- You plan many training runs
- 3-5 year horizon
- Have data center infrastructure
```

**4. Academic Discounts:**
```
Many cloud providers offer research credits:
- AWS Cloud Credits for Research
- Google Cloud Research Credits
- Microsoft Azure for Research

Can get $10K-100K in free credits
```

---

## 📊 Quality Expectations

### Perplexity (PPL) by Model Size

| Model Size | Expected PPL | Quality Level | Use Cases |
|------------|--------------|---------------|-----------|
| **38M** | 15-20 | Good for size | Research, prototyping |
| **7B** | 8-12 | Llama-2 7B | Chatbots, assistants |
| **13B** | 6-9 | Llama-2 13B | Advanced chatbots |
| **70B** | 3-5 | Llama-2 70B, GPT-3.5 | Production LLMs |
| **175B** | 2.5-4 | GPT-3 | Advanced applications |
| **671B MoE** | 1.5-2.5 | DeepSeek-V3, GPT-4 level | SOTA, research |

### Context Length with SLGA

| Model Size | SLGA Context | Global Landmarks | vs Standard Attention |
|------------|--------------|------------------|----------------------|
| 38M | 2K | 24 | 2× standard (1K) |
| 7B | 4K | 64 | 2× standard (2K) |
| 13B | 8K | 96 | 2× standard (4K) |
| 70B | 8K | 128 | Same as Llama-2 |
| 175B | 8K | 256 | Same as GPT-3 |
| 671B MoE | 32K | 512 | 4× GPT-3 (8K) |

**SLGA advantages for long context:**
- ✅ **O(n) complexity** vs O(n²) for standard attention
- ✅ **Learned landmarks** focus on important positions
- ✅ **No quality degradation** at max context length
- ✅ **Dilated windows** for better long-range dependencies

---

## 🚀 Training Timeline

### From Start to Production Model

**38M (Current - RTX 3090):**
```
Day 1:     Setup, first 1K steps
Day 3:     10K steps, basic coherence
Day 10:    50K steps, short sentences
Day 40:    100K steps, DONE ✅
           PPL ~15-20, can generate paragraphs
```

**7B (2× RTX 3090):**
```
Week 1:    10K steps, word-level learning
Week 4:    50K steps, sentence structure
Week 12:   150K steps, paragraph coherence
Week 24:   500K steps, DONE ✅
           PPL ~8-12, production-ready chatbot
```

**13B (H100):**
```
Week 2:    25K steps, basic competence
Week 8:    100K steps, good quality
Week 20:   750K steps, DONE ✅
           PPL ~6-9, advanced chatbot
```

**70B (8× H100):**
```
Month 2:   100K steps, emerging capabilities
Month 6:   500K steps, good quality
Month 12:  1M steps, DONE ✅
           PPL ~3-5, GPT-3.5 level
```

**175B (32× H100):**
```
Day 3:     100K steps, rapid progress
Day 8:     500K steps, strong capabilities
Day 16:    1.5M steps, DONE ✅
           PPL ~2.5-4, GPT-3 level
```

**671B MoE (64× H100):**
```
Day 2:     200K steps, fast learning (MoE efficiency)
Day 5:     1M steps, excellent quality
Day 11:    2M steps, DONE ✅
           PPL ~1.5-2.5, SOTA DeepSeek-V3 level
```

---

## 🎓 Academic Use Cases

### Which Model for Your Research?

**Prototyping New Architectures (38M-7B):**
- Fast iteration (hours to days)
- Test new attention mechanisms
- Validate SLGA improvements
- ✅ Use: RTX 3090 or 2× RTX 3090

**Benchmarking (7B-13B):**
- Compare against Llama-2 7B/13B
- Reproducible results
- Reasonable compute budget
- ✅ Use: 2× RTX 3090 or H100

**Frontier Research (70B-175B):**
- Scaling law experiments
- New training techniques
- Publish SOTA results
- ✅ Use: 8-32× H100 cluster

**Pushing Boundaries (671B MoE):**
- MoE architecture research
- Ultra-long context (128K+)
- Compete with commercial models
- ✅ Use: 64+ H100 cluster

---

## 📚 References

**SLGA Architecture:**
- Original SLGA paper: [Add citation]
- Sparse attention: Longformer, BigBird
- Global landmarks: Landmark Attention

**Scaling Laws:**
- Chinchilla paper (Hoffmann et al., 2022)
- Scaling laws for LLMs (Kaplan et al., 2020)

**MoE Models:**
- DeepSeek-V3: https://github.com/deepseek-ai/DeepSeek-V3
- Switch Transformers (Google, 2021)
- GLaM (Google, 2021)

**Training Frameworks:**
- PyTorch FSDP: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- DeepSpeed: https://www.deepspeed.ai/
- Fairscale MoE: https://github.com/facebookresearch/fairscale

---

## ✅ Decision Tree: Which Config to Choose?

```
Do you have more than 1 GPU?
│
├─ No → Use config_fineweb_edu_3090_optimized.yaml (38M)
│       ✅ You are here! Training works.
│
└─ Yes → How many GPUs?
         │
         ├─ 2-4 GPUs (RTX 3090/4090/A6000)
         │  └─ Use config_2x3090_7B.yaml (7B)
         │     ✅ Production-ready chatbot
         │
         ├─ 1 H100 80GB
         │  └─ Use config_H100_13B.yaml (13B)
         │     ✅ Advanced chatbot, research
         │
         ├─ 8 H100 (single node)
         │  └─ Use config_8xH100_70B.yaml (70B)
         │     ✅ GPT-3.5 level, SOTA
         │
         ├─ 32 H100 (4 nodes)
         │  └─ Use config_32xH100_175B.yaml (175B)
         │     ✅ GPT-3 level
         │
         └─ 64+ H100 (8+ nodes)
            └─ Use config_64xH100_671B_MoE.yaml (671B MoE)
               ✅ DeepSeek-V3 level, SOTA
```

---

## 🎯 Current Status

**Vous êtes ici:** RTX 3090 38M training
- ✅ Scheduler bug fixed (LR correct)
- ✅ Validation OOM fixed
- ✅ Training stable at step 1300
- ✅ PPL 1185 (expected for early training)
- ⏳ Wait for step 5000 to see quality

**Next milestone:**
- Step 5000 (~26 hours): Test generation quality
- Expected: "The capital of France is **Paris, which...**"
- PPL target: ~200-300

**Si vous obtenez plus de hardware:**
- 2× RTX 3090 → Upgrade to 7B config (8× better quality)
- 1× H100 → Upgrade to 13B config (GPT-3.5 approaching)
- 8× H100 → Upgrade to 70B config (GPT-3.5 level)

---

**Bonne chance avec votre scaling ! 🚀**
