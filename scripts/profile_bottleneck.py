#!/usr/bin/env python3
"""
Performance Bottleneck Profiler for SLGA Training

Profiles data loading vs GPU compute to identify bottlenecks.
Measures:
- Data loading time
- GPU forward/backward time
- CPU vs GPU utilization
- Throughput (tokens/sec)

Usage:
    python scripts/profile_bottleneck.py --config config/config_3090.yaml --steps 100

Expected Results:
    num_workers=0: GPU util ~5%, data_time >> compute_time
    num_workers=4: GPU util ~80%, data_time < compute_time
"""

import os
import sys
import yaml
import time
import torch
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import Config, LLMTransformer
from src.data import get_tokenizer, load_text_dataset, CollatorLocal
from torch.utils.data import DataLoader
from accelerate import Accelerator


def profile_training(config_path: str, num_steps: int = 100):
    """Profile training to identify bottlenecks"""

    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("="*80)
    print("🔍 SLGA PERFORMANCE PROFILING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Config file: {config_path}")
    print(f"  Batch size: {cfg['train']['batch_size']}")
    print(f"  Gradient accumulation: {cfg['train']['accum_steps']}")
    print(f"  Effective batch: {cfg['train']['batch_size'] * cfg['train']['accum_steps']}")
    print(f"  Sequence length: {cfg['train'].get('seq_len_start', 512)}")
    print(f"  Num workers: {cfg['data'].get('num_workers', 0)}")
    print(f"  AMP enabled: {cfg['train'].get('amp', True)}")
    print(f"  AMP dtype: {cfg['train'].get('amp_dtype', 'bf16')}")

    # Setup
    accelerator = Accelerator()
    device = accelerator.device

    # Model
    model_cfg = Config(**cfg["model"])
    model = LLMTransformer(model_cfg)
    model = model.to(device)

    # Data
    tokenizer = get_tokenizer(cfg["tokenizer"])

    print(f"\n📦 Loading dataset...")
    ds_train = load_text_dataset(
        cfg["data"]["dataset"],
        cfg["data"].get("subset"),
        cfg["data"]["split_train"],
    )

    # Limit for profiling
    if len(ds_train) > 10000:
        ds_train = ds_train.select(range(10000))

    print(f"  Dataset size: {len(ds_train)} samples")

    # Collator
    seq_len = cfg["train"].get("seq_len_start", 512)
    collate_fn = CollatorLocal(tokenizer, seq_len)

    # Dataloader
    num_workers = cfg["data"].get("num_workers", 0)
    train_loader = DataLoader(
        ds_train,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Prepare
    model, train_loader = accelerator.prepare(model, train_loader)
    model.train()

    # AMP
    amp_enabled = cfg["train"].get("amp", True)
    amp_dtype_str = cfg["train"].get("amp_dtype", "bf16").lower()

    if amp_dtype_str == "bf16" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    print(f"\n⚙️  AMP: {amp_enabled}, dtype: {amp_dtype}")

    # Profiling metrics
    data_times = []
    compute_times = []
    total_tokens = 0

    print(f"\n🔬 Profiling {num_steps} steps...")
    print(f"   (This will take ~{num_steps * 0.2:.1f}s with num_workers={num_workers})\n")

    iterator = iter(train_loader)

    # Warmup (skip first few batches for accurate timing)
    for _ in range(5):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)

    # Start profiling
    torch.cuda.synchronize()
    profile_start = time.time()

    for step in tqdm(range(num_steps), desc="Profiling"):
        # Measure data loading time
        data_start = time.time()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)

        torch.cuda.synchronize()
        data_end = time.time()
        data_time = data_end - data_start

        # Measure compute time
        compute_start = time.time()

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward + backward
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            logits = model(input_ids)

            # Simple CE loss
            logits_shifted = logits[:, :-1, :].contiguous()
            labels_shifted = labels[:, :-1].contiguous()

            loss = torch.nn.functional.cross_entropy(
                logits_shifted.view(-1, logits_shifted.size(-1)),
                labels_shifted.view(-1),
                ignore_index=-100,
            )

        # Backward
        accelerator.backward(loss)

        # Reset gradients (simulate real training)
        model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        compute_end = time.time()
        compute_time = compute_end - compute_start

        # Track metrics
        data_times.append(data_time * 1000)  # ms
        compute_times.append(compute_time * 1000)  # ms
        total_tokens += input_ids.numel()

    torch.cuda.synchronize()
    profile_end = time.time()

    total_time = profile_end - profile_start

    # Analysis
    avg_data_time = sum(data_times) / len(data_times)
    avg_compute_time = sum(compute_times) / len(compute_times)
    avg_step_time = avg_data_time + avg_compute_time

    throughput = total_tokens / total_time

    # GPU utilization estimate
    gpu_util = (avg_compute_time / avg_step_time) * 100

    # Bottleneck detection
    if avg_data_time > avg_compute_time * 1.5:
        bottleneck = "DATA LOADING"
        severity = "🔴 CRITICAL" if avg_data_time > avg_compute_time * 2 else "⚠️ HIGH"
    else:
        bottleneck = "COMPUTE"
        severity = "✅ GOOD"

    print("\n" + "="*80)
    print("📊 PROFILING RESULTS")
    print("="*80)

    print(f"\n⏱️  Timing Breakdown (per step):")
    print(f"  Data loading:    {avg_data_time:6.2f} ms")
    print(f"  GPU compute:     {avg_compute_time:6.2f} ms")
    print(f"  Total step:      {avg_step_time:6.2f} ms")
    print(f"  Steps/sec:       {1000 / avg_step_time:6.2f}")

    print(f"\n🎯 Bottleneck Analysis:")
    print(f"  Primary bottleneck: {severity} {bottleneck}")
    print(f"  GPU utilization:    {gpu_util:.1f}%")
    print(f"  Data/Compute ratio: {avg_data_time / avg_compute_time:.2f}x")

    if avg_data_time > avg_compute_time * 1.5:
        print(f"\n  ❌ GPU is STARVING for data!")
        print(f"  ❌ GPU spends {100 - gpu_util:.1f}% time WAITING")
        print(f"\n  💡 SOLUTION: Increase num_workers")
        current_workers = cfg["data"].get("num_workers", 0)
        recommended = max(4, current_workers * 2)
        print(f"     Current:     num_workers={current_workers}")
        print(f"     Recommended: num_workers={recommended}")
    else:
        print(f"\n  ✅ Data loading is efficient!")
        print(f"  ✅ GPU utilization is good ({gpu_util:.1f}%)")

    print(f"\n🚀 Throughput:")
    print(f"  Tokens/sec:      {throughput:,.0f}")
    print(f"  Total tokens:    {total_tokens:,}")
    print(f"  Total time:      {total_time:.2f}s")

    # Projections
    steps_per_hour = (3600 / avg_step_time) * 1000
    total_steps = cfg["train"]["max_steps"]
    hours_to_complete = total_steps / steps_per_hour

    print(f"\n📈 Projections:")
    print(f"  Steps/hour:      {steps_per_hour:,.0f}")
    print(f"  Total steps:     {total_steps:,}")
    print(f"  Time to finish:  {hours_to_complete:.1f} hours")

    # Memory usage
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9

        print(f"\n💾 Memory Usage:")
        print(f"  Allocated:       {mem_allocated:.2f} GB")
        print(f"  Reserved:        {mem_reserved:.2f} GB")
        print(f"  Total:           {mem_total:.2f} GB")
        print(f"  Utilization:     {(mem_allocated / mem_total) * 100:.1f}%")

    print("\n" + "="*80)

    # Recommendations
    print("\n💡 RECOMMENDATIONS:\n")

    if avg_data_time > avg_compute_time * 1.5:
        print("1. 🔴 CRITICAL: Increase num_workers to 4-6")
        print("   Expected speedup: 2-4×")
        print("   Expected GPU util: 75-85%\n")

    if cfg["train"]["batch_size"] < 12 and mem_allocated / mem_total < 0.7:
        print("2. ⚠️  Consider increasing batch_size to utilize more VRAM")
        print(f"   Current: {cfg['train']['batch_size']}")
        print(f"   Suggested: 12-16 (you have {(1 - mem_allocated / mem_total) * 100:.0f}% VRAM headroom)\n")

    if not cfg["data"].get("num_workers", 0):
        print("3. 🔧 Enable persistent_workers=True to avoid worker recreation overhead\n")

    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile SLGA training bottlenecks")
    parser.add_argument("--config", type=str, default="config/config_3090.yaml", help="Path to config file")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to profile")
    args = parser.parse_args()

    profile_training(args.config, args.steps)
