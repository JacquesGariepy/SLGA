#!/usr/bin/env python3
"""
Inspecter l'état du trainer pour comprendre l'état d'optimisation.
"""
import torch
import sys
import os

def inspect_trainer_state(checkpoint_path: str):
    """Inspecte le trainer_state.pt"""

    state_file = os.path.join(checkpoint_path, "trainer_state.pt")

    if not os.path.exists(state_file):
        print(f"❌ Trainer state not found: {state_file}")
        return

    print("=" * 80)
    print(f"=== Trainer State Inspection: {checkpoint_path} ===")
    print("=" * 80)
    print()

    # Charger l'état
    state = torch.load(state_file, map_location="cpu")

    print("📊 Training Progress")
    print("-" * 40)
    print(f"  Current step:     {state.get('step', 'N/A')}")
    print(f"  Current epoch:    {state.get('epoch', 'N/A')}")
    tokens_seen = state.get('tokens_seen', 'N/A')
    if isinstance(tokens_seen, (int, float)):
        print(f"  Tokens seen:      {tokens_seen:,}")
    else:
        print(f"  Tokens seen:      {tokens_seen}")
    print()

    print("📉 Loss History")
    print("-" * 40)
    if 'train_loss' in state:
        print(f"  Last train loss:  {state['train_loss']:.4f}")
    if 'val_loss' in state:
        print(f"  Last val loss:    {state['val_loss']:.4f}")
    print()

    print("🔧 Optimizer State")
    print("-" * 40)
    if 'optimizer' in state:
        opt_state = state['optimizer']
        print(f"  State dict keys:  {list(opt_state.keys())}")

        # Learning rate actuel
        if 'param_groups' in opt_state:
            for i, pg in enumerate(opt_state['param_groups']):
                lr = pg.get('lr', 'N/A')
                print(f"  Param group {i} LR: {lr}")

        # State de momentum/variance
        if 'state' in opt_state:
            num_params = len(opt_state['state'])
            print(f"  Parameters tracked: {num_params}")

            # Sample first param state
            if num_params > 0:
                first_key = list(opt_state['state'].keys())[0]
                first_state = opt_state['state'][first_key]
                print(f"  First param state keys: {list(first_state.keys())}")

                if 'exp_avg' in first_state:
                    exp_avg_norm = first_state['exp_avg'].norm().item()
                    print(f"  First param exp_avg norm: {exp_avg_norm:.6f}")

                if 'exp_avg_sq' in first_state:
                    exp_avg_sq_norm = first_state['exp_avg_sq'].norm().item()
                    print(f"  First param exp_avg_sq norm: {exp_avg_sq_norm:.6f}")
    print()

    print("📅 Scheduler State")
    print("-" * 40)
    if 'scheduler' in state:
        sched = state['scheduler']
        print(f"  Scheduler keys: {list(sched.keys())}")
        if 'last_epoch' in sched:
            print(f"  Last epoch: {sched['last_epoch']}")
        if '_step_count' in sched:
            print(f"  Step count: {sched['_step_count']}")
    print()

    print("🎲 RNG State")
    print("-" * 40)
    if 'rng_state' in state:
        print(f"  CPU RNG: {len(state['rng_state'])} bytes")
    if 'cuda_rng_state' in state:
        print(f"  CUDA RNG: {len(state['cuda_rng_state'])} bytes")
    print()

    print("💾 State Summary")
    print("-" * 40)
    total_keys = len(state.keys())
    print(f"  Total state keys: {total_keys}")
    print(f"  Keys: {list(state.keys())}")
    print()

    # Calcul du LR théorique au step actuel
    print("🧮 Learning Rate Calculation")
    print("-" * 40)
    step = state.get('step', 0)

    # Config par défaut (à adapter selon config.yaml)
    lr_max = 2.0e-4
    warmup_steps = 2000
    max_steps = 100000

    if step < warmup_steps:
        # Warmup linéaire
        lr_theoretical = lr_max * (step / warmup_steps)
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        lr_theoretical = lr_max * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        lr_theoretical = lr_theoretical.item()

    print(f"  Step: {step}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Max steps: {max_steps}")
    print(f"  LR max: {lr_max}")
    print(f"  Theoretical LR: {lr_theoretical:.6e}")

    if 'optimizer' in state and 'param_groups' in state['optimizer']:
        actual_lr = state['optimizer']['param_groups'][0]['lr']
        print(f"  Actual LR: {actual_lr:.6e}")
        print(f"  Match: {'✓' if abs(actual_lr - lr_theoretical) < 1e-8 else '✗'}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_trainer_state.py <checkpoint_path>")
        print("Example: python inspect_trainer_state.py out_slga/ckpt_16000")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    inspect_trainer_state(checkpoint_path)
