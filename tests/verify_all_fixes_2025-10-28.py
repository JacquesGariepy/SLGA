#!/usr/bin/env python3
"""
Script de vérification des fixes appliqués le 2025-10-28.

Vérifie que tous les changements ont été correctement appliqués:
1. Config: lr, warmup_steps, keep_last_checkpoints, debug_checkpoints
2. Utils: save_checkpoint avec rotation, get_memory_usage fix
3. Train: cache_ids fix, gradient monitoring fix, keep_last_n usage

Usage:
    python tests/verify_all_fixes_2025-10-28.py
"""

import os
import sys
import yaml
import re

# Couleurs pour terminal
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def success(msg):
    print(f"{GREEN}✓{RESET} {msg}")

def failure(msg):
    print(f"{RED}✗{RESET} {msg}")

def warning(msg):
    print(f"{YELLOW}⚠{RESET} {msg}")

def check_config():
    """Vérifie les changements dans config.wikipedia.yaml"""
    print("\n" + "="*60)
    print("1. Vérification CONFIG")
    print("="*60)

    config_path = "config/config.wikipedia.yaml"
    if not os.path.exists(config_path):
        failure(f"Config file not found: {config_path}")
        return False

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    checks = []

    # Check lr
    lr = cfg["train"]["lr"]
    if lr == 1.5e-4:
        success(f"lr = {lr} (correct, was 2.0e-4)")
        checks.append(True)
    else:
        failure(f"lr = {lr} (should be 1.5e-4)")
        checks.append(False)

    # Check warmup_steps
    warmup = cfg["train"]["warmup_steps"]
    if warmup == 5000:
        success(f"warmup_steps = {warmup} (correct, was 2000)")
        checks.append(True)
    else:
        failure(f"warmup_steps = {warmup} (should be 5000)")
        checks.append(False)

    # Check keep_last_checkpoints
    keep_last = cfg["train"].get("keep_last_checkpoints")
    if keep_last == 5:
        success(f"keep_last_checkpoints = {keep_last} (correct, new parameter)")
        checks.append(True)
    else:
        failure(f"keep_last_checkpoints = {keep_last} (should be 5)")
        checks.append(False)

    # Check debug_checkpoints
    debug_ckpt = cfg["train"].get("debug_checkpoints")
    if debug_ckpt == False:
        success(f"debug_checkpoints = {debug_ckpt} (correct, new parameter)")
        checks.append(True)
    else:
        failure(f"debug_checkpoints = {debug_ckpt} (should be false)")
        checks.append(False)

    return all(checks)

def check_utils():
    """Vérifie les changements dans scripts/utils.py"""
    print("\n" + "="*60)
    print("2. Vérification UTILS")
    print("="*60)

    utils_path = "scripts/utils.py"
    if not os.path.exists(utils_path):
        failure(f"Utils file not found: {utils_path}")
        return False

    with open(utils_path) as f:
        utils_code = f.read()

    checks = []

    # Check save_checkpoint signature
    if "keep_last_n: int = None" in utils_code:
        success("save_checkpoint() has keep_last_n parameter")
        checks.append(True)
    else:
        failure("save_checkpoint() missing keep_last_n parameter")
        checks.append(False)

    # Check rotation logic
    if "all_ckpts = sorted([d for d in os.listdir(out_dir)" in utils_code:
        success("Checkpoint rotation logic present")
        checks.append(True)
    else:
        failure("Checkpoint rotation logic missing")
        checks.append(False)

    # Check shutil.rmtree
    if "shutil.rmtree(old_path)" in utils_code or "shutil.rmtree(os.path.join(out_dir, old))" in utils_code:
        success("Old checkpoint deletion logic present")
        checks.append(True)
    else:
        failure("Old checkpoint deletion logic missing")
        checks.append(False)

    # Check memory free fix
    if "free = total - reserved" in utils_code:
        success("get_memory_usage() fixed (free = total - reserved)")
        checks.append(True)
    else:
        failure("get_memory_usage() not fixed (should use reserved, not allocated)")
        checks.append(False)

    # Check load_latest_checkpoint helper
    if "def load_latest_checkpoint" in utils_code:
        success("load_latest_checkpoint() helper function added")
        checks.append(True)
    else:
        warning("load_latest_checkpoint() helper not found (optional)")
        checks.append(True)  # Not critical

    return all(checks)

def check_train():
    """Vérifie les changements dans scripts/train.py"""
    print("\n" + "="*60)
    print("3. Vérification TRAIN")
    print("="*60)

    train_path = "scripts/train.py"
    if not os.path.exists(train_path):
        failure(f"Train file not found: {train_path}")
        return False

    with open(train_path) as f:
        train_code = f.read()

    checks = []

    # Check cache_ids truncation fix
    if "cache_ids = cache_ids[mask].unsqueeze(0) if mask.any() else None" in train_code:
        success("cache_ids truncation fix present")
        checks.append(True)
    else:
        failure("cache_ids truncation fix missing")
        checks.append(False)

    # Check gradient monitoring fix (named_parameters)
    gradient_monitoring_pattern = r"for name, param in model\.named_parameters\(\):"
    if re.search(gradient_monitoring_pattern, train_code):
        success("Gradient monitoring uses named_parameters() (crash fix)")
        checks.append(True)
    else:
        failure("Gradient monitoring not fixed (should use named_parameters)")
        checks.append(False)

    # Check keep_last_n usage in checkpoint save
    if "keep_last = cfg[\"train\"].get(\"keep_last_checkpoints\")" in train_code:
        success("keep_last_n retrieved from config")
        checks.append(True)
    else:
        failure("keep_last_n not retrieved from config")
        checks.append(False)

    if "save_checkpoint(model, optimizer, scheduler, out_dir, step, accelerator, keep_last_n=keep_last)" in train_code:
        success("save_checkpoint() called with keep_last_n parameter")
        checks.append(True)
    else:
        failure("save_checkpoint() not called with keep_last_n parameter")
        checks.append(False)

    # Check debug_checkpoints conditional
    if "debug_ckpt = cfg[\"train\"].get(\"debug_checkpoints\"" in train_code:
        success("Debug checkpoint logging is conditional")
        checks.append(True)
    else:
        failure("Debug checkpoint logging not conditional")
        checks.append(False)

    return all(checks)

def main():
    """Main verification routine"""
    print("\n" + "="*60)
    print("VÉRIFICATION DES FIXES 2025-10-28")
    print("="*60)

    results = []

    # Check config
    results.append(("CONFIG", check_config()))

    # Check utils
    results.append(("UTILS", check_utils()))

    # Check train
    results.append(("TRAIN", check_train()))

    # Summary
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)

    all_passed = True
    for name, passed in results:
        if passed:
            success(f"{name}: All checks passed")
        else:
            failure(f"{name}: Some checks failed")
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print(f"{GREEN}✓ TOUS LES FIXES SONT CORRECTEMENT APPLIQUÉS{RESET}")
        print("="*60)
        print("\nVous pouvez relancer le training:")
        print("  python scripts/train.py --config config/config.wikipedia.yaml --resume")
        return 0
    else:
        print(f"{RED}✗ CERTAINS FIXES SONT MANQUANTS{RESET}")
        print("="*60)
        print("\nVeuillez vérifier les erreurs ci-dessus et réappliquer les fixes.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
