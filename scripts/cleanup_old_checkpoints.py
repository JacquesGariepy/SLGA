#!/usr/bin/env python3
"""
Script de nettoyage manuel des vieux checkpoints.

Utilité:
- Nettoyer les checkpoints accumulés avant l'implémentation de la rotation
- Garder seulement les N derniers checkpoints
- Libérer de l'espace disque

Usage:
    # Dry-run (affiche ce qui serait supprimé)
    python scripts/cleanup_old_checkpoints.py --out-dir out_slga --keep 5 --dry-run

    # Vraie suppression
    python scripts/cleanup_old_checkpoints.py --out-dir out_slga --keep 5

    # Interactive (demande confirmation)
    python scripts/cleanup_old_checkpoints.py --out-dir out_slga --keep 5 --interactive
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

def get_checkpoint_dirs(out_dir):
    """Retourne la liste des répertoires de checkpoints triés par step"""
    if not os.path.exists(out_dir):
        print(f"Error: Directory {out_dir} does not exist")
        return []

    checkpoints = [d for d in os.listdir(out_dir) if d.startswith("ckpt_")]

    # Trier par step number
    checkpoints_sorted = sorted(checkpoints, key=lambda x: int(x.split("_")[1]))

    return checkpoints_sorted

def get_checkpoint_size(out_dir, ckpt_name):
    """Calcule la taille d'un checkpoint en MB"""
    ckpt_path = os.path.join(out_dir, ckpt_name)
    total_size = 0

    for dirpath, dirnames, filenames in os.walk(ckpt_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)

    return total_size / (1024 * 1024)  # MB

def main():
    parser = argparse.ArgumentParser(description="Clean up old checkpoints")
    parser.add_argument("--out-dir", type=str, default="out_slga",
                       help="Output directory containing checkpoints")
    parser.add_argument("--keep", type=int, default=5,
                       help="Number of most recent checkpoints to keep (default: 5)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be deleted without actually deleting")
    parser.add_argument("--interactive", action="store_true",
                       help="Ask for confirmation before deleting each checkpoint")
    args = parser.parse_args()

    out_dir = args.out_dir
    keep_last_n = args.keep

    print("="*70)
    print("CHECKPOINT CLEANUP UTILITY")
    print("="*70)
    print(f"Output directory: {out_dir}")
    print(f"Keep last: {keep_last_n} checkpoints")
    print(f"Mode: {'DRY RUN (no deletion)' if args.dry_run else 'LIVE (will delete)'}")
    print("="*70)

    # Get all checkpoints
    checkpoints = get_checkpoint_dirs(out_dir)

    if not checkpoints:
        print(f"\nNo checkpoints found in {out_dir}")
        return 0

    print(f"\nFound {len(checkpoints)} checkpoints:")

    total_size = 0
    for i, ckpt in enumerate(checkpoints):
        size_mb = get_checkpoint_size(out_dir, ckpt)
        total_size += size_mb
        step = ckpt.split("_")[1]
        status = "KEEP" if i >= len(checkpoints) - keep_last_n else "DELETE"
        marker = "✓" if status == "KEEP" else "✗"
        print(f"  {marker} {ckpt} (step {step}, {size_mb:.1f} MB) - {status}")

    print(f"\nTotal size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")

    # Determine which to delete
    if len(checkpoints) <= keep_last_n:
        print(f"\n✓ Only {len(checkpoints)} checkpoints present (≤ {keep_last_n}), nothing to delete")
        return 0

    to_delete = checkpoints[:-keep_last_n]
    to_keep = checkpoints[-keep_last_n:]

    delete_size = sum(get_checkpoint_size(out_dir, ckpt) for ckpt in to_delete)

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Checkpoints to DELETE: {len(to_delete)}")
    print(f"Checkpoints to KEEP: {len(to_keep)}")
    print(f"Space to free: {delete_size:.1f} MB ({delete_size/1024:.2f} GB)")
    print(f"{'='*70}")

    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No files will be deleted")
        print("\nWould delete:")
        for ckpt in to_delete:
            print(f"  - {ckpt}")
        print("\nTo actually delete, run without --dry-run")
        return 0

    # Confirmation
    if not args.interactive:
        print("\n⚠️  WARNING: This will permanently delete checkpoints!")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0

    # Delete
    deleted_count = 0
    deleted_size = 0

    for ckpt in to_delete:
        ckpt_path = os.path.join(out_dir, ckpt)
        size_mb = get_checkpoint_size(out_dir, ckpt)

        if args.interactive:
            response = input(f"\nDelete {ckpt} ({size_mb:.1f} MB)? [y/N]: ")
            if response.lower() != 'y':
                print(f"  Skipped {ckpt}")
                continue

        try:
            shutil.rmtree(ckpt_path)
            print(f"  ✓ Deleted {ckpt} ({size_mb:.1f} MB)")
            deleted_count += 1
            deleted_size += size_mb
        except Exception as e:
            print(f"  ✗ Error deleting {ckpt}: {e}")

    print(f"\n{'='*70}")
    print(f"CLEANUP COMPLETE")
    print(f"{'='*70}")
    print(f"Deleted: {deleted_count}/{len(to_delete)} checkpoints")
    print(f"Freed: {deleted_size:.1f} MB ({deleted_size/1024:.2f} GB)")
    print(f"Remaining: {len(to_keep)} checkpoints")
    print(f"{'='*70}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
