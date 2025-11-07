#!/usr/bin/env python3
"""
Script pour patcher train.py avec Live Metrics Display.

Usage:
    python scripts/patch_train_with_live_metrics.py [--dry-run]

Options:
    --dry-run: Affiche les changements sans les appliquer
"""

import sys
import os
import argparse
from pathlib import Path


def create_backup(file_path: Path) -> Path:
    """Crée un backup du fichier"""
    backup_path = file_path.with_suffix('.py.backup')
    backup_path.write_text(file_path.read_text())
    return backup_path


def patch_imports(content: str) -> str:
    """Ajoute l'import du LiveMetricsDisplay"""

    # Trouver la ligne après les imports from src
    import_marker = "from src.data import get_tokenizer"

    if import_marker in content:
        content = content.replace(
            import_marker,
            f"{import_marker}, load_text_dataset, CollatorLocal, CollatorLocalGlobal\nfrom src.live_metrics import LiveMetricsDisplay"
        )
    else:
        # Ajouter après le dernier import src
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('from src.') and i + 1 < len(lines):
                # Insérer après
                lines.insert(i + 1, 'from src.live_metrics import LiveMetricsDisplay')
                content = '\n'.join(lines)
                break

    return content


def patch_initialization(content: str) -> str:
    """Remplace progress_bar par live_display"""

    # Trouver et remplacer
    old_code = '''    progress_bar = tqdm(
        total=total_steps,
        desc="Training",
        disable=not accelerator.is_main_process,
    )'''

    new_code = '''    # Live metrics display
    if accelerator.is_main_process:
        live_display = LiveMetricsDisplay(
            max_steps=total_steps,
            log_every=cfg["train"].get("log_every", 50),
            width=100,
            compact=False,  # Set True for condensed view
        )
    else:
        live_display = None

    # Keep tqdm for backup (optional)
    progress_bar = tqdm(
        total=total_steps,
        desc="Training",
        disable=not accelerator.is_main_process or live_display is not None,
    )'''

    if old_code in content:
        content = content.replace(old_code, new_code)

    return content


def patch_logging_section(content: str) -> str:
    """Remplace la section de logging"""

    # Trouver la section de logging
    marker_start = "# Logging\n            if accelerator.is_main_process and step % cfg[\"train\"].get(\"log_every\""

    if marker_start not in content:
        return content

    # Trouver la fin de la section (avant "step_start_time = time.time()")
    lines = content.split('\n')
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if '# Logging' in line and 'accelerator.is_main_process' in lines[i+1]:
            start_idx = i
        if start_idx and 'step_start_time = time.time()' in line:
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        return content

    # Nouvelle section de logging
    new_logging = '''            # Logging avec Live Metrics
            if accelerator.is_main_process and step % cfg["train"].get("log_every", 50) == 0:
                # Gather loss (multi-GPU)
                loss_gathered = accelerator.gather(loss_ce.detach()).mean().item()
                lr_current = scheduler.get_last_lr()[0]
                ppl = math.exp(min(loss_gathered, 10))

                # Performance metrics
                elapsed_time = time.time() - step_start_time
                steps_per_sec = steps_since_log / elapsed_time if elapsed_time > 0 else 0
                tokens_per_sec = steps_per_sec * cfg["train"]["batch_size"] * current_seq_len

                # GPU memory
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1e9  # GB
                    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                else:
                    mem_allocated = 0
                    mem_total = 0

                # Update live display
                if live_display:
                    live_display.update(
                        step=step,
                        loss=loss_gathered,
                        ppl=ppl,
                        lr=lr_current,
                        grad_norm=last_grad_norm,
                        seq_len=current_seq_len,
                        global_weight=global_weight,
                        tokens_per_sec=tokens_per_sec,
                        gpu_memory_gb=mem_allocated,
                        gpu_memory_total_gb=mem_total,
                        num_landmarks=last_num_landmarks,
                        spacing_loss=last_spacing_loss,
                        sparsity_loss=last_spar_loss,
                    )

                # W&B (keep existing)
                if cfg["log"].get("wandb", False):
                    log_dict = {
                        "step": step,
                        "epoch": epoch,
                        "loss": loss_gathered,
                        "perplexity": ppl,
                        "lr": lr_current,
                        "seq_len": current_seq_len,
                        "global_weight": global_weight,
                        "grad_norm": last_grad_norm,
                    }
                    wandb.log(log_dict, step=step)

                # TensorBoard (keep existing)
                if writer is not None:
                    writer.add_scalar("train/loss", loss_gathered, step)
                    writer.add_scalar("train/ppl", ppl, step)
                    writer.add_scalar("train/lr", lr_current, step)
                    writer.add_scalar("train/grad_norm", last_grad_norm, step)

                # Reset timing'''

    # Remplacer
    lines[start_idx:end_idx] = new_logging.split('\n')
    content = '\n'.join(lines)

    return content


def patch_validation(content: str) -> str:
    """Ajoute live display update dans validation"""

    # Trouver la fin de la fonction validate
    marker = 'print(f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:>7.2f}")'

    if marker in content:
        addition = '''

    # Update live display with validation metrics
    if accelerator.is_main_process and 'live_display' in globals() and live_display:
        # Get current training metrics (last logged)
        live_display.update(
            step=step,
            val_loss=val_loss,
            val_ppl=val_ppl,
        )
'''
        content = content.replace(marker, marker + addition)

    return content


def main():
    parser = argparse.ArgumentParser(description="Patch train.py with Live Metrics")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    parser.add_argument("--train-file", type=str, default="scripts/train.py", help="Path to train.py")

    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent
    train_path = project_root / args.train_file

    if not train_path.exists():
        print(f"❌ File not found: {train_path}")
        sys.exit(1)

    print("=" * 80)
    print("🔧 Patching train.py with Live Metrics Display")
    print("=" * 80)
    print()

    # Read original
    content = train_path.read_text()
    original_content = content

    print("📝 Applying patches...")

    # Apply patches
    content = patch_imports(content)
    print("  ✓ Added import")

    content = patch_initialization(content)
    print("  ✓ Replaced progress_bar with live_display")

    content = patch_logging_section(content)
    print("  ✓ Updated logging section")

    content = patch_validation(content)
    print("  ✓ Added validation display")

    # Count changes
    original_lines = original_content.split('\n')
    new_lines = content.split('\n')
    diff_count = sum(1 for a, b in zip(original_lines, new_lines) if a != b)
    diff_count += abs(len(original_lines) - len(new_lines))

    print()
    print(f"📊 Changes: {diff_count} lines modified/added")
    print()

    if args.dry_run:
        print("🔍 DRY RUN - No changes written")
        print()
        print("Changes preview:")
        print("-" * 80)

        # Show first difference
        for i, (old, new) in enumerate(zip(original_lines, new_lines)):
            if old != new:
                print(f"Line {i+1}:")
                print(f"  - {old}")
                print(f"  + {new}")
                if i > 10:
                    print("  ... (more changes)")
                    break

        print()
        print("Run without --dry-run to apply changes")

    else:
        # Create backup
        backup_path = create_backup(train_path)
        print(f"💾 Backup created: {backup_path}")

        # Write patched version
        train_path.write_text(content)
        print(f"✅ Patched file written: {train_path}")

        print()
        print("✨ Patching complete!")
        print()
        print("Next steps:")
        print("  1. Review changes: diff scripts/train.py scripts/train.py.backup")
        print("  2. Test display: python -c \"from src.live_metrics import test_display; test_display()\"")
        print("  3. Run training: python scripts/train.py --config config.yaml")
        print()

    print("=" * 80)


if __name__ == "__main__":
    main()
