#!/bin/bash
# Script to restart training after scheduler bug fix

echo "=========================================="
echo "SLGA Training Restart Script"
echo "=========================================="
echo ""

# 1. Kill current training
echo "1. Stopping current training..."
pkill -f "train.py.*config_fineweb" || echo "   No training process found"
sleep 2
echo "   ✓ Training stopped"
echo ""

# 2. Delete bad checkpoints
echo "2. Deleting corrupted checkpoints..."
if [ -d "out_slga_fineweb" ]; then
    echo "   Found out_slga_fineweb/"

    # Keep tensorboard logs
    if [ -d "out_slga_fineweb/tensorboard" ]; then
        mv out_slga_fineweb/tensorboard /tmp/slga_tensorboard_backup
        echo "   ✓ Backed up tensorboard logs"
    fi

    # Delete all checkpoints
    rm -rf out_slga_fineweb/ckpt_*
    echo "   ✓ Deleted bad checkpoints (trained with wrong LR)"

    # Restore tensorboard
    if [ -d "/tmp/slga_tensorboard_backup" ]; then
        mkdir -p out_slga_fineweb
        mv /tmp/slga_tensorboard_backup out_slga_fineweb/tensorboard
        echo "   ✓ Restored tensorboard logs"
    fi
else
    echo "   Directory out_slga_fineweb/ not found, creating..."
    mkdir -p out_slga_fineweb
fi
echo ""

# 3. Clear Python cache
echo "3. Clearing Python cache..."
find . -name "*.pyc" -delete 2>/dev/null
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
echo "   ✓ Cache cleared"
echo ""

# 4. Verify fix applied
echo "4. Verifying scheduler fix..."
if grep -q "warmup_steps // accum_steps" scripts/train.py; then
    echo "   ✓ Scheduler fix detected in train.py"
else
    echo "   ❌ WARNING: Scheduler fix not found!"
    echo "   Please apply the fix manually (see docs/CRITICAL_SCHEDULER_BUG.md)"
    exit 1
fi
echo ""

# 5. Restart training
echo "5. Restarting training with corrected scheduler..."
echo ""
echo "=========================================="
echo "TRAINING STARTED"
echo "=========================================="
echo ""

# Run training
python scripts/train.py \
  --config config/config_fineweb_edu_3090_optimized.yaml \
  --max-steps 100000

echo ""
echo "=========================================="
echo "TRAINING COMPLETED OR INTERRUPTED"
echo "=========================================="
