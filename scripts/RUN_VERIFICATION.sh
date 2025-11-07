#!/bin/bash
# 🔍 SLGA Verification Script
# Run this to verify dataset and training pipeline before full training

set -e  # Exit on error

echo "================================================================================"
echo "SLGA Training Verification"
echo "================================================================================"
echo ""

# Check environment
echo "Step 0: Checking environment..."
if ! python -c "import torch" 2>/dev/null; then
    echo "❌ PyTorch not found. Activate conda environment first:"
    echo "   conda activate slga"
    exit 1
fi
echo "✅ PyTorch available"
echo ""

# Check config
if [ ! -f "config_3090.yaml" ]; then
    echo "❌ config_3090.yaml not found"
    exit 1
fi
echo "✅ Config found: config_3090.yaml"
echo ""

# Step 1: Inspect training batches
echo "================================================================================"
echo "Step 1: Inspecting Training Batches"
echo "================================================================================"
echo ""
python scripts/inspect_training_batch.py --config config_3090.yaml --num-batches 3

echo ""
read -p "Press Enter to continue to quick training test, or Ctrl+C to stop..."
echo ""

# Step 2: Quick training test (50 steps)
echo "================================================================================"
echo "Step 2: Quick Training Test (50 steps)"
echo "================================================================================"
echo ""
echo "This will train for 50 steps to verify the loop works..."
echo ""

# Backup config and use config_3090
if [ -f "config.yaml" ]; then
    cp config.yaml config.yaml.backup_verification
    echo "✅ Backed up config.yaml → config.yaml.backup_verification"
fi
cp config_3090.yaml config.yaml
echo "✅ Using config_3090.yaml"
echo ""

# Modify max_steps temporarily for quick test
python -c "
import yaml
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['train']['max_steps'] = 50
cfg['train']['save_every'] = 1000  # Don't save during test
with open('config_test.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"

echo "Running training for 50 steps..."
python scripts/train.py --config config_test.yaml

echo ""
echo "✅ Training test completed"
rm config_test.yaml
echo ""

read -p "Press Enter to continue to architecture diagnostic, or Ctrl+C to stop..."
echo ""

# Step 3: Architecture diagnostic
echo "================================================================================"
echo "Step 3: Architecture Diagnostic"
echo "================================================================================"
echo ""
python scripts/diagnose.py

echo ""
echo "================================================================================"
echo "Verification Complete!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Review TensorBoard logs:"
echo "   tensorboard --logdir out_slga/tensorboard --port 6006"
echo "   Open: http://localhost:6006"
echo ""
echo "2. If all checks passed, clean and start full training:"
echo "   bash scripts/clean_restart.sh"
echo "   python scripts/train.py"
echo ""
echo "3. Monitor training:"
echo "   python scripts/monitor.py"
echo ""
echo "See VERIFICATION_REPORT.md for detailed explanation of results."
echo ""
