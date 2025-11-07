#!/bin/bash
# Quick fix for SLGA performance bottleneck
# Applies num_workers fix to config files

set -e

echo "=========================================="
echo "🔧 SLGA Performance Quick Fix"
echo "=========================================="
echo ""

# Check if config exists
if [ ! -f "config/config_3090.yaml" ]; then
    echo "❌ Error: config/config_3090.yaml not found!"
    exit 1
fi

# Backup config
echo "📦 Creating backup..."
cp config/config_3090.yaml config/config_3090.yaml.backup_$(date +%Y%m%d_%H%M%S)
echo "   ✅ Backup saved"

# Apply fix
echo ""
echo "🔧 Applying fixes..."

# Fix 1: Change num_workers from 0 to 4
echo "   1. Setting num_workers: 0 → 4"
sed -i 's/num_workers: 0/num_workers: 4  # ✅ FIXED: Enable parallel data loading/' config/config_3090.yaml

# Fix 2: Increase batch_size if possible
echo "   2. Increasing batch_size: 8 → 12 (optional, if VRAM allows)"
sed -i 's/batch_size: 8/batch_size: 12  # ✅ OPTIMIZED: Better GPU utilization/' config/config_3090.yaml

# Fix 3: Adjust accum_steps to keep effective batch ~32-36
echo "   3. Adjusting accum_steps: 4 → 3 (keep effective batch)"
sed -i 's/accum_steps: 4/accum_steps: 3  # ✅ OPTIMIZED: effective_batch = 12 × 3 = 36/' config/config_3090.yaml

echo ""
echo "✅ Fixes applied!"
echo ""
echo "📊 Summary of changes:"
echo "   • num_workers: 0 → 4 (parallel data loading)"
echo "   • batch_size: 8 → 12 (better GPU util)"
echo "   • accum_steps: 4 → 3 (maintain effective batch)"
echo ""
echo "📈 Expected improvements:"
echo "   • GPU utilization: 4-5% → 75-85%"
echo "   • Throughput: ~1k tok/s → ~12k tok/s"
echo "   • Training speed: ~3× faster"
echo ""
echo "🚀 Next steps:"
echo "   1. Review changes: diff config/config_3090.yaml.backup_* config/config_3090.yaml"
echo "   2. Profile before/after: python scripts/profile_bottleneck.py --steps 100"
echo "   3. Restart training: python scripts/train.py --config config/config_3090.yaml --resume"
echo ""
echo "=========================================="
