#!/bin/bash
# Script pour nettoyer et redémarrer l'entraînement SLGA

echo "=========================================="
echo "SLGA Clean Restart Script"
echo "=========================================="
echo ""

# Vérifier si on est dans le bon répertoire
if [ ! -f "config.yaml" ]; then
    echo "❌ Error: config.yaml not found"
    echo "   Please run this script from the SLGA project root"
    exit 1
fi

# Demander confirmation
echo "⚠️  WARNING: This will delete ALL existing checkpoints and logs"
echo ""
echo "The following will be removed:"
echo "  - out_slga/ckpt_*"
echo "  - out_slga/tensorboard/*"
echo ""
read -p "Are you sure you want to continue? (yes/no): " -r
echo ""

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Sauvegarder la config actuelle
echo "📋 Backing up config.yaml..."
cp config.yaml config.yaml.backup
echo "✓ Backup saved to config.yaml.backup"
echo ""

# Supprimer les anciens checkpoints
echo "🗑️  Removing old checkpoints..."
rm -rf out_slga/ckpt_*
echo "✓ Checkpoints removed"
echo ""

# Nettoyer TensorBoard
echo "🗑️  Cleaning TensorBoard logs..."
rm -rf out_slga/tensorboard/*
echo "✓ TensorBoard logs cleaned"
echo ""

# Créer les répertoires nécessaires
echo "📁 Creating directories..."
mkdir -p out_slga/tensorboard
echo "✓ Directories ready"
echo ""

# Résumé
echo "=========================================="
echo "✅ Clean restart ready!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review your config.yaml if needed"
echo "  2. Start training: python scripts/train.py"
echo "  3. Monitor with: tensorboard --logdir out_slga/tensorboard"
echo ""
echo "Expected results with fixed code:"
echo "  - Step 1000: PPL ~1000-2000 (not ~12000!)"
echo "  - Step 10000: PPL ~200-400"
echo "  - Step 30000: PPL ~60-120"
echo ""
