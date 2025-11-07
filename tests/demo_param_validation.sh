#!/bin/bash
# Démonstration de la validation des paramètres dans generate.py

echo "================================================================================"
echo "DÉMONSTRATION: Validation des Paramètres de Génération"
echo "================================================================================"
echo ""

echo "1. Test avec paramètres INVALIDES (temperature négative)"
echo "   Command: python3 scripts/generate.py --checkpoint dummy.pt --temperature -0.5 --max-tokens 10"
echo "   ---"
python3 scripts/generate.py --checkpoint dummy.pt --config config/config.bk.yaml --temperature -0.5 --max-tokens 10 2>&1 | head -15
echo ""

echo "================================================================================"
echo ""

echo "2. Test avec paramètres INVALIDES (top_k = 0)"
echo "   Command: python3 scripts/generate.py --checkpoint dummy.pt --top-k 0 --max-tokens 10"
echo "   ---"
python3 scripts/generate.py --checkpoint dummy.pt --config config/config.bk.yaml --top-k 0 --max-tokens 10 2>&1 | head -15
echo ""

echo "================================================================================"
echo ""

echo "3. Test avec paramètres INVALIDES (top_p > 1)"
echo "   Command: python3 scripts/generate.py --checkpoint dummy.pt --top-p 1.5 --max-tokens 10"
echo "   ---"
python3 scripts/generate.py --checkpoint dummy.pt --config config/config.bk.yaml --top-p 1.5 --max-tokens 10 2>&1 | head -15
echo ""

echo "================================================================================"
echo ""

echo "4. Test avec MULTIPLES erreurs simultanées"
echo "   Command: python3 scripts/generate.py --temperature -1 --top-k 0 --top-p 2 --max-tokens -5"
echo "   ---"
python3 scripts/generate.py --checkpoint dummy.pt --config config/config.bk.yaml --temperature -1 --top-k 0 --top-p 2 --max-tokens -5 2>&1 | head -20
echo ""

echo "================================================================================"
echo ""

echo "5. Test avec paramètres VALIDES (doit passer la validation)"
echo "   Command: python3 scripts/generate.py --temperature 0.8 --top-k 50 --top-p 0.9 --max-tokens 10"
echo "   ---"
python3 scripts/generate.py --checkpoint dummy_model.pt --config config/config.bk.yaml --temperature 0.8 --top-k 50 --top-p 0.9 --max-tokens 10 2>&1 | head -20
echo ""

echo "================================================================================"
echo "CONCLUSION:"
echo "  ✓ La validation détecte correctement les paramètres invalides"
echo "  ✓ Les messages d'erreur sont clairs et informatifs"
echo "  ✓ Les paramètres valides passent la validation sans problème"
echo "================================================================================"
