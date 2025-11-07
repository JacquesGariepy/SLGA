# 🔴 DIAGNOSTIC URGENT - Sparsity Loss Bloquée à 4.25

**Date**: 2025-10-28  
**Problème**: Sparsity loss constante, domine l'entraînement

## 📊 RÉSULTATS TRAINING 1000 STEPS

Step 1000:
- Loss CE: 8.15
- Spacing: 0.0097 (0.1% de la loss) ❌
- Sparsity: 4.2500 (52% de la loss!) ❌
- LM: 48→48 ✅

**Sparsity loss = 4.25 dans 100% des steps (constante!)**

## 🐛 BUG: Mon fix de sparsity_loss est incorrect

La formule calcule toujours ~4.25:
- effective_size ≈ L (toujours)
- active_fraction = L/L = 1.0
- target = 48/384 * 1.2 = 0.15
- ReLU(1.0 - 0.15) = 0.85
- loss = 5.0 * 0.85 = 4.25

## ✅ SOLUTION IMMÉDIATE

Désactiver sparsity_loss temporairement:
```yaml
lambda_sparsity: 0.0
```

Relancer training ensuite.
