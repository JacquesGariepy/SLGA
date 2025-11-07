# 🚨 Quick Fix: Out of Memory

## Problème
```
CUDA out of memory. GPU 0 has 24 GB total, 0 bytes free.
PyTorch allocated: 35.45 GB (impossible sur 24 GB GPU!)
```

## Solutions Immédiates

### Solution 1: Réduire batch_size (RAPIDE)
```yaml
# config/config.wikipedia_2.yaml
train:
  batch_size: 4  # Au lieu de 8
  accum_steps: 8  # Au lieu de 4 (garde même effective batch)
```

### Solution 2: Réduire sequence length au resume
```yaml
train:
  seq_len_start: 256  # Au lieu de 384
  seq_len_final: 1024  # Au lieu de 2048
```

### Solution 3: Désactiver grad_checkpointing s'il est activé
```yaml
train:
  grad_checkpointing: false
```

### Solution 4: Vider cache avant resume
```bash
python -c "import torch; torch.cuda.empty_cache()"
python scripts/train.py --resume --config config/config.wikipedia_2.yaml
```

---

## Diagnostic

Le message "35.45 GB allocated" sur GPU 24 GB suggère :
1. **Fuite mémoire** du run précédent
2. **Checkpoint corrompu** avec mauvaise config
3. **Process zombie** tenant la mémoire

### Vérification
```bash
# Voir processes CUDA
nvidia-smi

# Tuer process zombie si présent
pkill -9 python

# Nettoyer et relancer
python scripts/train.py --resume --config config/config.wikipedia_2.yaml
```

---

## Fix Permanent (Bug #32)

Ajouter nettoyage agressif au resume:

```python
# scripts/train.py après ligne 600 (après load checkpoint)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # Force garbage collection
    import gc
    gc.collect()
    torch.cuda.empty_cache()
```

---

## Recommandation Immédiate

**Essayer dans cet ordre**:

1. Tuer processes: `pkill -9 python`
2. Réduire batch_size à 4 dans config
3. Relancer training
