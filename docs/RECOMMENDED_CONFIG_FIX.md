# 🔧 Configuration Recommandée pour Éviter OOM

## Situation Actuelle
- GPU: RTX 3090 (24 GB)
- Mémoire libre: 23.3 GB
- Dernière tentative: OOM crash

## Recommandation: Réduire batch_size

### Changement Suggéré

```yaml
# config/config.wikipedia_2.yaml
train:
  batch_size: 4        # ⬅️ Changer de 8 à 4
  accum_steps: 8       # ⬅️ Changer de 4 à 8

  # Effective batch reste identique: 4 × 8 = 32
  # (au lieu de 8 × 4 = 32)

  # OU plus conservateur:
  batch_size: 2        # Ultra-safe
  accum_steps: 16      # Garde effective batch = 32
```

### Pourquoi?

Le crash suggère que `batch_size=8` avec `seq_len` grandissant dépasse 24 GB:
- Début (seq_len=384): ~12 GB ✅
- Milieu (seq_len=1024): ~20 GB ⚠️
- Fin (seq_len=2048): ~28 GB ❌ OOM!

Avec `batch_size=4`:
- Fin (seq_len=2048): ~14 GB ✅

---

## Commande de Lancement

```bash
# 1. Modifier config (batch_size: 4, accum_steps: 8)

# 2. Relancer
python scripts/train.py --resume --config config/config.wikipedia_2.yaml 2>&1 | tee training.log
```

---

## Alternative: Réduire seq_len_final

Si vous voulez garder batch_size=8:

```yaml
train:
  seq_len_final: 1024  # Au lieu de 2048
```

---

## Note

Le GPU a actuellement 23 GB libres, donc le training devrait démarrer.
Le crash précédent a bien nettoyé la mémoire.
