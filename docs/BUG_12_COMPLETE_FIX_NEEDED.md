# Bug #12: Besoin d'un Fix Complet

## Le Problème Actuel

Mon fix partiel:
```python
# Part 1: Force logits à EOS
if stop_on_eos and finished.any():
    logits[finished] = float('-inf')
    logits[finished, eos_token_id] = 1e4

# Part 2: Force next_token à EOS
if stop_on_eos and finished.any():
    next_token[finished] = eos_token_id

# Résultat: Ajoute des EOS répétés après le premier
```

## Ce Que Le Test Attend

**Option 1**: Séquence s'arrête EXACTEMENT au premier EOS
```
Seq: "Hello world<EOS>"  ← Rien après
```

**Option 2 (acceptable)**: Séquences finies ajoutent seulement EOS
```
Seq: "Hello world<EOS><EOS><EOS>..."  ← Répétitions EOS OK
```

## Solution Recommandée

**Approche post-processing** (plus propre que bloquer pendant génération):

```python
# Après la boucle de génération, tronquer au premier EOS
if stop_on_eos:
    for b_idx in range(input_ids.size(0)):
        tokens = input_ids[b_idx]
        eos_positions = (tokens == eos_token_id).nonzero(as_tuple=True)[0]

        if len(eos_positions) > 0:
            # Tronquer juste après le premier EOS
            first_eos = eos_positions[0].item()
            input_ids[b_idx, first_eos+1:] = tokenizer.pad_token_id
```

Cette approche :
- ✅ Plus simple
- ✅ Pas d'impact sur la boucle de génération
- ✅ Nettoie après coup
- ✅ Passe le test

Mais mon approche actuelle (forcer EOS) est aussi valide fonctionnellement, juste pas ce que le test attend.

## Décision

Pour passer le test ET avoir un comportement propre, je devrais soit:
1. Post-process pour tronquer
2. OU modifier le test pour accepter EOS répétés

Je pense que l'approche post-process est meilleure.
