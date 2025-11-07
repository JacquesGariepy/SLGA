# scripts/create_mixed_dataset.py (VERSION FINALE ROBUSTE)
"""
Crée un dataset mixé équilibré pour combattre le biais "water/food"

Mix (avec fallbacks automatiques):
  - 50% FineWeb-Edu (science, éducation)
  - 30% WikiText-103 (connaissances générales) ✅ MARCHE
  - 20% FineWeb standard (diversité) ← Alternative à OpenWebText
"""

from datasets import load_dataset, interleave_datasets, DatasetDict
import os
import re

print("="*80)
print("CRÉATION DATASET MIXÉ ÉQUILIBRÉ (VERSION ROBUSTE)")
print("="*80)

# ============================================================================
# 1. CHARGER LES DATASETS SOURCES
# ============================================================================
print("\n📥 Chargement des datasets sources...")

# FineWeb-Edu (50%)
print("  [1/3] FineWeb-Edu (science/éducation)...")
try:
    ds_fineweb_edu = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train[:500000]",
        streaming=False
    )
    print(f"    ✓ Loaded: {len(ds_fineweb_edu):,} documents")
except Exception as e:
    print(f"    ❌ CRITICAL ERROR: {e}")
    raise

# WikiText-103 (30%)
print("  [2/3] WikiText-103 (connaissances générales)...")
try:
    ds_wiki = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="train",
        streaming=False
    )
    print(f"    ✓ Loaded: {len(ds_wiki):,} documents")
except Exception as e:
    print(f"    ❌ ERROR: {e}")
    print("    Skipping WikiText...")
    ds_wiki = None

# FineWeb Standard (20%) - Alternative à OpenWebText
print("  [3/3] FineWeb Standard (diversité)...")
try:
    # Essayer d'abord la version CC-MAIN-2024-10 (plus récente)
    ds_diverse = load_dataset(
        "HuggingFaceFW/fineweb",
        "CC-MAIN-2024-10",
        split="train[:200000]",
        streaming=False
    )
    print(f"    ✓ Loaded: {len(ds_diverse):,} documents (FineWeb standard)")
except Exception as e1:
    print(f"    ⚠️  FineWeb standard failed: {e1}")
    
    # Fallback: Utiliser une autre partie de FineWeb-Edu
    try:
        print("    Fallback: Using different FineWeb-Edu slice...")
        ds_diverse = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "sample-10BT",
            split="train[500000:700000]",  # Différente tranche
            streaming=False
        )
        print(f"    ✓ Loaded: {len(ds_diverse):,} documents (FineWeb-Edu different slice)")
    except Exception as e2:
        print(f"    ❌ ERROR: {e2}")
        ds_diverse = None

# Vérifier qu'on a au moins 2 datasets
datasets_loaded = sum([
    ds_fineweb_edu is not None,
    ds_wiki is not None,
    ds_diverse is not None
])

if datasets_loaded < 2:
    print("\n❌ ERROR: Besoin d'au moins 2 datasets!")
    print("Seul FineWeb-Edu a été chargé. Utilise FineWeb-Edu uniquement.")
    raise RuntimeError("Insufficient datasets loaded")

print(f"\n✓ {datasets_loaded}/3 datasets chargés avec succès")

# ============================================================================
# 2. NORMALISER LES COLONNES
# ============================================================================
print("\n🔄 Normalisation des colonnes...")

def normalize_text(example):
    """Extrait colonne 'text' peu importe le dataset"""
    if "text" in example:
        text = example["text"]
    elif "content" in example:
        text = example["content"]
    else:
        # Fallback: première valeur string
        text = next((v for v in example.values() if isinstance(v, str)), "")
    
    # Filtrer documents vides ou trop courts
    if len(text.strip()) < 100:
        return None
    
    return {"text": text}

print("  Normalizing FineWeb-Edu...")
ds_fineweb_edu = ds_fineweb_edu.map(normalize_text, num_proc=4)
ds_fineweb_edu = ds_fineweb_edu.filter(lambda x: x is not None, num_proc=4)
ds_fineweb_edu = ds_fineweb_edu.remove_columns(
    [col for col in ds_fineweb_edu.column_names if col != "text"]
)

if ds_wiki is not None:
    print("  Normalizing WikiText...")
    ds_wiki = ds_wiki.map(normalize_text, num_proc=4)
    ds_wiki = ds_wiki.filter(lambda x: x is not None, num_proc=4)
    ds_wiki = ds_wiki.remove_columns(
        [col for col in ds_wiki.column_names if col != "text"]
    )

if ds_diverse is not None:
    print("  Normalizing Diverse dataset...")
    ds_diverse = ds_diverse.map(normalize_text, num_proc=4)
    ds_diverse = ds_diverse.filter(lambda x: x is not None, num_proc=4)
    ds_diverse = ds_diverse.remove_columns(
        [col for col in ds_diverse.column_names if col != "text"]
    )

print("  ✓ Colonnes normalisées")

# ============================================================================
# 3. NETTOYAGE DES NEWLINES EXCESSIVES
# ============================================================================
print("\n🧹 Nettoyage des newlines excessives...")

def clean_text(example):
    """Nettoie les newlines qui causent mode collapse"""
    text = example["text"]
    
    # Remplacer 3+ newlines consécutives par 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remplacer tabs par espaces
    text = text.replace('\t', '    ')
    
    # Normaliser espaces multiples
    text = re.sub(r' {3,}', '  ', text)
    
    # Filtrer si trop court après nettoyage
    if len(text.strip()) < 50:
        return None
    
    return {"text": text}

print("  Cleaning FineWeb-Edu...")
ds_fineweb_edu = ds_fineweb_edu.map(clean_text, num_proc=4)
ds_fineweb_edu = ds_fineweb_edu.filter(lambda x: x is not None, num_proc=4)

if ds_wiki is not None:
    print("  Cleaning WikiText...")
    ds_wiki = ds_wiki.map(clean_text, num_proc=4)
    ds_wiki = ds_wiki.filter(lambda x: x is not None, num_proc=4)

if ds_diverse is not None:
    print("  Cleaning Diverse dataset...")
    ds_diverse = ds_diverse.map(clean_text, num_proc=4)
    ds_diverse = ds_diverse.filter(lambda x: x is not None, num_proc=4)

print("  ✓ Nettoyage terminé")

# ============================================================================
# 4. CRÉER TRAIN/VAL SPLITS
# ============================================================================
print("\n✂️  Création des splits train/val (98%/2%)...")

ds_fineweb_split = ds_fineweb_edu.train_test_split(test_size=0.02, seed=42)
print(f"  FineWeb-Edu: {len(ds_fineweb_split['train']):,} train, {len(ds_fineweb_split['test']):,} val")

if ds_wiki is not None:
    ds_wiki_split = ds_wiki.train_test_split(test_size=0.02, seed=42)
    print(f"  WikiText:    {len(ds_wiki_split['train']):,} train, {len(ds_wiki_split['test']):,} val")

if ds_diverse is not None:
    ds_diverse_split = ds_diverse.train_test_split(test_size=0.02, seed=42)
    print(f"  Diverse:     {len(ds_diverse_split['train']):,} train, {len(ds_diverse_split['test']):,} val")

# ============================================================================
# 5. INTERLEAVE (MÉLANGE) LES DATASETS
# ============================================================================
print("\n🔀 Mélange des datasets...")

# Déterminer la composition selon les datasets disponibles
datasets_train = [ds_fineweb_split["train"]]
datasets_val = [ds_fineweb_split["test"]]
dataset_names = ["FineWeb-Edu"]

if ds_wiki is not None and ds_diverse is not None:
    # Cas idéal: 3 datasets
    datasets_train.extend([ds_wiki_split["train"], ds_diverse_split["train"]])
    datasets_val.extend([ds_wiki_split["test"], ds_diverse_split["test"]])
    dataset_names.extend(["WikiText", "Diverse"])
    probabilities = [0.5, 0.3, 0.2]
    composition_str = "50% FineWeb-Edu, 30% WikiText, 20% Diverse"
    
elif ds_wiki is not None:
    # 2 datasets: FineWeb + Wiki
    datasets_train.append(ds_wiki_split["train"])
    datasets_val.append(ds_wiki_split["test"])
    dataset_names.append("WikiText")
    probabilities = [0.6, 0.4]
    composition_str = "60% FineWeb-Edu, 40% WikiText"
    
elif ds_diverse is not None:
    # 2 datasets: FineWeb + Diverse
    datasets_train.append(ds_diverse_split["train"])
    datasets_val.append(ds_diverse_split["test"])
    dataset_names.append("Diverse")
    probabilities = [0.7, 0.3]
    composition_str = "70% FineWeb-Edu, 30% Diverse"
    
else:
    # Fallback: FineWeb seulement (pas de mélange)
    print("  ⚠️  WARNING: Seulement FineWeb-Edu disponible!")
    probabilities = [1.0]
    composition_str = "100% FineWeb-Edu"

print(f"  Composition: {composition_str}")

# Interleave
ds_train_mixed = interleave_datasets(
    datasets_train,
    probabilities=probabilities,
    seed=42,
    stopping_strategy="all_exhausted"
)

ds_val_mixed = interleave_datasets(
    datasets_val,
    probabilities=probabilities,
    seed=42,
    stopping_strategy="all_exhausted"
)

print(f"  ✓ Train: {len(ds_train_mixed):,} documents")
print(f"  ✓ Val:   {len(ds_val_mixed):,} documents")

# ============================================================================
# 6. CRÉER DatasetDict ET SAUVEGARDER
# ============================================================================
print("\n💾 Sauvegarde du dataset mixé...")

dataset_mixed = DatasetDict({
    "train": ds_train_mixed,
    "test": ds_val_mixed
})

output_path = "data/mixed_balanced"
os.makedirs("data", exist_ok=True)

dataset_mixed.save_to_disk(output_path)

print(f"  ✓ Dataset sauvegardé dans: {output_path}")

# ============================================================================
# 7. STATISTIQUES FINALES
# ============================================================================
print("\n" + "="*80)
print("📊 STATISTIQUES DU DATASET MIXÉ")
print("="*80)

# Compter mots-clés pour vérifier équilibre
keywords = ["water", "food", "paris", "france", "science", "history", "math", "wikipedia"]
keyword_counts = {kw: 0 for kw in keywords}

print("\n🔍 Analyse de 10,000 documents...")
sample_size = min(10000, len(ds_train_mixed))

for i in range(sample_size):
    text_lower = ds_train_mixed[i]["text"].lower()
    for kw in keywords:
        keyword_counts[kw] += text_lower.count(kw)

print(f"\nFréquence des mots-clés (sur {sample_size:,} docs):")
for kw, count in sorted(keyword_counts.items(), key=lambda x: -x[1]):
    freq_per_doc = count / sample_size
    print(f"  '{kw:12s}': {count:6d} mentions ({freq_per_doc:.3f}/doc)")

# Comparaison avec FineWeb-Edu seul
print("\n📊 COMPARAISON:")
print("  ┌─────────────────────────────────────┐")
print("  │ FineWeb-Edu seul (original)         │")
print("  │   water: 1.03/doc (mode collapse)   │")
print("  ├─────────────────────────────────────┤")
print(f"  │ Mixed balanced (nouveau)            │")
print(f"  │   water: {keyword_counts['water']/sample_size:.2f}/doc", end="")

water_reduction = (1.03 - keyword_counts['water']/sample_size) / 1.03 * 100
if water_reduction > 0:
    print(f" (-{water_reduction:.1f}%) ✅  │")
else:
    print(f" (+{-water_reduction:.1f}%) ⚠️  │")

print("  └─────────────────────────────────────┘")

# ============================================================================
# 8. RÉSUMÉ FINAL
# ============================================================================
print("\n" + "="*80)
print("✅ DATASET MIXÉ CRÉÉ AVEC SUCCÈS!")
print("="*80)

print(f"\n📁 Location: {output_path}")
print(f"📊 Train: {len(ds_train_mixed):,} documents")
print(f"📊 Val:   {len(ds_val_mixed):,} documents")

print(f"\n🎨 Composition:")
for i, name in enumerate(dataset_names):
    prob_pct = probabilities[i] * 100
    print(f"  • {prob_pct:.0f}% {name}")

print(f"\n🔧 Utilisation dans config:")
print(f"  data:")
print(f"    dataset: \"{output_path}\"")
print(f"    subset: null")
print(f"    split_train: \"train\"")
print(f"    split_val: \"test\"")

print(f"\n💡 Bénéfices attendus:")
print(f"  • Réduction mode collapse 'water': -{water_reduction:.0f}%")
print(f"  • Diversité augmentée: {len(dataset_names)} sources")
print(f"  • Meilleure généralisation")

print("="*80)