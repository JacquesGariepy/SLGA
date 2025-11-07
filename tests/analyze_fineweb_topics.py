# tests/analyze_fineweb_topics.py
from datasets import load_dataset
from collections import Counter
import re

print("="*80)
print("ANALYSE DES TOPICS - FINEWEB-EDU")
print("="*80)

ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train")

# Compter occurrences de mots clés
keywords = {
    'water': 0,
    'food': 0,
    'waste': 0,
    'energy': 0,
    'environment': 0,
    'science': 0,
    'math': 0,
    'history': 0,
    'france': 0,
    'paris': 0,
}

n_samples = 10000
total_words = 0

for i in range(n_samples):
    text = ds[i]['text'].lower()
    words = re.findall(r'\b\w+\b', text)
    total_words += len(words)
    
    for keyword in keywords:
        keywords[keyword] += text.count(keyword)

print(f"\nAnalysé {n_samples:,} documents ({total_words:,} mots)\n")

print("📊 FRÉQUENCE DES TOPICS:")
sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
for keyword, count in sorted_keywords:
    freq = count / n_samples
    ppm = (count / total_words) * 1_000_000  # Parts per million
    print(f"  '{keyword:15s}': {count:6d} occurrences ({freq:5.2f}/doc, {ppm:6.1f} ppm)")

print("\n" + "="*80)
print("COMPARAISON AVEC PRÉDICTIONS DU MODÈLE")
print("="*80)

model_bias = {
    'water': 40.0,
    'food': 7.0,
    'waste': 3.0,
}

print("\nCorrection entre dataset et modèle:")
for word in model_bias:
    dataset_freq = keywords[word] / n_samples
    model_freq = model_bias[word]
    print(f"  '{word}':")
    print(f"    Dataset: {dataset_freq:.2f} mentions/doc")
    print(f"    Model generation: {model_freq:.1f}%")
    if dataset_freq > 2.0:
        print(f"    ✓ Corrélation forte - topic surreprésenté dans dataset")
    else:
        print(f"    ✗ Modèle sur-biaité au-delà du dataset")

# Analyse des catégories de documents
print("\n" + "="*80)
print("CATÉGORIES DE DOCUMENTS")
print("="*80)

categories = {
    'science/environment': ['water', 'climate', 'environment', 'ecosystem'],
    'food/health': ['food', 'nutrition', 'health', 'diet'],
    'math': ['equation', 'calculate', 'mathematics', 'algebra'],
    'history': ['history', 'century', 'ancient', 'historical'],
}

for category, terms in categories.items():
    count = sum(keywords.get(term, 0) for term in terms if term in keywords)
    print(f"  {category:25s}: {count:6d} mentions")