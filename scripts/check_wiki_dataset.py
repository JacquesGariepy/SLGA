# check_actual_training_data.py (CORRIGÉ)
import yaml
from datasets import load_dataset
from collections import Counter
import re

# Charger config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

print("="*80)
print("ANALYZING ACTUAL TRAINING DATASET")
print("="*80)

# Vérifier source du dataset
data_cfg = cfg.get("data", {})
print(f"\nConfig data section:")
for key, value in data_cfg.items():
    print(f"  {key}: {value}")

# Extraire info dataset
dataset_name = data_cfg.get('dataset') or data_cfg.get('dataset_name')
dataset_config = data_cfg.get('dataset_config') or data_cfg.get('config_name')
text_column = data_cfg.get('text_column', 'text')

if not dataset_name:
    print("\n❌ No dataset configured!")
    exit(1)

print(f"\nLoading dataset: {dataset_name}")
if dataset_config:
    print(f"Config: {dataset_config}")

# Charger dataset
try:
    if dataset_config:
        ds = load_dataset(
            dataset_name,
            dataset_config,
            split='train',
            streaming=True,
            trust_remote_code=True
        )
    else:
        ds = load_dataset(
            dataset_name,
            split='train',
            streaming=True,
            trust_remote_code=True
        )
    
    print("✓ Dataset loaded successfully")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("\nTrying alternative loading method...")
    try:
        # Pour wikimedia/wikipedia, format spécial
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",  # Date snapshot
            split='train',
            streaming=True,
            trust_remote_code=True
        )
        print("✓ Dataset loaded with alternative method")
    except Exception as e2:
        print(f"❌ Also failed: {e2}")
        exit(1)

# Analyser échantillons
print("\n" + "="*80)
print("ANALYZING SAMPLES")
print("="*80)

char_issues = Counter()
pattern_issues = Counter()
samples_analyzed = 0
samples_with_issues = 0
sample_lengths = []

try:
    for i, example in enumerate(ds):
        if samples_analyzed >= 100:
            break
        
        text = example.get(text_column, example.get('text', ''))
        if not text:
            print(f"⚠️  Sample {i}: No text found")
            continue
            
        samples_analyzed += 1
        sample_lengths.append(len(text))
        
        has_issue = False
        
        # Check 1: Caractères non-ASCII suspects
        non_ascii_chars = []
        for c in text:
            if ord(c) > 127:
                # Exclure caractères Unicode légitimes (accents européens)
                if ord(c) > 591:  # Au-delà des accents latins
                    char_issues[c] += 1
                    non_ascii_chars.append(c)
        
        if non_ascii_chars and len(non_ascii_chars) > 5:
            has_issue = True
        
        # Check 2: Patterns Wiki suspects
        if '----' in text or '====' in text:
            pattern_issues['wiki_formatting'] += 1
            has_issue = True
        
        if '}}' in text or '{{' in text:
            pattern_issues['wiki_templates'] += 1
            has_issue = True
        
        if '|' in text and text.count('|') > 5:
            pattern_issues['wiki_tables'] += 1
            has_issue = True
        
        # Check 3: Encodage corrompu
        if 'Ã' in text and ('Â' in text or '©' in text):
            pattern_issues['encoding_errors'] += 1
            has_issue = True
        
        # Check 4: Métadonnées
        if re.search(r'\b(Category|Template|File|Image):', text):
            pattern_issues['wiki_metadata'] += 1
            has_issue = True
        
        if re.search(r'\[\[.*?\]\]', text):
            pattern_issues['wiki_links'] += 1
        
        if has_issue:
            samples_with_issues += 1
        
        # Afficher premiers exemples problématiques
        if samples_with_issues <= 3 and has_issue:
            print(f"\n--- Sample {i} (HAS ISSUES) ---")
            print(f"Length: {len(text)} chars")
            print("First 400 chars:")
            print(repr(text[:400]))

except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user")
except Exception as e:
    print(f"\n❌ Error during analysis: {e}")

# Résumé
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Samples analyzed: {samples_analyzed}")
print(f"Samples with issues: {samples_with_issues} ({samples_with_issues/samples_analyzed*100:.1f}%)")

if sample_lengths:
    avg_len = sum(sample_lengths) / len(sample_lengths)
    print(f"\nAverage sample length: {avg_len:.0f} characters")
    print(f"Min length: {min(sample_lengths)}")
    print(f"Max length: {max(sample_lengths)}")

if char_issues:
    print(f"\nTop 10 problematic non-ASCII characters:")
    for char, count in char_issues.most_common(10):
        print(f"  '{char}' (U+{ord(char):04X}): {count} occurrences")

if pattern_issues:
    print(f"\nPattern issues found:")
    total_samples = samples_analyzed
    for pattern, count in sorted(pattern_issues.items(), key=lambda x: -x[1]):
        pct = count / total_samples * 100
        print(f"  {pattern}: {count} samples ({pct:.1f}%)")

# Verdict
print(f"\n{'='*80}")
print("VERDICT")
print(f"{'='*80}")

issue_rate = samples_with_issues / samples_analyzed if samples_analyzed > 0 else 0

if issue_rate > 0.5:
    print("🚨 CRITICAL: Dataset is heavily polluted (>50% samples with issues)")
    print("   → Training quality severely compromised")
    print("   → RESTART with clean dataset HIGHLY recommended")
elif issue_rate > 0.3:
    print("⚠️  WARNING: Dataset has significant issues (>30% samples)")
    print("   → This explains poor generation quality")
    print("   → Consider switching to cleaner dataset")
elif issue_rate > 0.1:
    print("🟡 CAUTION: Dataset has moderate issues (>10% samples)")
    print("   → May impact generation quality")
    print("   → Acceptable but not ideal")
else:
    print("✅ OK: Dataset appears relatively clean (<10% issues)")
    print("   → Problem may be elsewhere")
    print("   → Check perplexity and architecture")

print(f"\n💡 Recommendation:")
if issue_rate > 0.3:
    print("   Use 'Skylion007/openwebtext' or 'bookcorpus' instead")
    print("   These are pre-cleaned and production-ready")