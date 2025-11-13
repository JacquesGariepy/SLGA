# tests/analyze_model_knowledge.py
import torch
import yaml
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.legacy.model import Config, LLMTransformer
from src.legacy.data import get_tokenizer
from collections import Counter

print("="*80)
print("ANALYSE DES CONNAISSANCES DU MODÈLE - Step 4000")
print("="*80)

# Load
with open('config/config.wikipedia.yaml') as f:
    cfg = yaml.safe_load(f)

tokenizer = get_tokenizer(cfg['tokenizer'])
model_cfg = Config(**cfg['model'])
model = LLMTransformer(model_cfg).cuda()
state_dict = torch.load('out_slga/ckpt_2000/model.pt')
model.load_state_dict(state_dict)
model.eval()

print(f"✓ Model loaded: {model.get_num_params()/1e6:.2f}M params\n")

# Test prompts de différentes catégories
test_prompts = [
    # Faits simples
    ("The capital of France is", "Paris"),
    ("The sun is a", "star"),
    ("Water is made of", "hydrogen and oxygen"),
    
    # Mathématiques basiques
    ("2 + 2 =", "4"),
    ("One plus one equals", "two"),
    
    # Grammaire
    ("I am", "..."),
    ("She is", "..."),
    ("They are", "..."),
    
    # Continuation logique
    ("Once upon a time", "there was"),
    ("In conclusion,", "..."),
    ("First, second,", "third"),
]

print("="*80)
print("TEST DE COMPÉTENCES")
print("="*80)

for prompt, expected in test_prompts:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    
    with torch.no_grad():
        logits = model(input_ids)
        last_logits = logits[0, -1, :]
        
        # Top 5 prédictions
        probs = torch.softmax(last_logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=5)
        
        predictions = []
        for prob, token_id in zip(top_probs, top_ids):
            token = tokenizer.decode([token_id.item()])
            predictions.append((token, prob.item()))
        
        print(f"\nPrompt: '{prompt}'")
        print(f"Expected: '{expected}'")
        print(f"Top 5 predictions:")
        for i, (token, prob) in enumerate(predictions):
            marker = "✓" if expected.lower() in token.lower() else " "
            print(f"  {marker} {i+1}. '{token}' (prob: {prob:.4f})")

# Analyse des biais du modèle
print("\n" + "="*80)
print("ANALYSE DES BIAIS")
print("="*80)

# Générer 100 tokens et compter fréquences
prompt = "The"
input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

generated_tokens = []
for _ in range(100):
    with torch.no_grad():
        logits = model(input_ids)
        last_logits = logits[0, -1, :]
        
        # Temperature 0.8, suppress newlines
        last_logits[tokenizer.encode('\n')[0]] -= 20.0
        last_logits = last_logits / 0.8
        
        probs = torch.softmax(last_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        token_str = tokenizer.decode([next_token.item()])
        generated_tokens.append(token_str)
        
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

# Compter fréquences
token_freq = Counter(generated_tokens)
print("\nTop 10 tokens les plus générés (sur 100):")
for token, count in token_freq.most_common(10):
    print(f"  '{token}': {count} fois ({count}%)")

# Analyse de diversité
unique_tokens = len(set(generated_tokens))
print(f"\nDiversité lexicale: {unique_tokens}/100 tokens uniques ({unique_tokens}%)")

# Check si mode collapse
top_token_ratio = token_freq.most_common(1)[0][1] / 100
if top_token_ratio > 0.3:
    print(f"⚠️  MODE COLLAPSE: Token '{token_freq.most_common(1)[0][0]}' représente {top_token_ratio*100:.1f}%!")
elif top_token_ratio > 0.15:
    print(f"⚠️  Biais fort: Token '{token_freq.most_common(1)[0][0]}' représente {top_token_ratio*100:.1f}%")
else:
    print(f"✓ Distribution saine")

print("\n" + "="*80)