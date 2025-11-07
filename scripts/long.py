import torch
import yaml
from transformers import GPT2Tokenizer
import sys
import os
import argparse

# Ajouter le répertoire racine au path pour imports absolus
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.model import LLMTransformer, Config

# Parse arguments
parser = argparse.ArgumentParser(description="Test landmark selection with long sequences")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
parser.add_argument("--config", type=str, required=True, help="Path to config file")
args = parser.parse_args()

# Load model
with open(args.config) as f:
    config_dict = yaml.safe_load(f)
cfg = Config(**config_dict['model'])

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LLMTransformer(cfg).to(device)
checkpoint_path = os.path.join(args.checkpoint, 'model.pt')
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Prompt LONG (pour avoir assez de tokens)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
long_prompt = "The capital of France is Paris, which is located in the northern part of the country. It is known for its art, fashion, gastronomy, and culture. The Eiffel Tower is one of the most famous landmarks in Paris."

input_ids = tokenizer.encode(long_prompt, return_tensors="pt").to(device)
L = input_ids.size(1)

print(f"Sequence length: {L}")
print(f"Num landmarks to select: {cfg.global_k * 2}")

with torch.no_grad():
    logits, aux = model(input_ids, return_aux=True)

scores = aux['landmark_scores'][0]  # (L,)
indices = aux['landmark_indices'][0]  # (G,)

print(f"\nLandmark scores statistics:")
print(f"  Min: {scores.min().item():.6f}")
print(f"  Max: {scores.max().item():.6f}")  
print(f"  Mean: {scores.mean().item():.6f}")
print(f"  Std: {scores.std().item():.6f}")

# Vérifier si le selector discrimine
if scores.std() > 0.001:
    print("\n✅ Le selector DISCRIMINE entre les tokens!")
else:
    print("\n❌ Le selector ne discrimine PAS (scores uniformes)!")

# Top-5 tokens sélectionnés
print(f"\nTop-5 landmarks sélectionnés:")
for i in range(min(5, len(indices))):
    idx = indices[i].item()
    token = tokenizer.decode([input_ids[0, idx].item()])
    score = scores[idx].item()
    print(f"  {i+1}. Position {idx}: '{token}' (score: {score:.6f})")