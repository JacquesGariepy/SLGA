import torch
import yaml
import sys
import os
import argparse

# Ajouter le répertoire racine au path pour imports absolus
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import LLMTransformer, Config
from src.landmarks import landmark_spacing_loss
from transformers import GPT2Tokenizer

with open('config/config.wikipedia.yaml') as f:
    cfg_dict = yaml.safe_load(f)
cfg = Config(**cfg_dict['model'])

model = LLMTransformer(cfg).cuda()
ckpt = torch.load('out_slga/ckpt_9000/model.pt')
model.load_state_dict(ckpt)
model.train()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "The capital of France is Paris. " * 40  # Longue séquence
input_ids = tokenizer.encode(text, return_tensors='pt').cuda()

logits, aux = model(input_ids, return_aux=True)

print(f"Sequence length: {input_ids.size(1)}")
print(f"Num landmarks: {aux['landmark_indices'].size(1)}")
print(f"Ideal spacing: {input_ids.size(1) / aux['landmark_indices'].size(1):.1f}")

# Calculer spacing actuel
sorted_idx = torch.sort(aux['landmark_indices'], dim=-1)[0]
gaps = sorted_idx[:, 1:] - sorted_idx[:, :-1]
current_spacing = gaps.float().mean().item()
print(f"Current spacing: {current_spacing:.1f}\n")

# Test différents lambdas
import torch.nn.functional as F
labels = input_ids.clone()
loss_lm = F.cross_entropy(
    logits[:, :-1, :].reshape(-1, logits.size(-1)),
    labels[:, 1:].reshape(-1)
)
print(f"Loss LM: {loss_lm.item():.4f}\n")

for lambda_val in [0.1, 1.0, 10.0, 50.0, 100.0]:
    loss = landmark_spacing_loss(
        aux['landmark_indices'], 
        input_ids.size(1), 
        lambda_reg=lambda_val,
        selection_scores=aux['landmark_scores']
    )
    ratio = (loss.item() / loss_lm.item()) * 100
    print(f"Lambda {lambda_val:6.1f}: Loss = {loss.item():.6f} ({ratio:5.2f}% of LM loss)")
