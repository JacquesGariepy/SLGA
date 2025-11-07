# tests/debug_generation_fixed.py
import torch
import yaml
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.model import Config, LLMTransformer
from src.data import get_tokenizer

print("="*80)
print("DIAGNOSTIC GÉNÉRATION - Step 4000 (WITH NEWLINE SUPPRESSION)")
print("="*80)

# Load config
with open('config/config_fineweb_edu.yaml') as f:
    cfg = yaml.safe_load(f)

# Load tokenizer
tokenizer = get_tokenizer(cfg['tokenizer'])
print(f"\n✓ Tokenizer loaded")
print(f"  Vocab size: {len(tokenizer)}")
print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

# Get newline token ID
newline_id = tokenizer.encode('\n')[0]
print(f"  Newline token ID: {newline_id}")

# Load model
model_cfg = Config(**cfg['model'])
model = LLMTransformer(model_cfg).cuda()
state_dict = torch.load('out_slga_fineweb/ckpt_4000/model.pt')
model.load_state_dict(state_dict)
model.eval()

print(f"\n✓ Model loaded: {model.get_num_params()/1e6:.2f}M params")

# Test prompt
prompt = "The capital of France is "
input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

print(f"\n{'='*80}")
print(f"PROMPT: '{prompt}'")
print(f"Input IDs: {input_ids.tolist()}")
print(f"{'='*80}\n")

# Generation settings
max_tokens = 30
temperature = 0.8  # ← Plus raisonnable
top_k = 40
suppress_newlines = True
newline_penalty = 20.0  # ← Pénalité forte

output_tokens = []

for step in range(max_tokens):
    print(f"--- Step {step+1}/{max_tokens} ---")
    
    with torch.no_grad():
        logits = model(input_ids)
        last_logits = logits[0, -1, :].clone()
    
    # Stats AVANT suppression
    nl_logit_before = last_logits[newline_id].item()
    
    # 🔧 CRITICAL: Supprimer newlines
    if suppress_newlines:
        last_logits[newline_id] -= newline_penalty
    
    nl_logit_after = last_logits[newline_id].item()
    
    print(f"Newline logit: {nl_logit_before:.2f} → {nl_logit_after:.2f} (Δ={nl_logit_before-nl_logit_after:.2f})")
    
    # Temperature
    last_logits = last_logits / temperature
    
    # Top-K
    topk_vals, topk_idxs = torch.topk(last_logits, k=min(top_k, len(last_logits)))
    
    print(f"Top-5 after filtering:")
    for i in range(min(5, len(topk_vals))):
        tid = topk_idxs[i].item()
        token = tokenizer.decode([tid])
        # Affichage lisible
        if tid == newline_id:
            token_display = "'\\n'"
        elif token.strip() == '':
            token_display = f"'{token}' (space)"
        else:
            token_display = f"'{token}'"
        print(f"  {i+1}. {token_display:20s} (ID: {tid:5d}, logit: {topk_vals[i].item():7.2f})")
    
    # Filter
    logits_filtered = torch.full_like(last_logits, float('-inf'))
    logits_filtered.scatter_(0, topk_idxs, topk_vals)
    
    # Softmax
    probs = torch.softmax(logits_filtered, dim=-1)
    
    # Sample
    next_token_id = torch.multinomial(probs, num_samples=1).item()
    next_token_str = tokenizer.decode([next_token_id])
    next_prob = probs[next_token_id].item()
    
    # Display
    if next_token_id == newline_id:
        token_display = "'\\n'"
    else:
        token_display = f"'{next_token_str}'"
    
    print(f"✓ Selected: {token_display} (prob: {next_prob:.4f})\n")
    
    # Check for EOS
    if next_token_id == tokenizer.eos_token_id:
        print("⚠️  EOS generated")
        break
    
    # Append
    next_token_tensor = torch.tensor([[next_token_id]], device=input_ids.device)
    input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
    output_tokens.append(next_token_str)

print(f"\n{'='*80}")
print("FINAL OUTPUT:")
print(f"{'='*80}")
full_output = tokenizer.decode(input_ids[0].tolist())
print(full_output)
print(f"\n{'='*80}")
print("GENERATED ONLY:")
print(''.join(output_tokens))
print(f"{'='*80}")