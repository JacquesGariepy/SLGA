#!/usr/bin/env python3
"""
Script de Diagnostic: Pourquoi les Landmarks ne sont pas utilisés?

Ce script va tracer exactement ce qui se passe avec les landmarks
pendant la génération et l'entraînement.
"""

import torch
import yaml
from transformers import GPT2Tokenizer
import sys
import os

# Ajouter le répertoire racine au path pour imports absolus
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.legacy.model import LLMTransformer, Config


def debug_landmark_selection(checkpoint_path, config_path):
    """
    Debug détaillé de la sélection des landmarks
    """
    print("\n" + "="*70)
    print("🔍 DIAGNOSTIC: Sélection des Landmarks")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load config
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    cfg = Config(**config_dict['model'])
    
    print(f"\nConfig:")
    print(f"  learned_landmarks: {cfg.learned_landmarks}")
    print(f"  global_k: {cfg.global_k}")
    print(f"  local_window: {cfg.local_window}")
    
    # Create model
    model = LLMTransformer(cfg).to(device)
    
    # Load checkpoint
    state_dict = torch.load(f"{checkpoint_path}/model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"\n✓ Modèle chargé: {model.get_num_params() / 1e6:.2f}M paramètres")
    
    # Test avec un prompt
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Input shape: {input_ids.shape}")
    
    # Forward avec return_aux pour voir les landmarks
    with torch.no_grad():
        logits, aux = model(input_ids, return_aux=True)
    
    print("\n" + "-"*70)
    print("RÉSULTATS DU FORWARD:")
    print("-"*70)
    
    if 'landmark_indices' in aux and aux['landmark_indices'] is not None:
        landmark_indices = aux['landmark_indices']
        print(f"\n✓ landmark_indices trouvé!")
        print(f"  Shape: {landmark_indices.shape}")
        print(f"  Device: {landmark_indices.device}")
        print(f"  Dtype: {landmark_indices.dtype}")
        print(f"  Valeurs:\n{landmark_indices}")
        
        # Vérifier si les indices sont valides
        L = input_ids.size(1)
        if torch.all(landmark_indices < L):
            print(f"\n  ✓ Tous les indices sont < L={L} (valides)")
        else:
            print(f"\n  ❌ Certains indices sont >= L={L} (INVALIDES!)")
            invalid_mask = landmark_indices >= L
            print(f"     Indices invalides: {landmark_indices[invalid_mask]}")
        
        # Vérifier si les indices sont uniques
        unique_indices = torch.unique(landmark_indices)
        print(f"\n  Indices uniques: {len(unique_indices.tolist())}/{landmark_indices.numel()}")
        if len(unique_indices) < landmark_indices.numel():
            print(f"  ⚠️  Il y a des DUPLICATES!")
        
        # Extraire les landmarks comme le fait le modèle
        B, L, D = input_ids.size(0), input_ids.size(1), cfg.embed_dim
        
        # Position embeddings
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        tok_emb = model.token_emb(input_ids)
        pos_emb = model.pos_emb(pos)
        x = model.emb_dropout(tok_emb + pos_emb)
        
        # Extraire les landmarks
        G = landmark_indices.size(1)
        landmark_indices_exp = landmark_indices.unsqueeze(-1).expand(B, G, D)
        
        print(f"\n  Extraction des landmarks:")
        print(f"    x.shape: {x.shape}")
        print(f"    landmark_indices_exp.shape: {landmark_indices_exp.shape}")
        
        try:
            landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)
            print(f"    landmark_states.shape: {landmark_states.shape}")
            print(f"\n  ✓ Extraction réussie!")
            
            # Vérifier si les landmarks sont des zéros
            if torch.all(landmark_states == 0):
                print(f"  ❌ Tous les landmarks sont des ZÉROS!")
            else:
                mean_val = landmark_states.mean().item()
                std_val = landmark_states.std().item()
                print(f"  ✓ Landmarks ont des valeurs non-nulles")
                print(f"    Mean: {mean_val:.6f}, Std: {std_val:.6f}")
        
        except Exception as e:
            print(f"  ❌ Erreur lors de l'extraction: {e}")
    
    else:
        print(f"\n❌ landmark_indices est None ou absent!")
        print(f"   Clés dans aux: {list(aux.keys())}")
    
    if 'landmark_scores' in aux and aux['landmark_scores'] is not None:
        landmark_scores = aux['landmark_scores']
        print(f"\n✓ landmark_scores trouvé!")
        print(f"  Shape: {landmark_scores.shape}")
        print(f"  Min: {landmark_scores.min().item():.6f}")
        print(f"  Max: {landmark_scores.max().item():.6f}")
        print(f"  Mean: {landmark_scores.mean().item():.6f}")
        
        # Top-5 positions avec les plus hauts scores
        top_scores, top_positions = torch.topk(landmark_scores[0], k=min(5, landmark_scores.size(1)))
        print(f"\n  Top-5 positions par score:")
        for i, (pos, score) in enumerate(zip(top_positions.tolist(), top_scores.tolist())):
            token = tokenizer.decode([input_ids[0, pos].item()])
            print(f"    {i+1}. Position {pos}: '{token}' (score: {score:.6f})")
    else:
        print(f"\n⚠️  landmark_scores est None ou absent!")
    
    # Test avec le LearnableLandmarkSelector directement
    if model.landmark_selector is not None:
        print("\n" + "-"*70)
        print("TEST DIRECT DU LearnableLandmarkSelector:")
        print("-"*70)
        
        # Refaire les embeddings
        pos = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
        tok_emb = model.token_emb(input_ids)
        pos_emb = model.pos_emb(pos)
        x = model.emb_dropout(tok_emb + pos_emb)
        
        with torch.no_grad():
            indices, masks, scores = model.landmark_selector(x)
        
        print(f"\nSortie du LandmarkSelector:")
        print(f"  indices.shape: {indices.shape}")
        print(f"  indices:\n{indices}")
        print(f"\n  masks.shape: {masks.shape}")
        print(f"  masks:\n{masks}")
        print(f"\n  scores.shape: {scores.shape}")
        print(f"  scores min/max/mean: {scores.min():.6f}/{scores.max():.6f}/{scores.mean():.6f}")
        
        # Vérifier ce qui est envoyé à l'attention
        print(f"\n  Indices envoyés à l'attention: {indices[0, :10].tolist()}...")
        print(f"  Ces indices correspondent aux tokens:")
        for i, idx in enumerate(indices[0, :10].tolist()):
            if idx < input_ids.size(1):
                token = tokenizer.decode([input_ids[0, idx].item()])
                print(f"    {i}. idx={idx}: '{token}'")
            else:
                print(f"    {i}. idx={idx}: INVALIDE (>= L)")
    
    else:
        print(f"\n⚠️  model.landmark_selector est None!")
        print(f"   learned_landmarks={cfg.learned_landmarks} mais selector pas initialisé?")
    
    print("\n" + "="*70)


def check_forward_propagation(checkpoint_path, config_path):
    """
    Vérifier si les landmarks sont bien propagés dans les blocs
    """
    print("\n" + "="*70)
    print("🔍 DIAGNOSTIC: Propagation dans les Blocs")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    cfg = Config(**config_dict['model'])
    
    model = LLMTransformer(cfg).to(device)
    state_dict = torch.load(f"{checkpoint_path}/model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Hook pour capturer les landmarks passés à l'attention
    landmarks_captured = []
    
    def capture_hook(module, input_tuple, output):
        # input_tuple[0] est x, on cherche cache_global dans kwargs
        # Mais les hooks ne capturent pas kwargs facilement...
        # On va plutôt patch la méthode forward du bloc
        pass
    
    # Patch temporaire pour voir ce qui est passé
    original_block_forward = model.blocks[0].forward
    
    def patched_forward(x, cache_global=None, global_weight=1.0):
        if cache_global is not None:
            print(f"\n  Bloc 0 reçoit cache_global:")
            print(f"    Shape: {cache_global.shape}")
            print(f"    Mean: {cache_global.mean().item():.6f}")
            print(f"    Std: {cache_global.std().item():.6f}")
        else:
            print(f"\n  ❌ Bloc 0 reçoit cache_global=None!")
        return original_block_forward(x, cache_global, global_weight)
    
    model.blocks[0].forward = patched_forward
    
    # Forward
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"\nForward avec prompt: '{prompt}'")
    
    with torch.no_grad():
        logits = model(input_ids)
    
    print("\n✓ Forward terminé")
    
    # Restaurer
    model.blocks[0].forward = original_block_forward
    
    print("\n" + "="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug landmarks")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/config.wikipedia.yaml")
    args = parser.parse_args()
    
    debug_landmark_selection(args.checkpoint, args.config)
    check_forward_propagation(args.checkpoint, args.config)
    
    print("\n" + "="*70)
    print("✓ Diagnostic terminé")
    print("="*70)