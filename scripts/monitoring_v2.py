# scripts/monitor_v2.py
"""Enhanced monitoring for V2 training"""
import torch
import sys
import os
import glob
sys.path.insert(0, '.')

from src.model import Config, LLMTransformer
from src.data import get_tokenizer
from collections import Counter
import yaml
import matplotlib.pyplot as plt
from datetime import datetime

def load_checkpoint_metrics(ckpt_path):
    """Load metrics from checkpoint"""
    try:
        state = torch.load(f"{ckpt_path}/trainer_state.pt", map_location='cpu')
        return {
            'step': state.get('step', 0),
            'loss': state.get('loss', 0),
            'best_loss': state.get('best_loss', 0),
        }
    except:
        return None

def test_collapse(ckpt_path, config_path):
    """Test mode collapse"""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    tokenizer = get_tokenizer(cfg['tokenizer'])
    model_cfg = Config(**cfg['model'])
    model = LLMTransformer(model_cfg).cuda()
    
    state_dict = torch.load(f"{ckpt_path}/model.pt", map_location='cuda')
    model.load_state_dict(state_dict)
    model.eval()
    
    # Generate
    prompt = "The"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    newline_id = tokenizer.encode('\n')[0]
    
    generated = []
    for _ in range(200):
        with torch.no_grad():
            logits = model(input_ids)
            last_logits = logits[0, -1, :].clone()
            
            last_logits[newline_id] -= 20.0
            last_logits = last_logits / 0.8
            
            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            token_str = tokenizer.decode([next_token.item()])
            generated.append(token_str)
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    token_freq = Counter(generated)
    top_token, top_count = token_freq.most_common(1)[0]
    
    return {
        'collapse_score': top_count / 200,
        'top_token': top_token,
        'diversity': len(set(generated)) / 200,
    }

def main():
    print("="*80)
    print("V2 TRAINING MONITORING")
    print("="*80)
    
    # Find all checkpoints
    v1_ckpts = sorted(glob.glob('out_slga_fineweb/ckpt_*'))
    v2_ckpts = sorted(glob.glob('out_slga_fineweb_v2/ckpt_*'))
    
    all_data = []
    
    # Process V1 checkpoints
    print("\n📊 V1 Checkpoints (Original Config):")
    for ckpt in v1_ckpts:
        step = int(ckpt.split('_')[-1])
        metrics = load_checkpoint_metrics(ckpt)
        
        if metrics:
            collapse = test_collapse(ckpt, 'config/config_fineweb_edu.yaml')
            
            data = {
                'version': 'V1',
                'step': step,
                'loss': metrics['loss'],
                'collapse_score': collapse['collapse_score'],
                'top_token': collapse['top_token'],
                'diversity': collapse['diversity'],
            }
            all_data.append(data)
            
            # Print
            status = "🔴" if collapse['collapse_score'] > 0.3 else "🟡" if collapse['collapse_score'] > 0.15 else "🟢"
            print(f"  Step {step:5d}: Loss {metrics['loss']:.4f} | "
                  f"{status} Collapse {collapse['collapse_score']*100:5.1f}% "
                  f"('{collapse['top_token']}') | "
                  f"Diversity {collapse['diversity']*100:4.1f}%")
    
    # Process V2 checkpoints
    if v2_ckpts:
        print("\n📊 V2 Checkpoints (Anti-Collapse Config):")
        for ckpt in v2_ckpts:
            step = int(ckpt.split('_')[-1])
            metrics = load_checkpoint_metrics(ckpt)
            
            if metrics:
                collapse = test_collapse(ckpt, 'config/config_fineweb_edu_v2.yaml')
                
                data = {
                    'version': 'V2',
                    'step': step,
                    'loss': metrics['loss'],
                    'collapse_score': collapse['collapse_score'],
                    'top_token': collapse['top_token'],
                    'diversity': collapse['diversity'],
                }
                all_data.append(data)
                
                # Print
                status = "🔴" if collapse['collapse_score'] > 0.3 else "🟡" if collapse['collapse_score'] > 0.15 else "🟢"
                print(f"  Step {step:5d}: Loss {metrics['loss']:.4f} | "
                      f"{status} Collapse {collapse['collapse_score']*100:5.1f}% "
                      f"('{collapse['top_token']}') | "
                      f"Diversity {collapse['diversity']*100:4.1f}%")
    
    # Plot comparison
    if len(all_data) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        v1_data = [d for d in all_data if d['version'] == 'V1']
        v2_data = [d for d in all_data if d['version'] == 'V2']
        
        # Plot 1: Loss
        if v1_data:
            axes[0, 0].plot([d['step'] for d in v1_data], 
                           [d['loss'] for d in v1_data], 
                           'o-', label='V1', color='red')
        if v2_data:
            axes[0, 0].plot([d['step'] for d in v2_data], 
                           [d['loss'] for d in v2_data], 
                           's-', label='V2', color='green')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Collapse Score
        if v1_data:
            axes[0, 1].plot([d['step'] for d in v1_data], 
                           [d['collapse_score']*100 for d in v1_data], 
                           'o-', label='V1', color='red')
        if v2_data:
            axes[0, 1].plot([d['step'] for d in v2_data], 
                           [d['collapse_score']*100 for d in v2_data], 
                           's-', label='V2', color='green')
        axes[0, 1].axhline(y=30, color='r', linestyle='--', label='Severe threshold')
        axes[0, 1].axhline(y=15, color='orange', linestyle='--', label='Moderate threshold')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Collapse Score (%)')
        axes[0, 1].set_title('Mode Collapse')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Diversity
        if v1_data:
            axes[1, 0].plot([d['step'] for d in v1_data], 
                           [d['diversity']*100 for d in v1_data], 
                           'o-', label='V1', color='red')
        if v2_data:
            axes[1, 0].plot([d['step'] for d in v2_data], 
                           [d['diversity']*100 for d in v2_data], 
                           's-', label='V2', color='green')
        axes[1, 0].axhline(y=30, color='orange', linestyle='--', label='Min healthy')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Token Diversity (%)')
        axes[1, 0].set_title('Generation Diversity')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 4: Summary
        axes[1, 1].axis('off')
        summary_text = "TRAINING COMPARISON\n\n"
        
        if v1_data:
            last_v1 = v1_data[-1]
            summary_text += f"V1 (Last @ Step {last_v1['step']}):\n"
            summary_text += f"  Loss: {last_v1['loss']:.4f}\n"
            summary_text += f"  Collapse: {last_v1['collapse_score']*100:.1f}% ('{last_v1['top_token']}')\n"
            summary_text += f"  Diversity: {last_v1['diversity']*100:.1f}%\n\n"
        
        if v2_data:
            last_v2 = v2_data[-1]
            summary_text += f"V2 (Last @ Step {last_v2['step']}):\n"
            summary_text += f"  Loss: {last_v2['loss']:.4f}\n"
            summary_text += f"  Collapse: {last_v2['collapse_score']*100:.1f}% ('{last_v2['top_token']}')\n"
            summary_text += f"  Diversity: {last_v2['diversity']*100:.1f}%\n\n"
            
            if v1_data:
                improvement = ((last_v1['collapse_score'] - last_v2['collapse_score']) / 
                              last_v1['collapse_score'] * 100)
                summary_text += f"Improvement:\n"
                summary_text += f"  Collapse: {improvement:+.1f}%\n"
        
        axes[1, 1].text(0.1, 0.5, summary_text, 
                       fontsize=12, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'training_comparison_{timestamp}.png'
        plt.savefig(filename, dpi=150)
        print(f"\n📊 Plot saved: {filename}")

if __name__ == "__main__":
    main()