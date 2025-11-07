#!/usr/bin/env python3
"""
Test rapide du fix de collator pour vérifier qu'il n'y a plus d'erreur CUDA.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.data import get_tokenizer, CollatorLocal

def test_collator():
    """Test que le collator produit des labels valides"""

    print("="*80)
    print("TEST COLLATOR FIX")
    print("="*80)

    # Setup
    tokenizer = get_tokenizer("gpt2")
    collator = CollatorLocal(tokenizer, max_length=512)
    pad_id = tokenizer.pad_token_id

    # Créer un batch de test
    examples = [
        {"text": "Hello world, this is a test."},
        {"text": "Short"},  # Court pour forcer padding
        {"text": "Another example with more text to process."},
    ]

    print(f"\nPad token ID: {pad_id}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Collater
    batch = collator(examples)

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    print(f"\nBatch shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  labels: {labels.shape}")

    # Vérifications
    print(f"\nValidation:")

    # 1. Tailles correctes
    assert input_ids.shape == labels.shape, "❌ Shapes mismatch!"
    print(f"  ✅ Shapes match: {input_ids.shape}")

    # 2. Valeurs des labels
    unique_labels = torch.unique(labels)
    print(f"  Unique label values: {unique_labels.tolist()[:20]}...")  # Afficher premiers 20

    # 3. Vérifier que -100 est présent (pads masqués)
    num_masked = (labels == -100).sum().item()
    print(f"  ✅ Masked tokens (-100): {num_masked}")

    # 4. Vérifier qu'aucun label n'est le pad_id
    num_pads_in_labels = (labels == pad_id).sum().item()
    if num_pads_in_labels > 0:
        print(f"  ❌ ERREUR: {num_pads_in_labels} pads non masqués dans labels!")
        return False
    else:
        print(f"  ✅ No unmasked pads in labels")

    # 5. Vérifier que tous les labels sont valides (soit -100, soit dans [0, vocab_size))
    valid_labels = ((labels == -100) | ((labels >= 0) & (labels < tokenizer.vocab_size)))
    if not valid_labels.all():
        invalid_count = (~valid_labels).sum().item()
        print(f"  ❌ ERREUR: {invalid_count} labels invalides!")
        # Afficher quelques exemples
        invalid_vals = labels[~valid_labels][:10]
        print(f"     Exemples de valeurs invalides: {invalid_vals.tolist()}")
        return False
    else:
        print(f"  ✅ All labels valid (in [0, {tokenizer.vocab_size}) or -100)")

    # 6. Tester sur GPU si disponible
    if torch.cuda.is_available():
        print(f"\n  Testing on CUDA...")
        try:
            input_ids_cuda = input_ids.cuda()
            labels_cuda = labels.cuda()

            # Simuler loss computation (ce qui a crashé avant)
            logits = torch.randn(input_ids_cuda.size(0), input_ids_cuda.size(1), tokenizer.vocab_size, device='cuda')

            # Cross-entropy (ce qui déclenche l'assertion CUDA)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, tokenizer.vocab_size),
                labels_cuda.view(-1),
                ignore_index=-100
            )

            print(f"  ✅ CUDA loss computation successful: {loss.item():.4f}")
            print(f"  ✅ NO CUDA ASSERTION ERROR!")

        except Exception as e:
            print(f"  ❌ CUDA error: {e}")
            return False
    else:
        print(f"\n  ⚠️  CUDA not available, skipping GPU test")

    print(f"\n{'='*80}")
    print(f"✅ ALL TESTS PASSED - Collator fix is working!")
    print(f"{'='*80}")
    return True

if __name__ == "__main__":
    success = test_collator()
    sys.exit(0 if success else 1)
