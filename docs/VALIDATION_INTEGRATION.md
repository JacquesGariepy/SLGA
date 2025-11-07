# Validation Integration Guide

## Overview

Le module `src/validation.py` fournit des utilitaires complets pour valider la configuration et surveiller l'entraînement du modèle SLGA. Ce guide montre comment l'intégrer dans votre workflow.

## 1. Validation au Startup (scripts/train.py)

### Intégration de base

```python
from src.validation import (
    ConfigValidator,
    RuntimeValidator,
    print_validation_results,
    validate_training_step
)

def main():
    # Load configurations
    model_config = {
        'embed_dim': 512,
        'num_heads': 8,
        'local_window': 256,
        'global_k': 64,
        'max_seq_len': 2048,
        'dropout_rate': 0.1,
        'attn_drop': 0.1,
        'proj_drop': 0.1
    }

    training_config = {
        'batch_size': 32,
        'lr': 1e-4,
        'epochs': 10,
        'seq_len_start': 128,
        'seq_len_final': 512,
        'max_grad_norm': 1.0
    }

    # ✅ VALIDATION STARTUP
    print("=" * 60)
    print("VALIDATING CONFIGURATION")
    print("=" * 60)

    passed, results = ConfigValidator.validate_all(model_config, training_config)
    print_validation_results(results)

    if not passed:
        print("\n❌ Configuration validation failed. Fix errors before training.")
        return

    print("\n✅ Configuration valid, proceeding with training...\n")

    # Initialize model with validated config
    model = SLGAModel(**model_config)

    # ... rest of training code
```

### Validation avancée avec logging

```python
import logging
from src.validation import ConfigValidator, print_validation_results

def setup_training_with_validation(model_config, training_config, log_file="validation.log"):
    """Setup avec validation et logging détaillé"""

    # Setup logger
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    # Validate configurations
    logger.info("Validating model configuration...")
    model_results = ConfigValidator.validate_slga_config(model_config)

    logger.info("Validating training configuration...")
    train_results = ConfigValidator.validate_training_config(training_config)

    all_results = model_results + train_results

    # Log all results
    for result in all_results:
        level = {
            "error": logging.ERROR,
            "warning": logging.WARNING,
            "info": logging.INFO
        }[result.severity]

        logger.log(level, f"[{result.severity.upper()}] {result.message}")

    # Print to console
    print_validation_results(all_results)

    # Check for errors
    has_errors = any(r.severity == "error" and not r.passed for r in all_results)

    if has_errors:
        logger.error("Configuration validation failed")
        raise ValueError("Cannot proceed with invalid configuration")

    logger.info("Configuration validation passed")
    return True
```

## 2. Validation Pendant l'Entraînement

### Hook après chaque epoch

```python
def train_epoch(model, dataloader, optimizer, criterion, epoch, device):
    """Training epoch avec validation runtime"""

    model.train()
    total_loss = 0

    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        landmarks = batch.get('landmarks', None)

        # Forward pass
        outputs = model(input_ids, landmark_indices=landmarks)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

        # ✅ VALIDATION RUNTIME
        if step % 100 == 0:
            # Validate training step (before backward)
            seq_len = input_ids.size(1)
            if not validate_training_step(model, loss, step, landmarks, seq_len):
                print(f"⚠️  Validation warning at epoch {epoch}, step {step}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # ✅ CHECK GRADIENTS après backward
        if step % 100 == 0:
            grad_result = RuntimeValidator.check_gradients(model)
            if grad_result.severity == "error":
                print(f"❌ Gradient error at step {step}: {grad_result.message}")
                # Option: skip this step ou ajuster learning rate
                continue
            elif grad_result.severity == "warning":
                print(f"⚠️  {grad_result.message}")

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

### Validation complète après chaque epoch

```python
def validate_epoch_end(model, epoch, avg_loss, best_loss):
    """Validation complète en fin d'epoch"""

    print(f"\n{'=' * 60}")
    print(f"EPOCH {epoch} VALIDATION")
    print(f"{'=' * 60}")

    results = []

    # 1. Check loss progression
    loss_result = RuntimeValidator.check_loss(
        torch.tensor(avg_loss),
        step=epoch
    )
    results.append(loss_result)

    # 2. Check if loss is improving
    if avg_loss > best_loss * 1.1:
        results.append(ValidationResult(
            passed=False,
            message=f"Loss increasing: {avg_loss:.4f} > {best_loss:.4f}",
            severity="warning"
        ))

    # 3. Check gradient health
    grad_result = RuntimeValidator.check_gradients(model)
    results.append(grad_result)

    # 4. Check for NaN parameters
    nan_params = sum(1 for p in model.parameters() if torch.isnan(p).any())
    if nan_params > 0:
        results.append(ValidationResult(
            passed=False,
            message=f"Found {nan_params} parameters with NaN values",
            severity="error"
        ))
    else:
        results.append(ValidationResult(
            passed=True,
            message="All parameters valid (no NaN)",
            severity="info"
        ))

    # Print results
    print_validation_results(results, verbose=True)

    # Return whether training should continue
    has_errors = any(r.severity == "error" and not r.passed for r in results)
    return not has_errors
```

## 3. Validation des Landmarks

### Validation lors de la génération de landmarks

```python
from src.validation import RuntimeValidator

def generate_landmarks_with_validation(model, input_ids, num_landmarks):
    """Generate landmarks avec validation"""

    # Generate landmarks (your existing logic)
    landmark_indices = model.generate_landmarks(input_ids, num_landmarks)

    # ✅ VALIDATE landmarks
    seq_len = input_ids.size(1)
    landmark_result = RuntimeValidator.check_landmarks(
        landmark_indices,
        seq_len=seq_len,
        min_unique_ratio=0.9
    )

    if not landmark_result.passed:
        print(f"⚠️  Landmark validation: {landmark_result.message}")

        if landmark_result.severity == "error":
            # Fallback to heuristic landmarks
            print("Using heuristic landmarks instead")
            B = input_ids.size(0)
            G = num_landmarks
            landmark_indices = torch.linspace(
                0, seq_len - 1, G, device=input_ids.device
            ).long().unsqueeze(0).expand(B, -1)

    return landmark_indices
```

## 4. Validation des Outputs du Modèle

### Après forward pass

```python
from src.validation import RuntimeValidator

def forward_with_validation(model, input_ids, vocab_size):
    """Forward pass avec validation des outputs"""

    # Forward pass
    outputs = model(input_ids)

    # ✅ VALIDATE outputs
    output_result = RuntimeValidator.check_model_outputs(
        outputs,
        vocab_size=vocab_size,
        check_logits=True
    )

    if not output_result.passed:
        print(f"⚠️  Output validation: {output_result.message}")

        if output_result.severity == "error":
            raise RuntimeError(f"Invalid model outputs: {output_result.message}")

    return outputs
```

## 5. Script d'Entraînement Complet avec Validation

Voici un exemple complet d'intégration dans `scripts/train.py`:

```python
#!/usr/bin/env python3
"""
Training script avec validation complète
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from src.model import SLGAModel
from src.validation import (
    ConfigValidator,
    RuntimeValidator,
    print_validation_results,
    validate_training_step
)


def main():
    # ==================== CONFIGURATION ====================
    model_config = {
        'vocab_size': 50257,
        'embed_dim': 512,
        'num_heads': 8,
        'num_layers': 6,
        'local_window': 256,
        'global_k': 64,
        'max_seq_len': 2048,
        'dropout_rate': 0.1,
        'attn_drop': 0.1,
        'proj_drop': 0.1
    }

    training_config = {
        'batch_size': 32,
        'lr': 1e-4,
        'epochs': 10,
        'seq_len_start': 128,
        'seq_len_final': 512,
        'max_grad_norm': 1.0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # ==================== VALIDATION STARTUP ====================
    print("\n" + "=" * 70)
    print("CONFIGURATION VALIDATION")
    print("=" * 70)

    passed, results = ConfigValidator.validate_all(model_config, training_config)
    print_validation_results(results)

    if not passed:
        print("\n❌ Configuration validation failed. Exiting.")
        return

    print("\n✅ Configuration valid!\n")

    # ==================== SETUP ====================
    device = training_config['device']
    model = SLGAModel(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config['lr'])
    criterion = nn.CrossEntropyLoss()

    # Load dataset (placeholder)
    # train_loader = DataLoader(...)

    # ==================== TRAINING LOOP ====================
    best_loss = float('inf')

    for epoch in range(training_config['epochs']):
        print(f"\n{'=' * 70}")
        print(f"EPOCH {epoch + 1}/{training_config['epochs']}")
        print(f"{'=' * 70}")

        model.train()
        total_loss = 0

        # for step, batch in enumerate(train_loader):
        #     ... (training code with validation as shown above)

        # ✅ VALIDATION FIN D'EPOCH
        avg_loss = total_loss / 100  # len(train_loader)

        should_continue = validate_epoch_end(model, epoch, avg_loss, best_loss)

        if not should_continue:
            print("\n❌ Validation failed, stopping training")
            break

        # Update best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save checkpoint
            torch.save(model.state_dict(), f"checkpoints/best_model.pt")
            print(f"✅ Saved best model (loss: {best_loss:.4f})")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

## 6. Checklist d'Intégration

### ✅ Au startup (avant training):
- [ ] Valider `model_config` avec `ConfigValidator.validate_slga_config()`
- [ ] Valider `training_config` avec `ConfigValidator.validate_training_config()`
- [ ] Afficher les résultats avec `print_validation_results()`
- [ ] Arrêter si erreurs critiques détectées

### ✅ Pendant training (tous les 100 steps):
- [ ] Valider la loss avec `RuntimeValidator.check_loss()`
- [ ] Vérifier les gradients avec `RuntimeValidator.check_gradients()`
- [ ] Valider les landmarks avec `RuntimeValidator.check_landmarks()`
- [ ] Logger les warnings dans fichier de log

### ✅ Après chaque epoch:
- [ ] Validation complète avec `validate_epoch_end()`
- [ ] Vérifier progression de la loss
- [ ] Vérifier santé des gradients
- [ ] Vérifier absence de NaN dans les paramètres
- [ ] Sauvegarder checkpoint si validation OK

### ✅ Avant génération/inference:
- [ ] Valider les outputs avec `RuntimeValidator.check_model_outputs()`
- [ ] Valider les landmarks générés
- [ ] Vérifier température et top-k/top-p valides

## 7. Gestion des Erreurs

### Stratégies de récupération

```python
def handle_validation_error(result, step, optimizer):
    """Handle validation errors gracefully"""

    if result.severity == "error":
        if "NaN" in result.message or "Inf" in result.message:
            print(f"❌ Critical error at step {step}: {result.message}")
            print("Attempting recovery:")
            print("  1. Reducing learning rate by 50%")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"  2. New learning rate: {optimizer.param_groups[0]['lr']:.2e}")
            return "continue"

        elif "gradient" in result.message.lower():
            print(f"⚠️  Gradient issue: {result.message}")
            print("Skipping this optimization step")
            return "skip"

    elif result.severity == "warning":
        print(f"⚠️  Warning: {result.message}")
        return "continue"

    return "ok"
```

## Conclusion

En intégrant ces validations, vous pouvez:
1. **Détecter les erreurs de configuration** avant de lancer l'entraînement
2. **Surveiller la santé du modèle** pendant l'entraînement
3. **Intervenir rapidement** en cas de problèmes (NaN, gradients explosifs, etc.)
4. **Logger toutes les anomalies** pour diagnostic post-mortem

Cette approche réduit considérablement les bugs silencieux et les entraînements qui échouent après plusieurs heures.
