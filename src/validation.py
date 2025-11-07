"""
Validation utilities for SLGA model
Prevents runtime errors with comprehensive checks

Examples:
    >>> # Validate model configuration before training
    >>> from validation import ConfigValidator, print_validation_results
    >>>
    >>> config = {
    ...     'embed_dim': 512,
    ...     'num_heads': 8,
    ...     'local_window': 256,
    ...     'global_k': 64,
    ...     'max_seq_len': 2048,
    ...     'dropout_rate': 0.1
    ... }
    >>>
    >>> results = ConfigValidator.validate_slga_config(config)
    >>> print_validation_results(results)
    >>>
    >>> # Check gradients during training
    >>> from validation import RuntimeValidator
    >>>
    >>> grad_result = RuntimeValidator.check_gradients(model)
    >>> if not grad_result.passed and grad_result.severity == "error":
    ...     print(f"Critical: {grad_result.message}")
    ...     # Take corrective action
    >>>
    >>> # Validate loss values
    >>> loss_result = RuntimeValidator.check_loss(loss, step=100)
    >>> if not loss_result.passed:
    ...     print(f"Loss issue at step 100: {loss_result.message}")
    >>>
    >>> # Check landmark validity
    >>> landmark_result = RuntimeValidator.check_landmarks(
    ...     landmark_indices, seq_len=512
    ... )
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Résultat d'une validation"""
    passed: bool
    message: str
    severity: str  # "info", "warning", "error"


class ConfigValidator:
    """Validateur de configuration modèle"""

    @staticmethod
    def validate_slga_config(config: Dict[str, Any]) -> List[ValidationResult]:
        """Valide configuration SLGA

        Args:
            config: Dictionary containing SLGA configuration parameters
                Required keys: embed_dim, num_heads, local_window, max_seq_len, global_k

        Returns:
            List of ValidationResult objects indicating any issues found

        Example:
            >>> config = {
            ...     'embed_dim': 512,
            ...     'num_heads': 8,
            ...     'local_window': 256,
            ...     'max_seq_len': 2048,
            ...     'global_k': 64
            ... }
            >>> results = ConfigValidator.validate_slga_config(config)
            >>> assert all(r.passed or r.severity != "error" for r in results)
        """
        results = []

        # Check required keys
        required_keys = ['embed_dim', 'num_heads', 'local_window', 'max_seq_len', 'global_k']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            results.append(ValidationResult(
                passed=False,
                message=f"Missing required config keys: {missing_keys}",
                severity="error"
            ))
            return results  # Can't validate further without these keys

        # Check embed_dim divisible par num_heads
        if config['embed_dim'] % config['num_heads'] != 0:
            results.append(ValidationResult(
                passed=False,
                message=f"embed_dim={config['embed_dim']} must be divisible by num_heads={config['num_heads']}",
                severity="error"
            ))

        # Check local_window raisonnable
        if config['local_window'] > config['max_seq_len'] // 2:
            results.append(ValidationResult(
                passed=False,
                message=f"local_window={config['local_window']} too large for max_seq_len={config['max_seq_len']}",
                severity="warning"
            ))

        # Check global_k efficace
        if config['global_k'] > config['local_window']:
            results.append(ValidationResult(
                passed=False,
                message=f"global_k={config['global_k']} should be < local_window={config['local_window']}",
                severity="warning"
            ))

        # Check global_k not too small
        if config['global_k'] < 16:
            results.append(ValidationResult(
                passed=False,
                message=f"global_k={config['global_k']} might be too small (recommend >= 16)",
                severity="warning"
            ))

        # Check dropouts valides
        for key in ['dropout_rate', 'attn_drop', 'proj_drop']:
            if key in config:
                if not (0.0 <= config[key] < 1.0):
                    results.append(ValidationResult(
                        passed=False,
                        message=f"{key}={config[key]} must be in [0, 1)",
                        severity="error"
                    ))

        # Check positive integers
        for key in ['embed_dim', 'num_heads', 'local_window', 'max_seq_len', 'global_k']:
            if config[key] <= 0:
                results.append(ValidationResult(
                    passed=False,
                    message=f"{key}={config[key]} must be > 0",
                    severity="error"
                ))

        # If no issues found, add success message
        if not results:
            results.append(ValidationResult(
                passed=True,
                message="All SLGA config parameters valid",
                severity="info"
            ))

        return results

    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> List[ValidationResult]:
        """Valide configuration d'entraînement

        Args:
            config: Dictionary containing training configuration
                Expected keys: batch_size, lr, seq_len_start, seq_len_final, etc.

        Returns:
            List of ValidationResult objects

        Example:
            >>> train_config = {
            ...     'batch_size': 32,
            ...     'lr': 1e-4,
            ...     'seq_len_start': 128,
            ...     'seq_len_final': 512
            ... }
            >>> results = ConfigValidator.validate_training_config(train_config)
        """
        results = []

        # Check batch_size raisonnable
        if 'batch_size' in config:
            if config['batch_size'] < 1:
                results.append(ValidationResult(
                    passed=False,
                    message=f"batch_size must be >= 1, got {config['batch_size']}",
                    severity="error"
                ))
            elif config['batch_size'] > 512:
                results.append(ValidationResult(
                    passed=False,
                    message=f"batch_size={config['batch_size']} very large, may cause OOM",
                    severity="warning"
                ))

        # Check learning rate range
        if 'lr' in config:
            if not (1e-6 <= config['lr'] <= 1e-2):
                results.append(ValidationResult(
                    passed=False,
                    message=f"lr={config['lr']} outside typical range [1e-6, 1e-2]",
                    severity="warning"
                ))

        # Check curriculum seq_len progression
        if 'seq_len_start' in config and 'seq_len_final' in config:
            if config['seq_len_start'] >= config['seq_len_final']:
                results.append(ValidationResult(
                    passed=False,
                    message="seq_len_start must be < seq_len_final",
                    severity="error"
                ))

            # Check reasonable progression ratio
            ratio = config['seq_len_final'] / config['seq_len_start']
            if ratio > 8:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Curriculum ratio {ratio:.1f}x might be too aggressive",
                    severity="warning"
                ))

        # Check epochs
        if 'epochs' in config:
            if config['epochs'] < 1:
                results.append(ValidationResult(
                    passed=False,
                    message=f"epochs must be >= 1, got {config['epochs']}",
                    severity="error"
                ))

        # Check gradient clipping
        if 'max_grad_norm' in config:
            if config['max_grad_norm'] <= 0:
                results.append(ValidationResult(
                    passed=False,
                    message=f"max_grad_norm must be > 0, got {config['max_grad_norm']}",
                    severity="error"
                ))

        if not results:
            results.append(ValidationResult(
                passed=True,
                message="All training config parameters valid",
                severity="info"
            ))

        return results

    @staticmethod
    def validate_all(model_config: Dict[str, Any],
                     training_config: Dict[str, Any]) -> Tuple[bool, List[ValidationResult]]:
        """Valide toutes les configurations

        Args:
            model_config: SLGA model configuration
            training_config: Training configuration

        Returns:
            Tuple of (all_passed: bool, all_results: List[ValidationResult])

        Example:
            >>> passed, results = ConfigValidator.validate_all(model_cfg, train_cfg)
            >>> if not passed:
            ...     print_validation_results(results)
            ...     raise ValueError("Configuration validation failed")
        """
        all_results = []
        all_results.extend(ConfigValidator.validate_slga_config(model_config))
        all_results.extend(ConfigValidator.validate_training_config(training_config))

        # Check for any errors
        has_errors = any(r.severity == "error" and not r.passed for r in all_results)

        return not has_errors, all_results


class RuntimeValidator:
    """Validateur pendant training"""

    @staticmethod
    def check_gradients(model: nn.Module,
                       warn_threshold: float = 1e-7,
                       error_threshold: float = 100.0) -> ValidationResult:
        """Vérifie gradient flow

        Args:
            model: PyTorch model to check
            warn_threshold: Minimum gradient norm to warn about vanishing gradients
            error_threshold: Maximum gradient norm to warn about exploding gradients

        Returns:
            ValidationResult indicating gradient health

        Example:
            >>> result = RuntimeValidator.check_gradients(model)
            >>> if result.severity == "error":
            ...     # Scale down learning rate or clip gradients
            ...     optimizer.param_groups[0]['lr'] *= 0.5
        """
        grad_norms = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_norms.append((name, grad_norm))

        if not grad_norms:
            return ValidationResult(
                passed=False,
                message="No gradients found (possible detach issue)",
                severity="error"
            )

        # Check for vanishing gradients
        min_grad = min(norm for _, norm in grad_norms)
        min_name = [name for name, norm in grad_norms if norm == min_grad][0]

        if min_grad < warn_threshold:
            return ValidationResult(
                passed=False,
                message=f"Vanishing gradients detected (min={min_grad:.2e} at {min_name})",
                severity="warning"
            )

        # Check for exploding gradients
        max_grad = max(norm for _, norm in grad_norms)
        max_name = [name for name, norm in grad_norms if norm == max_grad][0]

        if max_grad > error_threshold:
            return ValidationResult(
                passed=False,
                message=f"Exploding gradients detected (max={max_grad:.2e} at {max_name})",
                severity="warning"
            )

        return ValidationResult(
            passed=True,
            message=f"Gradients healthy (range: {min_grad:.2e} - {max_grad:.2e})",
            severity="info"
        )

    @staticmethod
    def check_loss(loss: torch.Tensor, step: int,
                   max_loss: Optional[float] = None) -> ValidationResult:
        """Vérifie validité de la loss

        Args:
            loss: Loss tensor to check
            step: Current training step
            max_loss: Optional maximum acceptable loss value

        Returns:
            ValidationResult indicating loss validity

        Example:
            >>> result = RuntimeValidator.check_loss(loss, step=100, max_loss=10.0)
            >>> if not result.passed:
            ...     print(f"Step {step}: {result.message}")
            ...     if result.severity == "error":
            ...         break  # Stop training
        """
        if torch.isnan(loss):
            return ValidationResult(
                passed=False,
                message=f"NaN loss at step {step}",
                severity="error"
            )

        if torch.isinf(loss):
            return ValidationResult(
                passed=False,
                message=f"Inf loss at step {step}",
                severity="error"
            )

        loss_val = loss.item()

        if loss_val < 0:
            return ValidationResult(
                passed=False,
                message=f"Negative loss at step {step}: {loss_val:.4f}",
                severity="error"
            )

        if max_loss is not None and loss_val > max_loss:
            return ValidationResult(
                passed=False,
                message=f"Loss {loss_val:.4f} exceeds maximum {max_loss:.4f} at step {step}",
                severity="warning"
            )

        return ValidationResult(
            passed=True,
            message=f"Loss valid: {loss_val:.4f}",
            severity="info"
        )

    @staticmethod
    def check_landmarks(landmark_indices: Optional[torch.Tensor],
                       seq_len: int,
                       min_unique_ratio: float = 0.9) -> ValidationResult:
        """Vérifie validité des landmarks

        Args:
            landmark_indices: Tensor of shape (B, G) with landmark positions
            seq_len: Sequence length
            min_unique_ratio: Minimum ratio of unique landmarks expected

        Returns:
            ValidationResult indicating landmark validity

        Example:
            >>> result = RuntimeValidator.check_landmarks(landmarks, seq_len=512)
            >>> if result.severity == "warning":
            ...     print(f"Landmark issue: {result.message}")
        """
        if landmark_indices is None:
            return ValidationResult(
                passed=True,
                message="No landmarks (using heuristic)",
                severity="info"
            )

        # Check range
        if (landmark_indices < 0).any() or (landmark_indices >= seq_len).any():
            return ValidationResult(
                passed=False,
                message=f"Landmark indices out of range [0, {seq_len})",
                severity="error"
            )

        # Check uniqueness
        B, G = landmark_indices.shape
        unique_per_batch = [len(torch.unique(landmark_indices[b])) for b in range(B)]
        avg_unique = sum(unique_per_batch) / B

        if avg_unique < G * min_unique_ratio:
            return ValidationResult(
                passed=False,
                message=f"Many duplicate landmarks (avg {avg_unique:.1f}/{G})",
                severity="warning"
            )

        return ValidationResult(
            passed=True,
            message=f"Landmarks valid ({avg_unique:.1f}/{G} unique avg)",
            severity="info"
        )

    @staticmethod
    def check_model_outputs(outputs: torch.Tensor,
                          vocab_size: int,
                          check_logits: bool = True) -> ValidationResult:
        """Vérifie les sorties du modèle

        Args:
            outputs: Model output tensor (logits or probabilities)
            vocab_size: Expected vocabulary size
            check_logits: Whether outputs are logits (vs probabilities)

        Returns:
            ValidationResult indicating output validity
        """
        if torch.isnan(outputs).any():
            return ValidationResult(
                passed=False,
                message="NaN values in model outputs",
                severity="error"
            )

        if torch.isinf(outputs).any():
            return ValidationResult(
                passed=False,
                message="Inf values in model outputs",
                severity="error"
            )

        # Check last dimension matches vocab_size
        if outputs.shape[-1] != vocab_size:
            return ValidationResult(
                passed=False,
                message=f"Output dim {outputs.shape[-1]} != vocab_size {vocab_size}",
                severity="error"
            )

        # If probabilities, check they sum to 1
        if not check_logits:
            probs_sum = outputs.sum(dim=-1)
            if not torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-3):
                return ValidationResult(
                    passed=False,
                    message="Probabilities don't sum to 1",
                    severity="warning"
                )

        return ValidationResult(
            passed=True,
            message=f"Model outputs valid (shape: {outputs.shape})",
            severity="info"
        )


def print_validation_results(results: List[ValidationResult],
                            verbose: bool = True) -> None:
    """Affiche résultats de validation avec couleurs

    Args:
        results: List of ValidationResult objects to display
        verbose: If False, only show errors and warnings

    Example:
        >>> results = ConfigValidator.validate_slga_config(config)
        >>> print_validation_results(results)
        ✅ All validations passed!

        >>> # Or with issues:
        >>> print_validation_results(results)
        ❌ ERRORS:
          ❌ embed_dim=513 must be divisible by num_heads=8
        ⚠️  WARNINGS:
          ⚠️  global_k=512 should be < local_window=256
    """
    icons = {
        "error": "❌",
        "warning": "⚠️ ",
        "info": "ℹ️ "
    }

    errors = [r for r in results if r.severity == "error"]
    warnings = [r for r in results if r.severity == "warning"]
    infos = [r for r in results if r.severity == "info"]

    if errors:
        print("\n❌ ERRORS:")
        for r in errors:
            print(f"  {icons[r.severity]} {r.message}")

    if warnings:
        print("\n⚠️  WARNINGS:")
        for r in warnings:
            print(f"  {icons[r.severity]} {r.message}")

    if verbose and infos:
        print("\nℹ️  INFO:")
        for r in infos:
            print(f"  {icons[r.severity]} {r.message}")

    if not errors and not warnings:
        print("\n✅ All validations passed!")


def validate_training_step(model: nn.Module,
                          loss: torch.Tensor,
                          step: int,
                          landmarks: Optional[torch.Tensor] = None,
                          seq_len: Optional[int] = None) -> bool:
    """Validation complète d'un step d'entraînement

    Args:
        model: Model being trained
        loss: Current loss value
        step: Training step number
        landmarks: Optional landmark indices
        seq_len: Optional sequence length for landmark validation

    Returns:
        True if all validations pass (no errors), False otherwise

    Example:
        >>> # In training loop
        >>> loss = criterion(outputs, targets)
        >>>
        >>> if not validate_training_step(model, loss, step, landmarks, seq_len):
        ...     print("Validation failed, stopping training")
        ...     break
        >>>
        >>> loss.backward()
        >>> optimizer.step()
    """
    results = []

    # Check loss
    results.append(RuntimeValidator.check_loss(loss, step))

    # Check gradients if they exist
    has_grads = any(p.grad is not None for p in model.parameters())
    if has_grads:
        results.append(RuntimeValidator.check_gradients(model))

    # Check landmarks if provided
    if landmarks is not None and seq_len is not None:
        results.append(RuntimeValidator.check_landmarks(landmarks, seq_len))

    # Print any issues
    errors = [r for r in results if r.severity == "error" and not r.passed]
    warnings = [r for r in results if r.severity == "warning" and not r.passed]

    if errors or warnings:
        print(f"\n⚠️  Validation issues at step {step}:")
        print_validation_results(results, verbose=False)

    # Return False only if there are errors
    return len(errors) == 0
