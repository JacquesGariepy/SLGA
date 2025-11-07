"""
VALIDATION SNIPPET FOR scripts/train.py
Copy-paste these sections into your existing train.py

This file shows exactly what to add and where.
"""

# ============================================================================
# SECTION 1: IMPORTS (Add to top of train.py)
# ============================================================================
"""
Add these imports after your existing imports:
"""
from src.validation import (
    ConfigValidator,
    RuntimeValidator,
    print_validation_results,
    validate_training_step,
    ValidationResult
)


# ============================================================================
# SECTION 2: STARTUP VALIDATION (Add in main() before model init)
# ============================================================================
"""
Add this RIGHT AFTER loading your config, BEFORE initializing the model:
"""
def add_startup_validation(model_config, training_config):
    print("\n" + "=" * 70)
    print("VALIDATING CONFIGURATION")
    print("=" * 70)

    # Validate all configurations
    passed, results = ConfigValidator.validate_all(model_config, training_config)
    print_validation_results(results)

    if not passed:
        print("\n❌ Configuration validation failed. Fix errors before training.")
        print("See /mnt/d/ai/SLGA/docs/VALIDATION_INTEGRATION.md for details")
        sys.exit(1)

    print("\n✅ Configuration valid, proceeding with training...\n")


# ============================================================================
# SECTION 3: TRAINING STEP VALIDATION (Add in training loop)
# ============================================================================
"""
Add this in your training loop, AFTER forward pass, BEFORE backward:
"""
def add_step_validation_before_backward(model, loss, step, landmarks, seq_len):
    # Validate every 100 steps (adjust frequency as needed)
    if step % 100 == 0:
        print(f"\n[Step {step}] Running validation checkpoint...")

        # All-in-one validation
        is_valid = validate_training_step(
            model=model,
            loss=loss,
            step=step,
            landmarks=landmarks,
            seq_len=seq_len
        )

        if not is_valid:
            print(f"⚠️  Validation issues detected at step {step}")
            # Optionally: reduce learning rate or take other action
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] *= 0.9


"""
Add this AFTER backward pass, BEFORE optimizer.step():
"""
def add_gradient_validation_after_backward(model, step, optimizer):
    # Check gradients every 100 steps
    if step % 100 == 0:
        grad_result = RuntimeValidator.check_gradients(model)

        if grad_result.severity == "error":
            print(f"❌ Critical gradient error: {grad_result.message}")
            print("Skipping this optimization step")
            return "skip"  # Signal to skip optimizer.step()

        elif grad_result.severity == "warning":
            print(f"⚠️  {grad_result.message}")
            # Optionally reduce learning rate
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] *= 0.95

    return "continue"


# ============================================================================
# SECTION 4: EPOCH-END VALIDATION (Add after each epoch)
# ============================================================================
"""
Add this at the END of each epoch, before saving checkpoints:
"""
def add_epoch_end_validation(model, epoch, avg_loss, best_loss):
    print(f"\n{'=' * 70}")
    print(f"EPOCH {epoch} VALIDATION")
    print(f"{'=' * 70}")

    results = []

    # 1. Check loss validity
    loss_result = RuntimeValidator.check_loss(
        torch.tensor(avg_loss),
        step=epoch,
        max_loss=None  # Set a threshold if desired
    )
    results.append(loss_result)

    # 2. Check loss progression
    if epoch > 0 and avg_loss > best_loss * 1.1:
        results.append(ValidationResult(
            passed=False,
            message=f"Loss increasing: {avg_loss:.4f} > {best_loss * 1.1:.4f}",
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

    # Print validation results
    print_validation_results(results, verbose=True)

    # Return whether training should continue
    has_errors = any(r.severity == "error" and not r.passed for r in results)

    if has_errors:
        print("\n❌ Epoch validation failed with critical errors.")
        return False  # Signal to stop training

    return True  # Continue training


# ============================================================================
# SECTION 5: COMPLETE EXAMPLE INTEGRATION
# ============================================================================
"""
Here's a complete example showing where everything goes:
"""

def train_with_full_validation():
    """Example showing complete integration"""

    # Load configurations
    model_config = {...}
    training_config = {...}

    # ========== VALIDATION CHECKPOINT 1: STARTUP ==========
    print("\n" + "=" * 70)
    print("CONFIGURATION VALIDATION")
    print("=" * 70)

    passed, results = ConfigValidator.validate_all(model_config, training_config)
    print_validation_results(results)

    if not passed:
        print("\n❌ Configuration validation failed.")
        return

    print("\n✅ Configuration valid!\n")

    # Initialize model, optimizer, etc.
    model = SLGAModel(**model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config['lr'])
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')

    # Training loop
    for epoch in range(training_config['epochs']):
        print(f"\n{'=' * 70}")
        print(f"EPOCH {epoch + 1}")
        print(f"{'=' * 70}")

        model.train()
        total_loss = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            landmarks = batch.get('landmarks', None)

            # Forward pass
            outputs = model(input_ids, landmark_indices=landmarks)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            # ========== VALIDATION CHECKPOINT 2: STEP ==========
            if step % 100 == 0:
                seq_len = input_ids.size(1)
                validate_training_step(model, loss, step, landmarks, seq_len)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # ========== VALIDATION CHECKPOINT 3: GRADIENTS ==========
            if step % 100 == 0:
                grad_result = RuntimeValidator.check_gradients(model)
                if grad_result.severity == "error":
                    print(f"❌ {grad_result.message}")
                    continue  # Skip this step

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=training_config['max_grad_norm']
            )

            optimizer.step()
            total_loss += loss.item()

        # ========== VALIDATION CHECKPOINT 4: EPOCH END ==========
        avg_loss = total_loss / len(train_loader)

        should_continue = add_epoch_end_validation(
            model, epoch, avg_loss, best_loss
        )

        if not should_continue:
            print("\n❌ Stopping training due to validation errors.")
            break

        # Update best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print(f"\n✅ Saved checkpoint (loss: {best_loss:.4f})")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


# ============================================================================
# SECTION 6: MINIMAL INTEGRATION (If you want just the essentials)
# ============================================================================
"""
If you want the absolute minimum, add just these 3 checkpoints:
"""

def minimal_integration():
    """Minimal validation integration"""

    # 1. AT STARTUP
    passed, results = ConfigValidator.validate_all(model_config, training_config)
    if not passed:
        print_validation_results(results)
        sys.exit(1)

    # 2. IN TRAINING LOOP (every 100 steps)
    if step % 100 == 0:
        validate_training_step(model, loss, step, landmarks, seq_len)

    # 3. AFTER EACH EPOCH
    grad_result = RuntimeValidator.check_gradients(model)
    if grad_result.severity == "error":
        print(f"❌ {grad_result.message}")
        break  # Stop training


# ============================================================================
# SECTION 7: LANDMARK GENERATION VALIDATION
# ============================================================================
"""
If you generate landmarks dynamically, add this validation:
"""

def validate_landmark_generation(input_ids, landmark_indices, num_landmarks):
    """Validate generated landmarks"""
    seq_len = input_ids.size(1)

    result = RuntimeValidator.check_landmarks(
        landmark_indices,
        seq_len=seq_len,
        min_unique_ratio=0.9
    )

    if not result.passed:
        print(f"⚠️  Landmark validation: {result.message}")

        if result.severity == "error":
            # Fallback to heuristic landmarks
            print("  Using heuristic landmarks instead")
            B = input_ids.size(0)
            G = num_landmarks
            landmark_indices = torch.linspace(
                0, seq_len - 1, G, device=input_ids.device
            ).long().unsqueeze(0).expand(B, -1)

    return landmark_indices


# ============================================================================
# SECTION 8: OUTPUT VALIDATION (For inference/generation)
# ============================================================================
"""
When generating text or running inference, validate outputs:
"""

def validate_inference_outputs(outputs, vocab_size):
    """Validate model outputs during inference"""

    result = RuntimeValidator.check_model_outputs(
        outputs,
        vocab_size=vocab_size,
        check_logits=True
    )

    if not result.passed:
        print(f"⚠️  Output validation: {result.message}")

        if result.severity == "error":
            raise RuntimeError(f"Invalid model outputs: {result.message}")

    return outputs


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
"""
HOW TO INTEGRATE INTO YOUR EXISTING train.py:

1. Copy Section 1 imports to top of train.py

2. Add Section 2 (startup validation) in main() before model initialization

3. Add Section 3 (step validation) in your training loop:
   - Before backward: validate_training_step()
   - After backward: check_gradients()

4. Add Section 4 (epoch validation) after each epoch completes

5. Optional: Add Section 7 if you generate landmarks dynamically

6. Optional: Add Section 8 if you have inference/generation code

VALIDATION FREQUENCY:
- Startup: Once (before training starts)
- Step validation: Every 100 steps (adjustable)
- Gradient check: Every 100 steps (adjustable)
- Epoch validation: After every epoch

PERFORMANCE IMPACT:
- Negligible (~1-2% overhead with default frequency)
- Adjust validation frequency if needed (e.g., every 200 steps)

LOGGING:
- All validation results print to console
- Optionally add file logging (see VALIDATION_INTEGRATION.md)

ERROR HANDLING:
- Errors: Stop training immediately
- Warnings: Log and continue (optionally adjust hyperparameters)
- Info: Log for reference

For complete documentation, see:
/mnt/d/ai/SLGA/docs/VALIDATION_INTEGRATION.md
"""
