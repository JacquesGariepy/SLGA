# Oneshot Python Scripts

This directory contains standalone Python scripts that are NOT part of the pytest framework. These scripts are meant to be run independently for validation, diagnostics, debugging, testing, and analysis purposes.

## Directory Structure

```
oneshot/python/
├── validation/     - Scripts to validate bug fixes and system correctness
├── diagnostic/     - Scripts to diagnose issues and analyze system behavior
├── debug/          - Scripts for debugging specific components
├── testing/        - Standalone test scripts (not pytest)
└── analysis/       - Scripts for data and model analysis
```

## How to Run Oneshot Scripts

Unlike pytest tests, these scripts are executed directly with Python:

```bash
# General pattern
python tests/oneshot/python/<category>/<script_name>.py

# Examples
python tests/oneshot/python/validation/validate_critical_fixes.py
python tests/oneshot/python/diagnostic/diagnose_landmarks.py
python tests/oneshot/python/debug/debug_generation.py
```

## Categories

### Validation Scripts (`validation/`)

Scripts that validate bug fixes and verify system correctness.

| Script | Purpose | Usage |
|--------|---------|-------|
| `validate_critical_fixes.py` | Validates 5 critical patches | `python tests/oneshot/python/validation/validate_critical_fixes.py` |
| `validate_sparsity_fix.py` | Validates sparsity computation fix | `python tests/oneshot/python/validation/validate_sparsity_fix.py` |
| `verify_all_fixes_2025-10-28.py` | Comprehensive fix verification | `python tests/oneshot/python/validation/verify_all_fixes_2025-10-28.py` |
| `verify_memory_leak_fix.py` | Memory leak detection | `python tests/oneshot/python/validation/verify_memory_leak_fix.py` |
| `verify_step_counting.py` | Training step counter validation | `python tests/oneshot/python/validation/verify_step_counting.py` |

### Diagnostic Scripts (`diagnostic/`)

Scripts for diagnosing issues and analyzing system behavior.

| Script | Purpose | Usage |
|--------|---------|-------|
| `diagnose.py` | General model diagnostics | `python tests/oneshot/python/diagnostic/diagnose.py --checkpoint out_slga/ckpt_2000` |
| `diagnose_landmarks.py` | Landmark behavior analysis | `python tests/oneshot/python/diagnostic/diagnose_landmarks.py` |
| `diagnose_logits.py` | Logit distribution analysis | `python tests/oneshot/python/diagnostic/diagnose_logits.py` |
| `diagnose_scorer_problem.py` | Scorer issue diagnosis | `python tests/oneshot/python/diagnostic/diagnose_scorer_problem.py` |
| `diagnose_step1000.py` | Specific step debugging | `python tests/oneshot/python/diagnostic/diagnose_step1000.py` |
| `diagnose_training_health.py` | Training health check | `python tests/oneshot/python/diagnostic/diagnose_training_health.py` |

### Debug Scripts (`debug/`)

Scripts for debugging specific components.

| Script | Purpose | Usage |
|--------|---------|-------|
| `debug_generation.py` | Generation debugging | `python tests/oneshot/python/debug/debug_generation.py` |
| `debug_labels.py` | Label debugging | `python tests/oneshot/python/debug/debug_labels.py` |
| `debug_landmarks.py` | Landmark debugging | `python tests/oneshot/python/debug/debug_landmarks.py` |

### Testing Scripts (`testing/`)

Standalone test scripts (not pytest-based).

**Note:** These scripts were located in `scripts/` and need to be moved here. They include:
- `test_inference_bugs.py` - Test 6 inference-specific bugs
- `test_fixes.py` - Test generation fixes
- `test_complete.py` - Complete system test
- `test_determinism.py` - Determinism validation
- And more...

### Analysis Scripts (`analysis/`)

Scripts for data and model analysis.

| Script | Purpose | Usage |
|--------|---------|-------|
| `analyze_fineweb_topics.py` | Dataset topic analysis | `python tests/oneshot/python/analysis/analyze_fineweb_topics.py` |
| `analyze_model_knowledge.py` | Model knowledge probe | `python tests/oneshot/python/analysis/analyze_model_knowledge.py` |

## When to Use Oneshot Scripts vs Pytest

### Use Oneshot Scripts When:
- ✅ Need to run a one-time validation or diagnostic
- ✅ Want standalone execution without test framework overhead
- ✅ Need custom output formatting or interactive behavior
- ✅ Analyzing specific issues or behaviors
- ✅ Running ad-hoc investigations

### Use Pytest Tests When:
- ✅ Writing repeatable unit/integration tests
- ✅ Need test fixtures and parameterization
- ✅ Want automated test discovery
- ✅ Need coverage reporting
- ✅ Integration with CI/CD pipelines

## Common Patterns

Most oneshot scripts follow these patterns:

### Validation Pattern
```python
#!/usr/bin/env python
"""Validate specific bug fix or feature."""

def validate_fix():
    """Run validation checks."""
    # Setup
    # Execute
    # Verify
    # Report results
    pass

if __name__ == "__main__":
    validate_fix()
```

### Diagnostic Pattern
```python
#!/usr/bin/env python
"""Diagnose system behavior."""

import argparse

def diagnose(checkpoint_path):
    """Analyze model behavior."""
    # Load checkpoint
    # Run diagnostic tests
    # Print analysis
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    diagnose(args.checkpoint)
```

## Tips

1. **Always check script requirements** - Some scripts may need specific checkpoints or data files
2. **Read script docstrings** - Many scripts have detailed usage instructions in their docstrings
3. **Use absolute paths** - When specifying checkpoint or data paths
4. **Check script exit codes** - Many validation scripts return non-zero on failure
5. **Redirect output** - Use `> output.txt` to save diagnostic output for analysis

## Integration with Pytest

While these scripts are standalone, some can be converted to pytest tests:

```python
# Original oneshot
if __name__ == "__main__":
    validate_fix()

# Can be made pytest-compatible
def test_fix_validation():
    """Pytest wrapper for oneshot validation."""
    result = validate_fix()
    assert result.success
```

## Contributing

When adding new oneshot scripts:

1. Place in appropriate category directory
2. Add shebang line: `#!/usr/bin/env python`
3. Include docstring with purpose and usage
4. Add entry to this README
5. Make executable: `chmod +x script.py` (Unix)
6. Consider adding `--help` argument support

---

**Last Updated:** 2025-11-12
**Maintainer:** SLGA2 Development Team
