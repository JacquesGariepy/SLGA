# Oneshot Shell Scripts

This directory contains standalone shell scripts for orchestrating tests, managing operations, and running validation workflows.

## Directory Structure

```
oneshot/shell/
├── orchestration/  - Scripts that orchestrate multiple tests or operations
├── operations/     - Scripts for system operations (restart, cleanup, etc.)
└── validation/     - Shell-based validation workflows
```

## Categories

### Orchestration Scripts (`orchestration/`)

Scripts that run multiple tests or coordinate complex workflows.

**Expected scripts** (to be moved from original locations):
- `run_all_bug_tests.sh` - Run BUG #10, #11, #12 tests
- `run_all.sh` (from tests/integration/) - Run all integration tests
- `RUN_VERIFICATION.sh` (from scripts/) - Run verification suite

**Usage:**
```bash
bash tests/oneshot/shell/orchestration/run_all_bug_tests.sh
bash tests/oneshot/shell/orchestration/run_all.sh
bash tests/oneshot/shell/orchestration/RUN_VERIFICATION.sh
```

### Operations Scripts (`operations/`)

Scripts for system-level operations.

**Expected scripts** (to be moved from scripts/):
- `clean_restart.sh` - Clean environment restart
- `RESTART_TRAINING.sh` - Restart training pipeline
- `quick_fix_performance.sh` - Quick performance fix

**Usage:**
```bash
bash tests/oneshot/shell/operations/clean_restart.sh
bash tests/oneshot/shell/operations/RESTART_TRAINING.sh
bash tests/oneshot/shell/operations/quick_fix_performance.sh
```

### Validation Scripts (`validation/`)

Shell wrappers for validation workflows.

**Expected scripts** (to be moved from tests/):
- `validate_straight_through_fix.sh` - Validate straight-through fix
- `demo_param_validation.sh` - Demo parameter validation

**Usage:**
```bash
bash tests/oneshot/shell/validation/validate_straight_through_fix.sh
bash tests/oneshot/shell/validation/demo_param_validation.sh
```

## General Usage

Shell scripts in this directory should be executable:

```bash
# Make executable (if needed)
chmod +x tests/oneshot/shell/orchestration/run_all_bug_tests.sh

# Run directly
./tests/oneshot/shell/orchestration/run_all_bug_tests.sh

# Or with bash
bash tests/oneshot/shell/orchestration/run_all_bug_tests.sh
```

## Common Patterns

Most shell scripts follow these patterns:

### Orchestration Pattern
```bash
#!/bin/bash
# Run multiple tests in sequence

set -e  # Exit on error

echo "Running test suite..."

python tests/test_bug10.py
python tests/test_bug11.py
python tests/test_bug12.py

echo "All tests completed!"
```

### Operations Pattern
```bash
#!/bin/bash
# Perform system operation

set -e

echo "Cleaning environment..."
rm -rf out_slga/
rm -rf checkpoints/

echo "Restarting training..."
python scripts/train.py --config config/slga.yaml
```

### Validation Pattern
```bash
#!/bin/bash
# Validate fix with shell wrapper

set -e

echo "Validating straight-through fix..."
python tests/oneshot/python/validation/validate_sparsity_fix.py

if [ $? -eq 0 ]; then
    echo "✓ Validation passed"
else
    echo "✗ Validation failed"
    exit 1
fi
```

## Tips

1. **Always use `set -e`** - Exit on first error
2. **Add error messages** - Use `echo` to provide context
3. **Check dependencies** - Verify required files/dirs exist before operations
4. **Use absolute paths** - Or `cd` to project root first
5. **Document required environment** - Note any required env vars or setup

## Windows Compatibility

For Windows users, these scripts can be run via:

### Git Bash (Recommended)
```bash
bash tests/oneshot/shell/orchestration/run_all_bug_tests.sh
```

### WSL (Windows Subsystem for Linux)
```bash
wsl bash tests/oneshot/shell/orchestration/run_all_bug_tests.sh
```

### PowerShell Alternatives
If bash is not available, consider creating `.ps1` equivalents:
```powershell
# PowerShell version
python tests/test_bug10.py
python tests/test_bug11.py
python tests/test_bug12.py
```

## Integration with CI/CD

These scripts are designed to be CI/CD friendly:

```yaml
# GitHub Actions example
name: Run Test Suite
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run regression tests
        run: bash tests/oneshot/shell/orchestration/run_all_bug_tests.sh
```

## Contributing

When adding new shell scripts:

1. Place in appropriate category directory
2. Add shebang line: `#!/bin/bash`
3. Use `set -e` for error handling
4. Add comments explaining purpose
5. Make executable: `chmod +x script.sh`
6. Add entry to this README
7. Test on both Unix and Windows (Git Bash)

---

**Last Updated:** 2025-11-12
**Maintainer:** SLGA2 Development Team
