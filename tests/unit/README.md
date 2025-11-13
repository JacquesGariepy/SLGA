# Unit Tests for SLGA-Plus

This directory contains comprehensive unit tests for the SLGA-Plus model, organized by module.

## Structure

```
tests/unit/
├── attention/
│   └── test_slga_module.py          (100+ tests)
├── landmarks/
│   ├── test_learnable_landmark_selector.py  (80+ tests)
│   └── test_landmark_losses.py      (60+ tests)
├── model/
│   ├── test_transformer_block.py    (100+ tests)
│   └── test_config.py               (40+ tests)
└── README.md
```

## Running Tests

### Run all unit tests:
```bash
pytest tests/unit/ -v
```

### Run with coverage:
```bash
pytest tests/unit/ --cov=src --cov-report=html --cov-report=term-missing
```

### Run specific module:
```bash
pytest tests/unit/attention/ -v
pytest tests/unit/landmarks/ -v
pytest tests/unit/model/ -v
```

### Run by marker:
```bash
pytest tests/unit/ -m unit            # All unit tests
pytest tests/unit/ -m slow            # Slow tests only
pytest tests/unit/ -m gpu             # GPU tests only
pytest tests/unit/ -m property        # Property-based tests
```

## Test Coverage

### Attention Module (test_slga_module.py) - 100+ tests
- **Initialization**: Parameter validation, configuration, buffers
- **Local Attention**: Windowing, causal masking, dilation
- **Global Attention**: Top-K selection, diverse top-K, cache handling
- **Fusion**: Gated fusion, additive fusion, learned weights
- **Gradient Flow**: Through all components
- **Device Compatibility**: CPU/CUDA
- **Edge Cases**: Short/long sequences, extreme values
- **Numerical Stability**: NaN/Inf protection
- **Monitoring**: Debug metrics capture

### Landmarks Module (test_learnable_landmark_selector.py) - 80+ tests
- **Initialization**: Scorer network, temperature settings
- **Forward Pass**: Gumbel-Softmax, straight-through estimator
- **Temperature Annealing**: Decay schedule, minimum temperature
- **Gradient Flow**: Through scorer, with both selection methods
- **NaN Protection**: Bug #17 fix, float16/float32 handling
- **Device Compatibility**: CPU/CUDA
- **Edge Cases**: Single landmark, many landmarks, short sequences
- **Numerical Stability**: Extreme scores, repeated calls

### Landmark Losses (test_landmark_losses.py) - 60+ tests
- **Spacing Loss**: Differentiable version, Bug #16 fix (G=1)
- **Diversity Loss**: Entropy-based (deprecated)
- **Sparsity Loss**: Concentration metric, adaptive target
- **Gradient Flow**: Through all losses
- **Edge Cases**: G=0, G=1, G=L
- **Integration**: Combined losses, trade-offs
- **Numerical Stability**: Property-based testing

### Transformer Block (test_transformer_block.py) - 100+ tests
- **Initialization**: SLGA attention, FFN, layer norms
- **Forward Pass**: With/without global cache, global weights
- **Pre-Norm Architecture**: LayerNorm placement
- **Gradient Checkpointing**: Memory efficiency
- **Progressive Dilation**: Layer-wise dilation progression
- **Gradient Flow**: Through all sub-layers
- **Device Compatibility**: CPU/CUDA
- **Edge Cases**: Short/long sequences, extreme values
- **FFN Sub-module**: GELU activation, dropout, dimensions

### Config (test_config.py) - 40+ tests
- **Default Values**: All parameters
- **Customization**: Model sizes, SLGA config
- **Validation**: Positive values, dropout ranges
- **Serialization**: Dict conversion, reconstruction
- **Presets**: Small/base/large model configs

## Test Markers

- `@pytest.mark.unit`: Fast, isolated unit tests
- `@pytest.mark.slow`: Tests taking >1s
- `@pytest.mark.gpu`: Tests requiring CUDA
- `@pytest.mark.property`: Property-based tests (Hypothesis)
- `@pytest.mark.integration`: Multi-component tests
- `@pytest.mark.performance`: Performance benchmarks

## Key Testing Features

### 1. Parametrized Tests
Tests run with multiple parameter combinations:
```python
@pytest.mark.parametrize("batch,seq_len", [(1,64), (4,128), (8,256)])
def test_various_sizes(batch, seq_len):
    ...
```

### 2. Fixtures
Reusable test components:
```python
@pytest.fixture
def slga():
    return SLGAModule(embed_dim=512, num_heads=8)
```

### 3. Property-Based Testing
Random input generation with Hypothesis:
```python
@given(
    batch=st.integers(1, 8),
    seq_len=st.integers(16, 256)
)
def test_property_always_finite(batch, seq_len):
    ...
```

### 4. Bug Fix Validation
Tests specifically for reported bugs:
```python
def test_bug16_fix_single_landmark():
    """Test Bug #16 fix: G=1 should not cause NaN."""
    ...
```

## Coverage Goals

- **Line Coverage**: >95%
- **Branch Coverage**: >90%
- **Function Coverage**: 100%

Current coverage by module:
- `src/slga.py`: ~98%
- `src/landmarks.py`: ~96%
- `src/model.py`: ~94%

## Test Categories

### Functional Tests (70%)
- Correct output shapes
- Correct computations
- Expected behavior

### Gradient Tests (10%)
- Gradient flow through all paths
- No gradient leaks
- Proper backpropagation

### Edge Case Tests (10%)
- Boundary conditions
- Extreme values
- Error conditions

### Stability Tests (10%)
- NaN/Inf protection
- Numerical stability
- Repeated operations

## Writing New Tests

### Test Naming Convention
```python
def test_<what>_<condition>_<expected>():
    """Test that <what> does <expected> when <condition>."""
```

### Test Structure (Arrange-Act-Assert)
```python
def test_example():
    # Arrange: Setup test inputs
    x = torch.randn(2, 128, 256)

    # Act: Execute the operation
    out = module(x)

    # Assert: Verify expectations
    assert out.shape == (2, 128, 256)
    assert torch.isfinite(out).all()
```

### Required Checks
1. **Shape**: Output dimensions match expectations
2. **Finite**: No NaN or Inf in outputs
3. **Gradients**: Gradients flow when required
4. **Device**: Correct device placement

## Debugging Failed Tests

### Verbose output:
```bash
pytest tests/unit/attention/test_slga_module.py::TestSLGALocalAttention::test_forward_basic_shape -vv
```

### Show print statements:
```bash
pytest tests/unit/ -s
```

### Stop on first failure:
```bash
pytest tests/unit/ -x
```

### Debug with pdb:
```bash
pytest tests/unit/ --pdb
```

## Continuous Integration

Tests run automatically on:
- Pull requests
- Commits to main branch
- Nightly builds

Required:
- All tests pass
- Coverage >95%
- No new warnings

## Contributing

When adding new features:
1. Write tests FIRST (TDD)
2. Ensure tests cover:
   - Happy path
   - Edge cases
   - Error conditions
   - Gradient flow
3. Update this README if adding new test files
4. Maintain >95% coverage
