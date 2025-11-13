# SLGA-Plus Integration Tests

Comprehensive end-to-end integration tests for SLGA-Plus pipelines.

## Overview

This directory contains **150+ integration tests** covering all major pipelines:

### Test Categories

1. **Data Pipeline (50 tests)** - `test_data_pipeline.py`
   - Dataset loading → Processing → Tokenization → Collation
   - Full pipeline execution with real data
   - Memory efficiency tests
   - Cache consistency tests
   - Edge cases (empty text, long sequences, special characters)

2. **Training Pipeline (40 tests)** - `test_training_pipeline.py`
   - Model initialization → Forward pass → Loss → Backward → Optimizer step
   - Multi-step training loops
   - Checkpoint saving/loading
   - Gradient accumulation
   - Learning rate scheduling
   - Validation pipeline

3. **Model-Data Integration (30 tests)** - `test_model_data.py`
   - DataLoader → Model forward pass
   - Various batch sizes and sequence lengths
   - Edge cases (empty batches, very long sequences)
   - Device transfers (CPU ↔ GPU)
   - Batch variations
   - Memory scaling

4. **Generation Pipeline (20 tests)** - `test_generation_pipeline.py`
   - Model → Generator → Sampling → Decoding
   - Greedy vs stochastic sampling
   - Top-K and Top-P (nucleus) sampling
   - Temperature control
   - Repetition penalty
   - EOS stopping criteria

5. **Attention Pipeline (10 tests)** - `test_attention_pipeline.py`
   - Embedding → Local attention → Global attention → Fusion
   - Landmark selection → Attention computation
   - Multi-head integration
   - Gradient flow
   - Causal masking

## Running Tests

### Run All Integration Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run with coverage
pytest tests/integration/ -v --cov=src --cov-report=html

# Run specific category
pytest tests/integration/test_data_pipeline.py -v
pytest tests/integration/test_training_pipeline.py -v
pytest tests/integration/test_model_data.py -v
pytest tests/integration/test_generation_pipeline.py -v
pytest tests/integration/test_attention_pipeline.py -v
```

### Run Slow Tests

Some tests are marked as `@pytest.mark.slow`:

```bash
# Include slow tests
pytest tests/integration/ -v --runslow

# Exclude slow tests (default)
pytest tests/integration/ -v
```

### Run GPU Tests

GPU tests are skipped automatically if CUDA is not available:

```bash
# Run on GPU (if available)
pytest tests/integration/ -v

# The following tests require GPU:
# - test_model_data.py::TestDeviceTransfers::test_cuda_inference
# - test_model_data.py::TestDeviceTransfers::test_cpu_to_cuda_transfer
# - test_training_pipeline.py::TestForwardBackwardPass::test_mixed_precision_forward
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest tests/integration/ -v -n auto
```

## Test Structure

### Fixtures (conftest.py)

Shared fixtures available to all tests:

- `sample_texts` - Sample text data
- `long_sample_texts` - Longer text samples
- `tokenizer` - GPT-2 tokenizer
- `tiny_config` - Tiny model config for fast testing
- `tiny_model` - Tiny model instance
- `collator_local` - Local-only collator
- `collator_global` - Local+global collator
- `sample_dataset` - HuggingFace dataset
- `device` - Available device (CUDA or CPU)

### Example Test

```python
def test_full_pipeline_execution(self, sample_texts, tokenizer):
    """Test complete data pipeline from raw text to model input."""
    # Create dataset
    dataset = Dataset.from_dict({"text": sample_texts})

    # Create collator
    collator = CollatorLocal(tokenizer, max_length=128)

    # Process batch
    batch = collator(list(dataset))

    # Verify output
    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape[0] == len(sample_texts)
```

## Test Coverage

Target coverage metrics:

- **Statements**: >80%
- **Branches**: >75%
- **Functions**: >80%
- **Lines**: >80%

Current coverage:

```bash
# Generate coverage report
pytest tests/integration/ --cov=src --cov-report=term-missing
```

## Performance Benchmarks

Key performance tests:

1. **Memory Efficiency** (`test_memory_efficiency_many_batches`)
   - Verifies memory doesn't grow unbounded
   - Target: <50MB growth over 100 batches

2. **Batch Processing Speed** (`test_batch_processing_time`)
   - Verifies reasonable processing speed
   - Target: 100 batches in <5 seconds

3. **Training Speed** (`test_validation_faster_than_training`)
   - Verifies validation is faster than training
   - Ensures no unnecessary compute during eval

## Edge Cases Tested

- Empty text
- Very long text (>1000 words)
- Unicode characters (Chinese, Arabic, Hebrew)
- Special characters and symbols
- Whitespace-only text
- Mixed language text
- Code snippets
- Repeated text
- Single token text

## Best Practices

1. **Isolation**: Each test is independent
2. **Fast by Default**: Tests run quickly (<100ms each)
3. **Clear Assertions**: Tests verify specific behaviors
4. **Descriptive Names**: Test names explain what and why
5. **Arrange-Act-Assert**: Clear test structure

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Integration Tests
  run: |
    pytest tests/integration/ -v --cov=src --cov-report=xml

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Debugging Failed Tests

```bash
# Run with detailed output
pytest tests/integration/ -vv --tb=long

# Run single test
pytest tests/integration/test_data_pipeline.py::TestDataPipelineBasics::test_full_pipeline_local_collator -v

# Drop into debugger on failure
pytest tests/integration/ --pdb

# Show print statements
pytest tests/integration/ -s
```

## Adding New Tests

When adding new tests:

1. Choose appropriate test file based on category
2. Use existing fixtures from `conftest.py`
3. Follow naming convention: `test_<what>_<scenario>`
4. Add docstring explaining test purpose
5. Verify test runs quickly (<1s)
6. Mark slow tests with `@pytest.mark.slow`
7. Skip GPU tests when CUDA unavailable

Example:

```python
@pytest.mark.slow
def test_large_scale_training(self, tiny_model, tiny_config):
    """Test training on large dataset (slow)."""
    # Test implementation
    pass
```

## Dependencies

Required packages (from requirements.txt):

- `pytest>=7.0.0`
- `torch>=2.0.0`
- `transformers>=4.30.0`
- `datasets>=2.0.0`
- `psutil>=5.9.0` (for memory tests)

Optional:

- `pytest-cov` (coverage reports)
- `pytest-xdist` (parallel execution)
- `pytest-benchmark` (performance benchmarking)

## Contact

For issues or questions about integration tests, please refer to the main project documentation.
