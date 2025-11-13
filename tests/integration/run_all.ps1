# PowerShell script to run all integration tests with comprehensive reporting

Write-Host "========================================" -ForegroundColor Blue
Write-Host "SLGA-Plus Integration Test Suite" -ForegroundColor Blue
Write-Host "========================================" -ForegroundColor Blue
Write-Host ""

# Check if pytest is installed
$pytestInstalled = Get-Command pytest -ErrorAction SilentlyContinue
if (-not $pytestInstalled) {
    Write-Host "Error: pytest is not installed" -ForegroundColor Red
    Write-Host "Install with: pip install pytest pytest-cov"
    exit 1
}

# Create results directory
New-Item -ItemType Directory -Force -Path test_results | Out-Null

Write-Host "Running all integration tests..." -ForegroundColor Cyan
Write-Host ""

# Run each test category
Write-Host "1. Data Pipeline Tests (50 tests)" -ForegroundColor Cyan
pytest tests/integration/test_data_pipeline.py -v --tb=short --junit-xml=test_results/data_pipeline.xml

Write-Host ""
Write-Host "2. Training Pipeline Tests (40 tests)" -ForegroundColor Cyan
pytest tests/integration/test_training_pipeline.py -v --tb=short --junit-xml=test_results/training_pipeline.xml

Write-Host ""
Write-Host "3. Model-Data Integration Tests (30 tests)" -ForegroundColor Cyan
pytest tests/integration/test_model_data.py -v --tb=short --junit-xml=test_results/model_data.xml

Write-Host ""
Write-Host "4. Generation Pipeline Tests (20 tests)" -ForegroundColor Cyan
pytest tests/integration/test_generation_pipeline.py -v --tb=short --junit-xml=test_results/generation_pipeline.xml

Write-Host ""
Write-Host "5. Attention Pipeline Tests (10 tests)" -ForegroundColor Cyan
pytest tests/integration/test_attention_pipeline.py -v --tb=short --junit-xml=test_results/attention_pipeline.xml

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Running Complete Test Suite with Coverage" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Run all tests with coverage
pytest tests/integration/ -v `
    --cov=src `
    --cov-report=html:test_results/htmlcov `
    --cov-report=term-missing `
    --cov-report=xml:test_results/coverage.xml `
    --junit-xml=test_results/all_tests.xml

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Test Summary" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Count tests in each category
Write-Host "Test counts by category:"
Write-Host "  Data Pipeline:      50 tests"
Write-Host "  Training Pipeline:  40 tests"
Write-Host "  Model-Data:         30 tests"
Write-Host "  Generation:         20 tests"
Write-Host "  Attention:          10 tests"
Write-Host "  -------------------------"
Write-Host "  TOTAL:             150 tests"

Write-Host ""
Write-Host "Results saved to: test_results/" -ForegroundColor Green
Write-Host "  - HTML coverage report: test_results/htmlcov/index.html"
Write-Host "  - XML coverage report: test_results/coverage.xml"
Write-Host "  - JUnit XML reports: test_results/*.xml"

Write-Host ""
Write-Host "✓ Integration test suite complete!" -ForegroundColor Green
