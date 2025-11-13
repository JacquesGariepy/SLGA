#!/bin/bash
# Run all integration tests with comprehensive reporting

set -e  # Exit on error

echo "========================================"
echo "SLGA-Plus Integration Test Suite"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Error: pytest is not installed"
    echo "Install with: pip install pytest pytest-cov"
    exit 1
fi

# Create results directory
mkdir -p test_results

echo -e "${BLUE}Running all integration tests...${NC}"
echo ""

# Run each test category
echo -e "${BLUE}1. Data Pipeline Tests (50 tests)${NC}"
pytest tests/integration/test_data_pipeline.py -v --tb=short --junit-xml=test_results/data_pipeline.xml

echo ""
echo -e "${BLUE}2. Training Pipeline Tests (40 tests)${NC}"
pytest tests/integration/test_training_pipeline.py -v --tb=short --junit-xml=test_results/training_pipeline.xml

echo ""
echo -e "${BLUE}3. Model-Data Integration Tests (30 tests)${NC}"
pytest tests/integration/test_model_data.py -v --tb=short --junit-xml=test_results/model_data.xml

echo ""
echo -e "${BLUE}4. Generation Pipeline Tests (20 tests)${NC}"
pytest tests/integration/test_generation_pipeline.py -v --tb=short --junit-xml=test_results/generation_pipeline.xml

echo ""
echo -e "${BLUE}5. Attention Pipeline Tests (10 tests)${NC}"
pytest tests/integration/test_attention_pipeline.py -v --tb=short --junit-xml=test_results/attention_pipeline.xml

echo ""
echo -e "${GREEN}========================================"
echo "Running Complete Test Suite with Coverage"
echo "========================================${NC}"
echo ""

# Run all tests with coverage
pytest tests/integration/ -v \
    --cov=src \
    --cov-report=html:test_results/htmlcov \
    --cov-report=term-missing \
    --cov-report=xml:test_results/coverage.xml \
    --junit-xml=test_results/all_tests.xml

echo ""
echo -e "${GREEN}========================================"
echo "Test Summary"
echo "========================================${NC}"
echo ""

# Count tests in each category
echo "Test counts by category:"
echo "  Data Pipeline:      50 tests"
echo "  Training Pipeline:  40 tests"
echo "  Model-Data:         30 tests"
echo "  Generation:         20 tests"
echo "  Attention:          10 tests"
echo "  -------------------------"
echo "  TOTAL:             150 tests"

echo ""
echo -e "${GREEN}Results saved to: test_results/${NC}"
echo "  - HTML coverage report: test_results/htmlcov/index.html"
echo "  - XML coverage report: test_results/coverage.xml"
echo "  - JUnit XML reports: test_results/*.xml"

echo ""
echo -e "${GREEN}✓ Integration test suite complete!${NC}"
