#!/bin/bash
# Quick validation script for straight-through estimator fix
# Run this to verify the fix is working correctly

set -e  # Exit on error

echo "============================================================"
echo "VALIDATION: STRAIGHT-THROUGH ESTIMATOR FIX"
echo "============================================================"
echo ""
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "src/landmarks.py" ]; then
    echo -e "${RED}❌ ERROR: src/landmarks.py not found${NC}"
    echo "Please run this script from the SLGA root directory"
    exit 1
fi

# Check if test file exists
if [ ! -f "tests/test_straight_through_improvement.py" ]; then
    echo -e "${RED}❌ ERROR: Test file not found${NC}"
    exit 1
fi

echo -e "${YELLOW}Step 1: Running full test suite...${NC}"
echo ""
python tests/test_straight_through_improvement.py

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ All tests passed!${NC}"
else
    echo ""
    echo -e "${RED}❌ Tests failed!${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 2: Quick smoke test...${NC}"
echo ""
python -c "
import torch
import sys
sys.path.insert(0, 'src')
from landmarks import LearnableLandmarkSelector

# Quick smoke test
B, L, D, G = 2, 32, 64, 4
selector = LearnableLandmarkSelector(embed_dim=D, num_landmarks=G)
x = torch.randn(B, L, D, requires_grad=True)

# Test straight-through
selector.train()
indices, states, scores = selector(x, use_gumbel=False)
loss = states.sum()
loss.backward()

assert x.grad is not None, 'Gradients not propagating!'
assert indices.shape == (B, G), f'Wrong indices shape: {indices.shape}'
assert states.shape == (B, G, D), f'Wrong states shape: {states.shape}'

print('✅ Smoke test passed!')
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Smoke test passed!${NC}"
else
    echo -e "${RED}❌ Smoke test failed!${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 3: Checking code implementation...${NC}"
echo ""

# Check for key improvements in the code
if grep -q "sigmoid((scores - threshold) / temp)" src/landmarks.py; then
    echo -e "${GREEN}✅ Sigmoid soft-thresholding found${NC}"
else
    echo -e "${RED}❌ Sigmoid soft-thresholding NOT found${NC}"
    exit 1
fi

if grep -q "threshold = topk_vals\[:, -1:\].detach()" src/landmarks.py; then
    echo -e "${GREEN}✅ Adaptive threshold found${NC}"
else
    echo -e "${RED}❌ Adaptive threshold NOT found${NC}"
    exit 1
fi

if grep -q "selection_onehot + selection_soft - selection_soft.detach()" src/landmarks.py; then
    echo -e "${GREEN}✅ Improved straight-through formula found${NC}"
else
    echo -e "${RED}❌ Improved straight-through formula NOT found${NC}"
    exit 1
fi

echo ""
echo "============================================================"
echo -e "${GREEN}✅ VALIDATION COMPLETE: ALL CHECKS PASSED${NC}"
echo "============================================================"
echo ""
echo "📋 Summary:"
echo "  - Code implementation: ✅ Correct"
echo "  - Unit tests: ✅ Pass"
echo "  - Smoke test: ✅ Pass"
echo "  - Gradient flow: ✅ Verified"
echo ""
echo "💡 Next steps:"
echo "  1. Integrate into training (set use_gumbel=True in config)"
echo "  2. Benchmark on full SLGA training"
echo "  3. Compare metrics (spacing loss, convergence)"
echo ""
echo "📚 Documentation:"
echo "  - Full details: docs/STRAIGHT_THROUGH_FIX.md"
echo "  - Summary: docs/STRAIGHT_THROUGH_FIX_SUMMARY.md"
echo "  - Tests: tests/test_straight_through_improvement.py"
echo ""
