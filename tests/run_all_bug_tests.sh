#!/bin/bash
# Run all critical bug tests
# Tests for BUG #10, #11, #12

set -e

echo "================================================================================"
echo "CRITICAL BUGS TEST SUITE"
echo "================================================================================"
echo ""
echo "Testing fixes for:"
echo "  BUG #10: Token IDs used as positions (data.py)"
echo "  BUG #11: Frozen landmarks during generation (model.py)"
echo "  BUG #12: Sequences continue after EOS (model.py)"
echo ""
echo "================================================================================"
echo ""

# Change to repo root
cd "$(dirname "$0")/.."

# Track results
PASSED=0
FAILED=0

# BUG #10 Test
echo ""
echo "================================================================================"
echo "TEST 1/3: BUG #10 - Landmark Positions vs Token IDs"
echo "================================================================================"
if python tests/test_bug10_landmark_positions.py; then
    echo "✅ BUG #10 TEST PASSED"
    PASSED=$((PASSED + 1))
else
    echo "❌ BUG #10 TEST FAILED"
    FAILED=$((FAILED + 1))
fi

# BUG #11 Test
echo ""
echo "================================================================================"
echo "TEST 2/3: BUG #11 - Dynamic Landmark Updates"
echo "================================================================================"
if python tests/test_bug11_dynamic_landmarks.py; then
    echo "✅ BUG #11 TEST PASSED"
    PASSED=$((PASSED + 1))
else
    echo "❌ BUG #11 TEST FAILED"
    FAILED=$((FAILED + 1))
fi

# BUG #12 Test
echo ""
echo "================================================================================"
echo "TEST 3/3: BUG #12 - EOS Stopping"
echo "================================================================================"
if python tests/test_bug12_eos_stopping.py; then
    echo "✅ BUG #12 TEST PASSED"
    PASSED=$((PASSED + 1))
else
    echo "❌ BUG #12 TEST FAILED"
    FAILED=$((FAILED + 1))
fi

# Summary
echo ""
echo "================================================================================"
echo "TEST RESULTS SUMMARY"
echo "================================================================================"
echo "Passed: $PASSED/3"
echo "Failed: $FAILED/3"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED - All critical bugs are FIXED!"
    echo ""
    echo "✅ BUG #10: Landmarks use correct positions (not token IDs)"
    echo "✅ BUG #11: Landmarks update dynamically during generation"
    echo "✅ BUG #12: Sequences stop cleanly at EOS token"
    echo ""
    exit 0
else
    echo "❌ SOME TESTS FAILED - $FAILED critical bug(s) still present"
    echo ""
    echo "See detailed analysis in:"
    echo "  docs/CRITICAL_BUGS_ANALYSIS_AND_FIXES.md"
    echo ""
    echo "Quick fixes:"
    if [ $FAILED -gt 0 ]; then
        echo "  1. Review test output above for specific failure details"
        echo "  2. Apply fixes from CRITICAL_BUGS_ANALYSIS_AND_FIXES.md"
        echo "  3. Re-run: ./tests/run_all_bug_tests.sh"
    fi
    echo ""
    exit 1
fi
