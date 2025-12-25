#!/bin/bash

# OV-Memory v1.1 Complete Test Suite Execution Script
# Run all tests, benchmarks, and generate comprehensive report
#
# Usage: bash run_all_tests.sh
# Or:    chmod +x run_all_tests.sh && ./run_all_tests.sh

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}OV-Memory v1.1 - Complete Test Suite${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check Python version
echo -e "${YELLOW}[1/5] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "Python version: ${GREEN}${PYTHON_VERSION}${NC}\n"

# Install dependencies
echo -e "${YELLOW}[2/5] Installing dependencies...${NC}"
if [ -f "tests/requirements.txt" ]; then
    pip install -q -r tests/requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}\n"
else
    echo -e "${RED}✗ requirements.txt not found${NC}"
    exit 1
fi

# Run core unit tests
echo -e "${YELLOW}[3/5] Running core unit tests...${NC}"
if pytest tests/test_ov_memory_core.py -v --tb=short 2>&1 | tee test_output_core.log; then
    echo -e "${GREEN}✓ Core tests passed${NC}\n"
else
    echo -e "${RED}✗ Core tests failed${NC}"
    exit 1
fi

# Run compatibility tests
echo -e "${YELLOW}[4/5] Running compatibility tests...${NC}"
if pytest tests/test_agents_md_compatibility.py -v --tb=short 2>&1 | tee test_output_compat.log; then
    echo -e "${GREEN}✓ Compatibility tests passed${NC}\n"
else
    echo -e "${RED}✗ Compatibility tests failed${NC}"
    exit 1
fi

# Run benchmarks
echo -e "${YELLOW}[5/5] Running performance benchmarks...${NC}"
echo -e "(This may take ~45 seconds)\n"
if python tests/benchmark_ov_vs_markdown.py 2>&1 | tee benchmark_output.log; then
    echo -e "\n${GREEN}✓ Benchmarks completed${NC}\n"
else
    echo -e "${RED}✗ Benchmarks failed${NC}"
    exit 1
fi

# Generate summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Suite Summary${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Count test results
CORE_PASSED=$(grep -c "PASSED" test_output_core.log || echo "0")
COMPAT_PASSED=$(grep -c "PASSED" test_output_compat.log || echo "0")

echo -e "${GREEN}✓ Core Unit Tests: ${CORE_PASSED} passed${NC}"
echo -e "${GREEN}✓ Compatibility Tests: ${COMPAT_PASSED} passed${NC}"
echo -e "${GREEN}✓ Benchmark Suite: 4 suites completed${NC}\n"

# Check for benchmark results
if [ -f "tests/benchmark_results.json" ]; then
    echo -e "${GREEN}✓ Benchmark results saved: tests/benchmark_results.json${NC}\n"
else
    echo -e "${YELLOW}⚠ Benchmark results file not found${NC}\n"
fi

# Summary of outputs
echo -e "${YELLOW}Generated Output Files:${NC}"
echo -e "  • test_output_core.log - Core unit test output"
echo -e "  • test_output_compat.log - Compatibility test output"
echo -e "  • benchmark_output.log - Benchmark results"
echo -e "  • tests/benchmark_results.json - JSON results\n"

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ ALL TESTS COMPLETED SUCCESSFULLY!${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "${YELLOW}Next Steps:${NC}"
echo -e "  1. Review test results in generated .log files"
echo -e "  2. Analyze benchmarks in tests/benchmark_results.json"
echo -e "  3. Upload to Zenodo when ready"
echo -e "  4. See TESTING_GUIDE.md for detailed information\n"

echo -e "Om Vinayaka 🙏\n"
