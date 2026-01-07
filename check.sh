#!/bin/bash

# Development quality checks script
# Runs formatting checks and tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Running development quality checks..."
echo "======================================"
echo ""

# Track failures
FAILED=0

# Check formatting
echo -e "${YELLOW}[1/2] Checking code formatting...${NC}"
if uv run black --check .; then
    echo -e "${GREEN}Formatting check passed.${NC}"
else
    echo -e "${RED}Formatting check failed.${NC}"
    echo "Run ./format.sh to fix formatting issues."
    FAILED=1
fi
echo ""

# Run tests
echo -e "${YELLOW}[2/2] Running tests...${NC}"
if uv run pytest backend/tests/ -v; then
    echo -e "${GREEN}Tests passed.${NC}"
else
    echo -e "${RED}Tests failed.${NC}"
    FAILED=1
fi
echo ""

# Summary
echo "======================================"
if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}Some checks failed.${NC}"
    exit 1
fi
