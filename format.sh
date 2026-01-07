#!/bin/bash

# Code formatting script using black

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

show_help() {
    echo "Usage: ./format.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  --check    Check formatting without making changes"
    echo "  --help     Show this help message"
    echo ""
    echo "Without options, formats all Python files in place."
}

if [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

if [[ "$1" == "--check" ]]; then
    echo "Checking code formatting..."
    if uv run black --check .; then
        echo -e "${GREEN}All files are properly formatted.${NC}"
        exit 0
    else
        echo -e "${RED}Some files need formatting. Run ./format.sh to fix.${NC}"
        exit 1
    fi
else
    echo "Formatting code with black..."
    uv run black .
    echo -e "${GREEN}Formatting complete.${NC}"
fi
