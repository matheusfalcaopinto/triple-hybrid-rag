#!/bin/bash
# ============================================================================
# Standalone Agent Complete Test Suite
# ============================================================================
# Runs all tests: unit tests (pytest) and tool harness (run_harness.py)
#
# Usage:
#   ./scripts/run_all_tests.sh
#   ./scripts/run_all_tests.sh --quick      # Skip slow tests
#   ./scripts/run_all_tests.sh --verbose    # Verbose output
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$REPO_ROOT/venv"
RESULTS_DIR="$REPO_ROOT/scripts/test-results"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
QUICK_MODE=false
VERBOSE=""
for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            ;;
        --verbose|-v)
            VERBOSE="-v"
            ;;
    esac
done

# ============================================================================
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Standalone Agent Complete Test Suite                   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Root: $REPO_ROOT"
echo ""

# Activate virtual environment
if [ -f "$VENV_DIR/bin/activate" ]; then
    echo -e "${GREEN}✓ Activating venv${NC}"
    source "$VENV_DIR/bin/activate"
else
    echo -e "${RED}✗ venv not found at $VENV_DIR${NC}"
    echo "  Run: python3 -m venv venv && source venv/bin/activate && pip install -e ."
    exit 1
fi

# Ensure PYTHONPATH includes src
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT"

# Create results directory
mkdir -p "$RESULTS_DIR"

# ============================================================================
# Phase 1: Unit Tests
# ============================================================================
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Phase 1: Unit Tests (pytest)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cd "$REPO_ROOT"

echo "Running pytest..."
pytest tests/ $VERBOSE --tb=short 2>&1 | tee "$RESULTS_DIR/unit_tests.log" || true

UNIT_PASSED=$(grep -E "passed" "$RESULTS_DIR/unit_tests.log" | tail -1 || echo "0 passed")
echo -e "${GREEN}Unit Tests: $UNIT_PASSED${NC}"

# ============================================================================
# Phase 2: Tool Harness (Utility, CRM, Communication, Calendar)
# ============================================================================
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Phase 2: Tool Harness Suites${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Utility Suite
echo "Running Utility Suite..."
python scripts/run_harness.py --suite utility --output "$RESULTS_DIR/utility_results.json" 2>&1 | tee "$RESULTS_DIR/utility_harness.log"

if [ "$QUICK_MODE" = false ]; then
    # Full Suite (CRM, Communication, Calendar)
    # Note: These require DB and environment to be fully set up
    
    echo ""
    echo "Running CRM Suite..."
    python scripts/run_harness.py --suite crm --output "$RESULTS_DIR/crm_results.json" 2>&1 | tee "$RESULTS_DIR/crm_harness.log" || true

    echo ""
    echo "Running Communication Suite..."
    python scripts/run_harness.py --suite communication --output "$RESULTS_DIR/comm_results.json" 2>&1 | tee "$RESULTS_DIR/comm_harness.log" || true
    
    echo ""
    echo "Running Communication Suite (Twilio Backend)..."
    python scripts/run_harness.py --suite communication_twilio --output "$RESULTS_DIR/comm_twilio_results.json" 2>&1 | tee "$RESULTS_DIR/comm_twilio_harness.log" || true
    
    echo ""
    echo "Running Calendar Suite..."
    python scripts/run_harness.py --suite calendar --output "$RESULTS_DIR/calendar_results.json" 2>&1 | tee "$RESULTS_DIR/calendar_harness.log" || true
fi

# ============================================================================
# Phase 3: Tool List Verification
# ============================================================================
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Phase 3: Tool Registration Verification${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

TOOL_COUNT=$(python -c "
from voice_agent.tools import get_all_tools
print(len(get_all_tools()))
" 2>/dev/null || echo "0")

echo -e "${GREEN}✓ Total tools registered: $TOOL_COUNT${NC}"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                      Test Summary                            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo ""

# Count results from JSON files
if command -v jq &> /dev/null; then
    for file in "$RESULTS_DIR"/*_results.json; do
        if [ -f "$file" ]; then
            suite=$(basename "$file" _results.json)
            passed=$(jq '[.[] | select(.success == true)] | length' "$file" 2>/dev/null || echo "?")
            total=$(jq '. | length' "$file" 2>/dev/null || echo "?")
            echo "  $suite: $passed/$total passed"
        fi
    done
else
    echo "  (Install jq for detailed summary)"
fi

echo ""
echo -e "${GREEN}Test run complete!${NC}"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
