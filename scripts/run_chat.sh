#!/bin/bash
# ==============================================================================
# Voice Agent Pipecat Chat - Interactive Terminal Chat
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           Voice Agent Pipecat - Interactive Chat           ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

cd "$PROJECT_DIR"

# Activate venv if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo -e "${GREEN}→ Starting chat...${NC}"
echo ""

python scripts/pipecat_chat.py "$@"
