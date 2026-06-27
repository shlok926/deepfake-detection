#!/bin/bash

# ==============================================================================
# ENTERPRISE FORENSICS PLATFORM UNIX SETUP SCRIPT (scripts/setup_dev.sh)
# ==============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}=== Starting AI Forensics Platform Unix Dev Setup ===${NC}"

# 1. Verify python
if ! command -v python3 &> /dev/null; then
    echo -e "Error: python3 is not installed."
    exit 1
fi

# 2. Bootstrap directories
echo -e "${YELLOW}Auto-generating developer folder structure...${NC}"
python3 app/utils/bootstrap.py

# 3. Create venv
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment in './venv'...${NC}"
    python3 -m venv venv
fi

# 4. Install dependencies
echo -e "${YELLOW}Installing packages and dependencies...${NC}"
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install black isort flake8 mypy pre-commit pytest pytest-cov httpx

# 5. Pre-commit
if [ -d ".git" ]; then
    echo -e "${YELLOW}Configuring git pre-commit hooks...${NC}"
    pre-commit install
fi

# 6. Copy template env
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating local .env from template...${NC}"
    cp .env.example .env
fi

echo -e "${GREEN}=== Dev environment successfully initialized! ===${NC}"
echo -e "${GREEN}Run 'source venv/bin/activate' to start working.${NC}"
