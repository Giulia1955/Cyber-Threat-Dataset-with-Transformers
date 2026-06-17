#!/bin/bash
# Setup script for Copilot MCP servers
# Configures GitHub, HuggingFace, Kaggle, SQLite, and Ollama MCPs

set -e

echo "🔧 Setting up Copilot MCP Servers..."
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 1. GitHub MCP Setup
echo -e "${BLUE}[1/5] GitHub MCP Setup${NC}"
if [ -z "$GITHUB_TOKEN" ]; then
    echo -e "${YELLOW}⚠️  GITHUB_TOKEN not set${NC}"
    echo "Get your token at: https://github.com/settings/tokens"
    echo "Then run: export GITHUB_TOKEN=<your-token>"
else
    echo -e "${GREEN}✓ GITHUB_TOKEN is set${NC}"
    gh auth status 2>/dev/null || echo "Install GitHub CLI: https://cli.github.com"
fi
echo ""

# 2. HuggingFace MCP Setup
echo -e "${BLUE}[2/5] HuggingFace MCP Setup${NC}"
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}⚠️  HF_TOKEN not set${NC}"
    echo "Get your token at: https://huggingface.co/settings/tokens"
    echo "Then run: export HF_TOKEN=<your-token>"
else
    echo -e "${GREEN}✓ HF_TOKEN is set${NC}"
    python3 -c "from huggingface_hub import get_home; print(f'HF home: {get_home()}')" 2>/dev/null || echo "Install huggingface_hub: pip install huggingface-hub"
fi
echo ""

# 3. Kaggle MCP Setup
echo -e "${BLUE}[3/5] Kaggle MCP Setup${NC}"
if [ -f ~/.kaggle/kaggle.json ]; then
    echo -e "${GREEN}✓ Kaggle credentials found at ~/.kaggle/kaggle.json${NC}"
    chmod 600 ~/.kaggle/kaggle.json 2>/dev/null || true
else
    echo -e "${YELLOW}⚠️  Kaggle credentials not found${NC}"
    echo "Download from: https://www.kaggle.com/settings/account"
    echo "Place at: ~/.kaggle/kaggle.json"
    echo "Then run: chmod 600 ~/.kaggle/kaggle.json"
fi
echo ""

# 4. SQLite Database Setup
echo -e "${BLUE}[4/5] SQLite Database Setup${NC}"
DB_PATH="./data/experiments.db"
if [ ! -f "$DB_PATH" ]; then
    echo "Creating SQLite database at $DB_PATH..."
    mkdir -p data
    python3 << 'EOF'
import sqlite3
import os

db_path = "./data/experiments.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create experiments table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        model_version TEXT,
        dataset_version TEXT,
        params TEXT,
        metrics TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

# Create training_history table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS training_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER NOT NULL,
        epoch INTEGER,
        loss REAL,
        f1_score REAL,
        precision REAL,
        recall REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
    )
''')

conn.commit()
conn.close()
print(f"✓ SQLite database created at {db_path}")
EOF
else
    echo -e "${GREEN}✓ SQLite database already exists at $DB_PATH${NC}"
fi
echo ""

# 5. Ollama Setup
echo -e "${BLUE}[5/5] Ollama Setup${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama is installed${NC}"
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo -e "${GREEN}✓ Ollama service is running on http://localhost:11434${NC}"
    else
        echo -e "${YELLOW}⚠️  Ollama service not responding${NC}"
        echo "Start it with: ollama serve"
    fi
else
    echo -e "${YELLOW}⚠️  Ollama not installed${NC}"
    echo "Install from: https://ollama.ai"
fi
echo ""

# Summary
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}MCP Setup Summary:${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Configuration file: .copilot-mcps.yml"
echo ""
echo "Next steps:"
echo "  1. Set missing environment variables"
echo "  2. Start Ollama: ollama serve"
echo "  3. Test MCPs with Copilot commands"
echo ""
echo "For more details, see:"
echo "  - .copilot-mcps.yml (configuration)"
echo "  - .github/copilot-instructions.md (MCP documentation)"
echo ""
