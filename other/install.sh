#!/bin/bash

# NQ Signal Bot Installation Script

set -e

echo "=========================================="
echo "NQ Intraday Momentum Signal Bot Installer"
echo "=========================================="

# Check if Python 3.8+ is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Detected Python version: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION < 3.8" | bc -l) -eq 1 ]]; then
    echo "Error: Python 3.8 or higher is required."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data logs

# Initialize database
echo "Initializing database..."
python3 -c "
from src.utils.database import DatabaseManager
db = DatabaseManager('data/nq_signals.db')
db.init_database()
print('Database initialized successfully')
"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "Please edit .env file with your configuration"
fi

# Make main.py executable
chmod +x main.py

echo ""
echo "=========================================="
echo "Installation completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Run the bot: python main.py"
echo "3. Follow the interactive menu"
echo ""
echo "For TradingView setup, see: docs/tradingview_setup.md"
echo "For strategy configuration, see: docs/strategy_examples.md"
echo ""
echo "To start the webhook server:"
echo "python main.py webhook"
echo ""
echo "Happy trading!"