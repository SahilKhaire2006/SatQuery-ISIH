#!/bin/bash

echo "Setting up SatQuery project..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Create data directories
mkdir -p data/raw data/processed data/models
mkdir -p logs

# Copy environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file. Please update with your configuration."
fi

echo "Setup complete!"
echo ""
echo "To start the API server:"
echo "  python main.py"
echo ""
echo "To start the CLI:"
echo "  python cli.py"
