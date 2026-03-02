#!/bin/bash

# Quick start script for Whispered Speech Recognition

set -e

echo "🎤 Whispered Speech Recognition - Quick Start"
echo "============================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.10+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Prepare data
echo "📊 Preparing synthetic dataset..."
python scripts/prepare_data.py --config configs/data_config.yaml

# Run tests
echo "🧪 Running tests..."
python -m pytest tests/ -v

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Train the model:"
echo "   python main.py train --config configs/config.yaml"
echo ""
echo "2. Run the demo:"
echo "   python main.py demo"
echo ""
echo "3. Transcribe an audio file:"
echo "   python main.py transcribe --audio path/to/audio.wav"
echo ""
echo "⚠️  Remember: This is for research and educational purposes only!"
echo "   See README.md for privacy and ethical guidelines."
