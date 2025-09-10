#!/bin/bash

# Zidio-Emotion Setup Script
echo "🧠 Setting up Zidio-Emotion Detection System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Create virtual environment (optional but recommended)
read -p "Do you want to create a virtual environment? (y/n): " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv zidio_emotion_env
    source zidio_emotion_env/bin/activate
    echo "✅ Virtual environment created and activated"
fi

# Install requirements
echo "📦 Installing required packages..."
pip install -r requirements.txt

# Check if models exist
if [[ ! -f "emotion_detection_model.h5" ]]; then
    echo "⚠️  Warning: emotion_detection_model.h5 not found. You may need to train the model first."
fi

if [[ ! -f "svm_model1.pkl" ]]; then
    echo "⚠️  Warning: svm_model1.pkl not found. You may need to train the model first."
fi

echo "🎉 Setup complete!"
echo ""
echo "To run the application:"
echo "  streamlit run app1.py"
echo ""
echo "The application will open in your browser at http://localhost:8501"