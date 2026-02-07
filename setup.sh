#!/bin/bash
# Setup script for macOS - Local GGUF Chat

echo "Setting up Local GGUF Chat..."

# Check if Python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found. Install from python.org or brew install python3"
    exit 1
fi

echo "Installing dependencies..."

# macOS Apple Silicon - Metal support
if [[ $(uname -m) == "arm64" ]]; then
    echo "Detected Apple Silicon - Enabling Metal GPU support"
    CMAKE_ARGS="-DLLAMA_METAL=on" pip3 install llama-cpp-python --force-reinstall --no-cache-dir
else
    echo "Detected Intel Mac"
    pip3 install llama-cpp-python
fi

# Install other dependencies
pip3 install flask webview

echo ""
echo "Setup complete!"
echo ""
echo "To run the app:"
echo "  python3 standalone_chat.py"
echo ""
echo "Place .gguf models in the same folder or ~/models/"
