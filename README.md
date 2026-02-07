# Local GGUF Chat App

A simple web interface for running local GGUF AI models using llama.cpp.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **For macOS (Apple Silicon M1/M2/M3):**
   ```bash
   CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
   pip install flask flask-cors
   ```

2. **Download a GGUF model:**
   - Get models from [HuggingFace](https://huggingface.co/models?sort=trending&search=gguf)
   - Popular models: Llama-2, Mistral, Phi, TinyLlama
   - Place the `.gguf` file in the project directory or `~/models/`

3. **Run the app:**
   ```bash
   python app.py
   ```

4. **Open your browser** to `http://localhost:5000`

## Usage

1. Enter the path to your `.gguf` model file (or click "Scan for Models")
2. Click "Load Model"
3. Adjust generation parameters (temperature, max tokens, etc.)
4. Start chatting!

## Features

- Load any GGUF format model
- Adjustable context size, temperature, top-p sampling
- Chat history with system/user/assistant roles
- Auto-scan for models in common directories
- GPU acceleration support (set `n_gpu_layers`)

## GPU Acceleration

**macOS (Apple Silicon):** Metal is enabled during pip install. GPU acceleration works automatically.

To use GPU on other platforms, modify `app.py` line 26:
```python
n_gpu_layers=-1  # Use all GPU layers
```

## File Structure

- `app.py` - Flask backend with API endpoints
- `model_manager.py` - GGUF model wrapper
- `index.html` - Chat interface
- `requirements.txt` - Python dependencies
