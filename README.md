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
   - Place `.gguf` file in project directory or `~/models/`

3. **Run app:**
   ```bash
   python app.py
   ```

4. **Open your browser** to `http://localhost:5000`

## Usage

1. Enter path to your `.gguf` model file (or click "Scan for Models")
2. Click "Load Model"
3. Adjust generation parameters (temperature, max tokens, etc.)
4. **Set a system prompt** to guide AI behavior (optional)
5. Start chatting!

## Features

- Load any GGUF format model
- **System Prompts**: Guide AI behavior with custom system prompts
- Adjustable context size, temperature, top-p sampling
- Chat history with system/user/assistant roles
- Auto-scan for models in common directories
- GPU acceleration support (set `n_gpu_layers`)
- Model search and download from HuggingFace
- Beautiful animated welcome screen

## System Prompts

The app now supports custom system prompts to guide AI behavior:

- **Custom Prompts**: Enter any system prompt in the Settings panel
- **Quick Presets**: Use preset buttons for common roles:
  - 🤖 Helpful Assistant
  - 🎨 Creative AI
  - 💻 Coding Expert
  - 📚 Teacher
- **Persistent**: System prompts are saved in browser storage
- **Flexible**: Works with any GGUF model

### Example System Prompts

```
You are a helpful AI assistant that provides clear, concise answers.
You are a creative AI that thinks outside the box and generates innovative ideas.
You are a coding expert. Provide clean, efficient code solutions with explanations.
You are a teacher. Explain complex concepts simply and with helpful examples.
```

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

## Update Log

### v1.2.0 - System Prompts Feature
- ✨ **NEW**: System prompts support to guide AI behavior
- 🎨 **NEW**: Quick preset buttons for common AI roles
- 💾 **NEW**: Persistent system prompt storage
- 🔧 **NEW**: Custom system prompt input field
- 📱 **UI**: Enhanced settings panel with system prompt section
- 🐛 **FIX**: Improved model search with REST API pagination
- 🔍 **FIX**: Better HuggingFace API integration

### v1.1.0 - Model Search & Download
- ✨ **NEW**: HuggingFace model search and download
- 🎨 **NEW**: Animated welcome screen with gradients
- 📱 **NEW**: Model filtering by family, parameters, quantization
- 🔄 **NEW**: Pagination support for browsing models
- 🔍 **NEW**: Auto-search on filter changes

### v1.0.0 - Initial Release
- ✨ Basic GGUF model loading and chat
- 🎨 Clean web interface
- ⚙️ Generation parameters
- 📁 Model scanning
