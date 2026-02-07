# Local GGUF Chat - Standalone

A **single-file desktop app** for chatting with local GGUF models. No browser needed - runs in its own window.

## Quick Setup (macOS)

```bash
# 1. Run setup (only once)
./setup.sh

# 2. Run the app
python3 standalone_chat.py
```

That's it!

## What is this?

- **One file**: `standalone_chat.py` has everything (backend + frontend + GUI)
- **Desktop app**: Opens in its own window, not a browser tab
- **Auto-detects port**: No more "port 5000 in use" errors
- **Metal GPU**: Automatically uses Apple Silicon GPU on M1/M2/M3

## Getting Models

Download `.gguf` files from:
- https://huggingface.co/models?search=gguf

Place them in the app folder or `~/models/`.

## File Structure

```
standalone_chat.py   # Main app (run this!)
setup.sh             # One-time setup script
```

## Manual Setup (if setup.sh fails)

**Apple Silicon (M1/M2/M3):**
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip3 install llama-cpp-python
pip3 install flask webview
python3 standalone_chat.py
```

**Intel Mac:**
```bash
pip3 install llama-cpp-python flask webview
python3 standalone_chat.py
```
