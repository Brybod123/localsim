from llama_cpp import Llama
import os

class ModelManager:
    def __init__(self):
        self.model = None
        self.model_path = None
    
    def load_model(self, path, n_ctx=4096, n_gpu_layers=0):
        """Load a GGUF model from the specified path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Check for known incompatible models
        filename = os.path.basename(path).lower()
        incompatible_patterns = [
            'mistral3', 'ministral-3', 'mistral-3', 
            'reasoning-2512', '3b-reasoning'
        ]
        
        for pattern in incompatible_patterns:
            if pattern in filename:
                raise ValueError(f"Model '{filename}' contains '{pattern}' architecture which is not supported by the current llama-cpp-python version. This model requires a very recent version. Please try:\n1. pip install --upgrade llama-cpp-python\n2. Or use a different model like:\n   - Llama-3-8B models\n   - Mistral-7B models\n   - Qwen-7B models\n   - Phi-3 models")
        
        try:
            print(f"Loading model from: {path}")
            print(f"Model file size: {os.path.getsize(path) / (1024**2):.1f} MB")
            
            # Try to load with different parameters for better compatibility
            try:
                self.model = Llama(
                    model_path=path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    verbose=True,
                    n_batch=512,
                    f16_kv=True
                )
            except Exception as first_error:
                print(f"Initial load failed, trying with compatibility mode: {first_error}")
                # Try with more conservative settings
                self.model = Llama(
                    model_path=path,
                    n_ctx=min(n_ctx, 2048),  # Reduce context size
                    n_gpu_layers=0,           # Disable GPU layers
                    verbose=True,
                    n_batch=256,            # Smaller batch size
                    f16_kv=False            # Disable KV optimization
                )
            
            self.model_path = path
            print("Model loaded successfully")
            return True
        except Exception as e:
            error_msg = str(e)
            print(f"Error loading model: {error_msg}")
            
            # Provide helpful error messages for common issues
            if "unknown model architecture" in error_msg:
                arch_match = error_msg.find("'") + 1
                arch_end = error_msg.find("'", arch_match)
                if arch_end > arch_match:
                    arch = error_msg[arch_match:arch_end]
                    raise ValueError(f"Unsupported model architecture: '{arch}'. This model requires a newer version of llama-cpp-python. Try: pip install --upgrade llama-cpp-python\n\nAlternatively, try these compatible models:\n- Llama-3-8B-Instruct-Q4_K_M.gguf\n- Mistral-7B-Instruct-v0.2-Q4_K_M.gguf\n- Qwen1.5-7B-Chat-Q4_K_M.gguf\n- phi-3-mini-4k-instruct-q4.gguf")
                else:
                    raise ValueError("Unsupported model architecture. Try: pip install --upgrade llama-cpp-python")
            elif "out of memory" in error_msg.lower():
                raise ValueError("Not enough memory to load model. Try a smaller model or reduce context size.")
            elif "file not found" in error_msg.lower():
                raise FileNotFoundError(f"Model file not found: {path}")
            else:
                raise ValueError(f"Failed to load model: {error_msg}")
            
            self.model = None
            self.model_path = None
    
    def generate(self, prompt, max_tokens=512, temperature=0.7, top_p=0.9, stop=None):
        """Generate a response from the loaded model."""
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        try:
            print(f"Generating response with max_tokens={max_tokens}, temp={temperature}")
            
            # Reset context to avoid sequence position issues
            self.model.reset()
            
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                echo=False
            )
            return output['choices'][0]['text'].strip()
        except Exception as e:
            print(f"Generation error: {e}")
            # Try to reset the model context on error
            try:
                if self.model:
                    self.model.reset()
                    print("Model context reset due to error")
            except:
                pass
            raise e
    
    def reset_context(self):
        """Reset the model context to clear sequence positions."""
        if self.model is not None:
            try:
                self.model.reset()
                print("Model context reset successfully")
            except Exception as e:
                print(f"Error resetting context: {e}")
                raise e
        
    def generate_stream(self, prompt, max_tokens=512, temperature=0.7, top_p=0.9, stop=None):
        """Generate a streaming response from the loaded model."""
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        try:
            print(f"Starting stream generation with max_tokens={max_tokens}, temp={temperature}")
            
            # Reset context to avoid sequence position issues
            self.model.reset()
            
            stream = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or [],
                stream=True
            )
            
            for chunk in stream:
                text = chunk['choices'][0].get('text', '')
                if text:
                    yield text
        except Exception as e:
            print(f"Stream generation error: {e}")
            # Try to reset the model context on error
            try:
                if self.model:
                    self.model.reset()
                    print("Model context reset due to error")
            except:
                pass
            yield f"[ERROR] {str(e)}"
    
    def is_loaded(self):
        return self.model is not None
