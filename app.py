import requests
import json
import time
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from model_manager import ModelManager
from huggingface_hub import HfApi, hf_hub_download
import os
import threading
import shutil
import platform

app = Flask(__name__)
CORS(app)

model_manager = ModelManager()
api = HfApi()

# Track download progress
download_status = {
    'downloading': False,
    'model_id': None,
    'progress': 0,
    'error': None
}

SYSTEM_PROMPT = """You are a helpful AI assistant. Provide clear, concise, and accurate responses."""

def get_drive_info():
    """Get information about available drives and their storage."""
    drives = []
    
    if platform.system() == 'Darwin':  # macOS
        # Get mounted volumes
        try:
            result = os.popen('df -h | grep -E "^/dev/"').read()
            for line in result.strip().split('\n'):
                if line:
                    parts = line.split()
                    if len(parts) >= 6:
                        device = parts[0]
                        size = parts[1]
                        used = parts[2]
                        avail = parts[3]
                        mount_point = parts[8] if len(parts) > 8 else parts[-1]
                        
                        # Skip system volumes
                        if not any(skip in mount_point for skip in ['/System', '/Library', '/private']):
                            drives.append({
                                'device': device,
                                'mount_point': mount_point,
                                'total_size': size,
                                'used': used,
                                'available': avail,
                                'is_external': mount_point.startswith('/Volumes/') and mount_point != '/'
                            })
        except Exception as e:
            print(f"Error getting drive info: {e}")
    
    elif platform.system() == 'Windows':
        # Windows drives
        import psutil
        for partition in psutil.disk_partitions():
            if 'removable' in partition.opts or partition.fstype == 'exfat':
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    drives.append({
                        'device': partition.device,
                        'mount_point': partition.mountpoint,
                        'total_size': f"{usage.total // (1024**3)}G",
                        'used': f"{usage.used // (1024**3)}G", 
                        'available': f"{usage.free // (1024**3)}G",
                        'is_external': True
                    })
                except:
                    pass
    
    return drives

def get_high_storage_drives(min_gb=50):
    """Get drives with high available storage."""
    drives = get_drive_info()
    high_storage = []
    
    for drive in drives:
        # Parse available storage (handle different formats)
        avail_str = drive['available'].replace('G', '').replace('M', '').replace('K', '')
        try:
            avail_gb = float(avail_str)
            if 'M' in drive['available']:
                avail_gb /= 1024
            elif 'K' in drive['available']:
                avail_gb /= 1024 * 1024
            
            if avail_gb >= min_gb:
                drive['available_gb'] = avail_gb
                high_storage.append(drive)
        except:
            continue
    
    return high_storage

def format_size(bytes_size):
    """Format bytes to human readable format."""
    for unit in ['B', 'K', 'M', 'G', 'T']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}P"

def format_prompt(messages):
    """Format chat messages into a prompt for the model."""
    formatted = ""
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        if role == 'system':
            formatted += f"System: {content}\n"
        elif role == 'user':
            formatted += f"User: {content}\n"
        elif role == 'assistant':
            formatted += f"Assistant: {content}\n"
    formatted += "Assistant:"
    return formatted

@app.route('/api/models/search', methods=['GET'])
def search_models():
    """Search for GGUF models using HuggingFace REST API with pagination."""
    try:
        search_query = request.args.get('search', '')
        limit = int(request.args.get('limit', 10))
        max_params = request.args.get('max_params', '')
        quantization = request.args.get('quantization', '')
        family = request.args.get('family', '')
        offset = int(request.args.get('offset', 0))
        
        print(f"Search params: query='{search_query}', family='{family}', limit={limit}, offset={offset}")
        
        # Build search URL for REST API
        api_url = "https://huggingface.co/api/models"
        
        # Use broader search terms for better results
        search_terms = []
        if family:
            search_terms.append(family)
        if search_query and search_query != 'trending':
            search_terms.append(search_query)
        
        # Always include GGUF in search
        search_terms.append('GGUF')
        
        params = {
            'search': ' '.join(search_terms) if search_terms else 'GGUF',
            'limit': 50,  # Get more results to filter through
            'sort': 'downloads',
            'direction': -1
        }
        
        print(f"API URL: {api_url}")
        print(f"Params: {params}")
        
        # Make request to HuggingFace REST API
        try:
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            
            print(f"API Response Status: {response.status_code}")
            
            if response.status_code != 200:
                return jsonify({'error': f'HuggingFace API error: {response.status_code}'}), response.status_code
            
            # Parse response - HuggingFace API returns a list directly
            try:
                models = response.json()
                if not isinstance(models, list):
                    # Sometimes it's wrapped in a dict
                    models = models.get('data', []) if isinstance(models, dict) else []
            except Exception as parse_error:
                print(f"JSON parse error: {parse_error}")
                models = []
            
            print(f"API returned {len(models)} raw models")
            
            # Apply additional filtering manually since REST API has limitations
            filtered_models = []
            for m in models:
                # Only include GGUF models
                model_id = m.get('id', '').lower()
                is_gguf = 'gguf' in model_id or any(tag.lower() == 'gguf' for tag in m.get('tags', []))
                if not is_gguf:
                    continue
                
                # Apply family filter
                if family and family.lower() not in model_id:
                    continue
                
                # Extract parameter size
                param_size = extract_param_size(m.get('id', ''), m.get('tags', []))
                
                # Apply max_params filter
                if param_size:
                    if max_params and param_size > float(max_params):
                        continue
                else:
                    # Skip models without parameter size info if filter is applied
                    if max_params:
                        continue
                
                # Calculate RAM usage
                if param_size:
                    estimated_file_bytes = param_size * 1e9 * 0.6
                    ram_usage = calculate_ram_usage(int(estimated_file_bytes))
                    file_size_display = format_fun_size(int(estimated_file_bytes))
                else:
                    ram_usage = "Unknown"
                    file_size_display = "Unknown"
                
                filtered_models.append({
                    'id': m.get('id', ''),
                    'downloads': m.get('downloads', 0),
                    'tags': m.get('tags', []),
                    'author': m.get('author', 'Unknown'),
                    'param_size': param_size,
                    'ram_usage': ram_usage,
                    'file_size': file_size_display
                })
            
            print(f"Filtered to {len(filtered_models)} GGUF models")
            
            # Apply offset for pagination
            if offset > 0:
                filtered_models = filtered_models[offset:]
            else:
                filtered_models = filtered_models[:limit]
            
            print(f"Returning {len(filtered_models)} models with offset {offset}")
            
        except requests.exceptions.RequestException as e:
            print(f"REST API Error: {e}")
            # Fallback: return some popular models manually
            filtered_models = [
                {
                    'id': 'microsoft/DialoGPT-2.7-1b-GGUF',
                    'downloads': 1000000,
                    'tags': ['gguf', 'text-generation'],
                    'author': 'Microsoft',
                    'param_size': 2.7,
                    'ram_usage': calculate_ram_usage(int(2.7 * 1e9 * 0.6)),
                    'file_size': format_fun_size(int(2.7 * 1e9 * 0.6))
                },
                {
                    'id': 'TheBloke/Mistral-7B-Instruct-v0.1-GGUF',
                    'downloads': 2000000,
                    'tags': ['gguf', 'text-generation'],
                    'author': 'TheBloke',
                    'param_size': 7.0,
                    'ram_usage': calculate_ram_usage(int(7.0 * 1e9 * 0.6)),
                    'file_size': format_fun_size(int(7.0 * 1e9 * 0.6))
                }
            ]
            
            if offset > 0:
                filtered_models = filtered_models[offset:]
            else:
                filtered_models = filtered_models[:limit]
        
        return jsonify({'models': filtered_models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def extract_param_size(model_id, tags):
    """Extract parameter size from model ID or tags."""
    import re
    
    # Check model ID for size patterns
    size_patterns = [
        r'(\d+(?:\.\d+)?)B',  # 1B, 7B, 8B, 70B, etc.
        r'(\d+(?:\.\d+)?)\s*[Bb]',  # 1 b, 7 b, etc.
        r'(\d+(?:\.\d+)?)\s*[Mm]illion',  # 1 million, 7 million, etc.
    ]
    
    for pattern in size_patterns:
        match = re.search(pattern, model_id, re.IGNORECASE)
        if match:
            size = float(match.group(1))
            if 'B' in match.group(0).upper():
                return size
            elif 'million' in match.group(0).lower():
                return size / 1000  # Convert million to billion
    
    # Check tags for size info
    if tags:
        for tag in tags:
            for pattern in size_patterns:
                match = re.search(pattern, tag, re.IGNORECASE)
                if match:
                    size = float(match.group(1))
                    if 'B' in match.group(0).upper():
                        return size
                    elif 'million' in match.group(0).lower():
                        return size / 1000
    
    return None

def format_fun_size(bytes_val):
    """Format bytes with auto-scaling units for fun (MB -> GB -> TB -> PB -> EB -> ZB -> YB)."""
    if bytes_val <= 0:
        return "Unknown"
    
    # Define units in order
    units = ['MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
    
    # Start with MB (bytes / 1024^2)
    size = bytes_val / (1024 ** 2)
    unit_index = 0
    
    # Scale up while over 1000 and we have more units
    while size >= 1000 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    # Format with appropriate precision
    if size >= 100:
        return f"{size:.0f} {units[unit_index]}"
    elif size >= 10:
        return f"{size:.1f} {units[unit_index]}"
    else:
        return f"{size:.2f} {units[unit_index]}"

def calculate_ram_usage(file_size_bytes):
    """Calculate RAM usage based on actual GGUF file size."""
    if file_size_bytes <= 0:
        return "Unknown"
    
    # GGUF models typically need 1.5-2x the file size in RAM
    # Use 1.8x as a good estimate for most models
    ram_needed_bytes = file_size_bytes * 1.8
    
    return format_fun_size(ram_needed_bytes)

@app.route('/api/models/download', methods=['POST'])
def download_model():
    """Download a GGUF model from HuggingFace."""
    global download_status
    
    data = request.json
    model_id = data.get('model_id')
    custom_path = data.get('download_path', '~/models')
    
    if not model_id:
        return jsonify({'error': 'Model ID is required'}), 400
    
    if download_status['downloading']:
        return jsonify({'error': 'Another download is in progress'}), 400
    
    try:
        download_status['downloading'] = True
        download_status['model_id'] = model_id
        download_status['progress'] = 0
        download_status['error'] = None
        
        def download_thread():
            try:
                # Handle path expansion - only expand if it starts with ~
                if custom_path.startswith('~'):
                    models_dir = os.path.expanduser(custom_path)
                else:
                    # Use path as-is for absolute paths
                    models_dir = custom_path
                
                # Security check - warn about dangerous paths
                dangerous_paths = [
                    '/', '/System', '/Library', '/usr', '/bin', '/sbin', '/etc',
                    '/var', '/tmp', '/private', '/dev', '/proc'
                ]
                
                is_dangerous = False
                for dangerous in dangerous_paths:
                    if models_dir == dangerous or models_dir.startswith(dangerous + '/'):
                        is_dangerous = True
                        break
                
                if is_dangerous:
                    download_status['error'] = f"⚠️  Dangerous path detected: {models_dir}. This could harm your system. Please choose a safer location like ~/models, ~/Downloads, or an external drive."
                    download_status['downloading'] = False
                    return
                
                # Create directory if it doesn't exist
                os.makedirs(models_dir, exist_ok=True)
                print(f"Models directory: {models_dir}")
                
                # Additional warning for system-adjacent directories
                system_adjacent = ['/Applications', '/Users', '/opt']
                for adjacent in system_adjacent:
                    if models_dir.startswith(adjacent + '/') and models_dir != adjacent:
                        print(f"⚠️  Warning: Downloading to system-adjacent directory: {models_dir}")
                        break
                
                # Find GGUF file in the model repository
                model_info = api.model_info(model_id)
                gguf_files = []
                
                # Check all files for .gguf extension
                for file in model_info.siblings:
                    if file.rfilename.endswith('.gguf'):
                        gguf_files.append(file)
                
                print(f"Found {len(gguf_files)} GGUF files")
                
                if not gguf_files:
                    download_status['error'] = 'No GGUF files found in this model'
                    download_status['downloading'] = False
                    return
                
                # Choose the best GGUF file (prefer Q4_K_M or smallest)
                best_file = None
                priority_patterns = ['Q4_K_M.gguf', 'Q4_K_S.gguf', 'Q4_0.gguf', 'Q5_K_M.gguf']
                
                for pattern in priority_patterns:
                    for file in gguf_files:
                        if pattern in file.rfilename:
                            best_file = file
                            break
                    if best_file:
                        break
                
                if not best_file:
                    best_file = gguf_files[0]  # Fallback to first file
                
                print(f"Selected file: {best_file.rfilename}")
                download_status['progress'] = 5
                
                # Get actual file size for RAM calculation
                file_size_bytes = best_file.size if hasattr(best_file, 'size') and best_file.size else 0
                
                # Calculate RAM usage based on actual file size
                # GGUF files are typically compressed, so RAM usage ≈ file size * 1.5
                if file_size_bytes > 0:
                    ram_usage_gb = (file_size_bytes / (1024**3)) * 1.5
                    download_status['ram_usage'] = f"{ram_usage_gb:.1f} GB"
                else:
                    download_status['ram_usage'] = "Unknown"
                
                # Update download status with RAM info
                download_status['progress'] = 10
                
                # Download the file
                print(f"Starting download of {best_file.rfilename} from {model_id}")
                download_status['progress'] = 20
                
                downloaded_path = hf_hub_download(
                    repo_id=model_id,
                    filename=best_file.rfilename,
                    local_dir=models_dir,
                    local_dir_use_symlinks=False
                )
                
                print(f"Download completed: {downloaded_path}")
                download_status['progress'] = 100
                download_status['downloading'] = False
                download_status['downloaded_path'] = downloaded_path
                
            except Exception as e:
                print(f"Download error: {e}")
                import traceback
                traceback.print_exc()
                download_status['error'] = str(e)
                download_status['downloading'] = False
        
        # Start download in background thread
        thread = threading.Thread(target=download_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({'message': f'Downloading {model_id}...'})
        
    except Exception as e:
        download_status['downloading'] = False
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/download/status', methods=['GET'])
def download_status_endpoint():
    """Get current download status."""
    return jsonify(download_status)

@app.route('/api/models/browse', methods=['POST'])
def browse_directory():
    """Browse for a directory using web-based suggestions with external drive support."""
    try:
        # Get common directories
        home = os.path.expanduser('~')
        common_dirs = [
            os.path.join(home, 'Desktop'),
            os.path.join(home, 'Documents'),
            os.path.join(home, 'Downloads'),
            os.path.join(home, 'models'),
            os.path.join(home, 'AI'),
            os.path.join(home, 'LLM'),
            home
        ]
        
        # Filter existing directories
        existing_dirs = [d for d in common_dirs if os.path.exists(d)]
        
        # Get external drives with high storage
        high_storage_drives = get_high_storage_drives(min_gb=50)
        external_dirs = []
        
        for drive in high_storage_drives:
            if drive['is_external']:
                # Add the external drive root
                external_dirs.append({
                    'path': drive['mount_point'],
                    'name': f"💾 {drive['mount_point'].split('/')[-1] or drive['mount_point']} ({drive['available']} free)",
                    'is_external': True,
                    'available_gb': drive['available_gb']
                })
                
                # Add common subdirectories on external drives
                for subdir in ['models', 'AI', 'LLM', 'Downloads']:
                    subdir_path = os.path.join(drive['mount_point'], subdir)
                    if os.path.exists(subdir_path):
                        external_dirs.append({
                            'path': subdir_path,
                            'name': f"📁 {subdir} on {drive['mount_point'].split('/')[-1] or 'External'}",
                            'is_external': True,
                            'available_gb': drive['available_gb']
                        })
        
        # Check for high storage alert
        high_storage_alert = None
        if high_storage_drives:
            max_storage_drive = max(high_storage_drives, key=lambda x: x.get('available_gb', 0))
            if max_storage_drive.get('available_gb', 0) >= 100:
                high_storage_alert = {
                    'message': f"💾 High storage available: {max_storage_drive['mount_point']} has {max_storage_drive['available']} GB free!",
                    'drive': max_storage_drive
                }
        
        return jsonify({
            'suggestions': existing_dirs,
            'external_drives': external_dirs,
            'high_storage_alert': high_storage_alert,
            'message': 'Choose a directory or enter any path (⚠️ be careful with system directories)',
            'allow_custom_path': True,
            'security_warning': '⚠️ You can now enter any path. Be careful - downloading to system directories could be dangerous!'
        })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/models', methods=['POST'])
def load_model():
    """Load a GGUF model from the specified path."""
    data = request.json
    path = data.get('path')
    n_ctx = data.get('n_ctx', 4096)
    n_gpu_layers = data.get('n_gpu_layers', 0)
    
    if not path:
        return jsonify({'error': 'Model path is required'}), 400
    
    try:
        model_manager.load_model(path, n_ctx, n_gpu_layers)
        return jsonify({
            'success': True,
            'message': f'Model loaded successfully from {path}',
            'n_ctx': n_ctx
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/status', methods=['GET'])
def model_status():
    """Check if a model is loaded."""
    return jsonify({
        'loaded': model_manager.is_loaded(),
        'model_path': model_manager.model_path
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Generate a chat completion."""
    if not model_manager.is_loaded():
        return jsonify({'error': 'No model loaded'}), 400
    
    data = request.json
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 512)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    system_prompt = data.get('system_prompt')
    
    if not messages:
        return jsonify({'error': 'Messages are required'}), 400
    
    # Add system prompt if provided or default if not present
    if system_prompt:
        messages.insert(0, {'role': 'system', 'content': system_prompt})
    elif not any(m.get('role') == 'system' for m in messages):
        messages.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT})
    
    prompt = format_prompt(messages)
    
    try:
        response = model_manager.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["User:", "System:"]
        )
        return jsonify({
            'response': response.strip(),
            'model': model_manager.model_path
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Generate a streaming chat completion."""
    if not model_manager.is_loaded():
        return jsonify({'error': 'No model loaded'}), 400
    
    data = request.json
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 512)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    system_prompt = data.get('system_prompt')
    
    if not messages:
        return jsonify({'error': 'Messages are required'}), 400
    
    # Add system prompt if provided or default if not present
    if system_prompt:
        messages.insert(0, {'role': 'system', 'content': system_prompt})
    elif not any(m.get('role') == 'system' for m in messages):
        messages.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT})
    
    prompt = format_prompt(messages)
    
    def generate():
        try:
            for chunk in model_manager.generate_stream(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["User:", "System:"]
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/models/scan', methods=['POST'])
def scan_custom_path():
    """Scan a specific custom path for GGUF models."""
    try:
        data = request.json
        custom_path = data.get('path', '')
        
        if not custom_path:
            return jsonify({'error': 'No path provided'}), 400
        
        # Handle path expansion - only expand if it starts with ~
        if custom_path.startswith('~'):
            expanded_path = os.path.expanduser(custom_path)
        else:
            # Use path as-is for absolute paths
            expanded_path = custom_path
        
        if not os.path.exists(expanded_path):
            return jsonify({'error': f'Path does not exist: {expanded_path}'}), 404
        
        if not os.path.isdir(expanded_path):
            return jsonify({'error': f'Path is not a directory: {expanded_path}'}), 400
        
        models = []
        for file in os.listdir(expanded_path):
            if file.endswith('.gguf'):
                full_path = os.path.join(expanded_path, file)
                try:
                    stat = os.stat(full_path)
                    models.append({
                        'name': file,
                        'path': full_path,
                        'size': stat.st_size
                    })
                except:
                    continue
        
        # Sort by size (largest first)
        models.sort(key=lambda x: x['size'], reverse=True)
        return jsonify({'models': models, 'scanned_path': expanded_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/reset', methods=['POST'])
def reset_model_context():
    """Reset the model context to clear sequence position issues."""
    try:
        model_manager.reset_context()
        return jsonify({'message': 'Model context reset successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/available', methods=['GET'])
def list_available_models():
    """List all available GGUF models in default directories."""
    try:
        # Simplified default paths
        default_paths = [
            '.',  # Current directory
            './models',
            os.path.expanduser('~/models'),
        ]
        
        models = []
        seen_paths = set()  # Avoid duplicates
        
        for path in default_paths:
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.endswith('.gguf'):
                        full_path = os.path.join(path, file)
                        if full_path not in seen_paths:
                            try:
                                stat = os.stat(full_path)
                                models.append({
                                    'name': file,
                                    'path': full_path,
                                    'size': stat.st_size
                                })
                                seen_paths.add(full_path)
                            except:
                                continue
        
        # Sort by size (largest first)
        models.sort(key=lambda x: x['size'], reverse=True)
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6768, debug=True)
