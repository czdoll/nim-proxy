from flask import Flask, request, jsonify, Response
import requests
import os

app = Flask(__name__)

# Get NVIDIA API key from environment variable
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', '')
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        # Get the request data
        data = request.get_json()
        
        # Extract parameters
        messages = data.get('messages', [])
        model = data.get('model', 'meta/llama-3.1-8b-instruct')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 1024)
        stream = data.get('stream', False)
        
        # Prepare NVIDIA NIM request
        nim_payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # Make request to NVIDIA NIM
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        if stream:
            # Handle streaming response
            nim_response = requests.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers=headers,
                json=nim_payload,
                stream=True
            )
            
            def generate():
                for line in nim_response.iter_lines():
                    if line:
                        yield line + b'\n'
            
            return Response(generate(), mimetype='text/event-stream')
        else:
            # Handle non-streaming response
            nim_response = requests.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers=headers,
                json=nim_payload
            )
            
            return jsonify(nim_response.json()), nim_response.status_code
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models"""
    try:
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}"
        }
        
        response = requests.get(
            f"{NVIDIA_BASE_URL}/models",
            headers=headers
        )
        
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
