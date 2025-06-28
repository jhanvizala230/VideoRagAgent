import subprocess
import requests


def run_llama_cli(prompt: str) -> str:
    # Adjust path to your llama.cpp executable
    executable_path = "./llama.cpp/build/bin/llama-cli"
    model_path = "models/llama-3.2/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

    command = [executable_path, "-m", model_path, "-p", prompt]

    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
        return output.strip()
    except subprocess.CalledProcessError as e:
        return f"Error running LLaMA CLI: {e.output.strip()}"
    

def run_llama_ollama(prompt: str) -> str:
    model_name = "llama3.2:3b-instruct-q4_K_M"
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        return f"Error contacting Ollama API: {str(e)}"