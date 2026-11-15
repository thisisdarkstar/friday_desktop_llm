"""Export script to package DesktopIntent into Ollama with early-stopping parameters."""

import os
import subprocess
from pathlib import Path

MODELFILE_CONTENT = """FROM {model_source}

# Zero temperature for deterministic tool calls
PARAMETER temperature 0.0
PARAMETER top_p 0.95
PARAMETER top_k 1

# Limit max output tokens to prevent runaway generation (tool calls are ~20 tokens)
PARAMETER num_predict 96

# Early-stopping sequences: terminate immediately upon closing array bracket
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

# Qwen2.5 ChatML template
TEMPLATE \"\"\"<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
\"\"\"
"""


def export_to_ollama(model_source: str, model_name: str = "desktop-intent"):
    modelfile_path = Path("./Modelfile")
    content = MODELFILE_CONTENT.format(model_source=model_source)

    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[*] Generated Modelfile targeting '{model_source}'")
    print(f"[*] Creating Ollama model '{model_name}'...")

    cmd = f"ollama create {model_name} -f Modelfile"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if res.returncode == 0:
        print(f"[+] Successfully registered '{model_name}' in Ollama!")
        print(res.stdout)
    else:
        print(f"[-] Failed to register Ollama model: {res.stderr}")


if __name__ == "__main__":
    import sys

    # Can be a GGUF file path, HuggingFace path, or merged directory
    source = sys.argv[1] if len(sys.argv) > 1 else "./models/desktop-intent.gguf"
    export_to_ollama(source)
