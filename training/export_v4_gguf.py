"""Exports DesktopIntent-v4 to Q8_0 GGUF and registers in Ollama with baked-in system prompt."""

import os
import subprocess
import sys
from pathlib import Path

MODEL_DIR = Path("D:/Projects/desktop-llm/models/desktop-intent-merged-v4")
GGUF_OUT = Path("D:/Projects/desktop-llm/models/desktop-intent-q8-v4.gguf")
CONVERTER = Path("D:/Projects/desktop-llm/training/llama_cpp/convert_hf_to_gguf.py")
PYTHON_EXE = Path("D:/Projects/desktop-llm/.venv/Scripts/python.exe")
MODELFILE_PATH = Path("D:/Projects/desktop-llm/Modelfile.v4")

SYSTEM_PROMPT = (
    "You are DesktopIntent, the real-time neural router for FRIDAY AI. For every user input, "
    "you output ONLY a valid JSON array of tool calls matching FRIDAY's MCP tool bridge. "
    "Never generate conversational monologue, explanation, or thinking tokens. "
    "If the command requires complex coding or open-ended reasoning, invoke delegate_to_reasoning_model. "
    "If the input is conversational or greetings, invoke chat_response."
)

MODELFILE_CONTENT = f"""FROM {GGUF_OUT.resolve().as_posix()}

PARAMETER temperature 0.0
PARAMETER top_p 0.95
PARAMETER top_k 1
PARAMETER num_predict 128
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"

TEMPLATE \"\"\"<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"
"""


def main():
    print(f"[*] Creating Modelfile with baked-in system prompt at {MODELFILE_PATH}...")
    with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
        f.write(MODELFILE_CONTENT)

    print(f"[*] Registering 'friday:v4-perfect' in Ollama...")
    res_ollama = subprocess.run(
        ["ollama", "create", "friday:v4-perfect", "-f", str(MODELFILE_PATH)],
        capture_output=True,
        text=True,
    )
    if res_ollama.returncode != 0:
        print(f"[-] Ollama creation failed:\n{res_ollama.stderr}")
        sys.exit(res_ollama.returncode)
    print(res_ollama.stdout)

    print(f"[*] Tagging 'friday:latest' and 'desktop-intent:latest'...")
    subprocess.run(["ollama", "cp", "friday:v4-perfect", "friday:latest"])
    subprocess.run(["ollama", "cp", "friday:v4-perfect", "desktop-intent:latest"])

    print("[+] All models registered successfully in Ollama with baked-in system prompt!")


if __name__ == "__main__":
    main()
