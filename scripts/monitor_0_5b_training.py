"""Monitors the Kaggle Friday-0.5B-Ultimate training run, downloads the GGUF model,
and automatically registers it in Ollama once complete.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

KERNEL_SLUG = "thisisdarkstar/friday-0-5b-ultimate"
ROOT_DIR = Path("D:/Projects/desktop-llm")
OUTPUT_DIR = ROOT_DIR / "models" / "friday-0.5b-ultimate"
GGUF_DEST = ROOT_DIR / "models" / "friday-0.5b-ultimate-q8.gguf"
KAGGLE_BIN = ROOT_DIR / ".venv" / "Scripts" / "kaggle.exe"

os.environ["KAGGLE_API_TOKEN"] = os.environ.get(
    "KAGGLE_API_TOKEN", "KGAT_638244114569e1e1c226299332ed59e8"
)


def get_status():
    cmd = [str(KAGGLE_BIN), "kernels", "status", KERNEL_SLUG]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.stdout.strip()


def download_output():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Downloading output artifacts to {OUTPUT_DIR}...")
    cmd = [
        str(KAGGLE_BIN),
        "kernels",
        "output",
        KERNEL_SLUG,
        "-p",
        str(OUTPUT_DIR),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)


def deploy_to_ollama():
    found_ggufs = list(OUTPUT_DIR.glob("*.gguf"))
    if not found_ggufs:
        print("[-] No GGUF files found in output directory!")
        return False

    src_gguf = found_ggufs[0]
    import shutil
    shutil.copy(str(src_gguf), str(GGUF_DEST))
    print(f"[+] Staged GGUF to: {GGUF_DEST} ({GGUF_DEST.stat().st_size / (1024**2):.1f} MB)")

    modelfile_path = ROOT_DIR / "Modelfile.0.5b-ultimate"
    gguf_str = str(GGUF_DEST).replace('\\', '/')
    modelfile_content = f"""FROM {gguf_str}

PARAMETER temperature 0.0
PARAMETER top_p 0.95
PARAMETER top_k 1
PARAMETER num_predict 128
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

SYSTEM \"\"\"You are DesktopIntent, the real-time neural router for FRIDAY AI. For every user input, you output ONLY a valid JSON array of tool calls matching FRIDAY's MCP tool bridge. Never generate conversational monologue, explanation, or thinking tokens. If the command requires complex coding or open-ended reasoning, invoke delegate_to_reasoning_model. If the input is conversational or greetings, invoke chat_response.\"\"\"

TEMPLATE \"\"\"<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"
"""
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    print(f"[*] Registering in Ollama as friday:0.5b-ultimate...")
    res = subprocess.run(["ollama", "create", "friday:0.5b-ultimate", "-f", str(modelfile_path)], capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

    print("[🎉] friday:0.5b-ultimate is live and ready in Ollama!")
    return True


def main():
    status = get_status()
    print(f"[+] Status of {KERNEL_SLUG}: {status}")

    if "COMPLETE" in status:
        download_output()
        deploy_to_ollama()
    elif "RUNNING" in status or "QUEUED" in status:
        print("[⏳] Kernel is currently training on Kaggle GPU.")
        print("     Estimated duration: ~15 to 25 minutes.")
    else:
        print(f"[*] Current status: {status}")


if __name__ == "__main__":
    main()
