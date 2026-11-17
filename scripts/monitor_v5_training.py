"""Monitors the Kaggle Friday-0.5B-v5-Precision training run, downloads the GGUF model,
and automatically registers it in Ollama once complete.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

KERNEL_SLUG = "thisisdarkstar/friday-0-5b-v5-precision"
ROOT_DIR = Path("D:/Projects/desktop-llm")
OUTPUT_DIR = ROOT_DIR / "models" / "friday-0.5b-v5"
GGUF_DEST = ROOT_DIR / "models" / "friday-0.5b-v5-q8.gguf"
KAGGLE_BIN = ROOT_DIR / ".venv" / "Scripts" / "kaggle.exe"

os.environ["KAGGLE_API_TOKEN"] = os.environ.get(
    "KAGGLE_API_TOKEN", ""
)


def get_status() -> str:
    cmd = [str(KAGGLE_BIN), "kernels", "status", KERNEL_SLUG]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.stdout.strip()


def download_output() -> bool:
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
        print("STDERR:", res.stderr)

    # Check for GGUF file
    gguf_files = list(OUTPUT_DIR.glob("*.gguf"))
    if gguf_files:
        src = gguf_files[0]
        import shutil
        shutil.copy2(src, GGUF_DEST)
        print(f"[+] Successfully copied GGUF to {GGUF_DEST} ({GGUF_DEST.stat().st_size:,} bytes)")
        return True
    return False


def deploy_to_ollama() -> bool:
    if not GGUF_DEST.exists():
        print(f"[-] GGUF destination {GGUF_DEST} does not exist!")
        return False

    modelfile_content = f"""FROM "{GGUF_DEST.as_posix()}"
TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
\"\"\"
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.0
"""
    modelfile_path = ROOT_DIR / "models" / "Modelfile_v5"
    modelfile_path.write_text(modelfile_content, encoding="utf-8")

    print("[*] Creating Ollama model 'friday:0.5b-v5'...")
    res = subprocess.run(["ollama", "create", "friday:0.5b-v5", "-f", str(modelfile_path)], capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
    return res.returncode == 0


def main():
    print("=" * 75)
    print(f"🛰️ MONITORING KAGGLE TRAINING: {KERNEL_SLUG}")
    print("=" * 75)

    last_status = ""
    while True:
        status = get_status()
        if status != last_status:
            print(f"[{time.strftime('%H:%M:%S')}] Status: {status}", flush=True)
            last_status = status

        if "COMPLETE" in status:
            print("\n🎉 Kernel run is COMPLETE! Fetching GGUF artifacts...", flush=True)
            if download_output():
                print("[*] Deploying to Ollama...", flush=True)
                if deploy_to_ollama():
                    print("\n✅ Deployment complete! 'friday:0.5b-v5' is ready for benchmarking.")
            break
        elif "FAILED" in status or "CANCELLED" in status:
            print(f"\n❌ Kernel execution ended with: {status}")
            break

        time.sleep(15)


if __name__ == "__main__":
    main()
