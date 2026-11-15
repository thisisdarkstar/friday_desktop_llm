"""Monitors the Kaggle 1.5B training run and downloads the GGUF model once complete."""

import os
import subprocess
import sys
import time
from pathlib import Path

KERNEL_SLUG = "thisisdarkstar/friday-1-5b-desktop-intent-training"
OUTPUT_DIR = Path("D:/Projects/desktop-llm/models/friday-1.5b")
KAGGLE_BIN = Path("D:/Projects/desktop-llm/.venv/Scripts/kaggle.exe")

# Ensure token is set in environment
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


def main():
    status = get_status()
    print(f"[+] Current Status: {status}")
    if "COMPLETE" in status:
        download_output()
        print(f"[🎉] Done! Model files saved in {OUTPUT_DIR}")
    elif "RUNNING" in status:
        print("[⏳] Notebook is actively training on Kaggle GPU.")
        print("     Run this script anytime to check progress or auto-download.")
    else:
        print(f"[*] Status: {status}")


if __name__ == "__main__":
    main()
