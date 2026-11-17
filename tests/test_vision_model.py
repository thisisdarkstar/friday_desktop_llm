"""Test script for qwen3-vl:2b in Ollama on real UI application screenshots."""

import base64
import json
import time
import urllib.request
from pathlib import Path

MODEL_NAME = "qwen3-vl:2b"
OLLAMA_CHAT_ENDPOINT = "http://127.0.0.1:11434/api/chat"

UPLOADED_DIR = Path(r"C:\Users\demo2\.gemini\antigravity\brain\d6253261-aafc-414d-9cf3-26bae1b118c7\.user_uploaded")
IMG_TABLE = UPLOADED_DIR / "media_1794823239724_614fc185.png"   # Ollama models table
IMG_CODE = UPLOADED_DIR / "media_1794808175204_58ebda7a.png"    # Kaggle code + progress bar


def encode_image(img_path: Path) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_vision(img_path: Path, prompt: str) -> dict:
    b64_img = encode_image(img_path)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64_img],
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_CHAT_ENDPOINT,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    lat_ms = (time.perf_counter() - t0) * 1000

    reply = res.get("message", {}).get("content", "")
    return {"reply": reply, "latency_ms": lat_ms, "raw": res}


def main():
    print("=" * 75)
    print(f"👁️ TESTING VISION MODEL: {MODEL_NAME}")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # TEST 1: Table OCR & Visual Detail Extraction
    # -------------------------------------------------------------------------
    print("\n--- TEST 1: Ollama Model Table OCR ---")
    prompt_1 = "Read the table in this image. List each model name along with its size and context window."
    res1 = query_vision(IMG_TABLE, prompt_1)
    print(f"Latency: {res1['latency_ms']:.1f} ms")
    print(f"Response:\n{res1['reply']}")

    # -------------------------------------------------------------------------
    # TEST 2: UI Coordinate Grounding for Clicking
    # -------------------------------------------------------------------------
    print("\n--- TEST 2: UI Coordinate Grounding for Clicking ---")
    prompt_2 = (
        "Locate the 'qwen3-vl:2b' row and the 'View all' link in this UI screenshot. "
        "Provide their normalized coordinates or bounding box so a mouse click can target them."
    )
    res2 = query_vision(IMG_TABLE, prompt_2)
    print(f"Latency: {res2['latency_ms']:.1f} ms")
    print(f"Response:\n{res2['reply']}")

    # -------------------------------------------------------------------------
    # TEST 3: Code & Progress Bar Understanding
    # -------------------------------------------------------------------------
    print("\n--- TEST 3: Kaggle Code & Progress Bar Understanding ---")
    prompt_3 = (
        "In this screenshot, what Python function is running? "
        "How many samples are being trained on, and what does the progress bar show (current step, total steps, epoch)?"
    )
    res3 = query_vision(IMG_CODE, prompt_3)
    print(f"Latency: {res3['latency_ms']:.1f} ms")
    print(f"Response:\n{res3['reply']}")

    print("\n" + "=" * 75)
    print("✅ All vision tests completed!")
    print("=" * 75)


if __name__ == "__main__":
    main()
