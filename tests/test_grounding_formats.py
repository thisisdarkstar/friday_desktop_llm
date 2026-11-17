import base64
import json
import urllib.request
from pathlib import Path

IMG = Path(r"C:\Users\demo2\.gemini\antigravity\brain\d6253261-aafc-414d-9cf3-26bae1b118c7\.user_uploaded\media_1794823239724_614fc185.png")

with open(IMG, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

prompts = [
    "Locate the text 'View all'. Output only the bounding box coordinates.",
    "Give the bounding box of the View all button.",
    "What are the coordinates (x, y) of the 'View all' link?",
    "Detect View all",
]

for p in prompts:
    print(f"\n--- Testing prompt: {p} ---", flush=True)
    payload = {
        "model": "qwen3-vl:2b",
        "messages": [
            {"role": "user", "content": p, "images": [b64]}
        ],
        "stream": False,
        "options": {"temperature": 0.0}
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    msg = res.get("message", {}).get("content", "")
    print(f"Response:\n{repr(msg)}\n---\n{msg}", flush=True)
