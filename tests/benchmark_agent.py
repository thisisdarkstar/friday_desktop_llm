"""Live benchmark script for DesktopIntent."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime.agent import run_command

tests = [
    "hi",
    "mute audio",
    "open whatsapp",
    "dim screen to 40% and volume to 50",
    "calculate 150 * 8",
    "check cpu usage and battery",
]

print("=" * 60)
print("BENCHMARKING DESKTOPINTENT ON RTX 2050")
print("=" * 60)

for t in tests:
    res = run_command(t)
    total = res.get("total_ms", 0)
    inf = res.get("inference_ms", 0)
    raw = res.get("raw_tool_calls", "")
    acts = res.get("results", [])

    print(f"\n[USER] > \"{t}\"")
    print(f"⚡ Latency: {total}ms  (Inference: {inf}ms)")
    print(f"📦 Model Output: {raw}")
    print("🛠️ Executed Actions:")
    for a in acts:
        tool = a.get("tool", "unknown")
        result = a.get("result", a.get("message", ""))
        print(f"   • {tool} => {result}")
    print("-" * 60)
