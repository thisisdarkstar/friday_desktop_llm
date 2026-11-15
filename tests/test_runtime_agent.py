import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime.agent import run_command

tests = [
    "mute audio",
    "what is 250 * 4 + 10",
    "open brave and set volume to 50",
    "write a python script for web scraping",
    "good morning friday",
]

for cmd in tests:
    print(f"\n==========================================")
    print(f"Command: {cmd!r}")
    res = run_command(cmd)
    print(f"Total Latency: {res['total_ms']}ms (Inference: {res['inference_ms']}ms | Exec: {res['execution_ms']}ms)")
    print(f"Raw Tool Call: {res['raw_tool_calls']}")
    print(f"Dispatched Results: {res['results']}")
