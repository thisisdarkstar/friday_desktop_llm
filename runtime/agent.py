"""DesktopIntent Runtime Agent.
Processes user voice/text commands -> queries DesktopIntent-LLM in Ollama -> executes native Windows tools.
"""

import time
import requests
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.dispatcher import dispatch

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "desktop-intent"

SYSTEM_PROMPT = (
    "You are DesktopIntent, a deterministic desktop system control brain. "
    "For every user input, you output ONLY a valid JSON array of tool calls. "
    "Never generate explanation, reasoning, or conversational monologue. "
    "If the input is conversational, greetings, or out-of-domain, invoke the chat_response tool."
)


def run_command(command: str, model: str = DEFAULT_MODEL) -> dict:
    """Send user command to Ollama DesktopIntent brain and execute tools."""
    start_time = time.perf_counter()

    payload = {
        "model": model,
        "prompt": command,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 0.95,
            "num_predict": 128,
            "stop": ["<|im_end|>", "]"],
        },
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
        resp.raise_for_status()
        res_data = resp.json()
        raw_output = res_data.get("response", "").strip()
    except Exception as e:
        return {
            "command": command,
            "status": "error",
            "message": f"Could not reach Ollama model '{model}': {str(e)}",
        }

    inference_ms = (time.perf_counter() - start_time) * 1000

    # Ensure closing bracket if stopped at ']'
    if raw_output.startswith("[") and not raw_output.endswith("]"):
        raw_output += "]"

    # Dispatch tool execution
    exec_start = time.perf_counter()
    action_results = dispatch(raw_output)
    exec_ms = (time.perf_counter() - exec_start) * 1000

    return {
        "command": command,
        "inference_ms": round(inference_ms, 1),
        "execution_ms": round(exec_ms, 1),
        "total_ms": round(inference_ms + exec_ms, 1),
        "raw_tool_calls": raw_output,
        "results": action_results,
    }


def interactive_cli(model: str = DEFAULT_MODEL):
    print("=" * 60)
    print(f"🚀 DesktopIntent Agent Active (Model: {model})")
    print("Type any desktop command (e.g. 'mute sound', 'dim screen to 40%')")
    print("Type 'exit' or 'quit' to close.")
    print("=" * 60)

    while True:
        try:
            cmd = input("\n[DesktopIntent] > ").strip()
            if not cmd:
                continue
            if cmd.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            result = run_command(cmd, model=model)
            print(f"⚡ Latency: {result.get('total_ms')}ms (Inference: {result.get('inference_ms')}ms | Exec: {result.get('execution_ms')}ms)")
            print(f"📦 Model Output: {result.get('raw_tool_calls')}")
            print("🛠️ Actions:")
            for act in result.get("results", []):
                tool = act.get("tool", "unknown")
                status = act.get("status", act.get("result", {}).get("status", "ok"))
                res = act.get("result", act.get("message", ""))
                print(f"   • [{tool}] -> {res}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    target_model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    interactive_cli(target_model)
