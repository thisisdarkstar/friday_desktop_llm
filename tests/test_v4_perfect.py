"""Comprehensive 10/10 Verification Suite for DesktopIntent-v4 / friday:v4-perfect.

Evaluates all 5 pillars + standard 55 tools + cloud delegation:
1. Dynamic Runtime Tool Schemas
2. Acoustic Noise & Disfluency Hardening
3. Stateful Anaphora & Coreference Resolution
4. Built-in Micro-Personality & Chit-Chat
5. Calibrated Ambiguity Gating
6. Standard Core Desktop Tools
7. Cloud Reasoning Boundary Delegation
"""

import json
import time
import requests

ENDPOINT = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "friday:v4-perfect"

BASE_SYSTEM_PROMPT = (
    "You are DesktopIntent, the real-time neural router for FRIDAY AI. For every user input, "
    "you output ONLY a valid JSON array of tool calls matching FRIDAY's MCP tool bridge. "
    "Never generate conversational monologue, explanation, or thinking tokens. "
    "If the command requires complex coding or open-ended reasoning, invoke delegate_to_reasoning_model. "
    "If the input is conversational or greetings, invoke chat_response."
)

DYNAMIC_SYSTEM_PROMPT = (
    f"{BASE_SYSTEM_PROMPT}\n\n"
    "Runtime Tools:\n"
    "[\n"
    '  {"name": "obs_record", "description": "Control OBS screen recording", "parameters": ["action"]},\n'
    '  {"name": "spotify_search", "description": "Search tracks on Spotify", "parameters": ["query"]},\n'
    '  {"name": "docker_action", "description": "Manage docker containers", "parameters": ["action", "container"]}\n'
    "]"
)

TEST_CASES = [
    # --- Pillar 1: Dynamic Runtime Tools ---
    {
        "category": "1. Dynamic Runtime Tools",
        "prompt": "start recording my screen in obs",
        "system": DYNAMIC_SYSTEM_PROMPT,
        "expected_tool": "obs_record",
    },
    {
        "category": "1. Dynamic Runtime Tools",
        "prompt": "search for Daft Punk on Spotify",
        "system": DYNAMIC_SYSTEM_PROMPT,
        "expected_tool": "spotify_search",
    },
    {
        "category": "1. Dynamic Runtime Tools",
        "prompt": "restart redis container",
        "system": DYNAMIC_SYSTEM_PROMPT,
        "expected_tool": "docker_action",
    },
    {
        "category": "1. Dynamic Runtime Tools (Baseline Coexistence)",
        "prompt": "mute audio",
        "system": DYNAMIC_SYSTEM_PROMPT,
        "expected_tool": "mute_audio",
    },

    # --- Pillar 2: Acoustic Noise & Disfluencies ---
    {
        "category": "2. Acoustic Noise & Disfluencies",
        "prompt": "open note— wait actually open chrome",
        "system": BASE_SYSTEM_PROMPT,
        "expected_tool": "launch_app",
        "expected_arg": ("name", "chrome"),
    },
    {
        "category": "2. Acoustic Noise & Disfluencies",
        "prompt": "open calculator wait no open notepad",
        "system": BASE_SYSTEM_PROMPT,
        "expected_tool": "launch_app",
        "expected_arg": ("name", "notepad"),
    },
    {
        "category": "2. Acoustic Noise & Disfluencies",
        "prompt": "the current time is 140",
        "system": BASE_SYSTEM_PROMPT,
        "expected_tool": "get_time",
    },
    {
        "category": "2. Acoustic Noise & Disfluencies",
        "prompt": "uh friday can you like turn up the sound",
        "system": BASE_SYSTEM_PROMPT,
        "expected_tool": "change_volume",
    },
    {
        "category": "2. Acoustic Noise & Disfluencies",
        "prompt": "open note pad",
        "system": BASE_SYSTEM_PROMPT,
        "expected_tool": "launch_app",
        "expected_arg": ("name", "notepad"),
    },

    # --- Pillar 3: Stateful Anaphora & Coreference ---
    {
        "category": "3. Stateful Anaphora",
        "prompt": '[Prior Action: {"tool": "launch_app", "args": {"name": "notepad"}}]\nClose it',
        "system": BASE_SYSTEM_PROMPT,
        "expected_tool": "close_app",
        "expected_arg": ("name", "notepad"),
    },
    {
        "category": "3. Stateful Anaphora",
        "prompt": '[Prior Action: {"tool": "launch_app", "args": {"name": "chrome"}}]\nMinimize that',
        "system": BASE_SYSTEM_PROMPT,
        "expected_tool": "manage_window",
        "expected_arg": ("action", "minimize"),
    },
    {
        "category": "3. Stateful Anaphora",
        "prompt": '[Prior Action: {"tool": "set_volume", "args": {"level": 50}}]\nTurn it up',
        "system": BASE_SYSTEM_PROMPT,
        "expected_tool": "change_volume",
    },
    {
        "category": "3. Stateful Anaphora",
        "prompt": '[Prior Action: {"tool": "take_screenshot", "args": {}}]\nDo that again',
        "system": BASE_SYSTEM_PROMPT,
        "expected_tool": "take_screenshot",
    },

    # --- Pillar 4: Built-in Micro-Personality Chit-Chat ---
    {
        "category": "4. Micro-Personality Chit-Chat",
        "prompt": "hello friday",
        "system": BASE_SYSTEM_PROMPT,
        "expected_tool": "chat_response",
    },
    {
        "category": "4. Micro-Personality Chit-Chat",
        "prompt": "who are you",
        "system": BASE_SYSTEM_PROMPT,
        "expected_tool": "chat_response",
    },
    {
        "category": "4. Micro-Personality Chit-Chat",
        "prompt": "thank you friday",
        "system": BASE_SYSTEM_PROMPT,
        "expected_tool": "chat_response",
    },

    # --- Pillar 5: Calibrated Ambiguity Gating ---
    {
        "category": "5. Ambiguity Gating",
        "prompt": "delete the project",
        "system": BASE_SYSTEM_PROMPT,
        "expected_tool": "clarification_required",
    },
    {
        "category": "5. Ambiguity Gating",
        "prompt": "send an email",
        "system": BASE_SYSTEM_PROMPT,
        "expected_tool": "clarification_required",
    },
    {
        "category": "5. Ambiguity Gating",
        "prompt": "make me coffee",
        "system": BASE_SYSTEM_PROMPT,
        "expected_tool": "chat_response",
    },

    # --- Standard Core Desktop Tools ---
    {
        "category": "6. Standard Core Tools",
        "prompt": "open notepad",
        "system": BASE_SYSTEM_PROMPT,
        "expected_tool": "launch_app",
        "expected_arg": ("name", "notepad"),
    },
    {
        "category": "6. Standard Core Tools",
        "prompt": "dim display to 30",
        "system": BASE_SYSTEM_PROMPT,
        "expected_tool": "set_brightness",
    },
    {
        "category": "6. Standard Core Tools",
        "prompt": "lock the pc",
        "system": BASE_SYSTEM_PROMPT,
        "expected_tool": "lock_screen",
    },

    # --- Cloud Delegation Boundary ---
    {
        "category": "7. Cloud Delegation Boundary",
        "prompt": "write a complete react typescript dashboard with charts and authentication",
        "system": BASE_SYSTEM_PROMPT,
        "expected_tool": "delegate_to_reasoning_model",
    },
]


def run_benchmark():
    print(f"\n=======================================================")
    print(f"[*] RUNNING DESKTOPINTENT-v4 (10/10) MASTER BENCHMARK")
    print(f"[*] Target Model: {MODEL_NAME}")
    print(f"=======================================================\n")

    passed = 0
    total = len(TEST_CASES)
    latencies = []

    for idx, tc in enumerate(TEST_CASES, 1):
        cat = tc["category"]
        prompt = tc["prompt"]
        system = tc.get("system", BASE_SYSTEM_PROMPT)
        exp_tool = tc["expected_tool"]
        exp_arg = tc.get("expected_arg")

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 0.95,
                "num_predict": 96,
                "stop": ["<|im_end|>", "]"],
            },
        }

        t0 = time.perf_counter()
        try:
            resp = requests.post(ENDPOINT, json=payload, timeout=5.0)
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            raw = resp.json().get("response", "").strip()
            if raw.startswith("[") and not raw.endswith("]"):
                raw += "]"

            tool_calls = json.loads(raw)
            actual_tools = [c.get("tool") for c in tool_calls]
            
            tool_match = exp_tool in actual_tools
            arg_match = True
            if exp_arg and tool_match:
                matching_call = next(c for c in tool_calls if c.get("tool") == exp_tool)
                k, v = exp_arg
                arg_match = matching_call.get("args", {}).get(k) == v

            success = tool_match and arg_match
            if success:
                passed += 1
                status = "PASS"
            else:
                status = "FAIL"

            print(f"[{status}] #{idx:02d} [{cat}]")
            print(f"       Prompt : {prompt!r}")
            print(f"       Output : {raw} ({latency_ms:.1f}ms)")
            if not success:
                print(f"       Wanted : tool={exp_tool!r}, arg={exp_arg}")
            print()

        except Exception as e:
            print(f"[FAIL] #{idx:02d} [{cat}] Error: {e}\n")

    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    accuracy = (passed / total) * 100

    print("=======================================================")
    print(f"BENCHMARK RESULTS: {passed}/{total} Passed ({accuracy:.1f}%)")
    print(f"Average Response Latency: {avg_lat:.1f}ms")
    print("=======================================================")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        MODEL_NAME = sys.argv[1]
    run_benchmark()
