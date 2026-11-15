"""Comprehensive Evaluation Benchmark for DesktopIntent-LLM.

Tests accuracy, JSON validity, multi-intent decomposition, and inference speed.
Can run against either:
1. The merged PyTorch / Transformers model directly (`./models/desktop-intent-merged`)
2. The Ollama endpoint (`http://localhost:11434/api/generate`)
"""

import json
import time
import sys
from pathlib import Path

# Test suite covering all 6 functional categories
BENCHMARK_PROMPTS = [
    # 1. Single OS Actions
    {"prompt": "mute audio", "expected_tool": "mute_audio"},
    {"prompt": "volume 75", "expected_tool": "set_volume"},
    {"prompt": "turn brightness to 80%", "expected_tool": "set_brightness"},
    {"prompt": "lock the screen", "expected_tool": "set_power_state"},
    {"prompt": "what time is it right now", "expected_tool": "get_time_or_date"},
    {"prompt": "check cpu and memory stats", "expected_tool": "get_system_stats"},

    # 2. Window Management & Presets
    {"prompt": "minimize Chrome", "expected_tool": "manage_window"},
    {"prompt": "snap VS Code to the left half", "expected_tool": "snap_window"},
    {"prompt": "switch to research workspace preset", "expected_tool": "set_workspace_preset"},

    # 3. Multi-Intent Compound Commands
    {
        "prompt": "mute the sound and minimize WhatsApp",
        "expected_tools": ["mute_audio", "manage_window"],
    },
    {
        "prompt": "open Brave, set brightness to 60, and set a 15 minute timer",
        "expected_tools": ["launch_app", "set_brightness", "create_timer"],
    },
    {
        "prompt": "look up quantum computing on youtube and snap browser to the right",
        "expected_tools": ["search_in_browser", "snap_window"],
    },

    # 4. Voice Noise & Polite Wrappers
    {
        "prompt": "hey friday could you please uh turn down the volume by 20",
        "expected_tool": "change_volume",
    },
    {
        "prompt": "would you kindly take a quick screenshot of the screen",
        "expected_tool": "take_screenshot",
    },

    # 5. Productivity & Calculations
    {"prompt": "what is 450 * 12 + 85", "expected_tool": "calculate"},
    {"prompt": "remind me in 30 minutes to drink water", "expected_tool": "create_reminder"},

    # 6. Delegation to Cloud Reasoning Model
    {
        "prompt": "write a python script using asyncio to scrape web pages concurrently",
        "expected_tool": "delegate_to_reasoning_model",
    },
    {"prompt": "explain quantum entanglement in simple terms", "expected_tool": "delegate_to_reasoning_model"},

    # 7. Conversational Greeting
    {"prompt": "good morning friday how are you", "expected_tool": "chat_response"},
]


def test_ollama(model_name: str = "friday:latest"):
    import requests

    print(f"\n==========================================")
    print(f"[*] Benchmarking DesktopIntent via Ollama: {model_name}")
    print(f"==========================================\n")

    total = len(BENCHMARK_PROMPTS)
    passed = 0
    total_time_ms = 0.0

    for i, test in enumerate(BENCHMARK_PROMPTS, 1):
        prompt = test["prompt"]
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 0.95,
                "num_predict": 128,
                "stop": ["<|im_end|>", "]"],
            },
        }

        start = time.perf_counter()
        try:
            resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=10)
            elapsed = (time.perf_counter() - start) * 1000
            total_time_ms += elapsed

            raw = resp.json().get("response", "").strip()
            if raw.startswith("[") and not raw.endswith("]"):
                raw += "]"

            parsed = json.loads(raw)
            tools_called = [c.get("tool") for c in parsed]

            # Verification
            ok = False
            if "expected_tool" in test:
                ok = test["expected_tool"] in tools_called
            elif "expected_tools" in test:
                ok = all(t in tools_called for t in test["expected_tools"])

            status = "✅ PASS" if ok else "❌ FAIL"
            if ok:
                passed += 1

            print(f"[{i:02d}/{total:02d}] {status} ({elapsed:.1f}ms) | Input: {prompt!r}")
            print(f"         Output: {raw}")

        except Exception as e:
            print(f"[{i:02d}/{total:02d}] ❌ ERROR: {e} | Input: {prompt!r}")

    avg_ms = total_time_ms / total if total else 0
    acc = (passed / total) * 100
    print(f"\n==========================================")
    print(f"[*] Results: {passed}/{total} Passed ({acc:.1f}% Accuracy)")
    print(f"[*] Average Latency: {avg_ms:.1f}ms per query")
    print(f"==========================================\n")


def test_transformers(model_dir: str = "./models/desktop-intent-merged"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n==========================================")
    print(f"[*] Benchmarking Merged Model via PyTorch: {model_dir}")
    print(f"==========================================\n")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.eval()

    system_prompt = (
        "You are DesktopIntent, the real-time neural router for FRIDAY AI. "
        "For every user input, you output ONLY a valid JSON array of tool calls matching FRIDAY's MCP tool bridge. "
        "Never generate conversational monologue, explanation, or thinking tokens. "
        "If the command requires complex coding or open-ended reasoning, invoke delegate_to_reasoning_model. "
        "If the input is conversational or greetings, invoke chat_response."
    )

    total = len(BENCHMARK_PROMPTS)
    passed = 0
    total_time_ms = 0.0

    for i, test in enumerate(BENCHMARK_PROMPTS, 1):
        prompt = test["prompt"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = (time.perf_counter() - start) * 1000
        total_time_ms += elapsed

        gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        raw = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        try:
            parsed = json.loads(raw)
            tools_called = [c.get("tool") for c in parsed]
            ok = False
            if "expected_tool" in test:
                ok = test["expected_tool"] in tools_called
            elif "expected_tools" in test:
                ok = all(t in tools_called for t in test["expected_tools"])
        except Exception:
            ok = False

        status = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            passed += 1

        print(f"[{i:02d}/{total:02d}] {status} ({elapsed:.1f}ms) | Input: {prompt!r}")
        print(f"         Output: {raw}")

    avg_ms = total_time_ms / total if total else 0
    acc = (passed / total) * 100
    print(f"\n==========================================")
    print(f"[*] Results: {passed}/{total} Passed ({acc:.1f}% Accuracy)")
    print(f"[*] Average Latency: {avg_ms:.1f}ms per query")
    print(f"==========================================\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ollama":
        test_ollama()
    else:
        test_transformers()
