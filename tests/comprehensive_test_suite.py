"""Comprehensive Benchmark Suite for DesktopIntent-LLM.

Evaluates:
1. Single Intent (Generic & FRIDAY system/app commands)
2. Multi-Intent Compound Chains (2-step and 3-step pipelines)
3. Conversational Single & Multi-Turn Queries
4. Mixed Conversational + Actions (Greetings + compound commands)
5. Real FRIDAY Log Scenarios, Speech Noise & Cloud Reasoning Delegation
"""

import json
import sys
import time
import requests
from typing import Dict, List, Any

OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "friday"

session = requests.Session()

TEST_CATEGORIES = {
    "1. Single Intent (Generic & FRIDAY)": [
        {
            "query": "mute audio",
            "expected": lambda calls: any(c.get("tool") == "mute_audio" for c in calls),
            "desc": "Audio mute",
        },
        {
            "query": "volume 85",
            "expected": lambda calls: any(c.get("tool") == "set_volume" and c.get("args", {}).get("level") == 85 for c in calls),
            "desc": "Absolute volume level",
        },
        {
            "query": "turn down the volume by 15",
            "expected": lambda calls: any(c.get("tool") == "change_volume" and c.get("args", {}).get("delta") in (-15, 15) for c in calls),
            "desc": "Relative volume delta",
        },
        {
            "query": "set screen brightness to 70%",
            "expected": lambda calls: any(c.get("tool") == "set_brightness" and c.get("args", {}).get("level") == 70 for c in calls),
            "desc": "Display brightness level",
        },
        {
            "query": "open Brave",
            "expected": lambda calls: any(c.get("tool") == "launch_app" and "brave" in str(c.get("args", {})).lower() for c in calls),
            "desc": "App launch",
        },
        {
            "query": "minimize Spotify",
            "expected": lambda calls: any(c.get("tool") in ("manage_window", "minimize_window") and (c.get("args", {}).get("action") == "minimize" or "spotify" in str(c.get("args", {})).lower()) for c in calls),
            "desc": "Window minimization",
        },
        {
            "query": "snap VS Code to the left",
            "expected": lambda calls: any(c.get("tool") == "snap_window" and c.get("args", {}).get("position") == "left" for c in calls),
            "desc": "Window snapping",
        },
        {
            "query": "switch to coding workspace layout",
            "expected": lambda calls: any(c.get("tool") in ("set_workspace_preset", "arrange_workspace_preset") and "coding" in str(c.get("args", {})).lower() for c in calls),
            "desc": "Workspace preset layout",
        },
        {
            "query": "what is 75 * 14",
            "expected": lambda calls: any(c.get("tool") == "calculate" for c in calls),
            "desc": "Math calculation",
        },
        {
            "query": "set a timer for 10 minutes",
            "expected": lambda calls: any(c.get("tool") in ("create_timer", "set_timer") and (c.get("args", {}).get("duration_seconds") == 600 or c.get("args", {}).get("seconds") == 600) for c in calls),
            "desc": "Timer creation",
        },
        {
            "query": "check cpu and battery usage",
            "expected": lambda calls: any(c.get("tool") in ("get_system_stats", "get_system_info", "get_detailed_hardware_stats") for c in calls),
            "desc": "System hardware health",
        },
    ],
    "2. Multi-Intent Compound Chains": [
        {
            "query": "mute audio and minimize Chrome",
            "expected": lambda calls: any(c.get("tool") == "mute_audio" for c in calls) and any(c.get("tool") in ("manage_window", "minimize_window") for c in calls),
            "desc": "2-step: Audio + Window",
        },
        {
            "query": "open Spotify and set volume to 40",
            "expected": lambda calls: any(c.get("tool") == "launch_app" for c in calls) and any(c.get("tool") == "set_volume" for c in calls),
            "desc": "2-step: App launch + Volume",
        },
        {
            "query": "close WhatsApp and open Notepad",
            "expected": lambda calls: any(c.get("tool") in ("close_app", "kill_app", "manage_window") for c in calls) and any(c.get("tool") == "launch_app" for c in calls),
            "desc": "2-step: App terminate + App launch",
        },
        {
            "query": "search quantum computing tutorials on youtube and snap browser to the right",
            "expected": lambda calls: any(c.get("tool") == "search_in_browser" for c in calls) and any(c.get("tool") == "snap_window" for c in calls),
            "desc": "2-step: Web search + Window snap",
        },
        {
            "query": "open Brave, set brightness to 60, and set a 15 minute timer",
            "expected": lambda calls: any(c.get("tool") == "launch_app" for c in calls) and any(c.get("tool") == "set_brightness" for c in calls) and any(c.get("tool") in ("create_timer", "set_timer") for c in calls),
            "desc": "3-step compound: App + Brightness + Timer",
        },
        {
            "query": "minimize discord, launch code, and switch to research layout",
            "expected": lambda calls: any(c.get("tool") in ("manage_window", "minimize_window") for c in calls) and any(c.get("tool") == "launch_app" for c in calls) and any(c.get("tool") in ("set_workspace_preset", "arrange_workspace_preset") for c in calls),
            "desc": "3-step compound: Window + App + Workspace preset",
        },
    ],
    "3. Pure Conversational Queries": [
        {
            "query": "hello friday",
            "expected": lambda calls: any(c.get("tool") == "chat_response" for c in calls),
            "desc": "Simple greeting",
        },
        {
            "query": "who are you and what can you do",
            "expected": lambda calls: any(c.get("tool") == "chat_response" for c in calls),
            "desc": "Identity & capabilities inquiry",
        },
        {
            "query": "thank you for your help, goodbye",
            "expected": lambda calls: any(c.get("tool") == "chat_response" for c in calls),
            "desc": "Gratitude & farewell",
        },
        {
            "query": "hey friday, how is everything running today?",
            "expected": lambda calls: any(c.get("tool") == "chat_response" for c in calls),
            "desc": "Conversational pleasantry",
        },
    ],
    "4. Mixed Conversational + Actions": [
        {
            "query": "good morning friday, can you mute the sound?",
            "expected": lambda calls: any(c.get("tool") == "mute_audio" for c in calls),
            "desc": "Greeting + Single action",
        },
        {
            "query": "hey friday, please open brave and set volume to 50",
            "expected": lambda calls: any(c.get("tool") == "launch_app" for c in calls) and any(c.get("tool") == "set_volume" for c in calls),
            "desc": "Polite greeting + Multi-action",
        },
        {
            "query": "thanks friday, now close chrome and lock the screen",
            "expected": lambda calls: any(c.get("tool") in ("close_app", "kill_app", "manage_window", "close_window") for c in calls) and any(c.get("tool") in ("press_shortcut", "set_power_state", "lock_pc", "lock_workstation") for c in calls),
            "desc": "Gratitude + App close + Lock PC",
        },
        {
            "query": "hello there, I am starting work so switch to coding layout, open vscode, and turn brightness to 90%",
            "expected": lambda calls: any(c.get("tool") in ("set_workspace_preset", "arrange_workspace_preset") for c in calls) and any(c.get("tool") == "launch_app" for c in calls) and any(c.get("tool") == "set_brightness" for c in calls),
            "desc": "Conversational preamble + 3-step action chain",
        },
        {
            "query": "hope you are having a good day, could you look up lofi beats on youtube and set a 25 min timer",
            "expected": lambda calls: any(c.get("tool") == "search_in_browser" for c in calls) and any(c.get("tool") in ("create_timer", "set_timer") for c in calls),
            "desc": "Polite sentiment + Search + Timer",
        },
    ],
    "5. Real FRIDAY Log Scenarios, STT Noise & Cloud Delegation": [
        {
            "query": "friday please paste my clipboard into active app",
            "expected": lambda calls: any(c.get("tool") == "press_shortcut" and c.get("args", {}).get("action") == "paste" for c in calls),
            "desc": "GUI shortcut paste from clipboard",
        },
        {
            "query": "take a quick screenshot of my screen",
            "expected": lambda calls: any(c.get("tool") == "take_screenshot" for c in calls),
            "desc": "Desktop screenshot capture",
        },
        {
            "query": "would you kindly kinda turn down the volumme by 20",
            "expected": lambda calls: any(c.get("tool") == "change_volume" for c in calls),
            "desc": "STT Typo ('volumme') + speech fillers ('kinda')",
        },
        {
            "query": "open calc and serch for weather in tokyo on google",
            "expected": lambda calls: any(c.get("tool") == "search_in_browser" for c in calls),
            "desc": "Multiple typos ('calc', 'serch')",
        },
        {
            "query": "write a python script with asyncio to scrape news headlines concurrently",
            "expected": lambda calls: any(c.get("tool") == "delegate_to_reasoning_model" for c in calls),
            "desc": "Cloud reasoning delegation: Asyncio scraping",
        },
        {
            "query": "summarize the 20-page financial quarterly earnings report pdf",
            "expected": lambda calls: any(c.get("tool") in ("delegate_to_reasoning_model", "read_pdf_document") for c in calls),
            "desc": "Cloud reasoning delegation: PDF document analysis",
        },
    ],
    "6. v4 Stateful Anaphora & Coreference": [
        {
            "query": '[Prior Action: {"tool": "launch_app", "args": {"name": "notepad"}}]\nClose it',
            "expected": lambda calls: any(c.get("tool") == "close_app" for c in calls),
            "desc": "Anaphora: Close it (notepad)",
        },
        {
            "query": '[Prior Action: {"tool": "set_volume", "args": {"level": 50}}]\nTurn it up',
            "expected": lambda calls: any(c.get("tool") == "change_volume" and c.get("args", {}).get("delta", 0) > 0 for c in calls),
            "desc": "Anaphora: Turn it up (volume delta)",
        },
        {
            "query": '[Prior Action: {"tool": "set_brightness", "args": {"level": 60}}]\nMake it brighter',
            "expected": lambda calls: any(c.get("tool") == "change_brightness" and c.get("args", {}).get("delta", 0) > 0 for c in calls),
            "desc": "Anaphora: Make it brighter",
        },
        {
            "query": '[Prior Action: {"tool": "mute_audio", "args": {"mute": true}}]\nUndo that',
            "expected": lambda calls: any(c.get("tool") == "mute_audio" and (c.get("args", {}).get("mute") is False or not c.get("args")) for c in calls),
            "desc": "Anaphora: Undo that (audio unmute)",
        },
        {
            "query": '[Prior Action: {"tool": "take_screenshot", "args": {}}]\nDo that again',
            "expected": lambda calls: any(c.get("tool") == "take_screenshot" for c in calls),
            "desc": "Anaphora: Repeat action (screenshot)",
        },
    ],
    "7. v4 Calibrated Ambiguity & Safety Gating": [
        {
            "query": "delete the project",
            "expected": lambda calls: any(c.get("tool") == "clarification_required" for c in calls),
            "desc": "Ambiguity gating: delete unspecified project",
        },
        {
            "query": "send an email",
            "expected": lambda calls: any(c.get("tool") == "clarification_required" for c in calls),
            "desc": "Ambiguity gating: send email without recipient",
        },
        {
            "query": "make me coffee",
            "expected": lambda calls: any(c.get("tool") == "chat_response" for c in calls),
            "desc": "Physical impossibility refusal: make coffee",
        },
        {
            "query": "clean my room",
            "expected": lambda calls: any(c.get("tool") == "chat_response" for c in calls),
            "desc": "Physical impossibility refusal: clean room",
        },
    ],
    "8. v4 Dynamic In-Context Runtime Tools": [
        {
            "query": "start recording my screen in obs",
            "system": (
                "You are DesktopIntent, the real-time neural router for FRIDAY AI. For every user input, "
                "you output ONLY a valid JSON array of tool calls matching FRIDAY's MCP tool bridge. "
                "Never generate conversational monologue, explanation, or thinking tokens. "
                "If the command requires complex coding or open-ended reasoning, invoke delegate_to_reasoning_model. "
                "If the input is conversational or greetings, invoke chat_response.\n\n"
                "Runtime Tools:\n"
                '[\n  {"name": "obs_record", "description": "Control OBS screen recording", "parameters": ["action"]}\n]'
            ),
            "expected": lambda calls: any(c.get("tool") == "obs_record" and c.get("args", {}).get("action") == "start" for c in calls),
            "desc": "Runtime tool injection: OBS recording",
        },
        {
            "query": "search for Daft Punk on Spotify",
            "system": (
                "You are DesktopIntent, the real-time neural router for FRIDAY AI. For every user input, "
                "you output ONLY a valid JSON array of tool calls matching FRIDAY's MCP tool bridge. "
                "Never generate conversational monologue, explanation, or thinking tokens. "
                "If the command requires complex coding or open-ended reasoning, invoke delegate_to_reasoning_model. "
                "If the input is conversational or greetings, invoke chat_response.\n\n"
                "Runtime Tools:\n"
                '[\n  {"name": "spotify_search", "description": "Search tracks on Spotify", "parameters": ["query"]}\n]'
            ),
            "expected": lambda calls: any(c.get("tool") == "spotify_search" for c in calls),
            "desc": "Runtime tool injection: Spotify search",
        },
        {
            "query": "restart redis container",
            "system": (
                "You are DesktopIntent, the real-time neural router for FRIDAY AI. For every user input, "
                "you output ONLY a valid JSON array of tool calls matching FRIDAY's MCP tool bridge. "
                "Never generate conversational monologue, explanation, or thinking tokens. "
                "If the command requires complex coding or open-ended reasoning, invoke delegate_to_reasoning_model. "
                "If the input is conversational or greetings, invoke chat_response.\n\n"
                "Runtime Tools:\n"
                '[\n  {"name": "docker_action", "description": "Manage docker containers", "parameters": ["action", "container"]}\n]'
            ),
            "expected": lambda calls: any(c.get("tool") == "docker_action" and c.get("args", {}).get("action") == "restart" for c in calls),
            "desc": "Runtime tool injection: Docker container",
        },
    ],
}


def query_ollama(prompt: str, system: Any = None) -> tuple[List[Dict[str, Any]], str, float]:
    """Queries desktop-intent on Ollama and returns (parsed_json, raw_text, latency_ms)."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 0.95,
            "num_predict": 128,
            "stop": ["<|im_end|>"],
        },
    }
    if system:
        payload["system"] = system

    start = time.perf_counter()
    resp = session.post(OLLAMA_ENDPOINT, json=payload, timeout=20)
    resp.raise_for_status()
    raw = resp.json().get("response", "").strip()
    latency_ms = (time.perf_counter() - start) * 1000

    if raw.startswith("[") and not raw.endswith("]"):
        raw += "]"

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = [parsed]
    except Exception:
        parsed = []

    return parsed, raw, latency_ms


def main():
    print("=" * 75, flush=True)
    print(f"🔬 RUNNING COMPREHENSIVE BENCHMARK SUITE FOR: {MODEL_NAME}", flush=True)
    print("=" * 75, flush=True)

    overall_total = 0
    overall_passed = 0
    overall_latency = 0.0
    category_summaries = []

    for cat_name, tests in TEST_CATEGORIES.items():
        print(f"\n📁 CATEGORY: {cat_name}", flush=True)
        print("-" * 75, flush=True)
        cat_passed = 0
        cat_latency = 0.0

        for i, t in enumerate(tests, 1):
            prompt = t["query"]
            desc = t["desc"]
            calls, raw, lat_ms = query_ollama(prompt, system=t.get("system"))

            cat_latency += lat_ms
            overall_latency += lat_ms
            overall_total += 1

            passed = False
            try:
                passed = t["expected"](calls)
            except Exception:
                passed = False

            if passed:
                cat_passed += 1
                overall_passed += 1
                status_icon = "✅ PASS"
            else:
                status_icon = "❌ FAIL"

            print(f"[{i:02d}/{len(tests):02d}] {status_icon} ({lat_ms:.1f}ms) | {desc}", flush=True)
            print(f"       Query : {prompt!r}", flush=True)
            print(f"       Output: {raw}", flush=True)

        cat_acc = (cat_passed / len(tests)) * 100
        cat_avg_ms = cat_latency / len(tests)
        category_summaries.append((cat_name, cat_passed, len(tests), cat_acc, cat_avg_ms))
        print(f"--> Category Result: {cat_passed}/{len(tests)} ({cat_acc:.1f}%) | Avg Latency: {cat_avg_ms:.1f}ms", flush=True)

    print("\n" + "=" * 75, flush=True)
    print("📊 OVERALL BENCHMARK SCORECARD", flush=True)
    print("=" * 75, flush=True)
    for name, p, tot, acc, avg_ms in category_summaries:
        bar = "█" * int(acc // 10) + "░" * (10 - int(acc // 10))
        print(f"{name:<55} | {bar} | {p}/{tot} ({acc:5.1f}%) | {avg_ms:6.1f}ms", flush=True)

    total_acc = (overall_passed / overall_total) * 100
    avg_total_ms = overall_latency / overall_total
    print("-" * 75, flush=True)
    print(f"TOTAL SCORE: {overall_passed}/{overall_total} PASSED ({total_acc:.1f}%) | AVERAGE LATENCY: {avg_total_ms:.1f}ms", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
