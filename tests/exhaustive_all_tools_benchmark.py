"""Exhaustive All-Tools Benchmark Suite for FRIDAY AI.

Verifies EVERY SINGLE ONE of the 55 tool capabilities across all 4 query archetypes:
- Type 1: Direct Imperative
- Type 2: Conversational & Polite Preambles
- Type 3: Multi-Intent Compound Chains
- Type 4: STT Noise, Slang & Phonetic Typos
Plus Cloud Reasoning Hard Boundaries & Standby Modes.
"""

import json
import time
import requests
from typing import Dict, List, Any

OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "friday"

http_session = requests.Session()


def c_any_tool(calls: List[Dict[str, Any]], *tools: str) -> bool:
    return any(c.get("tool") in tools for c in calls)


def c_tool_arg(calls: List[Dict[str, Any]], tool: str, arg_key: str, expected_val: Any) -> bool:
    for c in calls:
        if c.get("tool") == tool:
            val = c.get("args", {}).get(arg_key)
            if isinstance(expected_val, str) and isinstance(val, str):
                return expected_val.lower() in val.lower()
            return val == expected_val
    return False


BENCHMARK_MATRIX: List[Dict[str, Any]] = [
    # =========================================================================
    # 1. FILESYSTEM TOOLS (Directory, File, Move, Copy, Organize)
    # =========================================================================
    {
        "id": "FS01", "cat": "Filesystem", "type": "Direct",
        "query": "create a new directory named project_files",
        "desc": "Create directory (direct)",
        "validate": lambda calls: c_any_tool(calls, "create_directory") and c_tool_arg(calls, "create_directory", "path", "project_files")
    },
    {
        "id": "FS02", "cat": "Filesystem", "type": "Conversational",
        "query": "hey friday, could you please make a folder called invoices?",
        "desc": "Create directory (polite/conversational)",
        "validate": lambda calls: c_any_tool(calls, "create_directory") and c_tool_arg(calls, "create_directory", "path", "invoices")
    },
    {
        "id": "FS03", "cat": "Filesystem", "type": "Multi-Intent",
        "query": "create folder invoices and open file explorer",
        "desc": "Create directory + Launch explorer",
        "validate": lambda calls: c_any_tool(calls, "create_directory") and c_any_tool(calls, "launch_app", "press_shortcut")
    },
    {
        "id": "FS04", "cat": "Filesystem", "type": "STT Noise",
        "query": "mak a new directry calld project_files",
        "desc": "Create directory (typos: 'mak', 'directry', 'calld')",
        "validate": lambda calls: c_any_tool(calls, "create_directory")
    },
    {
        "id": "FS05", "cat": "Filesystem", "type": "Direct",
        "query": "delete file temp.txt",
        "desc": "Delete file (direct)",
        "validate": lambda calls: c_any_tool(calls, "delete_file") and c_tool_arg(calls, "delete_file", "path", "temp.txt")
    },
    {
        "id": "FS06", "cat": "Filesystem", "type": "Conversational",
        "query": "would you mind deleting file temp.txt?",
        "desc": "Delete file (polite)",
        "validate": lambda calls: c_any_tool(calls, "delete_file")
    },
    {
        "id": "FS07", "cat": "Filesystem", "type": "Direct",
        "query": "copy file data.csv to backup folder",
        "desc": "Copy file",
        "validate": lambda calls: c_any_tool(calls, "copy_file")
    },
    {
        "id": "FS08", "cat": "Filesystem", "type": "Direct",
        "query": "move file report.pdf to downloads",
        "desc": "Move file",
        "validate": lambda calls: c_any_tool(calls, "move_file")
    },
    {
        "id": "FS09", "cat": "Filesystem", "type": "Direct",
        "query": "organize my downloads directory by file type",
        "desc": "Organize directory by type",
        "validate": lambda calls: c_any_tool(calls, "organize_directory_by_type")
    },

    # =========================================================================
    # 2. CALENDAR & SCHEDULING
    # =========================================================================
    {
        "id": "CAL01", "cat": "Calendar", "type": "Direct",
        "query": "schedule team meeting tomorrow at 10am",
        "desc": "Create calendar event (direct)",
        "validate": lambda calls: c_any_tool(calls, "create_calendar_event") and c_tool_arg(calls, "create_calendar_event", "title", "team meeting")
    },
    {
        "id": "CAL02", "cat": "Calendar", "type": "Conversational",
        "query": "hey friday, could you add dentist appointment on Friday at 3pm to my calendar?",
        "desc": "Create calendar event (conversational)",
        "validate": lambda calls: c_any_tool(calls, "create_calendar_event")
    },
    {
        "id": "CAL03", "cat": "Calendar", "type": "Multi-Intent",
        "query": "schedule team meeting tomorrow at 10am and mute audio",
        "desc": "Create calendar event + Mute audio",
        "validate": lambda calls: c_any_tool(calls, "create_calendar_event") and c_any_tool(calls, "mute_audio", "set_volume_mute")
    },
    {
        "id": "CAL04", "cat": "Calendar", "type": "STT Noise",
        "query": "scheudle team meetng tomorow at 10am",
        "desc": "Create calendar event (STT typos)",
        "validate": lambda calls: c_any_tool(calls, "create_calendar_event")
    },
    {
        "id": "CAL05", "cat": "Calendar", "type": "Direct",
        "query": "what is on my calendar today",
        "desc": "List calendar events",
        "validate": lambda calls: c_any_tool(calls, "list_calendar_events")
    },
    {
        "id": "CAL06", "cat": "Calendar", "type": "Direct",
        "query": "delete calendar event team meeting",
        "desc": "Delete calendar event",
        "validate": lambda calls: c_any_tool(calls, "delete_calendar_event")
    },

    # =========================================================================
    # 3. COMMUNICATION, MESSAGING & OFFICE
    # =========================================================================
    {
        "id": "MSG01", "cat": "Communication", "type": "Direct",
        "query": "send a whatsapp message to John saying I will be late",
        "desc": "Send WhatsApp message (direct)",
        "validate": lambda calls: c_any_tool(calls, "send_whatsapp_message") and c_tool_arg(calls, "send_whatsapp_message", "contact", "John")
    },
    {
        "id": "MSG02", "cat": "Communication", "type": "Conversational",
        "query": "hey friday, please message Sarah on whatsapp see you at 6pm",
        "desc": "Send WhatsApp message (conversational)",
        "validate": lambda calls: c_any_tool(calls, "send_whatsapp_message")
    },
    {
        "id": "MSG03", "cat": "Communication", "type": "Direct",
        "query": "send telegram message to Alex meeting starting soon",
        "desc": "Send Telegram message",
        "validate": lambda calls: c_any_tool(calls, "send_telegram_message")
    },
    {
        "id": "MSG04", "cat": "Communication", "type": "Direct",
        "query": "read my latest emails",
        "desc": "Read emails inbox",
        "validate": lambda calls: c_any_tool(calls, "read_emails")
    },
    {
        "id": "MSG05", "cat": "Communication", "type": "Direct",
        "query": "compose email to boss@company.com subject quarterly report body report attached",
        "desc": "Compose email",
        "validate": lambda calls: c_any_tool(calls, "compose_email")
    },
    {
        "id": "MSG06", "cat": "Communication", "type": "Direct",
        "query": "create a word document named project_proposal.docx",
        "desc": "Create Word document",
        "validate": lambda calls: c_any_tool(calls, "create_word_document")
    },

    # =========================================================================
    # 4. VOICE, SPEECH & MEDIA
    # =========================================================================
    {
        "id": "VOX01", "cat": "Voice & Media", "type": "Direct",
        "query": "switch voice to female tts",
        "desc": "Switch TTS voice (female)",
        "validate": lambda calls: c_any_tool(calls, "switch_tts_voice")
    },
    {
        "id": "VOX02", "cat": "Voice & Media", "type": "Conversational",
        "query": "hey friday, could you change tts voice to Aria?",
        "desc": "Switch TTS voice (Aria)",
        "validate": lambda calls: c_any_tool(calls, "switch_tts_voice")
    },
    {
        "id": "VOX03", "cat": "Voice & Media", "type": "Direct",
        "query": "list available tts voices",
        "desc": "List TTS voices",
        "validate": lambda calls: c_any_tool(calls, "list_tts_voices")
    },
    {
        "id": "VOX04", "cat": "Voice & Media", "type": "Direct",
        "query": "play some jazz music",
        "desc": "Play music",
        "validate": lambda calls: c_any_tool(calls, "play_music")
    },
    {
        "id": "VOX05", "cat": "Voice & Media", "type": "Multi-Intent",
        "query": "play some jazz music and set volume to 50",
        "desc": "Play music + Set volume",
        "validate": lambda calls: c_any_tool(calls, "play_music") and c_any_tool(calls, "set_volume")
    },
    {
        "id": "VOX06", "cat": "Voice & Media", "type": "Direct",
        "query": "run terminal command ipconfig",
        "desc": "Run terminal command",
        "validate": lambda calls: c_any_tool(calls, "run_terminal_command")
    },

    # =========================================================================
    # 5. AUDIO, SOUND & HARDWARE TOGGLES
    # =========================================================================
    {
        "id": "SYS01", "cat": "System Audio", "type": "Direct",
        "query": "mute audio",
        "desc": "Mute audio (basic)",
        "validate": lambda calls: c_any_tool(calls, "mute_audio", "set_volume_mute")
    },
    {
        "id": "SYS02", "cat": "System Audio", "type": "Direct",
        "query": "unmute sound",
        "desc": "Unmute audio",
        "validate": lambda calls: c_any_tool(calls, "mute_audio")
    },
    {
        "id": "SYS03", "cat": "System Audio", "type": "Direct",
        "query": "set volume to 80",
        "desc": "Set volume level",
        "validate": lambda calls: c_any_tool(calls, "set_volume") and c_tool_arg(calls, "set_volume", "level", 80)
    },
    {
        "id": "SYS04", "cat": "System Audio", "type": "Direct",
        "query": "turn down the volumme by 20",
        "desc": "Change volume (typo 'volumme')",
        "validate": lambda calls: c_any_tool(calls, "change_volume")
    },
    {
        "id": "SYS05", "cat": "System Display", "type": "Direct",
        "query": "max brightness",
        "desc": "Brightness upper boundary",
        "validate": lambda calls: c_any_tool(calls, "set_brightness") and c_tool_arg(calls, "set_brightness", "level", 100)
    },
    {
        "id": "SYS06", "cat": "System Display", "type": "Direct",
        "query": "dim the screen to 20%",
        "desc": "Dim brightness to absolute 20%",
        "validate": lambda calls: c_any_tool(calls, "set_brightness") and c_tool_arg(calls, "set_brightness", "level", 20)
    },
    {
        "id": "SYS07", "cat": "System Hardware", "type": "Direct",
        "query": "turn off bluetooth",
        "desc": "Toggle bluetooth off",
        "validate": lambda calls: c_any_tool(calls, "toggle_bluetooth") and calls[0].get("args", {}).get("enabled") is False
    },
    {
        "id": "SYS08", "cat": "System Hardware", "type": "Direct",
        "query": "turn off wifi",
        "desc": "Toggle wifi off",
        "validate": lambda calls: c_any_tool(calls, "toggle_wifi")
    },
    {
        "id": "SYS09", "cat": "System Hardware", "type": "Direct",
        "query": "how much battery is left",
        "desc": "Battery status check",
        "validate": lambda calls: c_any_tool(calls, "get_battery_status", "get_system_info")
    },
    {
        "id": "SYS10", "cat": "System Hardware", "type": "Direct",
        "query": "check cpu and memory usage",
        "desc": "System info check",
        "validate": lambda calls: c_any_tool(calls, "get_system_info")
    },

    # =========================================================================
    # 6. APPS, WINDOWS & WORKSPACE PRESETS
    # =========================================================================
    {
        "id": "WIN01", "cat": "Apps & Windows", "type": "Direct",
        "query": "open file explorer",
        "desc": "Open File Explorer (app or shortcut)",
        "validate": lambda calls: c_any_tool(calls, "launch_app", "press_shortcut")
    },
    {
        "id": "WIN02", "cat": "Apps & Windows", "type": "Direct",
        "query": "launch visual studio code",
        "desc": "Launch VS Code",
        "validate": lambda calls: c_any_tool(calls, "launch_app")
    },
    {
        "id": "WIN03", "cat": "Apps & Windows", "type": "Direct",
        "query": "close spotify",
        "desc": "Close Spotify",
        "validate": lambda calls: c_any_tool(calls, "close_app", "kill_app")
    },
    {
        "id": "WIN04", "cat": "Apps & Windows", "type": "Direct",
        "query": "minimize active window",
        "desc": "Minimize window",
        "validate": lambda calls: c_any_tool(calls, "manage_window")
    },
    {
        "id": "WIN05", "cat": "Apps & Windows", "type": "Direct",
        "query": "snap window left",
        "desc": "Snap window left",
        "validate": lambda calls: c_any_tool(calls, "snap_window") and calls[0].get("args", {}).get("position") == "left"
    },
    {
        "id": "WIN06", "cat": "Apps & Windows", "type": "Direct",
        "query": "switch to coding workspace layout",
        "desc": "Workspace preset coding",
        "validate": lambda calls: c_any_tool(calls, "set_workspace_preset") and "coding" in str(calls[0].get("args", {})).lower()
    },
    {
        "id": "WIN07", "cat": "Apps & Windows", "type": "Direct",
        "query": "clean desktop layout",
        "desc": "Workspace preset clean",
        "validate": lambda calls: c_any_tool(calls, "set_workspace_preset") and "clean" in str(calls[0].get("args", {})).lower()
    },
    {
        "id": "WIN08", "cat": "Apps & Windows", "type": "Direct",
        "query": "open a new browser tab",
        "desc": "Browser new tab",
        "validate": lambda calls: c_any_tool(calls, "browser_tab_action")
    },
    {
        "id": "WIN09", "cat": "Apps & Windows", "type": "Direct",
        "query": "reload this webpage",
        "desc": "Browser reload tab",
        "validate": lambda calls: c_any_tool(calls, "browser_tab_action")
    },

    # =========================================================================
    # 7. SHORTCUTS, CLIPBOARD & TIMERS
    # =========================================================================
    {
        "id": "CUT01", "cat": "Shortcuts", "type": "Multi-Intent",
        "query": "select all and copy",
        "desc": "Shortcut select all + copy",
        "validate": lambda calls: len(calls) >= 2 and all(c.get("tool") == "press_shortcut" for c in calls)
    },
    {
        "id": "CUT02", "cat": "Shortcuts", "type": "Direct",
        "query": "undo my last action",
        "desc": "Shortcut undo",
        "validate": lambda calls: c_any_tool(calls, "press_shortcut") and calls[0].get("args", {}).get("action") == "undo"
    },
    {
        "id": "CUT03", "cat": "Shortcuts", "type": "Direct",
        "query": "get clipboard history",
        "desc": "Get clipboard history",
        "validate": lambda calls: c_any_tool(calls, "get_clipboard_history") or (c_any_tool(calls, "press_shortcut") and "clipboard" in str(calls).lower())
    },
    {
        "id": "CUT04", "cat": "Shortcuts", "type": "Direct",
        "query": "take a screenshot of my screen",
        "desc": "Take screenshot",
        "validate": lambda calls: c_any_tool(calls, "take_screenshot")
    },
    {
        "id": "CUT05", "cat": "Shortcuts", "type": "Direct",
        "query": "set a timer for 15 minutes",
        "desc": "Create timer (900s)",
        "validate": lambda calls: c_any_tool(calls, "create_timer") and c_tool_arg(calls, "create_timer", "duration_seconds", 900)
    },
    {
        "id": "CUT06", "cat": "Shortcuts", "type": "Direct",
        "query": "calculate 450 * 18",
        "desc": "Calculate expression",
        "validate": lambda calls: c_any_tool(calls, "calculate")
    },
    {
        "id": "CUT07", "cat": "Shortcuts", "type": "Direct",
        "query": "take a note: buy groceries tomorrow",
        "desc": "Take note",
        "validate": lambda calls: c_any_tool(calls, "take_note")
    },

    # =========================================================================
    # 8. COMPLEX 3-STEP & 4-STEP COMPOUND CHAINS
    # =========================================================================
    {
        "id": "CHN01", "cat": "Compound", "type": "Multi-Intent",
        "query": "reload tab and mute volume",
        "desc": "2-step: Reload tab + Mute audio",
        "validate": lambda calls: c_any_tool(calls, "browser_tab_action") and c_any_tool(calls, "mute_audio", "set_volume_mute")
    },
    {
        "id": "CHN02", "cat": "Compound", "type": "Multi-Intent",
        "query": "take a screenshot, open file explorer, and set volume to 80",
        "desc": "3-step: Screenshot + Explorer + Set volume",
        "validate": lambda calls: c_any_tool(calls, "take_screenshot") and c_any_tool(calls, "launch_app", "press_shortcut") and c_any_tool(calls, "set_volume")
    },
    {
        "id": "CHN03", "cat": "Compound", "type": "Multi-Intent",
        "query": "mute volume, close chrome, minimize all windows, and lock pc",
        "desc": "4-step: Mute + Close app + Minimize all + Lock pc",
        "validate": lambda calls: c_any_tool(calls, "mute_audio", "set_volume_mute") and c_any_tool(calls, "close_app", "kill_app") and c_any_tool(calls, "minimize_all_windows", "manage_window") and c_any_tool(calls, "lock_screen", "press_shortcut")
    },
    {
        "id": "CHN04", "cat": "Compound", "type": "Multi-Intent",
        "query": "enable dark mode, turn on night light, set brightness to 40, and set volume to 20",
        "desc": "4-step: Dark mode + Night light + Brightness + Volume",
        "validate": lambda calls: c_any_tool(calls, "toggle_dark_mode") and c_any_tool(calls, "toggle_night_light") and c_any_tool(calls, "set_brightness") and c_any_tool(calls, "set_volume")
    },
    {
        "id": "CHN05", "cat": "Compound", "type": "Multi-Intent",
        "query": "turn off bluetooth, enable dark mode, and switch to coding workspace",
        "desc": "3-step: Bluetooth off + Dark mode + Preset coding",
        "validate": lambda calls: c_any_tool(calls, "toggle_bluetooth") and c_any_tool(calls, "toggle_dark_mode") and c_any_tool(calls, "set_workspace_preset")
    },

    # =========================================================================
    # 9. ASSISTANT STANDBY & CONVERSATIONAL
    # =========================================================================
    {
        "id": "AST01", "cat": "Dialogue", "type": "Standby",
        "query": "friday go to sleep",
        "desc": "FRIDAY sleep instruction",
        "validate": lambda calls: c_any_tool(calls, "chat_response", "sleep_system", "lock_screen")
    },
    {
        "id": "AST02", "cat": "Dialogue", "type": "Standby",
        "query": "stand by friday",
        "desc": "FRIDAY standby",
        "validate": lambda calls: c_any_tool(calls, "chat_response")
    },
    {
        "id": "AST03", "cat": "Dialogue", "type": "Standby",
        "query": "stop listening for now",
        "desc": "FRIDAY pause listening",
        "validate": lambda calls: c_any_tool(calls, "chat_response")
    },
    {
        "id": "AST04", "cat": "Dialogue", "type": "Conversational",
        "query": "can you feel emotions?",
        "desc": "Emotion inquiry",
        "validate": lambda calls: c_any_tool(calls, "chat_response")
    },
    {
        "id": "AST05", "cat": "Dialogue", "type": "Conversational",
        "query": "who are you and what can you do",
        "desc": "Identity inquiry",
        "validate": lambda calls: c_any_tool(calls, "chat_response")
    },

    # =========================================================================
    # 10. CLOUD REASONING HARD BOUNDARIES (NO FICTIONAL TOOLS)
    # =========================================================================
    {
        "id": "CLD01", "cat": "Cloud Delegation", "type": "Coding",
        "query": "write a complete react typescript dashboard with charts and authentication",
        "desc": "Cloud reasoning: React TS dashboard",
        "validate": lambda calls: c_any_tool(calls, "delegate_to_reasoning_model")
    },
    {
        "id": "CLD02", "cat": "Cloud Delegation", "type": "Math",
        "query": "solve this calculus differential equation: dy/dx + 2y = e^(3x)",
        "desc": "Cloud reasoning: Calculus ODE",
        "validate": lambda calls: c_any_tool(calls, "delegate_to_reasoning_model")
    },
    {
        "id": "CLD03", "cat": "Cloud Delegation", "type": "Strategy",
        "query": "draft a comprehensive business plan for a specialty coffee shop",
        "desc": "Cloud reasoning: Business plan",
        "validate": lambda calls: c_any_tool(calls, "delegate_to_reasoning_model")
    },
    {
        "id": "CLD04", "cat": "Cloud Delegation", "type": "Architecture",
        "query": "compare postgresql and mongodb for real-time analytics with trade-offs",
        "desc": "Cloud reasoning: Database comparison",
        "validate": lambda calls: c_any_tool(calls, "delegate_to_reasoning_model")
    },
    {
        "id": "CLD05", "cat": "Cloud Delegation", "type": "Testing",
        "query": "generate a full pytest test suite with mocks for this fastapi user auth router",
        "desc": "Cloud reasoning: Pytest generation",
        "validate": lambda calls: c_any_tool(calls, "delegate_to_reasoning_model")
    },
    {
        "id": "CLD06", "cat": "Cloud Delegation", "type": "Scraping",
        "query": "write a python script with asyncio to scrape news headlines concurrently",
        "desc": "Cloud reasoning: Asyncio scraper",
        "validate": lambda calls: c_any_tool(calls, "delegate_to_reasoning_model")
    },
    {
        "id": "CLD07", "cat": "Cloud Delegation", "type": "Analysis",
        "query": "summarize the 20-page financial quarterly earnings report pdf",
        "desc": "Cloud reasoning: Document summarization",
        "validate": lambda calls: c_any_tool(calls, "delegate_to_reasoning_model")
    },
    {
        "id": "CLD08", "cat": "Cloud Delegation", "type": "Theory",
        "query": "explain the difference between transformer self-attention and mamba state space models",
        "desc": "Cloud reasoning: AI architecture theory",
        "validate": lambda calls: c_any_tool(calls, "delegate_to_reasoning_model")
    },
]


def query_ollama(model: str, prompt: str) -> tuple[List[Dict[str, Any]], str, float]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 0.95,
            "num_predict": 128,
            "stop": ["<|im_end|>", "]"],
        },
    }

    start = time.perf_counter()
    resp = http_session.post(OLLAMA_ENDPOINT, json=payload, timeout=20)
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


def run_benchmark(model_name: str = DEFAULT_MODEL):
    print("=" * 85, flush=True)
    print(f"🔬 EXHAUSTIVE ALL-TOOLS BENCHMARK FOR FRIDAY AI: {model_name}", flush=True)
    print(f"🎯 Total Verification Test Cases: {len(BENCHMARK_MATRIX)} covering all 55 MCP tools & 4 query types", flush=True)
    print("=" * 85, flush=True)

    print("[*] Warming up model in VRAM...", end=" ", flush=True)
    _, _, warmup_ms = query_ollama(model_name, "hello friday")
    print(f"Ready ({warmup_ms:.1f}ms warmup). Starting benchmark...\n", flush=True)

    passed_count = 0
    failures = []
    category_map = {}

    for i, test in enumerate(BENCHMARK_MATRIX, 1):
        tid = test["id"]
        cat = test["cat"]
        qtype = test["type"]
        query = test["query"]
        desc = test["desc"]
        validator = test["validate"]

        calls, raw, latency_ms = query_ollama(model_name, query)

        passed = False
        try:
            passed = bool(validator(calls))
        except Exception:
            passed = False

        if cat not in category_map:
            category_map[cat] = {"passed": 0, "total": 0, "latencies": []}
        category_map[cat]["total"] += 1
        category_map[cat]["latencies"].append(latency_ms)

        if passed:
            passed_count += 1
            category_map[cat]["passed"] += 1
            icon = "✅ PASS"
        else:
            icon = "❌ FAIL"
            failures.append({
                "id": tid, "cat": cat, "type": qtype, "query": query,
                "desc": desc, "got": calls if calls else raw
            })

        print(f"[{tid}] {icon} ({latency_ms:.1f}ms) [{cat} | {qtype}] {desc}")
        print(f"     Query : '{query}'")
        print(f"     Output: {calls if calls else raw!r}")

    print("\n" + "=" * 85, flush=True)
    print(f"📊 EXHAUSTIVE BENCHMARK SCORECARD: {model_name}", flush=True)
    print("=" * 85, flush=True)

    for cat, data in category_map.items():
        p = data["passed"]
        tot = data["total"]
        pct = (p / tot) * 100
        bars = int(pct / 10)
        bar_str = "█" * bars + "░" * (10 - bars)
        avg_lat = sum(data["latencies"]) / max(len(data["latencies"]), 1)
        print(f"{cat:<25} | {bar_str} | {p:>2}/{tot:>2} ({pct:5.1f}%) | {avg_lat:.1f}ms")

    total_pct = (passed_count / len(BENCHMARK_MATRIX)) * 100
    all_lats = [l for c in category_map.values() for l in c["latencies"]]
    total_avg_lat = sum(all_lats) / max(len(all_lats), 1)

    print("-" * 85, flush=True)
    print(f"TOTAL SCORE: {passed_count}/{len(BENCHMARK_MATRIX)} PASSED ({total_pct:.1f}%) | AVERAGE LATENCY: {total_avg_lat:.1f}ms", flush=True)
    print("=" * 85, flush=True)

    if failures:
        print(f"\n⚠️ {len(failures)} EDGE CASES IDENTIFIED:")
        for f in failures:
            print(f"• [{f['id']}] [{f['cat']}] {f['desc']}")
            print(f"  Query: '{f['query']}'")
            print(f"  Got  : {f['got']}\n")
    else:
        print("\n🏆 PERFECT SCORE: 100% ACCURACY ACROSS ALL FRIDAY MCP TOOLS!", flush=True)


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    run_benchmark(model)
