"""Comprehensive Verification for friday:v6-essentials."""
import json
import time
import urllib.request

MODEL = "friday:v6-essentials"
SYSTEM_PROMPT = (
    "You are DesktopIntent, the real-time neural router for FRIDAY AI. For every user input, "
    "you output ONLY a valid JSON array of tool calls matching FRIDAY's MCP tool bridge. "
    "Never generate conversational monologue, explanation, or thinking tokens. "
    "If the command requires complex coding or open-ended reasoning, invoke delegate_to_reasoning_model. "
    "If the input is conversational or greetings, invoke chat_response."
)

TEST_CASES = [
    # 1. New Daily Essentials
    ("read my unread emails", "read_unread_emails"),
    ("list upcoming calendar events for tomorrow", "list_calendar_events"),
    ("start screen recording", "start_screen_recording"),
    ("stop recording my screen", "stop_screen_recording"),
    ("lock my computer", "lock_pc"),
    ("put the pc to sleep", "sleep_system"),
    ("read the pdf document report.pdf", "read_pdf_file"),
    ("get clipboard text", "get_clipboard_text"),
    ("copy 'Hello FRIDAY' to clipboard", "set_clipboard_text"),
    ("minimize notepad", "minimize_window"),
    ("maximize chrome", "maximize_window"),
    ("bring vscode to front", "focus_window"),
    ("open a new browser tab", "browser_new_tab"),
    ("close current browser tab", "browser_close_tab"),
    ("add contact Bob with phone 555-1234", "add_contact"),

    # 2. Conversational & Idiomatic Precision (The 7 gaps from audit)
    ("my eyes are hurting, make the screen a bit dimmer", "decrease_brightness"),
    ("it's blinding in here, dim the display", "decrease_brightness"),
    ("my ears are ringing, quiet down please", "decrease_volume"),
    ("it's deafening in here, silence this PC", "mute_audio"),
    ("hold on, freeze my active timer for a moment", "pause_timer"),
    ("put the countdown on ice", "pause_timer"),
    ("unfreeze the timer", "resume_timer"),
    ("kill that annoying alarm", "cancel_alarm"),
    ("how much juice is left in this machine?", "get_battery_status"),
    ("am I running on battery or plugged into the wall?", "get_battery_status"),
    ("how many monitors do I have connected to this machine?", "get_monitors_info"),
    ("can you change Windows to night mode please?", "set_system_theme"),
    ("turn on night light", "toggle_night_light"),

    # 3. Core Anchors & v5 Precision Regressions
    ("resume the music", "media_play_pause"),
    ("mute system volume", "mute_audio"),
    ("search for files named report in downloads", "search_files"),
    ("screenshot of monitor 2", "capture_monitor_screenshot"),
    ("type 'Hello World' into document", "type_text"),
    ("set a timer for 10 minutes", "set_timer"),
    ("what is the current time right now?", "get_current_time"),
    ("good morning friday how are you", "chat_response"),
    ("write a quicksort in python", "delegate_to_reasoning_model"),
]

def query_model(prompt: str):
    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 0.95,
            "num_predict": 128
        }
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            res = json.loads(response.read().decode("utf-8"))
            dt = (time.time() - t0) * 1000
            return res.get("response", "").strip(), dt
    except Exception as e:
        return f"ERROR: {e}", 0

def run_tests():
    print(f"[*] Running Verification Benchmark on [{MODEL}] with ZERO safeguards...")
    print("=" * 80)
    passed = 0
    total = len(TEST_CASES)
    latencies = []

    for query, expected_tool in TEST_CASES:
        raw_res, latency = query_model(query)
        latencies.append(latency)
        is_pass = False
        parsed_tools = []

        try:
            parsed = json.loads(raw_res)
            if isinstance(parsed, list) and len(parsed) > 0:
                parsed_tools = [p.get("tool") for p in parsed if isinstance(p, dict)]
                if expected_tool in parsed_tools:
                    is_pass = True
        except Exception:
            is_pass = False

        status = "✅ PASS" if is_pass else "❌ FAIL"
        if is_pass:
            passed += 1

        print(f"{status} | {latency:5.1f}ms | \"{query[:42]:<42}\" -> {parsed_tools}")
        if not is_pass:
            print(f"      Expected: {expected_tool} | Raw: {raw_res[:80]}")

    print("=" * 80)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    pct = (passed / total) * 100
    print(f"Summary: {passed}/{total} Passed ({pct:.1f}%) | Avg Latency: {avg_lat:.1f} ms")

if __name__ == "__main__":
    run_tests()
