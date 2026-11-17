import json
import time
import urllib.request

MODEL = "friday:0.5b-v5"
SYSTEM_PROMPT = (
    "You are DesktopIntent, the real-time neural router for FRIDAY AI. For every user input, "
    "you output ONLY a valid JSON array of tool calls matching FRIDAY's MCP tool bridge. "
    "Never generate conversational monologue, explanation, or thinking tokens. "
    "If the command requires complex coding or open-ended reasoning, invoke delegate_to_reasoning_model. "
    "If the input is conversational or greetings, invoke chat_response."
)

TEST_CASES = [
    # 1. Media vs Volume vs Play
    ("resume the music", "media_play_pause"),
    ("pause playback", "media_play_pause"),
    ("mute audio", "mute_audio"),
    ("play Bohemian Rhapsody on Spotify", "play_music"),
    ("go back one song", "media_prev_track"),
    ("skip track", "media_next_track"),

    # 2. Local Search vs Web Search
    ("search for files named report in downloads", "search_files"),
    ("find contact Alice in my phonebook", "search_contacts"),
    ("search for workout playlist on spotify", "search_playlists"),
    ("search google for latest nvidia drivers", "search_in_browser"),

    # 3. Vision & Monitors
    ("screenshot of monitor 2", "capture_monitor_screenshot"),
    ("capture all screens", "capture_monitor_screenshot"),
    ("take a screenshot of chrome", "capture_window_screenshot"),
    ("read the text on my screen", "extract_text_ocr"),

    # 4. Canonical Timers & Reminders
    ("set a timer for 10 minutes", "set_timer"),
    ("pause the timer", "pause_timer"),
    ("resume timer", "resume_timer"),
    ("cancel the timer", "cancel_timer"),
    ("remind me to check the oven in 15 minutes", "set_reminder"),

    # 5. Telemetry & Ports
    ("check if port 8000 is open", "check_port_status"),
    ("ping google.com to test connectivity", "ping_host"),
    ("show my network adapters and IP addresses", "get_network_interfaces"),
    ("list open browser tabs", "browser_list_tabs"),

    # 6. Action Over Chitchat / Quoted Typing
    ("type 'Hello World from FRIDAY' into the document", "type_text"),
    ("press enter key", "press_key"),
    ("hello friday how are you", "chat_response"),

    # 7. Core Reflex Anchor
    ("open notepad", "launch_app"),
    ("turn brightness to 80", "set_brightness"),
    ("volume 50", "set_volume"),
    ("snap window to the right", "snap_window"),
]

def query_model(prompt: str) -> dict:
    payload = {
        "model": MODEL,
        "prompt": f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "stream": False,
        "options": {"temperature": 0.0}
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    lat = (time.perf_counter() - t0) * 1000
    text = res.get("response", "").strip()
    return {"text": text, "latency_ms": lat}

def main():
    print("=" * 80)
    print(f"🎯 TESTING RAW NEURAL MODEL: {MODEL} (ZERO SAFEGUARDS)")
    print("=" * 80)

    passed = 0
    total = len(TEST_CASES)

    for prompt, expected_tool in TEST_CASES:
        res = query_model(prompt)
        text = res["text"]
        lat = res["latency_ms"]

        try:
            calls = json.loads(text)
            actual_tool = calls[0].get("tool") if calls else None
            args = calls[0].get("args") if calls else {}
        except Exception:
            actual_tool = "INVALID_JSON"
            args = {}

        ok = (actual_tool == expected_tool)
        if ok:
            passed += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        print(f"{status} | {lat:5.1f}ms | \"{prompt[:40]:40s}\" -> {actual_tool} {args}")

    print("=" * 80)
    pct = (passed / total) * 100
    print(f"SCORE: {passed}/{total} ({pct:.1f}%) | Model: {MODEL}")
    print("=" * 80)

if __name__ == "__main__":
    main()
