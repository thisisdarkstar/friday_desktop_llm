"""Rigorous verification suite for dataset/v10_blend."""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
BLEND_DIR = ROOT / "dataset" / "v10_blend"

COMMON_SHORTCUTS = {
    "copy", "paste", "cut", "select_all", "save", "undo", "redo", "find",
    "new_tab", "close_tab", "reopen_tab", "switch_tab", "zoom_in", "zoom_out",
    "reset_zoom", "refresh", "hard_refresh", "show_desktop", "file_explorer",
    "settings", "task_manager", "lock_screen", "clipboard_history", "open_clipboard"
}

SINGLE_KEYS = {"enter", "return", "tab", "esc", "escape", "space", "backspace"}

PLANNED_TOOLS = {
    "calculate", "take_note", "change_brightness", "minimize_all_windows",
    "toggle_night_light", "obs_record", "slack_dm"
}

def verify_split(path: Path):
    print(f"\n--- Verifying {path.name} ---")
    total = 0
    tools = Counter()
    issues = Counter()
    planned_found = Counter()

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            total += 1
            sample = json.loads(line)
            messages = sample.get("messages", [])
            asst = next((m for m in messages if m.get("role") == "assistant"), None)
            assert asst, f"Row {idx}: missing assistant message"
            
            calls = json.loads(asst["content"])
            assert isinstance(calls, list), f"Row {idx}: content not a list"

            for c in calls:
                tool = c.get("tool")
                args = c.get("args", {})
                tools[tool] += 1

                if tool in PLANNED_TOOLS:
                    planned_found[tool] += 1

                # 1. Hallucination checks
                if tool == "full_screen_window":
                    issues["hallucinated_full_screen_window"] += 1

                # 2. press_shortcut checks
                if tool == "press_shortcut":
                    action = args.get("action")
                    if action in SINGLE_KEYS:
                        issues["single_key_in_press_shortcut"] += 1
                    if action not in COMMON_SHORTCUTS and not action.startswith("ctrl") and not action.startswith("win"):
                        issues["unrecognized_shortcut_action"] += 1

                # 3. press_key checks
                if tool == "press_key":
                    if "key" not in args:
                        issues["press_key_missing_key"] += 1

                # 4. Schema checks
                if tool in ("launch_app", "close_app"):
                    if "app_name" not in args:
                        issues[f"{tool}_missing_app_name"] += 1

                if tool == "kill_process":
                    if not args.get("confirmed"):
                        issues["kill_process_missing_confirmed"] += 1

                if tool in ("maximize_window", "minimize_window", "focus_window", "restore_window"):
                    if "title_keyword" not in args and "window_title" in args:
                        issues[f"{tool}_stale_window_title"] += 1

    print(f"Total rows: {total}")
    print(f"Unique tools: {len(tools)}")
    print(f"Planned tools preserved: {dict(planned_found)}")
    print(f"Issues detected: {dict(issues)}")
    assert len(issues) == 0, f"Verification failed with issues: {dict(issues)}"
    print("Verification: PASSED!")

def verify_specific_queries(path: Path):
    print(f"\n--- Verifying Targeted Prompt Resolutions in {path.name} ---")
    verified = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            u = next(m["content"] for m in d["messages"] if m["role"] == "user").lower()
            a = json.loads(next(m["content"] for m in d["messages"] if m["role"] == "assistant"))

            if "type 'test message' and press enter" in u or "type test message and press enter" in u:
                tools = [c["tool"] for c in a]
                if tools == ["type_text", "press_key"] and a[1].get("args", {}).get("key") == "enter":
                    verified["type_and_press_enter"] += 1

            if "make chrome full screen please" in u:
                tools = [c["tool"] for c in a]
                if tools == ["maximize_window"] and a[0].get("args", {}).get("title_keyword") == "chrome":
                    verified["make_chrome_full_screen"] += 1

            if "is dark mode currently turned on" in u:
                tools = [c["tool"] for c in a]
                if tools == ["get_system_theme"]:
                    verified["is_dark_mode_turned_on"] += 1

            if "what is the current brightness" in u:
                tools = [c["tool"] for c in a]
                if tools == ["get_brightness"]:
                    verified["what_is_current_brightness"] += 1

            if "give me a quick status check on my pc resources" in u:
                tools = [c["tool"] for c in a]
                if tools == ["get_system_info"]:
                    verified["status_check_pc_resources"] += 1

    print("Targeted queries verified:", dict(verified))
    for k in ["type_and_press_enter", "make_chrome_full_screen", "is_dark_mode_turned_on", "what_is_current_brightness", "status_check_pc_resources"]:
        assert verified[k] > 0, f"Targeted query {k} not found or failed!"
    print("Targeted Query Assertions: ALL PASSED!")

if __name__ == "__main__":
    verify_split(BLEND_DIR / "train.jsonl")
    verify_split(BLEND_DIR / "validation.jsonl")
    verify_specific_queries(BLEND_DIR / "train.jsonl")
    print("\nALL V10 BLEND VERIFICATIONS PASSED WITH ZERO ERRORS!")
