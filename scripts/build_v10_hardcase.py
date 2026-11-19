"""Build the v10 hard-case replay subset.

Incorporates:
  - All v8 curated cases (polite wrappers, compound close, launch vs search).
  - All v9 curated cases (version check, philosophical, obs record, calendar STT).
  - Targeted v10 fixes:
    1. Read-only introspection:
       - 'Is dark mode currently turned on?' -> get_system_theme
       - 'what is the current brightness'   -> get_brightness
       - 'is audio muted'                   -> get_volume_status
       - 'status check on pc resources'     -> get_system_info
    2. Single-key GUI actions:
       - 'type ... and press enter' -> type_text + press_key(key='enter')
       - 'press enter'              -> press_key(key='enter')
       - 'hit escape'               -> press_key(key='esc')
    3. Window Fullscreen without hallucinations:
       - 'Make Chrome full screen please' -> maximize_window(title_keyword='chrome')
       - 'make active window full screen' -> maximize_window(title_keyword='active')

Output:
    dataset/v10_hardcase/train.jsonl
    dataset/v10_hardcase/validation.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "dataset" / "v10_hardcase"

SYSTEM = (
    "You are DesktopIntent, the real-time neural router for FRIDAY AI. For every user input, "
    "you output ONLY a valid JSON array of tool calls matching FRIDAY's MCP tool bridge. "
    "Never generate conversational monologue, explanation, or thinking tokens. "
    "If the command requires complex coding or open-ended reasoning, invoke delegate_to_reasoning_model. "
    "If the input is conversational or greetings, invoke chat_response."
)

V10_NEW_CASES: List[Tuple[str, str, List[Dict[str, Any]]]] = [
    # --- Introspection: Theme (read-only) ------------------------------------
    ("v10_theme_read", "Is dark mode currently turned on?",
     [{"tool": "get_system_theme", "args": {}}]),
    ("v10_theme_read", "can you check if dark mode is active?",
     [{"tool": "get_system_theme", "args": {}}]),
    ("v10_theme_read", "tell me what system theme is currently running",
     [{"tool": "get_system_theme", "args": {}}]),
    ("v10_theme_read", "is light theme enabled right now?",
     [{"tool": "get_system_theme", "args": {}}]),

    # --- Introspection: Brightness & Audio (read-only) ------------------------
    ("v10_display_read", "what is the current brightness",
     [{"tool": "get_brightness", "args": {}}]),
    ("v10_display_read", "check my display brightness level",
     [{"tool": "get_brightness", "args": {}}]),
    ("v10_audio_read", "is the volume currently muted?",
     [{"tool": "get_volume_status", "args": {}}]),
    ("v10_audio_read", "what is the current volume level?",
     [{"tool": "get_volume_status", "args": {}}]),

    # --- Introspection: System Health & Info ---------------------------------
    ("v10_system_read", "Give me a quick status check on my PC resources",
     [{"tool": "get_system_info", "args": {}}]),
    ("v10_system_read", "how are my system resources looking right now?",
     [{"tool": "get_system_info", "args": {}}]),

    # --- GUI: Single Key Actions (press_key NOT press_shortcut) --------------
    ("v10_key_enter", "type 'test message' and press enter",
     [{"tool": "type_text", "args": {"text": "test message"}},
      {"tool": "press_key", "args": {"key": "enter"}}]),
    ("v10_key_enter", "type 'search query' and hit enter",
     [{"tool": "type_text", "args": {"text": "search query"}},
      {"tool": "press_key", "args": {"key": "enter"}}]),
    ("v10_key_enter", "type 'npm start' then hit the enter key",
     [{"tool": "type_text", "args": {"text": "npm start"}},
      {"tool": "press_key", "args": {"key": "enter"}}]),
    ("v10_key_press", "press enter please",
     [{"tool": "press_key", "args": {"key": "enter"}}]),
    ("v10_key_press", "hit escape",
     [{"tool": "press_key", "args": {"key": "esc"}}]),
    ("v10_key_press", "press the tab key",
     [{"tool": "press_key", "args": {"key": "tab"}}]),

    # --- Window: Fullscreen -> maximize_window -------------------------------
    ("v10_fullscreen", "Make Chrome full screen please",
     [{"tool": "maximize_window", "args": {"title_keyword": "chrome"}}]),
    ("v10_fullscreen", "make notepad full screen",
     [{"tool": "maximize_window", "args": {"title_keyword": "notepad"}}]),
    ("v10_fullscreen", "put active window in full screen",
     [{"tool": "maximize_window", "args": {"title_keyword": "active"}}]),
    ("v10_fullscreen", "put VS Code in full screen mode",
     [{"tool": "maximize_window", "args": {"title_keyword": "code"}}]),
]


def load_v8_and_v9_cases() -> List[Tuple[str, str, List[Dict[str, Any]]]]:
    # Load from build_v9_hardcase
    spec = importlib.util.spec_from_file_location(
        "v9_hc", ROOT / "scripts" / "build_v9_hardcase.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    v8_cases = mod.load_v8_cases()
    v9_cases = mod.V9_CASES
    return v8_cases + v9_cases


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def to_sample(prompt: str, calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant",
             "content": json.dumps(calls, ensure_ascii=False, separators=(",", ":"))},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    base_cases = load_v8_and_v9_cases()
    all_cases = base_cases + V10_NEW_CASES

    print(f"Total v10 hardcase replay cases: {len(all_cases)}")
    print(f"  - Base (v8 + v9): {len(base_cases)}")
    print(f"  - New v10 targeted: {len(V10_NEW_CASES)}")

    if args.report_only:
        return

    by_mode = defaultdict(list)
    for mode, prompt, calls in all_cases:
        by_mode[mode].append((prompt, calls))

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for mode, items in by_mode.items():
        if len(items) == 1:
            train.append(to_sample(items[0][0], items[0][1]))
        else:
            for p, c in items[:-1]:
                train.append(to_sample(p, c))
            val.append(to_sample(items[-1][0], items[-1][1]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", val)):
        path = OUT_DIR / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                        encoding="utf-8")
        print(f"Wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
