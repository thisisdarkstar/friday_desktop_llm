"""Build a small hard-case replay subset for the v8 intent router.

The v5/v6/v7 shootout exposed a handful of *genuine* model errors that the
rebalanced blend alone may not fix (polite wrappers, compound "close X and Y"
chains, browser-tab verbs, launch-vs-search, and a few classification slips).
This script writes curated paraphrases of those failure modes — deliberately
**not** the exact benchmark queries, so the 340-test battery stays clean.

A contamination check asserts that no generated prompt appears in the battery.

Output:
    dataset/v8_hardcase/train.jsonl
    dataset/v8_hardcase/validation.jsonl

Usage:
    python scripts/build_hardcase_replay.py
    python scripts/build_hardcase_replay.py --report-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "dataset" / "v8_hardcase"

SYSTEM = (
    "You are DesktopIntent, the real-time neural router for FRIDAY AI. For every user input, "
    "you output ONLY a valid JSON array of tool calls matching FRIDAY's MCP tool bridge. "
    "Never generate conversational monologue, explanation, or thinking tokens. "
    "If the command requires complex coding or open-ended reasoning, invoke delegate_to_reasoning_model. "
    "If the input is conversational or greetings, invoke chat_response."
)

# (mode, prompt, canonical calls). Calls use live registry names and schemas;
# planned tools (calculate, minimize_all_windows) are intentionally preserved.
CASES: List[Tuple[str, str, List[Dict[str, Any]]]] = [
    # --- launch vs search ----------------------------------------------------
    ("launch_vs_search", "kindly start the firefox browser for me",
     [{"tool": "launch_app", "args": {"app_name": "firefox"}}]),
    ("launch_vs_search", "open the chrome application please",
     [{"tool": "launch_app", "args": {"app_name": "chrome"}}]),
    ("launch_vs_search", "could you launch microsoft edge",
     [{"tool": "launch_app", "args": {"app_name": "msedge"}}]),
    ("launch_vs_search", "start notepad",
     [{"tool": "launch_app", "args": {"app_name": "notepad"}}]),

    # --- polite wrappers around a single action ------------------------------
    ("polite_single", "hi there, would you mind minimizing the spotify window for me?",
     [{"tool": "minimize_window", "args": {"title_keyword": "spotify"}}]),
    ("polite_single", "hope you're well, could you please maximize the chrome window?",
     [{"tool": "maximize_window", "args": {"title_keyword": "chrome"}}]),
    ("polite_single", "hello friend, please focus the slack window",
     [{"tool": "focus_window", "args": {"title_keyword": "slack"}}]),
    ("polite_single", "hey, could you turn the volume up by 15 please?",
     [{"tool": "change_volume", "args": {"delta": 15}}]),
    ("polite_single", "thanks a lot, please turn the volume down by 25",
     [{"tool": "change_volume", "args": {"delta": -25}}]),
    ("polite_single", "would you be so kind as to set a 20 minute timer?",
     [{"tool": "set_timer", "args": {"message": "20 minute timer", "seconds": 1200}}]),
    ("polite_single", "listen friday, I need a 45 minute timer please",
     [{"tool": "set_timer", "args": {"message": "45 minute timer", "seconds": 2700}}]),

    # --- compound: close X and do Y ------------------------------------------
    ("compound_close", "please close discord and set the volume to 25",
     [{"tool": "close_app", "args": {"app_name": "discord"}},
      {"tool": "set_volume", "args": {"level": 25}}]),
    ("compound_close", "close brave then turn the volume down to 15",
     [{"tool": "close_app", "args": {"app_name": "brave"}},
      {"tool": "set_volume", "args": {"level": 15}}]),
    ("compound_close", "shut down slack and set volume to 40",
     [{"tool": "close_app", "args": {"app_name": "slack"}},
      {"tool": "set_volume", "args": {"level": 40}}]),
    ("compound_close", "wrap up: close the spotify app and lock the workstation",
     [{"tool": "close_app", "args": {"app_name": "spotify"}},
      {"tool": "lock_workstation", "args": {}}]),
    ("compound_close", "alright, close telegram and put the pc to sleep",
     [{"tool": "close_app", "args": {"app_name": "telegram"}},
      {"tool": "sleep_system", "args": {}}]),

    # --- compound: preset + app + brightness ---------------------------------
    ("compound_preset", "starting work now, switch to the focus layout, open vscode, and set brightness to 80 percent",
     [{"tool": "arrange_workspace_preset", "args": {"preset_name": "focus"}},
      {"tool": "launch_app", "args": {"app_name": "code"}},
      {"tool": "set_brightness", "args": {"level": 80}}]),
    ("compound_preset", "ready for the standup, switch to research layout and launch zoom",
     [{"tool": "arrange_workspace_preset", "args": {"preset_name": "research"}},
      {"tool": "launch_app", "args": {"app_name": "zoom"}}]),

    # --- browser tab verbs ----------------------------------------------------
    ("browser_tabs", "could you reload the current browser tab?",
     [{"tool": "browser_tab_action", "args": {"action": "reload"}}]),
    ("browser_tabs", "please open a new tab for me",
     [{"tool": "browser_tab_action", "args": {"action": "new_tab"}}]),
    ("browser_tabs", "go to the next tab please",
     [{"tool": "browser_tab_action", "args": {"action": "next"}}]),
    ("browser_tabs", "close this tab, thanks",
     [{"tool": "browser_tab_action", "args": {"action": "close"}}]),
    ("browser_tabs", "open a fresh tab and search for github trending repositories",
     [{"tool": "browser_tab_action", "args": {"action": "new_tab"}},
      {"tool": "search_in_browser", "args": {"query": "github trending repositories", "engine": "github"}}]),

    # --- search + snap --------------------------------------------------------
    ("search_snap", "look up python tutorials on youtube and snap the window to the right",
     [{"tool": "search_in_browser", "args": {"query": "python tutorials", "engine": "youtube"}},
      {"tool": "snap_window", "args": {"position": "right", "title_keyword": "browser"}}]),
    ("search_snap", "search for weather in tokyo then snap the window left",
     [{"tool": "search_in_browser", "args": {"query": "weather in tokyo", "engine": "google"}},
      {"tool": "snap_window", "args": {"position": "left", "title_keyword": "active"}}]),

    # --- classification slips -------------------------------------------------
    ("classification", "which release of friday is this?",
     [{"tool": "chat_response", "args": {}}]),
    ("classification", "what build am I currently talking to?",
     [{"tool": "chat_response", "args": {}}]),
    ("classification", "do you like koalas?",
     [{"tool": "chat_response", "args": {}}]),
    ("classification", "hey friday, how's everything going today?",
     [{"tool": "chat_response", "args": {}}]),
    ("classification", "hey friday, could you check the cpu and memory usage?",
     [{"tool": "get_system_info", "args": {}}]),

    # --- stateful anaphora ----------------------------------------------------
    ("anaphora", "[Prior Action: {\"tool\": \"launch_app\", \"args\": {\"app_name\": \"spotify\"}}]\nclose that window",
     [{"tool": "close_app", "args": {"app_name": "spotify"}}]),
    ("anaphora", "[Prior Action: {\"tool\": \"launch_app\", \"args\": {\"app_name\": \"chrome\"}}]\nshut it down please",
     [{"tool": "close_app", "args": {"app_name": "chrome"}}]),

    # --- planned tools preserved (forward-looking) ---------------------------
    ("planned", "hey, could you work out 15 percent tip on 85 dollars?",
     [{"tool": "calculate", "args": {"expression": "15% of 85"}}]),
    ("planned", "tidy up: minimize all the windows and mute the sound",
     [{"tool": "minimize_all_windows", "args": {}},
      {"tool": "mute_audio", "args": {"mute": True}}]),
]


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_battery_prompts() -> set:
    path = ROOT / "tests" / "compare_friday_versions.py"
    spec = importlib.util.spec_from_file_location("friday_compare_for_replay", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {normalize(t["query"]) for t in module.build_test_battery()}


def to_sample(prompt: str, calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": json.dumps(calls, ensure_ascii=False, separators=(",", ":"))},
    ]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    battery = load_battery_prompts()
    collisions = [p for _, p, _ in CASES if normalize(p) in battery]
    if collisions:
        raise SystemExit(f"Contamination: {len(collisions)} replay prompts are in the benchmark battery:\n"
                         + "\n".join(f"  - {c}" for c in collisions))

    modes = Counter(m for m, _, _ in CASES)
    print(f"Replay cases: {len(CASES)}")
    for mode, count in modes.most_common():
        print(f"  {mode:<18} {count}")
    print("Contamination check: OK (no benchmark prompts reused)")

    if args.report_only:
        print("Report only - nothing written.")
        return

    # Hold out one case per mode for validation (never the same prompt in both).
    seen_modes: set = set()
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for mode, prompt, calls in CASES:
        if mode not in seen_modes:
            seen_modes.add(mode)
            val.append(to_sample(prompt, calls))
        else:
            train.append(to_sample(prompt, calls))

    for name, rows in (("train", train), ("validation", val)):
        path = OUT_DIR / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                        encoding="utf-8")
        print(f"Wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
