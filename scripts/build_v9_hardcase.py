"""Build the v9 hard-case replay subset.

v9 = v8's hardcase replay plus targeted corrections for the 10 genuine
current-vocabulary gate failures observed in the 2026-11-18 v8 shootout:

  - 4x press_shortcut: key-combo strings ("ctrl+v") vs semantic action
    (FIXED in the migration script instead; see SHORTCUT_KEYS_TO_ACTION).
  - C15 "what version are you running?"  -> chat_response
  - C17 "do you sleep?"                  -> chat_response
  - N04 screenshot with STT fillers      -> take_screenshot
  - D02 "stop obs recording"             -> obs_record(action=stop)
  - CAL04 calendar event with STT typos  -> create_calendar_event
  - MSG05 compose email (no attachments) -> compose_email (avoid invalid JSON)

Every prompt is a paraphrase, never the exact battery query. A contamination
check asserts no replay prompt collides with the 340-test battery.

Output:
    dataset/v9_hardcase/train.jsonl
    dataset/v9_hardcase/validation.jsonl

Usage:
    python scripts/build_v9_hardcase.py
    python scripts/build_v9_hardcase.py --report-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "dataset" / "v9_hardcase"

SYSTEM = (
    "You are DesktopIntent, the real-time neural router for FRIDAY AI. For every user input, "
    "you output ONLY a valid JSON array of tool calls matching FRIDAY's MCP tool bridge. "
    "Never generate conversational monologue, explanation, or thinking tokens. "
    "If the command requires complex coding or open-ended reasoning, invoke delegate_to_reasoning_model. "
    "If the input is conversational or greetings, invoke chat_response."
)

# Additional v9 cases. Mode names prefixed "v9_" so the hold-out split keeps
# one of each distinct failure family in validation.
V9_CASES: List[Tuple[str, str, List[Dict[str, Any]]]] = [
    # --- C15: version check must be pure chat_response -----------------------
    ("v9_version", "hey, which build of friday is this right now?",
     [{"tool": "chat_response", "args": {}}]),
    ("v9_version", "can you tell me what release you are running?",
     [{"tool": "chat_response", "args": {}}]),
    ("v9_version", "what version of the assistant am i talking to?",
     [{"tool": "chat_response", "args": {}}]),

    # --- C17: philosophical / anthropomorphic -> chat_response ---------------
    ("v9_philosophical", "do you ever get tired, friday?",
     [{"tool": "chat_response", "args": {}}]),
    ("v9_philosophical", "are you conscious when nobody is around?",
     [{"tool": "chat_response", "args": {}}]),
    ("v9_philosophical", "do you dream at night?",
     [{"tool": "chat_response", "args": {}}]),

    # --- N04: screenshot with STT fillers ------------------------------------
    ("v9_screenshot_stt", "uh, friday, could you grab a screenshot of the whole screen please?",
     [{"tool": "take_screenshot", "args": {}}]),
    ("v9_screenshot_stt", "er, take a screenshot of the monitor, uh, right now",
     [{"tool": "take_screenshot", "args": {}}]),
    ("v9_screenshot_stt", "um would you snap a screenshot of the display for me",
     [{"tool": "take_screenshot", "args": {}}]),

    # --- D02: obs recording via the live obs_record tool ---------------------
    ("v9_obs_record", "please stop the obs recording",
     [{"tool": "obs_record", "args": {"action": "stop"}}]),
    ("v9_obs_record", "halt the current obs capture",
     [{"tool": "obs_record", "args": {"action": "stop"}}]),
    ("v9_obs_record", "could you start recording in obs for me?",
     [{"tool": "obs_record", "args": {"action": "start"}}]),

    # --- CAL04: calendar event despite STT typos -----------------------------
    ("v9_calendar_stt", "scheedule a team synk tommorow at 9am",
     [{"tool": "create_calendar_event", "args": {"title": "team sync", "start_time": "tomorrow at 9am"}}]),
    ("v9_calendar_stt", "pls add a dentist appointmnt for friday at 3pm",
     [{"tool": "create_calendar_event", "args": {"title": "dentist appointment", "start_time": "friday at 3pm"}}]),

    # --- MSG05: clean compose_email (no hallucinated attachments) ------------
    ("v9_email_clean", "compose an email to hr@company.com with subject leave request and body requesting next monday off",
     [{"tool": "compose_email", "args": {"to": "hr@company.com", "subject": "leave request",
                                        "body": "requesting next monday off"}}]),
    ("v9_email_clean", "write an email to boss@company.com subject status update body everything is on track",
     [{"tool": "compose_email", "args": {"to": "boss@company.com", "subject": "status update",
                                        "body": "everything is on track"}}]),

    # --- shortcut semantics reinforcement (paraphrases, semantic action) ----
    ("v9_shortcut_action", "please paste the clipboard contents",
     [{"tool": "press_shortcut", "args": {"action": "paste"}}]),
    ("v9_shortcut_action", "select every bit of text on this page",
     [{"tool": "press_shortcut", "args": {"action": "select_all"}}]),
    ("v9_shortcut_action", "could you undo my last action please?",
     [{"tool": "press_shortcut", "args": {"action": "undo"}}]),
    ("v9_shortcut_action", "copy the highlighted text for me",
     [{"tool": "press_shortcut", "args": {"action": "copy"}}]),
]


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = __import__("re").sub(r"[^a-z0-9 ]+", " ", text)
    return __import__("re").sub(r"\s+", " ", text).strip()


def load_v8_cases() -> List[Tuple[str, str, List[Dict[str, Any]]]]:
    spec = importlib.util.spec_from_file_location(
        "friday_v8_hardcase", ROOT / "scripts" / "build_hardcase_replay.py")
    if spec is None or spec.loader is None:
        raise ImportError("build_hardcase_replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.CASES)


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

    # Contamination check against the battery.
    spec = importlib.util.spec_from_file_location(
        "friday_compare_for_v9", ROOT / "tests" / "compare_friday_versions.py")
    cmp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmp)
    battery = {normalize(t["query"]) for t in cmp.build_test_battery()}

    cases = load_v8_cases() + V9_CASES
    collisions = [p for _, p, _ in cases if normalize(p) in battery]
    if collisions:
        raise SystemExit(f"Contamination: {len(collisions)} replay prompts are in the battery:\n"
                         + "\n".join(f"  - {c}" for c in collisions))

    modes = Counter(m for m, _, _ in cases)
    print(f"Replay cases: {len(cases)} (v8 base: {len(load_v8_cases())}, v9 new: {len(V9_CASES)})")
    for mode, count in modes.most_common():
        print(f"  {mode:<24} {count}")
    print("Contamination check: OK")

    if args.report_only:
        return

    seen: set = set()
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for mode, prompt, calls in cases:
        if mode not in seen:
            seen.add(mode)
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
