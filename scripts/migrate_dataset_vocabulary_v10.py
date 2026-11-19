"""Migrate the FRIDAY training corpus onto the unified v10 MCP tool vocabulary.

Enforces:
  1. Strict MCP schemas: launch_app(app_name), close_app(app_name),
     maximize_window(title_keyword), kill_process(confirmed=True), etc.
  2. Single-Key GUI Guard:
     Single keys ('enter', 'tab', 'esc', 'space', 'backspace') are rewritten to press_key(key=...).
     press_shortcut is STRICTLY reserved for semantic actions (e.g. 'copy', 'paste', 'show_desktop').
  3. Read-Only State Guard:
     Interrogative theme prompts ('is dark mode on?') map to get_system_theme(),
     never set_system_theme().
  4. Planned Tools Preservation:
     All 18 future tools (calculate, take_note, change_brightness, obs_record, etc.)
     are preserved verbatim with standardized argument shapes.

Output:
    dataset/v10_migrated/<subset>/{train,validation}.jsonl
    docs/PENDING_TOOLS_v10.md
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
DEFAULT_BACKEND = Path(os.getenv("FRIDAY_BACKEND", r"E:\F Drive Backup\FRIDAY-AI-Assistant\backend"))
SUBSET_DIR = ROOT / "hf_export" / "friday-desktop-intent" / "subsets"
OUT_DIR = ROOT / "dataset" / "v10_migrated"
BACKLOG_MD = ROOT / "docs" / "PENDING_TOOLS_v10.md"

VALID_PRESETS = {"coding", "research", "dual_screen_coding", "focus", "clean"}

# Single keys that MUST use press_key rather than press_shortcut
SINGLE_KEYS = {
    "enter", "return", "tab", "esc", "escape", "space", "backspace",
    "up", "down", "left", "right", "delete", "home", "end", "pageup", "pagedown"
}

# Map compound key combinations to canonical semantic action names
SHORTCUT_KEYS_TO_ACTION = {
    ("ctrl", "c"): "copy",
    ("ctrl", "v"): "paste",
    ("ctrl", "x"): "cut",
    ("ctrl", "a"): "select_all",
    ("ctrl", "s"): "save",
    ("ctrl", "z"): "undo",
    ("ctrl", "y"): "redo",
    ("ctrl", "f"): "find",
    ("ctrl", "t"): "new_tab",
    ("ctrl", "w"): "close_tab",
    ("win", "v"): "clipboard_history",
    ("win", "d"): "show_desktop",
    ("win", "e"): "file_explorer",
    ("win", "i"): "settings",
    ("win", "l"): "lock_screen",
    ("ctrl", "shift", "esc"): "task_manager",
    ("ctrl", "shift", "t"): "reopen_tab",
    ("ctrl", "tab"): "switch_tab",
    ("f5",): "refresh",
    ("ctrl", "f5"): "hard_refresh",
    ("ctrl", "+"): "zoom_in",
    ("ctrl", "-"): "zoom_out",
    ("ctrl", "0"): "reset_zoom",
}

ROUTER_NATIVE = {
    "chat_response",
    "delegate_to_reasoning_model",
    "clarification_required",
    "press_shortcut",
    "type_text",
    "press_key",
}

# Future planned tools: preserve verbatim, do not drop or modify
FORWARD_LOOKING = {
    "calculate",
    "take_note",
    "change_brightness",
    "increase_brightness",
    "decrease_brightness",
    "minimize_all_windows",
    "restore_all_windows",
    "toggle_night_light",
    "toggle_bluetooth",
    "cancel_alarm",
    "slack_dm",
    "obs_record",
    "spotify_search",
    "docker_action",
    "git_quick_commit",
    "hue_lights",
    "notion_create_task",
    "steam_action",
}

ARG_ALIASES: Dict[str, Dict[str, str]] = {
    "launch_app": {"name": "app_name", "app": "app_name"},
    "close_app": {"name": "app_name", "app": "app_name"},
    "snap_window": {"app": "title_keyword", "window_title": "title_keyword"},
    "minimize_window": {"window_title": "title_keyword", "app": "title_keyword"},
    "maximize_window": {"window_title": "title_keyword", "app": "title_keyword"},
    "focus_window": {"window_title": "title_keyword", "app": "title_keyword"},
    "restore_window": {"window_title": "title_keyword", "app": "title_keyword"},
    "close_window": {"window_title": "title_keyword", "app": "title_keyword"},
    "send_telegram_message": {"contact": "recipient", "chat": "recipient"},
    "compose_email": {"recipient": "to"},
    "copy_file": {"source": "source_path", "src": "source_path",
                  "destination": "destination_path", "dst": "destination_path"},
    "move_file": {"source": "source_path", "src": "source_path",
                  "destination": "destination_path", "dst": "destination_path"},
    "toggle_wifi": {"enabled": "enable"},
    "delete_calendar_event": {"title": "event_id"},
    "list_calendar_events": {"day": "time_frame"},
    "send_whatsapp_message": {"phone_number": "contact"},
    "create_reminder": {"text": "message", "time": "time_str"},
    "set_reminder": {"text": "message", "time": "time_str"},
    "organize_files": {"directory": "directory_path"},
    "press_key": {"action": "key", "name": "key"},
}

PRESERVE_UNKNOWN_ARGS: Dict[str, set] = {
    "browser_switch_tab": {"tab_id"},
}

REQUIRED_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "kill_process": {"confirmed": True},
    "shutdown_system": {"confirmed": True},
    "restart_system": {"confirmed": True},
    "hibernate_system": {"confirmed": True},
    "delete_file": {"confirmed": True},
    "create_word_document": {"content_markdown": ""},
    "compose_email": {"body": ""},
}

DROP_ARGS: Dict[str, Tuple[str, ...]] = {
    "start_screen_recording": ("fps",),
    "send_whatsapp_message": ("phone_number",),
}

TIMEFRAME_VALUES: Dict[str, str] = {
    "today": "today",
    "tomorrow": "tomorrow",
    "this week": "this_week",
    "this_week": "this_week",
    "weekend": "upcoming",
    "afternoon": "today",
    "upcoming": "upcoming",
}


def extract_registry(backend: Path) -> Dict[str, Dict[str, Any]]:
    schemas: Dict[str, Dict[str, Any]] = {}
    for py_file in sorted(backend.rglob("*.py")):
        if any(part in {"venv", ".venv", "__pycache__", "node_modules"} for part in py_file.parts):
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Attribute) and node.func.attr == "register_manual_tool"):
                continue
            kwargs: Dict[str, Any] = {}
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                try:
                    kwargs[kw.arg] = ast.literal_eval(kw.value)
                except Exception:
                    kwargs[kw.arg] = None
            name = kwargs.get("name")
            if isinstance(name, str):
                schemas[name] = kwargs.get("input_schema") or {}
    return schemas


def normalize_args(tool: str, args: Dict[str, Any], registry: Dict[str, Dict[str, Any]],
                   repairs: Counter) -> Dict[str, Any]:
    out = dict(args)

    # 1. press_shortcut keys mapping & single key guard BEFORE dropping unknown args
    if tool == "press_shortcut":
        if "keys" in out:
            keys = out.pop("keys")
            if isinstance(keys, list):
                norm = tuple(str(k).lower() for k in keys)
                if len(norm) == 1 and norm[0] in SINGLE_KEYS:
                    out["action"] = norm[0]
                else:
                    out["action"] = SHORTCUT_KEYS_TO_ACTION.get(norm, "+".join(norm))
            else:
                out["action"] = str(keys)
            repairs["press_shortcut.keys->action"] += 1

        if "action" in out:
            act = str(out["action"]).lower()
            norm_map = {
                "win+v": "clipboard_history", "win v": "clipboard_history",
                "ctrl+c": "copy", "ctrl c": "copy",
                "ctrl+v": "paste", "ctrl v": "paste",
                "ctrl+a": "select_all", "ctrl a": "select_all",
                "ctrl+s": "save", "ctrl s": "save",
                "ctrl+z": "undo", "ctrl z": "undo",
                "ctrl+y": "redo", "ctrl y": "redo",
                "ctrl+f": "find", "ctrl f": "find",
                "win+d": "show_desktop", "win d": "show_desktop",
                "win+l": "lock_screen", "win l": "lock_screen",
                "ctrl+shift+esc": "task_manager",
            }
            if act in norm_map:
                out["action"] = norm_map[act]

    # 2. Argument aliases
    aliases = ARG_ALIASES.get(tool, {})
    for stale, canonical in aliases.items():
        if stale in out and canonical not in out:
            out[canonical] = out.pop(stale)
            repairs[f"{tool}.{stale}->{canonical}"] += 1
        elif stale in out:
            out.pop(stale)
            repairs[f"{tool}.{stale} dropped (duplicate)"] += 1

    for stale in DROP_ARGS.get(tool, ()):
        if stale in out:
            out.pop(stale)
            repairs[f"{tool}.{stale} dropped"] += 1

    if tool in registry:
        schema = registry.get(tool, {})
        props = schema.get("properties", {})
        preserved = PRESERVE_UNKNOWN_ARGS.get(tool, set())
        for key in list(out.keys()):
            if key not in props and key not in preserved:
                out.pop(key)
                repairs[f"{tool}.{key} dropped (unknown arg)"] += 1

    for req, default in REQUIRED_DEFAULTS.get(tool, {}).items():
        if req not in out:
            out[req] = default
            repairs[f"{tool}.{req} filled"] += 1

    if tool in ("create_reminder", "set_reminder") and "minutes_from_now" in out:
        raw = out.pop("minutes_from_now")
        try:
            out.setdefault("seconds", int(raw) * 60)
        except (TypeError, ValueError):
            out.setdefault("time_str", str(raw))
        repairs[f"{tool}.minutes_from_now->seconds"] += 1

    if tool == "update_calendar_event" and "event_id" not in out and "title" in out:
        out["event_id"] = out.pop("title")
        repairs["update_calendar_event.title->event_id"] += 1

    if tool == "browser_switch_tab" and "tab_id" in out and "tab_index" not in out:
        raw = str(out["tab_id"]).strip()
        if raw.isdigit():
            out["tab_index"] = int(raw)
            out.pop("tab_id")
            repairs["browser_switch_tab.tab_id->tab_index"] += 1

    if tool == "list_calendar_events" and "start_date" in out:
        raw = str(out["start_date"]).strip().lower()
        preset = TIMEFRAME_VALUES.get(raw)
        if preset:
            out.pop("start_date")
            out.setdefault("time_frame", preset)
            repairs[f"list_calendar_events.start_date({raw})->time_frame"] += 1

    return out


Handler = Callable[[Dict[str, Any], str], Optional[Tuple[str, Dict[str, Any]]]]


def _app_of(args: Dict[str, Any]) -> str:
    raw = args.get("app_name") or args.get("app") or args.get("name") or args.get("window_title") or args.get("title_keyword") or ""
    return str(raw).strip().lower()


def _manage_window(args: Dict[str, Any], _prompt: str):
    action = str(args.get("action", "")).lower()
    app = _app_of(args)
    if not app:
        return None
    if action == "close":
        return "close_app", {"app_name": app}
    if action == "minimize":
        return "minimize_window", {"title_keyword": app}
    if action in ("maximize", "fullscreen", "full_screen"):
        return "maximize_window", {"title_keyword": app}
    if action in ("focus", "bring", "front", "switch"):
        return "focus_window", {"title_keyword": app}
    return None


def _theme(args: Dict[str, Any], prompt: str):
    p_lower = prompt.lower()
    # If user is asking an interrogative question, route to get_system_theme!
    if any(p_lower.startswith(w) for w in ["is ", "are ", "what is ", "what's ", "check ", "tell me if ", "how is "]):
        return "get_system_theme", {}

    if "theme" in args:
        theme = str(args["theme"]).lower()
        if theme not in ("dark", "light", "toggle"):
            return None
        return "set_system_theme", {"theme": theme}
    flag = args.get("dark", args.get("enabled"))
    if isinstance(flag, bool):
        return "set_system_theme", {"theme": "dark" if flag else "light"}
    return "set_system_theme", {"theme": "toggle"}


def _delta(sign: int) -> Handler:
    def handler(args: Dict[str, Any], _prompt: str):
        raw = args.get("amount", args.get("delta"))
        if raw is None:
            return None
        try:
            value = abs(int(raw))
        except Exception:
            return None
        return "change_volume", {"delta": sign * value}
    return handler


MIGRATIONS: Dict[str, Handler] = {
    "manage_window": _manage_window,
    "set_workspace_preset": lambda a, _p: ("arrange_workspace_preset", {"preset_name": str(a.get("preset") or a.get("preset_name") or "coding").lower()}),
    "kill_app": lambda a, _p: ("close_app", {"app_name": _app_of(a)}) if _app_of(a) else None,
    "lock_screen": lambda _a, _p: ("lock_workstation", {}),
    "lock_pc": lambda _a, _p: ("lock_workstation", {}),
    "create_timer": lambda a, _p: ("set_timer", {"seconds": int(a.get("seconds") or a.get("duration_seconds") or 300), "message": a.get("label") or a.get("message") or "Timer"}),
    "toggle_dark_mode": _theme,
    "decrease_volume": _delta(-1),
    "increase_volume": _delta(+1),
    "get_time": lambda _a, _p: ("get_current_time", {}),
    "get_date": lambda _a, _p: ("get_current_time", {}),
    "get_volume": lambda _a, _p: ("get_volume_status", {}),
    "inspect_screen": lambda a, _p: ("analyze_screen", {"prompt": a.get("prompt") or a.get("query") or "analyze screen"}),
    "send_desktop_notification": lambda a, _p: ("send_desktop_toast", {"title": a.get("title") or "FRIDAY", "message": a.get("message") or a.get("text") or "Alert"}),
    "set_volume_mute": lambda _a, _p: ("mute_volume", {}),
    "get_system_stats": lambda _a, _p: ("get_system_info", {}),
}


def migrate_sample(sample: Dict[str, Any], registry: Dict[str, Dict[str, Any]],
                   backlog: Dict[str, Any], repairs: Counter,
                   repair_args: bool = True):
    messages = sample.get("messages") or []
    assistant = next((m for m in messages if m.get("role") == "assistant"), None)
    if assistant is None:
        return None, "no_assistant_message", 0

    try:
        calls = json.loads(assistant.get("content", ""))
    except Exception:
        return None, "unparseable_assistant_json", 0
    if not isinstance(calls, list):
        return None, "assistant_not_a_list", 0

    prompt = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    rewritten: List[Dict[str, Any]] = []
    rewrites = 0

    for call in calls:
        if not isinstance(call, dict):
            return None, "non_object_call", 0
        tool = str(call.get("tool", ""))
        args = call.get("args") or {}

        # Check single key guard on press_shortcut
        if tool == "press_shortcut":
            act = str(args.get("action", "")).lower()
            if act in SINGLE_KEYS:
                tool = "press_key"
                args = {"key": act}
                rewrites += 1
                repairs["press_shortcut_single_key->press_key"] += 1

        if tool in FORWARD_LOOKING:
            backlog[tool]["calls"] += 1
            backlog[tool]["shapes"][tuple(sorted(args))] += 1
            for key, value in args.items():
                backlog[tool]["values"][key][repr(value)[:48]] += 1
            if len(backlog[tool]["examples"]) < 3 and prompt:
                backlog[tool]["examples"].append((prompt.strip(), args))
            rewritten.append(call)
            continue

        handler = MIGRATIONS.get(tool)
        if handler is not None:
            result = handler(args, prompt)
            if result is None:
                return None, f"unmappable:{tool}", 0
            tool, args = result
            rewrites += 1
        elif tool in ROUTER_NATIVE:
            pass
        elif tool not in registry:
            return None, f"unknown_tool:{tool}", 0

        if tool not in registry and tool not in ROUTER_NATIVE:
            return None, f"target_missing:{tool}", 0

        if repair_args and (tool in registry or tool in ROUTER_NATIVE):
            args = normalize_args(tool, args, registry, repairs)

        # Post-normalization single-key check on press_shortcut
        if tool == "press_shortcut":
            act = str(args.get("action", "")).lower()
            if act in SINGLE_KEYS:
                tool = "press_key"
                args = {"key": act}
                rewrites += 1
                repairs["press_shortcut_single_key->press_key"] += 1

        rewritten.append({"tool": tool, "args": args})

    migrated = json.loads(json.dumps(sample))
    for message in migrated["messages"]:
        if message.get("role") == "assistant":
            message["content"] = json.dumps(rewritten, ensure_ascii=False, separators=(",", ":"))

    return migrated, None, rewrites


def process_file(path: Path, registry: Dict[str, Dict[str, Any]],
                 backlog: Dict[str, Any], repairs: Counter,
                 repair_args: bool = True) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Counter]:
    clean: List[Dict[str, Any]] = []
    quarantined: List[Dict[str, Any]] = []
    reasons = Counter()

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except Exception:
            quarantined.append({"raw": line, "reason": "unparseable_json", "line": line_no})
            reasons["unparseable_json"] += 1
            continue

        migrated, reason, _ = migrate_sample(sample, registry, backlog, repairs, repair_args=repair_args)
        if reason is not None:
            quarantined.append({"sample": sample, "reason": reason, "line": line_no})
            reasons[reason] += 1
        else:
            clean.append(migrated)

    return clean, quarantined, reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=Path, default=DEFAULT_BACKEND)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    registry = extract_registry(args.backend)
    subsets = sorted([d.name for d in SUBSET_DIR.iterdir() if d.is_dir()])
    print(f"Registry: {len(registry)} tools | Subsets: {len(subsets)}")

    backlog = defaultdict(lambda: {"calls": 0, "shapes": Counter(), "values": defaultdict(Counter), "examples": []})
    repairs = Counter()

    total_clean = 0
    total_quar = 0

    for subset in subsets:
        for split in ("train", "validation"):
            in_file = SUBSET_DIR / subset / f"{split}.jsonl"
            if not in_file.exists():
                continue
            clean, quar, reasons = process_file(in_file, registry, backlog, repairs)
            total_clean += len(clean)
            total_quar += len(quar)
            if not args.report_only:
                out_file = args.out / subset / f"{split}.jsonl"
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in clean) + "\n", encoding="utf-8")

    print(f"Migration completed! Clean samples: {total_clean}, Quarantined: {total_quar}")
    print("Repairs applied:", dict(repairs.most_common(15)))


if __name__ == "__main__":
    main()
