"""Migrate the FRIDAY training corpus onto the current MCP tool vocabulary.

Two distinct classes of "missing" tool show up in the corpus, and they need
opposite treatment:

  1. RENAMED - the registry has a live replacement under a different name.
     These are migrated (name + argument schema), e.g.
         manage_window{action:"minimize", app:"spotify"}
             -> minimize_window{title_keyword:"spotify"}
         set_workspace_preset{preset:"coding"}
             -> arrange_workspace_preset{preset_name:"coding"}
         create_timer{duration_seconds:300, label:"Timer"}
             -> set_timer{seconds:300, message:"Timer"}

  2. PLANNED - not implemented in FRIDAY yet, but intentionally represented in
     the corpus so the intent model is future-proof. These are PRESERVED
     verbatim and collected into an implementation backlog:
         calculate, take_note, change_brightness, minimize_all_windows,
         restore_all_windows, toggle_night_light, toggle_bluetooth, cancel_alarm

Only calls that are neither migrated, preserved, registered, nor router-native
are quarantined (genuinely unknown names).

Non-destructive: input files are never modified.

Output:
    dataset/v8_migrated/<subset>/{train,validation}.jsonl
    dataset/v8_migrated/quarantine/<subset>_<split>.jsonl
    docs/PENDING_TOOLS.md

Usage:
    python scripts/migrate_dataset_vocabulary.py                 # write output
    python scripts/migrate_dataset_vocabulary.py --report-only   # dry run
    python scripts/migrate_dataset_vocabulary.py --backend PATH
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
DEFAULT_BACKEND = Path(os.getenv("FRIDAY_BACKEND", r"E:\F Drive Backup\FRIDAY-AI-Assistant\backend"))
SUBSET_DIR = ROOT / "hf_export" / "friday-desktop-intent" / "subsets"
OUT_DIR = ROOT / "dataset" / "v8_migrated"
BACKLOG_MD = ROOT / "docs" / "PENDING_TOOLS.md"
SCHEMA_GAPS_MD = ROOT / "docs" / "SCHEMA_GAPS.md"

# Values accepted by arrange_workspace_preset (backend/tools/apps.py).
# "clean" is used by 6 corpus samples but is not in the enum yet.
VALID_PRESETS = {"coding", "research", "dual_screen_coding", "focus", "clean"}

# Reverse map for press_shortcut: original samples use ``keys: ["ctrl", "v"]``.
# The runtime (backend/tools/gui.py COMMON_SHORTCUTS) accepts either a semantic
# action name or a key list, but NOT a joined string like "ctrl+v" in ``action``.
# v8's migration produced that broken form and cost 4 gate tests; v9 maps back
# to the canonical semantic name instead.
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
}

# Pseudo-tools the router handles before MCP dispatch.
ROUTER_NATIVE = {
    "chat_response",
    "delegate_to_reasoning_model",
    "clarification_required",
    "press_shortcut",
    "type_text",
    "press_key",
}

# Class 2: intentional but not implemented. Preserved verbatim, never rewritten.
FORWARD_LOOKING = {
    "calculate",            # arithmetic evaluation
    "take_note",            # note capture
    "change_brightness",    # relative brightness delta
    "increase_brightness",
    "decrease_brightness",
    "minimize_all_windows", # bulk window management
    "restore_all_windows",
    "toggle_night_light",   # real toggle (open_system_setting only opens Settings)
    "toggle_bluetooth",
    "cancel_alarm",         # set_alarm exists, cancel does not
    "slack_dm",
    # Dynamic runtime tools: injected as schemas at runtime, not registered.
    "obs_record",
    "spotify_search",
    "docker_action",
    "git_quick_commit",
    "hue_lights",
    "notion_create_task",
    "steam_action",
}


# ---------------------------------------------------------------------------
# Argument-schema repair
# ---------------------------------------------------------------------------
# Even when the tool NAME is live, the corpus often uses an older argument key.
# Left alone, the model learns to emit calls the handler rejects. Each entry
# maps stale key -> canonical key; the stale key is removed after the move so
# the model learns exactly one spelling.
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
}

# Live tools whose schema cannot express a corpus intent yet. The argument is
# preserved (never dropped) and the tool is reported as a schema gap.
PRESERVE_UNKNOWN_ARGS: Dict[str, set] = {
    "browser_switch_tab": {"tab_id"},  # only accepts tab_index; corpus names tabs
}

KNOWN_SCHEMA_GAPS: Dict[str, str] = {
    "browser_switch_tab": "corpus targets tabs by name ('switch to docs tab'); "
                          "schema only accepts tab_index:int",
}

# Canonical values the corpus omits but the handler requires.
REQUIRED_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "kill_process": {"confirmed": True},
    "shutdown_system": {"confirmed": True},
    "restart_system": {"confirmed": True},
    "hibernate_system": {"confirmed": True},
    "delete_file": {"confirmed": True},
    "create_word_document": {"content_markdown": ""},
    "compose_email": {"body": ""},
}

# Keys that no live schema consumes and that carry no intent worth keeping.
DROP_ARGS: Dict[str, Tuple[str, ...]] = {
    "start_screen_recording": ("fps",),
    "send_whatsapp_message": ("phone_number",),  # already folded into contact
}

# Values seen in list_calendar_events{start_date} that are really time_frame presets.
TIMEFRAME_VALUES: Dict[str, str] = {
    "today": "today",
    "tomorrow": "tomorrow",
    "this week": "this_week",
    "this_week": "this_week",
    "weekend": "upcoming",
    "afternoon": "today",
    "upcoming": "upcoming",
    "all": "all",
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Argument repair for calls whose tool name is already live
# ---------------------------------------------------------------------------
def normalize_args(tool: str, args: Dict[str, Any], registry: Dict[str, Dict[str, Any]],
                   repairs: Counter) -> Dict[str, Any]:
    """Move stale argument keys onto the live schema and fill required defaults."""
    out = dict(args or {})
    schema = registry.get(tool) or {}
    props = schema.get("properties") or {}

    # create_reminder/set_reminder historically used minutes_from_now.
    if tool in ("create_reminder", "set_reminder") and "minutes_from_now" in out:
        raw = out.pop("minutes_from_now")
        try:
            out.setdefault("seconds", int(raw) * 60)
        except (TypeError, ValueError):
            out.setdefault("time_str", str(raw))
        repairs[f"{tool}.minutes_from_now->seconds"] += 1

    # press_shortcut used a list of keys. Map to the canonical *semantic* action
    # name where known (v8 emitted "ctrl+v" which the runtime/validator reject).
    if tool == "press_shortcut" and "keys" in out:
        keys = out.pop("keys")
        if isinstance(keys, list):
            norm = tuple(str(k).lower() for k in keys)
            out.setdefault("action", SHORTCUT_KEYS_TO_ACTION.get(norm, "+".join(norm)))
        else:
            out.setdefault("action", str(keys))
        repairs["press_shortcut.keys->action"] += 1

    # update_calendar_event used title as the target identifier.
    if tool == "update_calendar_event" and "event_id" not in out and "title" in out:
        out["event_id"] = out.pop("title")
        repairs["update_calendar_event.title->event_id"] += 1

    # browser_switch_tab only accepts an integer index.
    if tool == "browser_switch_tab" and "tab_id" in out and "tab_index" not in out:
        raw = str(out["tab_id"]).strip()
        if raw.isdigit():
            out["tab_index"] = int(raw)
            out.pop("tab_id")
            repairs["browser_switch_tab.tab_id->tab_index"] += 1

    # list_calendar_events start_date sometimes holds a preset keyword.
    if tool == "list_calendar_events" and "start_date" in out:
        raw = str(out["start_date"]).strip().lower()
        preset = TIMEFRAME_VALUES.get(raw)
        if preset:
            out.pop("start_date")
            out.setdefault("time_frame", preset)
            repairs[f"list_calendar_events.start_date({raw})->time_frame"] += 1

    # read_emails used filter="unread" instead of unread_only=True.
    if tool == "read_emails" and "filter" in out:
        raw = str(out.pop("filter")).strip().lower()
        if raw in ("unread", "unread_only", "unread only"):
            out.setdefault("unread_only", True)
        elif raw:
            out.setdefault("query", raw)
        repairs["read_emails.filter->unread_only/query"] += 1

    for old, new in ARG_ALIASES.get(tool, {}).items():
        if old not in out:
            continue
        if new not in out:
            out[new] = out.pop(old)
            repairs[f"{tool}.{old}->{new}"] += 1
        else:
            out.pop(old)
            repairs[f"{tool}.{old} dropped (duplicate)"] += 1

    for key in DROP_ARGS.get(tool, ()):
        if key in out:
            out.pop(key)
            repairs[f"{tool}.{key} dropped"] += 1

    for key, value in REQUIRED_DEFAULTS.get(tool, {}).items():
        if key not in out:
            out[key] = value
            repairs[f"{tool}.{key} filled"] += 1

    # Drop keys the schema does not define, but only when a schema is known and
    # the key is not an intentional gap preserved above.
    if props:
        preserved = PRESERVE_UNKNOWN_ARGS.get(tool, set())
        for key in list(out):
            if key not in props:
                if key in preserved:
                    repairs[f"{tool}.{key} preserved (schema gap)"] += 1
                else:
                    out.pop(key)
                    repairs[f"{tool}.{key} unknown dropped"] += 1

    return out


# ---------------------------------------------------------------------------
# Class 1 migrations: args -> (new_tool, new_args) or None when unmappable
# ---------------------------------------------------------------------------
Handler = Callable[[Dict[str, Any]], Optional[Tuple[str, Dict[str, Any]]]]


def _app_of(args: Dict[str, Any]) -> Optional[str]:
    return args.get("app") or args.get("name") or args.get("app_name") or args.get("title_keyword")


def _manage_window(args: Dict[str, Any]):
    action = str(args.get("action", "")).lower()
    app = _app_of(args)
    if not app:
        return None
    if action == "close":
        return "close_app", {"app_name": app}
    if action == "minimize":
        return "minimize_window", {"title_keyword": app}
    if action == "maximize":
        return "maximize_window", {"title_keyword": app}
    if action in ("focus", "bring", "front", "switch"):
        return "focus_window", {"title_keyword": app}
    return None


def _workspace_preset(args: Dict[str, Any]):
    preset = str(args.get("preset") or args.get("preset_name") or "").lower()
    if preset not in VALID_PRESETS:
        return None
    return "arrange_workspace_preset", {"preset_name": preset}


def _lock(_args: Dict[str, Any]):
    return "lock_workstation", {}


def _timer(args: Dict[str, Any]):
    message = args.get("label") or args.get("message") or "Timer"
    if "seconds" in args:
        seconds = int(args["seconds"])
    elif "duration_seconds" in args:
        seconds = int(args["duration_seconds"])
    elif "minutes" in args:
        seconds = int(args["minutes"]) * 60
    elif "duration_minutes" in args:
        seconds = int(args["duration_minutes"]) * 60
    else:
        return None
    if seconds <= 0:
        return None
    return "set_timer", {"seconds": seconds, "message": message}


def _theme(args: Dict[str, Any]):
    if "theme" in args:
        theme = str(args["theme"]).lower()
        if theme not in ("dark", "light", "toggle"):
            return None
        return "set_system_theme", {"theme": theme}
    flag = args.get("dark", args.get("enabled"))
    if isinstance(flag, bool):
        return "set_system_theme", {"theme": "dark" if flag else "light"}
    return None


def _delta(sign: int) -> Handler:
    def handler(args: Dict[str, Any]):
        raw = args.get("amount", args.get("delta"))
        if raw is None:
            return None
        try:
            value = abs(int(raw))
        except Exception:
            return None
        return "change_volume", {"delta": sign * value}

    return handler


def _current_time(_args: Dict[str, Any]):
    return "get_current_time", {}


def _volume_status(_args: Dict[str, Any]):
    return "get_volume_status", {}


def _analyze_screen(args: Dict[str, Any]):
    prompt = args.get("prompt") or args.get("query")
    if not prompt:
        return None
    return "analyze_screen", {"prompt": prompt}


def _desktop_toast(args: Dict[str, Any]):
    title = args.get("title") or "FRIDAY"
    message = args.get("message") or args.get("text") or args.get("body")
    if not message:
        return None
    return "send_desktop_toast", {"title": title, "message": message}


MIGRATIONS: Dict[str, Handler] = {
    "manage_window": _manage_window,
    "set_workspace_preset": _workspace_preset,
    "kill_app": lambda a: ("close_app", {"app_name": _app_of(a)}) if _app_of(a) else None,
    "lock_screen": _lock,
    "lock_pc": _lock,
    "create_timer": _timer,
    "toggle_dark_mode": _theme,
    "decrease_volume": _delta(-1),
    "increase_volume": _delta(+1),
    "get_time": _current_time,
    "get_date": _current_time,
    "get_volume": _volume_status,
    "inspect_screen": _analyze_screen,
    "send_desktop_notification": _desktop_toast,
    "set_volume_mute": lambda _a: ("mute_volume", {}),
    "get_system_stats": lambda _a: ("get_system_info", {}),
}

MIGRATION_TARGETS = {
    "close_app", "minimize_window", "maximize_window", "focus_window",
    "arrange_workspace_preset", "lock_workstation", "set_timer", "set_system_theme",
    "change_volume", "get_current_time", "get_volume_status", "analyze_screen",
    "send_desktop_toast", "mute_volume", "get_system_info",
}


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
def migrate_sample(sample: Dict[str, Any], registry: Dict[str, Dict[str, Any]],
                   backlog: Dict[str, Any], repairs: Counter,
                   repair_args: bool = True):
    """Return (migrated_sample, reason, rewrites). reason is None on success."""
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
            result = handler(args)
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

        if repair_args and tool in registry:
            args = normalize_args(tool, args, registry, repairs)

        rewritten.append({"tool": tool, "args": args})

    migrated = json.loads(json.dumps(sample))
    for message in migrated["messages"]:
        if message.get("role") == "assistant":
            message["content"] = json.dumps(rewritten, ensure_ascii=False, separators=(",", ":"))

    return migrated, None, rewrites


def new_backlog() -> Dict[str, Any]:
    return defaultdict(lambda: {
        "calls": 0,
        "shapes": Counter(),
        "values": defaultdict(Counter),
        "examples": [],
    })


def process_file(path: Path, registry: Dict[str, Dict[str, Any]],
                 stats: Dict[str, Any], quarantine: List[Dict[str, Any]],
                 backlog: Dict[str, Any], repairs: Counter,
                 repair_args: bool = True) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        stats["total"] += 1
        try:
            sample = json.loads(line)
        except Exception:
            stats["quarantined"] += 1
            stats["reasons"]["unparseable_sample"] += 1
            continue

        migrated, reason, rewrites = migrate_sample(sample, registry, backlog, repairs, repair_args)
        if reason is not None:
            stats["quarantined"] += 1
            stats["reasons"][reason] += 1
            quarantine.append({"source": f"{path.parent.name}/{path.name}", "reason": reason, "sample": sample})
            continue

        if rewrites:
            stats["rewritten_samples"] += 1
            stats["rewritten_calls"] += rewrites
        else:
            stats["unchanged_samples"] += 1
        kept.append(migrated)

    return kept


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_backlog(path: Path, backlog: Dict[str, Any]) -> None:
    total = sum(entry["calls"] for entry in backlog.values())
    lines = [
        "# Pending MCP Tools",
        "",
        "Tools the training corpus expects but FRIDAY does not register yet.",
        "These calls are preserved verbatim during vocabulary migration so the",
        "intent model stays future-proof. Implement them in the backend, or",
        "delete the intent from the corpus and tests together.",
        "",
        f"Total calls waiting on implementation: **{total}**",
        "",
        "| Tool | Calls | Inferred arguments |",
        "| :--- | ---: | :--- |",
    ]
    for tool, entry in sorted(backlog.items(), key=lambda kv: -kv[1]["calls"]):
        shape = entry["shapes"].most_common(1)
        args_txt = ", ".join(shape[0][0]) if shape and shape[0][0] else "(none)"
        lines.append(f"| `{tool}` | {entry['calls']} | {args_txt} |")

    lines += ["", "## Details", ""]
    for tool, entry in sorted(backlog.items(), key=lambda kv: -kv[1]["calls"]):
        lines.append(f"### `{tool}` - {entry['calls']} calls")
        lines.append("")
        lines.append("Argument shapes:")
        lines.append("")
        for shape, count in entry["shapes"].most_common(5):
            lines.append(f"- `{list(shape)}` x{count}")
        for arg, counter in entry["values"].items():
            top = ", ".join(f"{v} x{n}" for v, n in counter.most_common(4))
            lines.append(f"- values for `{arg}`: {top}")
        if entry["examples"]:
            lines.append("")
            lines.append("Example prompts:")
            for prompt, call_args in entry["examples"]:
                lines.append(f"- `{prompt}` -> `{json.dumps(call_args, ensure_ascii=False)}`")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_schema_gaps(path: Path, repairs: Counter) -> None:
    """Document live tools whose schema cannot express a corpus intent."""
    lines = [
        "# Schema Gaps",
        "",
        "Live tools whose registered argument schema cannot express an intent the",
        "training corpus contains. The corpus argument is preserved verbatim so",
        "the model keeps the intent; the handler should be extended to accept it.",
        "",
        "| Tool | Missing capability | Corpus calls |",
        "| :--- | :--- | ---: |",
    ]
    for tool, gap in sorted(KNOWN_SCHEMA_GAPS.items()):
        count = sum(n for key, n in repairs.items() if key.startswith(f"{tool}."))
        lines.append(f"| `{tool}` | {gap} | {count} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=Path, default=DEFAULT_BACKEND)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--backlog", type=Path, default=BACKLOG_MD)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--no-arg-repair", action="store_true",
                        help="skip argument-schema repair for live tools")
    args = parser.parse_args()
    repair_args = not args.no_arg_repair

    registry = extract_registry(args.backend)
    if not registry:
        raise SystemExit(f"No tool schemas found under {args.backend}")

    missing = sorted(MIGRATION_TARGETS - set(registry))
    if missing:
        raise SystemExit(f"Migration targets missing from registry: {missing}")

    subsets = sorted(p for p in SUBSET_DIR.iterdir() if p.is_dir())
    if not subsets:
        raise SystemExit(f"No subsets under {SUBSET_DIR}")

    print(f"Registry: {len(registry)} tools | subsets: {len(subsets)}")
    print(f"Migrate: {len(MIGRATIONS)} retired names | Preserve: {len(FORWARD_LOOKING)} planned tools\n")

    rows: List[Tuple[str, Dict[str, Any]]] = []
    totals: Dict[str, Any] = {
        "total": 0, "rewritten_samples": 0, "rewritten_calls": 0,
        "unchanged_samples": 0, "quarantined": 0, "reasons": Counter(),
        "repairs": Counter(),
    }
    backlog = new_backlog()

    if args.out.exists() and not args.report_only:
        shutil.rmtree(args.out)

    for subset in subsets:
        for split in ("train", "validation"):
            src = subset / f"{split}.jsonl"
            if not src.exists():
                continue
            stats: Dict[str, Any] = {
                "total": 0, "rewritten_samples": 0, "rewritten_calls": 0,
                "unchanged_samples": 0, "quarantined": 0, "reasons": Counter(),
            }
            quarantine: List[Dict[str, Any]] = []
            kept = process_file(src, registry, stats, quarantine, backlog,
                                totals["repairs"], repair_args)

            rows.append((f"{subset.name}/{split}", stats))
            for key in ("total", "rewritten_samples", "rewritten_calls",
                        "unchanged_samples", "quarantined"):
                totals[key] += stats[key]
            totals["reasons"].update(stats["reasons"])

            if not args.report_only:
                write_jsonl(args.out / subset.name / f"{split}.jsonl", kept)
                if quarantine:
                    write_jsonl(args.out / "quarantine" / f"{subset.name}_{split}.jsonl", quarantine)

    print(f"{'subset/split':<32} {'total':>7} {'rewritten':>10} {'calls':>7} {'quarantined':>12}")
    print("-" * 74)
    for label, stats in rows:
        print(f"{label:<32} {stats['total']:>7} {stats['rewritten_samples']:>10} "
              f"{stats['rewritten_calls']:>7} {stats['quarantined']:>12}")
    print("-" * 74)
    print(f"{'TOTAL':<32} {totals['total']:>7} {totals['rewritten_samples']:>10} "
          f"{totals['rewritten_calls']:>7} {totals['quarantined']:>12}")

    retained = totals["total"] - totals["quarantined"]
    print(f"\nRetained {retained}/{totals['total']} ({retained / max(totals['total'], 1) * 100:.1f}%)"
          f" | rewritten {totals['rewritten_calls']} calls in {totals['rewritten_samples']} samples")

    print("\nPreserved for future implementation:")
    for tool, entry in sorted(backlog.items(), key=lambda kv: -kv[1]["calls"]):
        shapes = entry["shapes"].most_common(1)
        args_txt = ", ".join(shapes[0][0]) if shapes and shapes[0][0] else "no args"
        print(f"  {tool:<26} {entry['calls']:>5} calls   args: {args_txt}")

    if repair_args and totals["repairs"]:
        print("\nArgument repairs (live tools with stale schemas):")
        for repair, count in totals["repairs"].most_common(60):
            print(f"  {repair:<52} {count:>6}")

    if totals["reasons"]:
        print("\nQuarantined (genuinely unknown):")
        for reason, count in totals["reasons"].most_common():
            print(f"  {reason:<32} {count:>6}")

    if args.report_only:
        print("\nReport only - nothing written.")
        return

    write_backlog(args.backlog, backlog)
    if repair_args and KNOWN_SCHEMA_GAPS:
        write_schema_gaps(SCHEMA_GAPS_MD, totals["repairs"])

    # ---- verify -----------------------------------------------------------
    residual: Counter = Counter()
    missing_required: Counter = Counter()
    written = 0
    for path in args.out.rglob("*.jsonl"):
        if "quarantine" in path.parts:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                calls = json.loads(json.loads(line)["messages"][-1]["content"])
            except Exception:
                continue
            written += 1
            if not isinstance(calls, list):
                continue
            for call in calls:
                if not isinstance(call, dict):
                    continue
                tool = call.get("tool")
                if tool in MIGRATIONS:
                    residual[tool] += 1
                if repair_args and tool in registry and tool not in KNOWN_SCHEMA_GAPS:
                    schema = registry.get(tool) or {}
                    args_now = call.get("args") or {}
                    for required_key in schema.get("required") or []:
                        if required_key not in args_now:
                            missing_required[f"{tool}.{required_key}"] += 1

    print("\nVerification:")
    print(f"  samples written: {written}")
    print("  un-migrated renamed tools: " + (str(dict(residual)) if residual else "none"))
    if residual:
        raise SystemExit("FAIL - renamed tools survived migration")
    if repair_args:
        print("  missing required args: " + (str(dict(missing_required)) if missing_required else "none"))
        if missing_required:
            raise SystemExit("FAIL - required arguments missing after repair")

    print(f"\nDataset: {args.out}")
    print(f"Backlog: {args.backlog}")
    if repair_args and KNOWN_SCHEMA_GAPS:
        print(f"Schema gaps: {SCHEMA_GAPS_MD}")


if __name__ == "__main__":
    main()
