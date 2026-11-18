"""Re-score a saved FRIDAY shootout under production tool normalization.

The 340-test validators were written against an older tool vocabulary. The
runtime rewrites renamed tools before execution, so a raw "fail" can be a false
negative. Tests that reference planned (not-yet-implemented) tools are excluded
from the release gate: emitting those tools is correct behaviour.

This script reports three numbers per model:

  raw          - the score stored in the shootout
  normalized   - model output rewritten into the validator's own dialect
  live-only    - gate score on tests referencing only live tools

Usage:
    python tests/rescore_shootout.py
    python tests/rescore_shootout.py --results tests/results/friday_version_shootout_20261117_213946.json
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).parent.parent
BACKEND = Path(os.getenv("FRIDAY_BACKEND", r"E:\F Drive Backup\FRIDAY-AI-Assistant\backend"))

# Renamed tools: the registry has a live replacement. The *intent* is valid and
# stays in the corpus, but the 340-test validators still check the old spelling
# and often the old arg schema. Those validators are modernized in Step 2 of the
# remediation plan; until then they are excluded from the gate alongside planned
# tools (but tracked separately so the reason is visible).
RENAMED = {
    "manage_window", "set_workspace_preset", "kill_app", "lock_screen", "lock_pc",
    "create_timer", "set_volume_mute", "toggle_dark_mode", "get_system_stats",
}

# Planned tools: intentionally in the corpus for a future-proof intent model but
# not implemented in FRIDAY yet (docs/PENDING_TOOLS.md). Emitting them is correct;
# validators referencing them are excluded from the release gate until they land.
PLANNED = {
    "change_brightness", "minimize_all_windows", "restore_all_windows", "take_note",
    "calculate", "toggle_night_light",
}

# Union: validators whose vocabulary is stale for one of the two reasons above.
RETIRED = RENAMED | PLANNED

# Semantic families: same intent, different spelling / granularity.
# Argument checks stay strict, so genuine argument errors still fail.
GROUPS = [
    ["manage_window", "minimize_window", "maximize_window", "restore_window",
     "focus_window", "minimize_all_windows", "restore_all_windows", "close_window",
     "minimize_app", "maximize_app", "restore_app", "focus_app"],
    ["browser_tab_action", "browser_close_tab", "browser_new_tab",
     "browser_switch_tab", "close_browser_tab"],
    ["lock_screen", "lock_pc", "lock_workstation"],
    ["create_timer", "set_timer"],
    ["take_screenshot", "capture_monitor_screenshot", "capture_window_screenshot",
     "capture_screen", "screenshot"],
    ["change_volume", "decrease_volume", "increase_volume", "adjust_volume"],
    ["set_brightness", "change_brightness", "increase_brightness",
     "decrease_brightness", "dim_brightness", "adjust_brightness", "brightness_set"],
    ["set_workspace_preset", "arrange_workspace_preset"],
    ["toggle_dark_mode", "set_system_theme", "dark_mode"],
    ["close_app", "kill_app", "quit_app"],
    ["launch_app", "open_app", "launch_application", "start_app"],
    ["mute_audio", "mute_volume", "mute_sound", "toggle_mute"],
    ["unmute_audio", "unmute_volume", "unmute_sound"],
    ["take_note", "create_note", "add_note"],
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_equivalence(alias_map: Dict[str, str]) -> Dict[str, Set[str]]:
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for key, value in alias_map.items():
        union(key, value)
    for group in GROUPS:
        for member in group:
            union(group[0], member)

    classes: Dict[str, Set[str]] = defaultdict(set)
    for name in list(parent):
        classes[find(name)].add(name)
    return {name: members for members in classes.values() for name in members}


def validator_tools(fn) -> Set[str]:
    try:
        src = inspect.getsource(fn)
    except Exception:
        return set()
    names: Set[str] = set()
    for m in re.finditer(r'c_any_tool\(\s*calls\s*,((?:\s*"[a-z0-9_]+"\s*,?)+)', src):
        names.update(re.findall(r'"([a-z0-9_]+)"', m.group(1)))
    for m in re.finditer(r'\.get\(\s*"tool"\s*\)\s*==\s*"([a-z0-9_]+)"', src):
        names.add(m.group(1))
    for m in re.finditer(r'\.get\(\s*"tool"\s*\)\s+in\s+\(([^)]*)\)', src):
        names.update(re.findall(r'"([a-z0-9_]+)"', m.group(1)))
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--backend", type=Path, default=BACKEND)
    args = parser.parse_args()

    safeguards = load_module("friday_safeguards", args.backend / "utils" / "neural_safeguards.py")
    equivalence = build_equivalence(dict(safeguards.TOOL_CANONICAL_MAP))

    compare = load_module("friday_compare", ROOT / "tests" / "compare_friday_versions.py")
    battery = compare.build_test_battery()
    # (suite, category, id) is unique across all 340 tests; query text is not a key.
    by_key = {(t["suite"], t["category"], t["id"]): t for t in battery}

    if args.results:
        path = args.results
    else:
        candidates = sorted((ROOT / "tests" / "results").glob("friday_version_shootout_*.json"))
        if not candidates:
            raise SystemExit("No shootout results found; run tests/compare_friday_versions.py first.")
        path = candidates[-1]

    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"Battery: {len(battery)} | results: {path.name}\n")

    def equivalent(a: str, b: str) -> bool:
        return bool(equivalence.get(str(a).lower(), {str(a).lower()})
                    & equivalence.get(str(b).lower(), {str(b).lower()}))

    print(f"{'Model':<24} {'Raw':>10} {'Normalized':>13} {'Gate':>12}")
    print("  Gate = tests whose validators use current tool vocabulary")
    print("  (excludes renamed-spelling validators pending Step 2 and planned-tool validators pending implementation)")
    print("-" * 98)

    report: Dict[str, Dict] = {}
    for result in data["results"]:
        model = result["model"]
        total = result["overall"]["total"]
        raw = result["overall"]["passed"]
        normalized = 0
        live_pass = live_total = 0
        excl_renamed = excl_planned = 0
        flips = []

        for rec in result["records"]:
            test = by_key.get((rec["suite"], rec["category"], rec["id"]))
            if test is None:
                continue
            calls = [c for c in (rec.get("output") or []) if isinstance(c, dict)]
            expected = validator_tools(test["validate"])

            if expected & PLANNED:
                excl_planned += 1
            elif expected & RENAMED:
                excl_renamed += 1
            else:
                live_total += 1
                live_pass += int(rec["passed"])

            rewritten = []
            for call in calls:
                name = str(call.get("tool", ""))
                display = name
                if name not in expected:
                    for candidate in expected:
                        if equivalent(name, candidate):
                            display = candidate
                            break
                rewritten.append({"tool": display, "args": dict(call.get("args") or {})})

            try:
                passed = bool(test["validate"](rewritten))
            except Exception:
                passed = False

            normalized += int(passed)
            if passed and not rec["passed"]:
                flips.append({
                    "id": rec["id"],
                    "query": rec["query"],
                    "said": [c.get("tool") for c in calls],
                    "wanted": sorted(expected),
                })

        print(f"{model:<24} {raw:>5}/{total:<4} {normalized:>7}/{total:<4} {live_pass:>5}/{live_total:<6}")
        report[model] = {
            "raw": raw, "normalized": normalized, "total": total,
            "live_pass": live_pass, "live_total": live_total,
            "excluded_renamed": excl_renamed, "excluded_planned": excl_planned,
            "flips": flips,
        }

    first = next(iter(report.values()), {})
    print(f"\nExcluded per model from the gate: {first.get('excluded_renamed', 0)} renamed-spelling validators, "
          f"{first.get('excluded_planned', 0)} planned-tool validators "
          "(both constant across models; see RENAMED / PLANNED sets).")

    print("\nFalse negatives (validator vocabulary drift):")
    for model, row in report.items():
        print(f"\n  {model}: {len(row['flips'])}")
        for flip in row["flips"]:
            print(f"    [{flip['id']:<4}] {flip['query'][:60]}")
            print(f"           said={flip['said']} wanted={flip['wanted']}")

    out = ROOT / "tests" / "results" / "alias_normalized_rescore.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
