"""Audit FRIDAY's tool vocabulary against the training corpus and test suite.

Answers three questions:

  1. Which tool names do the 340-test validators reference, and do those tools
     still exist in the MCP registry?
  2. For the renamed intents, what share of training samples still use a retired
     tool name?
  3. Which prompts are duplicated or carry conflicting labels across subsets?

Usage:
    python tests/audit_vocabulary.py
    python tests/audit_vocabulary.py --backend "E:/path/to/FRIDAY-AI-Assistant/backend"
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).parent.parent
DEFAULT_BACKEND = Path(os.getenv("FRIDAY_BACKEND", r"E:\F Drive Backup\FRIDAY-AI-Assistant\backend"))

# Names handled by the router / core, not registered as MCP tools.
ROUTER_NATIVE = {
    "chat_response", "delegate_to_reasoning_model", "clarification_required",
    "press_shortcut", "type_text", "press_key", "sleep_system", "set_power_state",
}

# Injected on purpose by the zero-shot runtime-tool category.
DYNAMIC_RUNTIME = {
    "obs_record", "spotify_search", "docker_action", "git_quick_commit",
    "hue_lights", "notion_create_task", "steam_action", "toggle_bluetooth",
}

# Retired: absent from the registry and from the runtime alias handler.
RETIRED = {
    "manage_window", "set_workspace_preset", "kill_app", "lock_screen", "lock_pc",
    "create_timer", "change_brightness", "set_volume_mute", "toggle_dark_mode",
    "minimize_all_windows", "restore_all_windows", "take_note", "calculate",
    "toggle_night_light", "get_system_stats",
}

SUBSETS = {
    "synthetic": "hf_export/friday-desktop-intent/subsets/synthetic_full/train.jsonl",
    "calibration": "hf_export/friday-desktop-intent/subsets/calibration/train.jsonl",
    "v4": "hf_export/friday-desktop-intent/subsets/v4_perfect/train.jsonl",
    "v5": "hf_export/friday-desktop-intent/subsets/v5_precision/train.jsonl",
    "v6": "hf_export/friday-desktop-intent/subsets/v6_daily_essentials/train.jsonl",
    "v7": "hf_export/friday-desktop-intent/subsets/v7_productivity/train.jsonl",
}

FAMILIES = {
    "workspace preset": (r"workspace|layout|preset", "set_workspace_preset", "arrange_workspace_preset"),
    "timer": (r"\btimer\b", "create_timer", "set_timer"),
    "window minimize": (r"minimi[sz]e", "manage_window", "minimize_window"),
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def extract_registry(backend: Path) -> Dict[str, Dict[str, Any]]:
    """Parse bridge.register_manual_tool(...) into name -> schema."""
    tools_dir = backend / "tools"
    schemas: Dict[str, Dict[str, Any]] = {}
    if not tools_dir.exists():
        return schemas
    for py_file in sorted(tools_dir.glob("*.py")):
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
                schemas[name] = {
                    "module": py_file.stem,
                    "description": kwargs.get("description") or "",
                    "input_schema": kwargs.get("input_schema") or {},
                }
    return schemas


def load_alias_map(backend: Path) -> Dict[str, str]:
    path = backend / "utils" / "neural_safeguards.py"
    if not path.exists():
        return {}
    spec = importlib.util.spec_from_file_location("friday_safeguards", path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return {}
    return dict(getattr(module, "TOOL_CANONICAL_MAP", {}))


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------
def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validator_tools(fn) -> Set[str]:
    """Extract tool names from either validator idiom."""
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


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def load_subset(path: Path) -> List[tuple]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            messages = data.get("messages", [])
            user = next((m["content"] for m in messages if m.get("role") == "user"), None)
            assistant = next((m["content"] for m in messages if m.get("role") == "assistant"), None)
            if user and assistant:
                rows.append((user.strip(), assistant.strip()))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=Path, default=DEFAULT_BACKEND)
    args = parser.parse_args()

    registry = extract_registry(args.backend)
    if not registry:
        raise SystemExit(f"No tool schemas found under {args.backend}")

    alias_map = load_alias_map(args.backend)
    compare = load_module("friday_compare", ROOT / "tests" / "compare_friday_versions.py")
    battery = compare.build_test_battery()

    # ---- 1. validators -----------------------------------------------------
    print("=" * 78)
    print(f"REGISTRY: {len(registry)} MCP tools | corpus aliases: {len(alias_map)}")
    print("=" * 78)

    reference_counts: Counter = Counter()
    retired_tests: List[tuple] = []
    for test in battery:
        names = validator_tools(test["validate"])
        reference_counts.update(names)
        bad = sorted(n for n in names if n in RETIRED)
        if bad:
            retired_tests.append((test["suite"], test["id"], test["query"], bad))

    print(f"\nTests referencing retired vocabulary: {len(retired_tests)}/{len(battery)}")
    suite_counts = Counter(s for s, *_ in retired_tests)
    print(f"  by suite: {dict(suite_counts)}")

    print("\nRetired-name references:")
    for name, count in reference_counts.most_common():
        if name in RETIRED:
            print(f"  {name:<28} {count:>3}x")

    unknown = [n for n in reference_counts
               if n not in registry and n not in ROUTER_NATIVE
               and n not in DYNAMIC_RUNTIME and n not in alias_map]
    if unknown:
        print("\nUnclassified names (neither registered, native, dynamic nor aliased):")
        for name in sorted(unknown):
            print(f"  {name:<28} {reference_counts[name]:>3}x")

    # ---- 2. corpus ---------------------------------------------------------
    corpus = {name: load_subset(ROOT / rel) for name, rel in SUBSETS.items()}
    corpus = {k: v for k, v in corpus.items() if v}

    print("\n" + "=" * 78)
    print("CORPUS: retired vs live labels for renamed intents")
    print("=" * 78)
    print(f"{'family':<20} {'retired':>9} {'live':>7} {'retired share':>15}")
    print("-" * 56)
    for family, (pattern, old, new) in FAMILIES.items():
        old_n = new_n = 0
        for rows in corpus.values():
            for prompt, label in rows:
                if not re.search(pattern, prompt, re.I):
                    continue
                old_n += f'"{old}"' in label
                new_n += f'"{new}"' in label
        total = old_n + new_n
        share = old_n / total * 100 if total else 0.0
        print(f"{family:<20} {old_n:>9} {new_n:>7} {share:>14.1f}%")

    # ---- 3. duplicates and conflicts --------------------------------------
    print("\n" + "=" * 78)
    print("CORPUS: overlap and label conflicts")
    print("=" * 78)
    owners: Dict[str, Set[str]] = defaultdict(set)
    labels: Dict[str, Set[str]] = defaultdict(set)
    for name, rows in corpus.items():
        for prompt, label in rows:
            key = prompt.lower()
            owners[key].add(name)
            labels[key].add(label)

    duplicated = sum(1 for v in owners.values() if len(v) > 1)
    conflicting = sum(1 for v in labels.values() if len(v) > 1)
    print(f"  unique prompts:            {len(owners)}")
    print(f"  prompts in >1 subset:      {duplicated}")
    print(f"  prompts with >1 label:     {conflicting}")

    if conflicting:
        print("\n  Sample conflicts:")
        shown = 0
        for prompt, values in labels.items():
            if len(values) < 2:
                continue
            print(f"    {prompt[:64]}")
            for value in sorted(values)[:2]:
                print(f"      {value[:96]}")
            shown += 1
            if shown >= 5:
                break


if __name__ == "__main__":
    main()
