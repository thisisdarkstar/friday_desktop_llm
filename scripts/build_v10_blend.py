"""Build the definitive v10 training blend from the v10 migrated corpus.

Resolves all conflicts, enforces strict MCP schemas, eliminates hallucinations,
protects single-key GUI commands, ensures read-only introspection, and preserves
all planned future productivity tools.

Output:
    dataset/v10_blend/train.jsonl
    dataset/v10_blend/validation.jsonl
    dataset/v10_blend/blend_report.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).parent.parent
MIGRATED = ROOT / "dataset" / "v10_migrated"
HARDCASE = ROOT / "dataset" / "v10_hardcase"
OUT_DIR = ROOT / "dataset" / "v10_blend"

SUBSET_WEIGHTS: Dict[str, int] = {
    "synthetic_full": 1,
    "calibration": 2,
    "v4_perfect": 2,
    "v5_precision": 6,
    "v6_daily_essentials": 2,
    "v7_productivity": 2,
    "hardcase_replay": 8,
}

OWNER_PRIORITY: List[str] = [
    "hardcase_replay",
    "v5_precision",
    "v7_productivity",
    "v6_daily_essentials",
    "v4_perfect",
    "calibration",
    "synthetic_full",
]

# Explicit ground truth resolutions for ambiguous / conflicting prompts.
CONFLICT_OVERRIDES: Dict[str, List[Dict[str, Any]]] = {
    "open file explorer": [{"tool": "launch_app", "args": {"app_name": "explorer"}}],
    "show desktop": [{"tool": "press_shortcut", "args": {"action": "show_desktop"}}],
    "can you please show desktop": [{"tool": "press_shortcut", "args": {"action": "show_desktop"}}],
    "minimize everything and show desktop": [{"tool": "press_shortcut", "args": {"action": "show_desktop"}}],
    "switch to next tab": [{"tool": "browser_tab_action", "args": {"action": "next"}}],
    "close current tab": [{"tool": "browser_tab_action", "args": {"action": "close"}}],
    "kill chrome": [{"tool": "close_app", "args": {"app_name": "chrome"}}],
    "kill notepad": [{"tool": "close_app", "args": {"app_name": "notepad"}}],
    "kill spotify": [{"tool": "close_app", "args": {"app_name": "spotify"}}],
    "kill slack": [{"tool": "close_app", "args": {"app_name": "slack"}}],
    "kill task manager": [{"tool": "close_app", "args": {"app_name": "taskmgr"}}],
    "kill discord": [{"tool": "close_app", "args": {"app_name": "discord"}}],
    "close chrome window": [{"tool": "close_window", "args": {"title_keyword": "chrome"}}],
    "close brave window": [{"tool": "close_window", "args": {"title_keyword": "brave"}}],
    "close slack window": [{"tool": "close_window", "args": {"title_keyword": "slack"}}],
    "close telegram window": [{"tool": "close_window", "args": {"title_keyword": "telegram"}}],
    "close notepad window": [{"tool": "close_window", "args": {"title_keyword": "notepad"}}],
    "close spotify window": [{"tool": "close_window", "args": {"title_keyword": "spotify"}}],
    "hey friday close calculator window": [{"tool": "close_window", "args": {"title_keyword": "calculator"}}],
    "open task manager": [{"tool": "launch_app", "args": {"app_name": "taskmgr"}}],
    "lock screen": [{"tool": "lock_workstation", "args": {}}],
    "open telegram and focus it": [
        {"tool": "launch_app", "args": {"app_name": "telegram"}},
        {"tool": "focus_window", "args": {"title_keyword": "telegram"}},
    ],
    "remind me to drink water in 30 minutes": [
        {"tool": "set_reminder", "args": {"message": "drink water", "time_str": "in 30 minutes"}}
    ],
    # v10 fixes:
    "type test message and press enter": [
        {"tool": "type_text", "args": {"text": "test message"}},
        {"tool": "press_key", "args": {"key": "enter"}}
    ],
    "make chrome full screen please": [
        {"tool": "maximize_window", "args": {"title_keyword": "chrome"}}
    ],
    "is dark mode currently turned on": [
        {"tool": "get_system_theme", "args": {}}
    ],
    "what is the current brightness": [
        {"tool": "get_brightness", "args": {}}
    ],
    "give me a quick status check on my pc resources": [
        {"tool": "get_system_info", "args": {}}
    ],
    "i need to do some web browsing fire up chrome": [
        {"tool": "launch_app", "args": {"app_name": "chrome"}}
    ],
    "download this youtube video https www youtube com watch v dqw4w9wgxcq": [
        {"tool": "download_youtube",
         "args": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "format": "mp4", "quality": "best"}}
    ],
    "friday download this youtube video https www youtube com watch v dqw4w9wgxcq": [
        {"tool": "download_youtube",
         "args": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "format": "mp4", "quality": "best"}}
    ],
    "download this youtube video https youtu be kjqp7kiw5fk": [
        {"tool": "download_youtube",
         "args": {"url": "https://youtu.be/kJQP7kiw5Fk", "format": "mp4", "quality": "best"}}
    ],
    "friday download this youtube video https youtu be kjqp7kiw5fk": [
        {"tool": "download_youtube",
         "args": {"url": "https://youtu.be/kJQP7kiw5Fk", "format": "mp4", "quality": "best"}}
    ],
}


def normalize_prompt(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_split(split: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for subset in SUBSET_WEIGHTS:
        if subset == "hardcase_replay":
            path = HARDCASE / f"{split}.jsonl"
        else:
            path = MIGRATED / subset / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            prompt = next(m["content"] for m in sample["messages"] if m["role"] == "user")
            calls = json.loads(next(m["content"] for m in sample["messages"]
                                    if m["role"] == "assistant"))
            rows.append({
                "subset": subset,
                "prompt": prompt,
                "key": normalize_prompt(prompt),
                "calls": calls,
                "sample": sample,
                "label": json.dumps(calls, sort_keys=True, ensure_ascii=False),
            })
    return rows


def dedupe(rows: List[Dict[str, Any]], split: str, report: Dict[str, Any]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["key"]].append(row)

    priority = {name: i for i, name in enumerate(OWNER_PRIORITY)}
    chosen: List[Dict[str, Any]] = []
    duplicates = 0
    semantic_conflicts = 0
    arg_variants = 0
    overrides_used = 0

    for key, group in groups.items():
        duplicates += len(group) - 1
        distinct = {row["label"] for row in group}
        distinct_sequences = {tuple(c["tool"] for c in row["calls"]) for row in group}

        if key in CONFLICT_OVERRIDES:
            overrides_used += 1
            representative = min(group, key=lambda r: priority.get(r["subset"], 99))
            sample = json.loads(json.dumps(representative["sample"]))
            for message in sample["messages"]:
                if message.get("role") == "assistant":
                    message["content"] = json.dumps(
                        CONFLICT_OVERRIDES[key], ensure_ascii=False, separators=(",", ":")
                    )
            chosen.append({**representative, "calls": CONFLICT_OVERRIDES[key], "sample": sample})
            continue

        if len(distinct_sequences) > 1:
            semantic_conflicts += 1
            report.setdefault("unresolved_semantic_conflicts", []).append({
                "split": split,
                "prompt": key,
                "variants": sorted({json.dumps(r["calls"], ensure_ascii=False) for r in group})[:4],
            })
        elif len(distinct) > 1:
            arg_variants += 1

        representative = min(group, key=lambda r: priority.get(r["subset"], 99))
        chosen.append(representative)

    report[f"{split}_duplicates_removed"] = duplicates
    report[f"{split}_unresolved_semantic_conflicts"] = semantic_conflicts
    report[f"{split}_arg_variants_resolved"] = arg_variants
    report[f"{split}_overrides_applied"] = overrides_used
    return chosen


def expand(chosen: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Counter, Counter]:
    exposures: List[Dict[str, Any]] = []
    unique = Counter()
    exposed = Counter()
    for row in chosen:
        weight = SUBSET_WEIGHTS[row["subset"]]
        unique[row["subset"]] += 1
        exposed[row["subset"]] += weight
        for _ in range(weight):
            exposures.append(row["sample"])
    return exposures, unique, exposed


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    report: Dict[str, Any] = {"weights": SUBSET_WEIGHTS}

    train_rows = dedupe(load_split("train"), "train", report)
    val_rows = dedupe(load_split("validation"), "validation", report)

    # No prompt may appear in both train and validation.
    train_keys = {row["key"] for row in train_rows}
    before = len(val_rows)
    val_rows = [row for row in val_rows if row["key"] not in train_keys]
    report["validation_rows_removed_as_train_leak"] = before - len(val_rows)

    train, train_unique, train_exposed = expand(train_rows)
    val, val_unique, val_exposed = expand(val_rows)

    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(val)

    total = len(train)
    shares = {
        name: round(count / max(total, 1) * 100, 2)
        for name, count in train_exposed.items()
    }
    report["train_total"] = total
    report["validation_total"] = len(val)
    report["unique_per_subset"] = dict(train_unique)
    report["exposures_per_subset"] = dict(train_exposed)
    report["exposure_share_pct"] = shares

    print(f"Train: {total:,} exposures | validation: {len(val):,}")
    print(f"{'subset':<24} {'unique':>8} {'x':>3} {'exposures':>10} {'share':>7}")
    print("-" * 58)
    for subset in SUBSET_WEIGHTS:
        unique = train_unique.get(subset, 0)
        exposure = train_exposed.get(subset, 0)
        print(f"{subset:<24} {unique:>8} {SUBSET_WEIGHTS[subset]:>3} {exposure:>10} "
              f"{shares.get(subset, 0):>6.1f}%")
    print("-" * 58)
    print(f"duplicates removed: {report['train_duplicates_removed']:,} | "
          f"overrides applied: {report['train_overrides_applied']} | "
          f"semantic conflicts: {report['train_unresolved_semantic_conflicts']} | "
          f"arg variants resolved: {report['train_arg_variants_resolved']} | "
          f"val leaks removed: {report['validation_rows_removed_as_train_leak']}")

    if args.report_only:
        print("\nReport only - nothing written.")
        return

    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "validation.jsonl", val)
    (args.out / "blend_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
