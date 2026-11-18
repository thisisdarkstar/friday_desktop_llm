"""Independently verify dataset/v8_blend against the FRIDAY tool registry.

Checks: no stale renamed tool names, no missing required args (except the
documented browser_switch_tab{tab_id} gap), no unknown args on tools that declare
a schema, zero train/validation prompt leakage, and planned-tool coverage.

Usage:
    python scripts/verify_v8_blend.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from migrate_dataset_vocabulary import (
    extract_registry, DEFAULT_BACKEND, FORWARD_LOOKING, MIGRATIONS,
)

reg = extract_registry(DEFAULT_BACKEND)
renamed = set(MIGRATIONS) | {"get_time", "get_date", "get_volume", "inspect_screen",
                             "send_desktop_notification", "decrease_volume", "increase_volume"}
stale = Counter()
missing = Counter()
unknown_args = Counter()
call_mix = Counter()
prompts = {"train": set(), "validation": set()}

for split in ("train", "validation"):
    for line in (ROOT / "dataset/v8_blend" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        msgs = obj["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        asst = next(m["content"] for m in msgs if m["role"] == "assistant")
        prompts[split].add(user.strip().lower())
        for call in json.loads(asst):
            tool = call["tool"]
            args = call.get("args") or {}
            call_mix[tool] += 1
            if tool in renamed:
                stale[tool] += 1
            if tool in reg:
                schema = reg[tool] or {}
                props = schema.get("properties") or {}
                for req in (schema.get("required") or []):
                    if req not in args and "tab_id" not in args:  # documented gap
                        missing[f"{tool}.{req}"] += 1
                # Only tools that declare a schema can have "unknown" args.
                # Empty props means unconstrained; the migrator preserves those.
                if props:
                    for k in args:
                        if k not in props and k != "tab_id":
                            unknown_args[f"{tool}.{k}"] += 1

print("stale renamed names:", dict(stale) or "NONE")
print("missing required args:", dict(missing) or "NONE")
print("unknown args (excl tab_id gap):", dict(unknown_args) or "NONE")
print("train/val prompt overlap:", len(prompts["train"] & prompts["validation"]))
print("total distinct tools:", len(call_mix))
print("forward-looking calls:", sum(call_mix[t] for t in FORWARD_LOOKING))
for t in sorted(FORWARD_LOOKING):
    if call_mix[t]:
        print(f"  {t}: {call_mix[t]}")
# hardcase signature
hc = [c for c in call_mix if c in {"chat_response", "get_system_info"}]
print("hardcase-ish tools present:", {t: call_mix[t] for t in hc})
