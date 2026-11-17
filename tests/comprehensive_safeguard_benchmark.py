"""Comprehensive Safeguard Diagnostic Benchmark for FRIDAY AI.

Executes a grand battery of 296 exhaustive tests across all archetypes:
1. Single Intent (all 55 MCP tools in isolation + synonyms)
2. Compound Multi-Intent Chains (2, 3, 4 steps with varied conjunctions)
3. Pure Conversational, Persona & Identity
4. Mixed Conversational + Actions (polite greetings, rants, emotional wrappers)
5. Calibrated Ambiguity & Destructive Safety Gating
6. Speech-to-Text Acoustic Noise & Disfluencies
7. Stateful Coreference & Anaphora
8. Dynamic Runtime In-Context Injected Tools

Diagnoses every failure, identifies the exact safeguard needed,
applies the safeguard pipeline in real time, and measures the delta!
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tests.stress_test_benchmark as stress_mod
import tests.exhaustive_all_tools_benchmark as ex_mod

MODEL_NAME = "friday:0.5b-v5"

# =============================================================================
# PRODUCTION SAFEGUARD ENGINE (DETERMINISTIC <1ms PYTHON LAYER)
# =============================================================================

TOOL_CANONICAL_MAP = {
    # Synonyms / Hallucinated variants -> Canonical FRIDAY MCP tool
    "check_system_info": "get_system_info",
    "system_info": "get_system_info",
    "get_version": "chat_response",
    "check_system": "get_system_info",
    "system_status": "get_system_info",
    "file_delete": "delete_file",
    "remove_file": "delete_file",
    "organize_directory": "organize_files",
    "change_volume": "set_volume",
    "volume_set": "set_volume",
    "adjust_volume": "set_volume",
    "brightness_set": "set_brightness",
    "adjust_brightness": "set_brightness",
    "speak": "chat_response",
    "assistant_reply": "chat_response",
    "conversation": "chat_response",
    "chat_message_send": "send_whatsapp_message",
    "whatsapp_send": "send_whatsapp_message",
    "telegram_send": "send_telegram_message",
}

APP_CANONICAL_MAP = {
    "google": "chrome",
    "google chrome": "chrome",
    "browser": "chrome",
    "internet": "chrome",
    "ms word": "winword",
    "word": "winword",
    "word document": "winword",
    "excel": "excel",
    "ms excel": "excel",
    "powerpoint": "powerpnt",
    "vscode": "code",
    "vs code": "code",
    "visual studio code": "code",
    "spotify music": "spotify",
    "whatsapp web": "whatsapp",
    "slack desktop": "slack",
    "discord app": "discord",
}

# Strong action trigger patterns that should NEVER be swallowed by a greeting
ACTION_PATTERNS = [
    (r"\b(?:set|turn|change|put)\s+volume\s+(?:to\s+)?(\d+)", "set_volume", lambda m: {"level": int(m.group(1))}),
    (r"\b(?:mute|unmute)\s+(?:audio|sound|volume)", "mute_audio", lambda m: {"mute": "unmute" not in m.group(0)}),
    (r"\b(?:set|turn|change)\s+brightness\s+(?:to\s+)?(\d+)", "set_brightness", lambda m: {"level": int(m.group(1))}),
    (r"\b(?:take|grab|capture)\s+(?:a\s+)?screenshot", "take_screenshot", lambda m: {}),
    (r"\b(?:open|launch|start)\s+([a-zA-Z0-9_\s]+)", "launch_app", lambda m: {"name": m.group(1).strip()}),
    (r"\b(?:close|quit|kill|exit)\s+([a-zA-Z0-9_\s]+)", "close_app", lambda m: {"name": m.group(1).strip()}),
]


def apply_safeguards(raw_calls: Optional[List[Dict[str, Any]]], user_query: str) -> List[Dict[str, Any]]:
    """Applies the deterministic multi-stage safeguard pipeline to raw tool calls."""
    if raw_calls is None:
        raw_calls = []

    safeguarded_calls = []

    # Safeguard 1 & 2: Tool Name & App Name Normalization
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        tool = call.get("tool", "")
        args = call.get("args", {})
        if not isinstance(args, dict):
            args = {}

        # 1. Normalize tool name
        clean_tool = TOOL_CANONICAL_MAP.get(tool.lower(), tool)

        # 2. Normalize app name if applicable
        if "name" in args and isinstance(args["name"], str):
            app_norm = args["name"].lower().strip()
            args["name"] = APP_CANONICAL_MAP.get(app_norm, args["name"])

        # 3. Type Coercion & Clamping
        if clean_tool in ("set_volume", "set_brightness") and "level" in args:
            try:
                lvl = int(re.sub(r"[^\d]", "", str(args["level"])))
                args["level"] = max(0, min(100, lvl))
            except Exception:
                pass

        if clean_tool == "manage_window" and "action" in args:
            args["action"] = str(args["action"]).lower()
            if args["action"] == "close" and "app" in args:
                clean_tool = "close_app"
                args = {"name": args.get("app")}

        safeguarded_calls.append({"tool": clean_tool, "args": args})

    # Safeguard 3: Action-over-Chitchat Precedence Check
    # If the model ONLY output chat_response, but the prompt clearly contains an imperative command:
    if len(safeguarded_calls) == 1 and safeguarded_calls[0]["tool"] == "chat_response":
        q_lower = user_query.lower()
        extracted_actions = []
        for pat, tool_name, arg_fn in ACTION_PATTERNS:
            match = re.search(pat, q_lower)
            if match:
                try:
                    args = arg_fn(match)
                    if "name" in args and isinstance(args["name"], str):
                        args["name"] = APP_CANONICAL_MAP.get(args["name"], args["name"])
                    extracted_actions.append({"tool": tool_name, "args": args})
                except Exception:
                    pass
        if extracted_actions:
            # Replace pure greeting with the actual requested action!
            safeguarded_calls = extracted_actions

    return safeguarded_calls


# =============================================================================
# BENCHMARK SUITE ASSEMBLY
# =============================================================================

def collect_all_tests() -> List[Dict[str, Any]]:
    tests = []
    seen_queries = set()

    # 1. Stress benchmark suite (225 tests)
    for cat_name, test_cases in stress_mod.BENCHMARK_SUITE.items():
        for t in test_cases:
            q = t["query"].strip()
            if q not in seen_queries:
                seen_queries.add(q)
                tests.append({
                    "id": t.get("id", f"S{len(tests)+1:03d}"),
                    "cat": cat_name,
                    "query": q,
                    "desc": t["desc"],
                    "system": t.get("system"),
                    "validate": t["validate"],
                })

    # 2. Exhaustive all-tools suite (71 tests)
    for t in ex_mod.BENCHMARK_MATRIX:
        q = t["query"].strip()
        if q not in seen_queries:
            seen_queries.add(q)
            tests.append({
                "id": t.get("id", f"E{len(tests)+1:03d}"),
                "cat": f"Exhaustive: {t.get('cat', 'General')}",
                "query": q,
                "desc": t["desc"],
                "system": None,
                "validate": t["validate"],
            })

    return tests


def main():
    all_tests = collect_all_tests()
    total_count = len(all_tests)
    print("=" * 85)
    print(f"🔬 RUNNING SAFEGUARD DIAGNOSTIC BENCHMARK: {MODEL_NAME}")
    print(f"   Total Unique Test Battery: {total_count} Tests")
    print("=" * 85)

    raw_passed = 0
    safeguarded_passed = 0
    failures_diagnosed = []
    safeguard_interventions = {
        "Tool Alias Normalization": 0,
        "App Name Canonicalization": 0,
        "Action Precedence Override": 0,
        "Argument Clamping / Coercion": 0,
        "Cascaded / Unresolved": 0,
    }

    t_start = time.perf_counter()

    for i, t in enumerate(all_tests, 1):
        q = t["query"]
        desc = t["desc"]
        cat = t["cat"]

        print(f"[{i:3d}/{total_count}] Testing: {q[:45]:<45} ... ", end="", flush=True)

        raw_calls, raw_text, lat = stress_mod.query_ollama(MODEL_NAME, q, system=t.get("system"))

        # 1. Test Raw Model Output
        raw_ok = False
        try:
            raw_ok = bool(t["validate"](raw_calls))
        except Exception:
            raw_ok = False

        if raw_ok:
            raw_passed += 1
            safeguarded_passed += 1
            print(f"PASS (Raw {lat:.0f}ms)")
            continue

        # 2. Raw Model Failed -> Apply Safeguard Pipeline
        t_sg0 = time.perf_counter()
        sg_calls = apply_safeguards(raw_calls, q)
        t_sg_lat_us = (time.perf_counter() - t_sg0) * 1_000_000

        sg_ok = False
        try:
            sg_ok = bool(t["validate"](sg_calls))
        except Exception:
            sg_ok = False

        rescued_by = "Tool / App Normalizer"
        if sg_ok:
            safeguarded_passed += 1
            if rescued_by != "1.5B Model Cascade":
                if len(raw_calls or []) == 1 and raw_calls[0].get("tool") == "chat_response" and sg_calls[0].get("tool") != "chat_response":
                    rescued_by = "Action Precedence Override"
                    safeguard_interventions["Action Precedence Override"] += 1
                elif any(c.get("tool") != r.get("tool") for c, r in zip(sg_calls, raw_calls or [])):
                    rescued_by = "Tool Alias Normalization"
                    safeguard_interventions["Tool Alias Normalization"] += 1
                elif any(c.get("args") != r.get("args") for c, r in zip(sg_calls, raw_calls or [])):
                    rescued_by = "App Name Canonicalization"
                    safeguard_interventions["App Name Canonicalization"] += 1
                else:
                    rescued_by = "Argument Clamping / Coercion"
                    safeguard_interventions["Argument Clamping / Coercion"] += 1

            print(f"RESCUED 🛡️ (+{t_sg_lat_us:.1f}µs) via [{rescued_by}]")
        else:
            safeguard_interventions["Cascaded / Unresolved"] += 1
            print(f"FAIL ❌")
            failures_diagnosed.append({
                "id": t["id"],
                "category": cat,
                "desc": desc,
                "query": q,
                "raw_calls": raw_calls,
                "safeguarded_calls": sg_calls,
            })

    total_time = time.perf_counter() - t_start

    print("\n" + "=" * 85)
    print("🏆 FINAL COMPREHENSIVE BENCHMARK SCORECARD")
    print("=" * 85)
    print(f"Total Test Cases Evaluated : {total_count}")
    print(f"Raw Model Accuracy (0.5B)  : {raw_passed}/{total_count} ({raw_passed/total_count*100:.2f}%)")
    print(f"Safeguarded Model Accuracy : {safeguarded_passed}/{total_count} ({safeguarded_passed/total_count*100:.2f}%)")
    print(f"Net Accuracy Gain          : +{(safeguarded_passed - raw_passed)/total_count*100:.2f}%")
    print(f"Total Benchmark Duration   : {total_time:.1f}s")
    print("=" * 85)

    print("\n🛡️ SAFEGUARD INTERVENTIONS BREAKDOWN:")
    for sg_name, count in safeguard_interventions.items():
        print(f"  • {sg_name:<30} : {count:>2} errors resolved instantly")

    if failures_diagnosed:
        print("\n🔍 REMAINING UNRESOLVED RESIDUALS (TARGETS FOR 1.5B CASCADE):")
        for f in failures_diagnosed:
            print(f"  [{f['id']}] {f['query']}")
            print(f"       Desc: {f['desc']}")
            print(f"       Raw : {json.dumps(f['raw_calls'])}")
            print(f"       Post: {json.dumps(f['safeguarded_calls'])}")
            print()

    # Save detailed JSON report
    report_data = {
        "model": MODEL_NAME,
        "total_tests": total_count,
        "raw_passed": raw_passed,
        "raw_accuracy": raw_passed / total_count * 100,
        "safeguarded_passed": safeguarded_passed,
        "safeguarded_accuracy": safeguarded_passed / total_count * 100,
        "interventions": safeguard_interventions,
        "unresolved_residuals": failures_diagnosed,
    }
    with open("tests/safeguard_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print("\n[+] Detailed benchmark report saved to tests/safeguard_benchmark_results.json")


if __name__ == "__main__":
    main()
