"""Head-to-head benchmark for FRIDAY 0.5B model generations (v5 / v6 / v7).

Runs the project's full 340-test battery against multiple Ollama tags under
identical conditions:

  1. Comprehensive Baseline  (44 tests)
  2. Stress Suite           (225 tests)
  3. Exhaustive All-Tools   ( 71 tests)

Every model receives the exact same system prompt, decoding options, and
test order. This is required because the deployed ``friday:v5-precision``
tag was built without a baked-in SYSTEM block, which would otherwise make
it emit non-JSON on every prompt.

Usage:
    python tests/compare_friday_versions.py
    python tests/compare_friday_versions.py friday:v5-precision friday:v6-essentials friday:v7-productivity
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

ROOT = Path(__file__).parent.parent
OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/generate"

DEFAULT_MODELS = [
    "friday:v5-precision",
    "friday:v6-essentials",
    "friday:v7-productivity",
]

SYSTEM_PROMPT = (
    "You are DesktopIntent, the real-time neural router for FRIDAY AI. For every user input, "
    "you output ONLY a valid JSON array of tool calls matching FRIDAY's MCP tool bridge. "
    "Never generate conversational monologue, explanation, or thinking tokens. "
    "If the command requires complex coding or open-ended reasoning, invoke delegate_to_reasoning_model. "
    "If the input is conversational or greetings, invoke chat_response."
)

OPTIONS = {
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 0.95,
    "num_predict": 128,
    "stop": ["<|im_end|>", "<|endoftext|>"],
}


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_test_battery() -> List[Dict[str, Any]]:
    """Flatten the three existing benchmark suites into one ordered battery."""
    comp = _load_module("friday_comp", "comprehensive_test_suite.py")
    stress = _load_module("friday_stress", "stress_test_benchmark.py")
    exhaust = _load_module("friday_exhaust", "exhaustive_all_tools_benchmark.py")

    battery: List[Dict[str, Any]] = []

    for category, cases in comp.TEST_CATEGORIES.items():
        for i, case in enumerate(cases, 1):
            battery.append({
                "suite": "Comprehensive",
                "category": category,
                "id": f"C{i:02d}",
                "desc": case["desc"],
                "query": case["query"],
                "system": case.get("system"),
                "validate": case["expected"],
            })

    for category, cases in stress.BENCHMARK_SUITE.items():
        for case in cases:
            battery.append({
                "suite": "Stress",
                "category": category,
                "id": case.get("id", "S"),
                "desc": case["desc"],
                "query": case["query"],
                "system": case.get("system"),
                "validate": case["validate"],
            })

    for case in exhaust.BENCHMARK_MATRIX:
        battery.append({
            "suite": "Exhaustive",
            "category": case.get("cat", "Tools"),
            "id": case.get("id", "E"),
            "desc": case["desc"],
            "query": case["query"],
            "system": None,
            "validate": case["validate"],
        })

    return battery


def query_model(model: str, prompt: str, system: str | None) -> Dict[str, Any]:
    payload = {
        "model": model,
        "system": system or SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": OPTIONS,
    }
    request = urllib.request.Request(
        OLLAMA_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
    latency_ms = (time.perf_counter() - start) * 1000

    raw = (body.get("response") or "").strip()
    if raw.startswith("[") and not raw.endswith("]"):
        raw += "]"

    parsed: List[Dict[str, Any]] = []
    try:
        value = json.loads(raw)
        parsed = value if isinstance(value, list) else [value]
    except Exception:
        parsed = []

    eval_count = body.get("eval_count") or 0
    eval_duration_ns = body.get("eval_duration") or 0
    tokens_per_sec = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns else 0.0

    return {
        "parsed": parsed,
        "raw": raw,
        "latency_ms": latency_ms,
        "eval_count": eval_count,
        "tokens_per_sec": tokens_per_sec,
    }


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    passed = sum(1 for r in records if r["passed"])
    latencies = [r["latency_ms"] for r in records]
    toks = [r["eval_count"] for r in records]
    tps = [r["tokens_per_sec"] for r in records if r["tokens_per_sec"] > 0]
    ordered = sorted(latencies)

    def percentile(p: float) -> float:
        if not ordered:
            return 0.0
        index = min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))
        return ordered[index]

    return {
        "passed": passed,
        "total": len(records),
        "accuracy": (passed / len(records) * 100) if records else 0.0,
        "invalid_json": sum(1 for r in records if r["invalid_json"]),
        "latency_mean": statistics.fmean(latencies) if latencies else 0.0,
        "latency_median": statistics.median(latencies) if latencies else 0.0,
        "latency_p95": percentile(0.95),
        "tokens_mean": statistics.fmean(toks) if toks else 0.0,
        "tokens_per_sec_mean": statistics.fmean(tps) if tps else 0.0,
    }


def run_model(model: str, battery: List[Dict[str, Any]]) -> Dict[str, Any]:
    print("=" * 92, flush=True)
    print(f"BENCHMARKING: {model}", flush=True)
    print("=" * 92, flush=True)

    print("[*] Warming up...", end=" ", flush=True)
    for _ in range(2):
        query_model(model, "hello friday", None)
    print("ready.\n", flush=True)

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    started = time.perf_counter()

    for i, test in enumerate(battery, 1):
        result = query_model(model, test["query"], test["system"])
        try:
            passed = bool(test["validate"](result["parsed"]))
        except Exception:
            passed = False

        record = {
            "suite": test["suite"],
            "category": test["category"],
            "id": test["id"],
            "desc": test["desc"],
            "query": test["query"],
            "passed": passed,
            "invalid_json": not result["parsed"],
            "latency_ms": result["latency_ms"],
            "eval_count": result["eval_count"],
            "tokens_per_sec": result["tokens_per_sec"],
            "output": result["parsed"] if result["parsed"] else result["raw"],
        }
        records.append(record)
        if not passed:
            failures.append(record)

        if i % 25 == 0 or i == len(battery):
            so_far = summarize(records)
            print(
                f"    [{i:>3}/{len(battery)}] {so_far['accuracy']:>5.1f}% "
                f"| {so_far['latency_mean']:>6.1f}ms avg",
                flush=True,
            )

    wall = time.perf_counter() - started

    suites: Dict[str, Dict[str, Any]] = {}
    for name in ("Comprehensive", "Stress", "Exhaustive"):
        suite_records = [r for r in records if r["suite"] == name]
        if suite_records:
            suites[name] = summarize(suite_records)

    categories: Dict[str, Dict[str, Any]] = {}
    for record in records:
        categories.setdefault(record["category"], []).append(record)
    category_summary = {
        name: summarize(group) for name, group in categories.items()
    }

    overall = summarize(records)
    overall["wall_seconds"] = wall

    return {
        "model": model,
        "overall": overall,
        "suites": suites,
        "categories": category_summary,
        "failures": failures,
        "records": records,
    }


def print_report(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 92, flush=True)
    print("FRIDAY 0.5B GENERATION SHOOTOUT", flush=True)
    print("=" * 92, flush=True)

    header = f"{'Model':<26} {'Score':>10} {'Accuracy':>9} {'Mean':>9} {'P50':>8} {'P95':>8} {'Invalid':>8}"
    print(header, flush=True)
    print("-" * 92, flush=True)

    best_acc = max(r["overall"]["accuracy"] for r in results)
    best_lat = min(r["overall"]["latency_mean"] for r in results)

    for result in results:
        o = result["overall"]
        flags = []
        if o["accuracy"] == best_acc:
            flags.append("BEST ACC")
        if o["latency_mean"] == best_lat:
            flags.append("FASTEST")
        suffix = f"  <- {' + '.join(flags)}" if flags else ""
        print(
            f"{result['model']:<26} {o['passed']:>4}/{o['total']:<4} "
            f"{o['accuracy']:>7.1f}% {o['latency_mean']:>7.1f}ms "
            f"{o['latency_median']:>7.1f}ms {o['latency_p95']:>7.1f}ms "
            f"{o['invalid_json']:>8}{suffix}",
            flush=True,
        )

    print("\nPer-suite accuracy:", flush=True)
    suite_names = ["Comprehensive", "Stress", "Exhaustive"]
    print(f"{'Model':<26} " + " ".join(f"{name:>15}" for name in suite_names), flush=True)
    print("-" * 92, flush=True)
    for result in results:
        cells = []
        for name in suite_names:
            suite = result["suites"].get(name)
            cells.append(
                f"{suite['passed']:>3}/{suite['total']:<3} {suite['accuracy']:>5.1f}%"
                if suite else f"{'-':>15}"
            )
        print(f"{result['model']:<26} " + " ".join(f"{cell:>15}" for cell in cells), flush=True)

    for result in results:
        print(f"\nFailures for {result['model']}: {len(result['failures'])}", flush=True)
        for failure in result["failures"][:25]:
            print(
                f"  [{failure['suite']} | {failure['category']}] {failure['id']} "
                f"{failure['desc']}",
                flush=True,
            )
            print(f"      query : {failure['query']!r}", flush=True)
            print(f"      output: {str(failure['output'])[:140]}", flush=True)


def main() -> None:
    models = sys.argv[1:] or DEFAULT_MODELS
    battery = build_test_battery()

    print("=" * 92, flush=True)
    print("FRIDAY AI HEAD-TO-HEAD BENCHMARK", flush=True)
    print(f"Models  : {', '.join(models)}", flush=True)
    print(f"Battery : {len(battery)} tests/model", flush=True)
    print(f"Decoding: temp=0.0 top_k=1 top_p=0.95 num_predict=128", flush=True)
    print("=" * 92, flush=True)

    results = [run_model(model, battery) for model in models]
    print_report(results)

    output_dir = ROOT / "tests" / "results"
    output_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"friday_version_shootout_{stamp}.json"
    output_file.write_text(
        json.dumps({"timestamp": stamp, "battery_size": len(battery), "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nFull results saved to: {output_file}", flush=True)


if __name__ == "__main__":
    main()
