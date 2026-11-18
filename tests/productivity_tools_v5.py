"""Benchmark / export the v7 productivity tools on the v5 router.

v5 does not know the v7 tools from training, but it routes tools that are
injected at runtime through the system prompt ("Runtime Tools" block). This
script:

  1. Extracts the real MCP tool schemas from FRIDAY's backend.
  2. Builds prompts from the v7 productivity dataset generator.
  3. Runs a model with those schemas injected into the system prompt.
  4. Scores tool selection + argument extraction.
  5. Exports every raw output as JSONL (usable as distillation data).

Examples:
    # quick 1-per-tool probe, v5 only
    python tests/productivity_tools_v5.py --per-tool 1

    # full 6-per-tool shootout, v5 vs v7
    python tests/productivity_tools_v5.py --per-tool 6 --compare friday:v7-productivity

    # export outputs for the entire 3,210-prompt v7 dataset
    python tests/productivity_tools_v5.py --all
"""

from __future__ import annotations

import argparse
import ast
import json
import statistics
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).parent.parent
FRIDAY_TOOLS = Path(r"E:\F Drive Backup\FRIDAY-AI-Assistant\backend\tools")
OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "friday:v5-precision"

BASE_SYSTEM = (
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

# v7 domain -> tools (from training/generate_v7_productivity.py)
DOMAINS: Dict[str, List[str]] = {
    "health_processes": [
        "get_system_health", "get_detailed_hardware_stats", "list_processes", "kill_process",
    ],
    "windowing_layouts": [
        "snap_window", "arrange_workspace_preset", "save_workspace_profile",
        "load_workspace_profile", "list_windows",
    ],
    "file_organization": [
        "organize_directory_by_type", "open_in_explorer", "rename_file",
        "search_files", "grep_files",
    ],
    "weather_info": ["get_weather", "search_wikipedia", "news_search"],
    "messaging_contacts": [
        "send_whatsapp_message", "send_telegram_message", "compose_email",
        "search_contacts",
    ],
    "multimedia": [
        "download_youtube", "convert_media_ffmpeg", "capture_monitor_screenshot",
    ],
    "voice_notifications": [
        "switch_tts_voice", "list_tts_voices", "send_desktop_toast",
    ],
}

TOOL_DOMAIN = {tool: domain for domain, tools in DOMAINS.items() for tool in tools}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
def extract_schemas(tools_dir: Path = FRIDAY_TOOLS) -> Dict[str, Dict[str, Any]]:
    """Parse bridge.register_manual_tool(...) calls into name -> schema."""
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
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "register_manual_tool"):
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
                    "description": kwargs.get("description") or "",
                    "input_schema": kwargs.get("input_schema") or {},
                }
    return schemas


def _params_of(schema: Dict[str, Any]) -> List[str]:
    return list((schema.get("input_schema") or {}).get("properties", {}).keys())


STRICT_RULES = (
    "\n\nRuntime Tools override all built-in tools. When a runtime tool can satisfy the request, "
    "you MUST call it by its exact name and never substitute a similar built-in tool. "
    "Use the exact argument names and value types from the schema. "
    "Only include arguments the user provided or the schema marks as required; "
    "otherwise let the tool use its default."
)


def runtime_tools_block(
    tool_names: Sequence[str],
    schemas: Dict[str, Dict[str, Any]],
    schema_format: str,
) -> str:
    tools = []
    for name in tool_names:
        schema = schemas.get(name, {})
        entry: Dict[str, Any] = {
            "name": name,
            "description": (schema.get("description") or name).strip(),
        }
        if schema_format == "full":
            entry["parameters"] = schema.get("input_schema") or {"type": "object", "properties": {}}
        else:
            entry["parameters"] = _params_of(schema)
        tools.append(entry)
    return "\n\nRuntime Tools:\n" + json.dumps(tools, indent=1)


def build_system(
    tool_names: Sequence[str],
    schemas: Dict[str, Dict[str, Any]],
    schema_format: str = "names",
    strict: bool = False,
    examples: Sequence[Dict[str, Any]] = (),
) -> str:
    parts = [BASE_SYSTEM]
    if strict:
        parts.append(STRICT_RULES)
    parts.append(runtime_tools_block(tool_names, schemas, schema_format))
    if examples:
        lines = ["\n\nExamples:"]
        for sample in examples:
            lines.append(f"User: {sample['q']}")
            lines.append(f"Assistant: {json.dumps(sample['t'], ensure_ascii=False)}")
        parts.append("\n".join(lines))
    return "".join(parts)


def pick_examples(
    tool_names: Sequence[str],
    pool: Sequence[Dict[str, Any]],
    exclude_prompt: str,
    count: int,
) -> List[Dict[str, Any]]:
    if count <= 0:
        return []
    chosen: List[Dict[str, Any]] = []
    for tool in tool_names:
        candidates = [
            s for s in pool
            if s["t"][0]["tool"] == tool and s["q"] != exclude_prompt
        ]
        if not candidates:
            continue
        if count >= len(candidates):
            chosen.extend(candidates[:count])
        else:
            step = (len(candidates) - 1) / max(count - 1, 1)
            indexes = sorted({round(i * step) for i in range(count)})
            chosen.extend(candidates[i] for i in indexes)
    return chosen


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_v7_samples() -> List[Dict[str, Any]]:
    sys.path.insert(0, str(ROOT / "training"))
    import generate_v7_productivity as g7  # noqa: WPS433

    raw = (
        g7.generate_domain_health_processes()
        + g7.generate_domain_windowing_layouts()
        + g7.generate_domain_file_organization()
        + g7.generate_domain_weather_info()
        + g7.generate_domain_messaging_contacts()
        + g7.generate_domain_multimedia_conversions()
        + g7.generate_domain_voice_notifications()
    )

    seen = set()
    samples = []
    for sample in raw:
        if sample["q"] in seen:
            continue
        seen.add(sample["q"])
        samples.append(sample)
    return samples


def select_samples(samples: List[Dict[str, Any]], per_tool: int | None) -> List[Dict[str, Any]]:
    if per_tool is None:
        return samples

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(sample["t"][0]["tool"], []).append(sample)

    selected: List[Dict[str, Any]] = []
    for tool in sorted(grouped):
        group = grouped[tool]
        if per_tool >= len(group):
            selected.extend(group)
            continue
        if per_tool == 1:
            indexes = [0]
        else:
            indexes = sorted({round(i * (len(group) - 1) / (per_tool - 1)) for i in range(per_tool)})
        selected.extend(group[i] for i in indexes)
    return selected


# ---------------------------------------------------------------------------
# Inference + scoring
# ---------------------------------------------------------------------------
def query_model(model: str, prompt: str, system: str, response_format: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": OPTIONS,
    }
    if response_format is not None:
        payload["format"] = response_format
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

    return {"raw": raw, "parsed": parsed, "latency_ms": latency_ms}


def _norm(value: Any) -> str:
    return str(value).strip().lower().replace("\\", "/")


def score_calls(expected: List[Dict[str, Any]], actual: List[Dict[str, Any]]) -> Tuple[bool, bool]:
    """Return (tool_ok, args_ok). Requires exact tool sequence and expected arg subset."""
    expected_tools = [call.get("tool") for call in expected]
    actual_tools = [call.get("tool") for call in actual if isinstance(call, dict)]
    tool_ok = expected_tools == actual_tools

    if not tool_ok:
        return False, False

    args_ok = True
    for exp, act in zip(expected, actual):
        exp_args = exp.get("args") or {}
        act_args = act.get("args") or {}
        for key, value in exp_args.items():
            if key not in act_args:
                args_ok = False
                break
            if _norm(act_args[key]) != _norm(value):
                args_ok = False
                break
        if not args_ok:
            break
    return tool_ok, args_ok


def response_schema(tool_names: Sequence[str]) -> Dict[str, Any]:
    """Ollama structured-output schema: valid JSON + tool names constrained."""
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": list(tool_names)},
                "args": {"type": "object"},
            },
            "required": ["tool", "args"],
        },
    }


def run_model(
    model: str,
    samples: List[Dict[str, Any]],
    schemas: Dict[str, Dict[str, Any]],
    schema_mode: str,
    schema_format: str = "names",
    strict: bool = False,
    example_pool: Sequence[Dict[str, Any]] = (),
    few_shot: int = 0,
    structured: bool = False,
) -> Dict[str, Any]:
    print("=" * 96, flush=True)
    print(
        f"MODEL: {model}   |   samples: {len(samples)}   |   "
        f"schemas: {schema_mode}/{schema_format}{'/strict' if strict else ''}"
        f"{f'/fewshot{few_shot}' if few_shot else ''}"
        f"{'/structured' if structured else ''}",
        flush=True,
    )
    print("=" * 96, flush=True)

    for _ in range(2):
        query_model(model, "hello friday", BASE_SYSTEM)

    records: List[Dict[str, Any]] = []
    started = time.perf_counter()

    for i, sample in enumerate(samples, 1):
        expected = sample["t"]
        domain = TOOL_DOMAIN.get(expected[0]["tool"])
        if schema_mode == "all":
            tool_names = sorted(TOOL_DOMAIN)
        elif schema_mode == "single":
            tool_names = sorted({call["tool"] for call in expected})
        else:
            tool_names = DOMAINS[domain]
        system = build_system(
            tool_names,
            schemas,
            schema_format,
            strict,
            pick_examples(tool_names, example_pool, sample["q"], few_shot),
        )

        result = query_model(
            model,
            sample["q"],
            system,
            response_schema(tool_names) if structured else None,
        )
        tool_ok, args_ok = score_calls(expected, result["parsed"])

        records.append({
            "prompt": sample["q"],
            "expected": expected,
            "actual": result["parsed"],
            "raw": result["raw"],
            "tool_ok": tool_ok,
            "args_ok": args_ok,
            "success": tool_ok and args_ok,
            "latency_ms": result["latency_ms"],
            "domain": domain,
        })

        if i % 25 == 0 or i == len(samples):
            ok = sum(1 for r in records if r["success"])
            print(f"    [{i:>4}/{len(samples)}] success {ok / i * 100:5.1f}%", flush=True)

    by_tool: Dict[str, Dict[str, Any]] = {}
    for record in records:
        tool = record["expected"][0]["tool"]
        bucket = by_tool.setdefault(tool, {"n": 0, "tool_ok": 0, "args_ok": 0, "success": 0})
        bucket["n"] += 1
        bucket["tool_ok"] += int(record["tool_ok"])
        bucket["args_ok"] += int(record["args_ok"])
        bucket["success"] += int(record["success"])

    return {
        "model": model,
        "schema_mode": schema_mode,
        "records": records,
        "by_tool": by_tool,
        "wall_seconds": time.perf_counter() - started,
    }


def print_report(result: Dict[str, Any]) -> None:
    total = len(result["records"])
    success = sum(1 for r in result["records"] if r["success"])
    tool_ok = sum(1 for r in result["records"] if r["tool_ok"])
    args_ok = sum(1 for r in result["records"] if r["args_ok"])
    latencies = [r["latency_ms"] for r in result["records"]]

    print("\n" + "-" * 96, flush=True)
    print(f"PER-TOOL RESULTS: {result['model']} ({result['schema_mode']} schemas)", flush=True)
    print(f"{'Tool':<32} {'n':>4} {'Tool':>7} {'Args':>7} {'Both':>7}", flush=True)
    print("-" * 96, flush=True)
    for tool in sorted(result["by_tool"]):
        b = result["by_tool"][tool]
        n = b["n"]
        print(
            f"{tool:<32} {n:>4} {b['tool_ok'] / n * 100:>6.1f}% "
            f"{b['args_ok'] / n * 100:>6.1f}% {b['success'] / n * 100:>6.1f}%",
            flush=True,
        )
    print("-" * 96, flush=True)
    print(
        f"{'OVERALL':<32} {total:>4} {tool_ok / total * 100:>6.1f}% "
        f"{args_ok / total * 100:>6.1f}% {success / total * 100:>6.1f}%",
        flush=True,
    )
    print(
        f"Latency: mean {statistics.fmean(latencies):.1f}ms | "
        f"median {statistics.median(latencies):.1f}ms | wall {result['wall_seconds']:.0f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--compare", nargs="*", default=[])
    parser.add_argument("--per-tool", type=int, default=6,
                        help="samples per tool (default 6); use --all for the full dataset")
    parser.add_argument("--all", action="store_true", help="run every v7 sample (3,210)")
    parser.add_argument("--schema-mode", choices=["domain", "all", "single"], default="domain")
    parser.add_argument("--schema-format", choices=["names", "full"], default="names")
    parser.add_argument("--strict", action="store_true",
                        help="add explicit runtime-tool override rules")
    parser.add_argument("--few-shot", type=int, default=0,
                        help="in-context examples per runtime tool (0 = none)")
    parser.add_argument("--structured", action="store_true",
                        help="constrain output with an Ollama JSON schema")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    schemas = extract_schemas()
    if not schemas:
        raise SystemExit(f"Could not read FRIDAY tool schemas from {FRIDAY_TOOLS}")

    all_samples = load_v7_samples()
    samples = select_samples(all_samples, None if args.all else args.per_tool)
    print(f"Loaded {len(samples)} prompts covering "
          f"{len({s['t'][0]['tool'] for s in samples})} tools.", flush=True)

    models = [args.model] + args.compare
    results = [
        run_model(
            model, samples, schemas, args.schema_mode, args.schema_format,
            args.strict, all_samples, args.few_shot, args.structured,
        )
        for model in models
    ]
    for result in results:
        print_report(result)

    if len(results) > 1:
        print("\n" + "=" * 96, flush=True)
        print("HEAD-TO-HEAD", flush=True)
        print("=" * 96, flush=True)
        base = results[0]
        base_map = {(r["prompt"], tuple(c["tool"] for c in r["expected"])): r for r in base["records"]}
        for other in results[1:]:
            other_map = {(r["prompt"], tuple(c["tool"] for c in r["expected"])): r for r in other["records"]}
            base_only = sum(1 for k, r in base_map.items() if r["success"] and not other_map[k]["success"])
            other_only = sum(1 for k, r in other_map.items() if r["success"] and not base_map[k]["success"])
            print(
                f"{base['model']} {sum(1 for r in base['records'] if r['success'])}/{len(base['records'])} "
                f"vs {other['model']} {sum(1 for r in other['records'] if r['success'])}/{len(other['records'])} "
                f"| {base['model'].split(':')[1]} wins {base_only}, "
                f"{other['model'].split(':')[1]} wins {other_only}",
                flush=True,
            )

    if not args.no_export:
        output_dir = ROOT / "tests" / "results"
        output_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for result in results:
            slug = result["model"].replace(":", "_").replace("/", "_")
            jsonl_path = output_dir / f"{slug}_productivity_outputs_{stamp}.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as handle:
                for record in result["records"]:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"\nOutputs exported: {jsonl_path}", flush=True)


if __name__ == "__main__":
    main()
