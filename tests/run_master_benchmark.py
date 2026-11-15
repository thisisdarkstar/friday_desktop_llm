"""Unified Master Benchmark Suite for FRIDAY AI.

Executes all evaluation benchmarks in sequence:
1. Comprehensive Suite (32 tests) - Original foundation baseline
2. Stress Benchmark (190 tests) - Limit testing across single, multi, conversational, STT
3. Exhaustive All-Tools Benchmark (71 tests) - Complete 55 MCP tools across 4 archetypes

Total: 293 unique verification tests against Ollama 'friday'.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tests.comprehensive_test_suite as comp_mod
import tests.stress_test_benchmark as stress_mod
import tests.exhaustive_all_tools_benchmark as exhaust_mod


def main():
    target_model = sys.argv[1] if len(sys.argv) > 1 else "friday"
    comp_mod.MODEL_NAME = target_model

    print("=" * 85, flush=True)
    print(f"🔥 FRIDAY AI GRAND MASTER BENCHMARK: {target_model}", flush=True)
    print(f"🎯 Total Test Battery: 340 tests across 3 comprehensive benchmark suites (v4 Full Battery)", flush=True)
    print("=" * 85, flush=True)

    suite_results = {}
    master_failures = []
    start_total_time = time.perf_counter()

    # =========================================================================
    # SUITE 1: COMPREHENSIVE SUITE (32 TESTS)
    # =========================================================================
    print(f"\n[SUITE 1/3] Running Comprehensive Baseline Benchmark (32 tests)...", flush=True)
    s1_passed = 0
    s1_total = 0
    s1_lat = 0.0

    for cat_name, tests in comp_mod.TEST_CATEGORIES.items():
        for t in tests:
            s1_total += 1
            prompt = t["query"]
            desc = t["desc"]
            calls, raw, lat = comp_mod.query_ollama(prompt, system=t.get("system"))
            s1_lat += lat

            passed = False
            try:
                passed = t["expected"](calls)
            except Exception:
                passed = False

            if passed:
                s1_passed += 1
            else:
                master_failures.append({
                    "suite": "Comprehensive",
                    "category": cat_name,
                    "desc": desc,
                    "query": prompt,
                    "output": calls if calls else raw,
                })

    s1_avg = s1_lat / max(s1_total, 1)
    s1_pct = (s1_passed / s1_total) * 100
    suite_results["1. Comprehensive Baseline"] = (s1_passed, s1_total, s1_pct, s1_avg)
    print(f"   -> Result: {s1_passed}/{s1_total} ({s1_pct:.1f}%) | Avg Latency: {s1_avg:.1f}ms", flush=True)

    # =========================================================================
    # SUITE 2: STRESS BENCHMARK (190 TESTS)
    # =========================================================================
    print(f"\n[SUITE 2/3] Running Stress Test Benchmark (190 tests)...", flush=True)
    s2_passed = 0
    s2_total = 0
    s2_lat = 0.0

    for cat_name, test_cases in stress_mod.BENCHMARK_SUITE.items():
        cat_p = 0
        cat_tot = len(test_cases)
        for t in test_cases:
            s2_total += 1
            tid = t.get("id", "S")
            query = t["query"]
            desc = t["desc"]
            calls, raw, lat = stress_mod.query_ollama(target_model, query, system=t.get("system"))
            s2_lat += lat

            passed = False
            try:
                passed = bool(t["validate"](calls))
            except Exception:
                passed = False

            if passed:
                s2_passed += 1
                cat_p += 1
            else:
                master_failures.append({
                    "suite": "Stress Suite",
                    "category": cat_name,
                    "desc": f"[{tid}] {desc}",
                    "query": query,
                    "output": calls if calls else raw,
                })
        print(f"      • {cat_name[:45]:<45}: {cat_p:>2}/{cat_tot:<2} passed", flush=True)

    s2_avg = s2_lat / max(s2_total, 1)
    s2_pct = (s2_passed / s2_total) * 100
    suite_results["2. Stress Benchmark"] = (s2_passed, s2_total, s2_pct, s2_avg)
    print(f"   -> Result: {s2_passed}/{s2_total} ({s2_pct:.1f}%) | Avg Latency: {s2_avg:.1f}ms", flush=True)

    # =========================================================================
    # SUITE 3: EXHAUSTIVE ALL-TOOLS BENCHMARK (71 TESTS)
    # =========================================================================
    print(f"\n[SUITE 3/3] Running Exhaustive All-Tools Benchmark (71 tests)...", flush=True)
    s3_passed = 0
    s3_total = 0
    s3_lat = 0.0

    for t in exhaust_mod.BENCHMARK_MATRIX:
        s3_total += 1
        tid = t.get("id", "E")
        cat = t.get("cat", "Tools")
        query = t["query"]
        desc = t["desc"]
        calls, raw, lat = exhaust_mod.query_ollama(target_model, query)
        s3_lat += lat

        passed = False
        try:
            passed = bool(t["validate"](calls))
        except Exception:
            passed = False

        if passed:
            s3_passed += 1
        else:
            master_failures.append({
                "suite": "All-Tools Matrix",
                "category": cat,
                "desc": f"[{tid}] {desc}",
                "query": query,
                "output": calls if calls else raw,
            })

    s3_avg = s3_lat / max(s3_total, 1)
    s3_pct = (s3_passed / s3_total) * 100
    suite_results["3. Exhaustive All-Tools"] = (s3_passed, s3_total, s3_pct, s3_avg)
    print(f"   -> Result: {s3_passed}/{s3_total} ({s3_pct:.1f}%) | Avg Latency: {s3_avg:.1f}ms", flush=True)

    # =========================================================================
    # GRAND TOTAL SCORECARD
    # =========================================================================
    total_passed = s1_passed + s2_passed + s3_passed
    total_count = s1_total + s2_total + s3_total
    total_pct = (total_passed / total_count) * 100
    total_time_s = time.perf_counter() - start_total_time
    avg_total_lat = (s1_lat + s2_lat + s3_lat) / total_count

    print("\n" + "=" * 85, flush=True)
    print(f"🏆 GRAND MASTER SCORECARD FOR: {target_model}", flush=True)
    print("=" * 85, flush=True)
    print(f"{'Benchmark Suite':<35} | {'Score':<14} | {'Accuracy':<10} | {'Avg Latency'}", flush=True)
    print("-" * 85, flush=True)

    for suite_name, (passed, count, pct, avg_ms) in suite_results.items():
        bars = int(pct / 10)
        bar_str = "█" * bars + "░" * (10 - bars)
        print(f"{suite_name:<35} | {bar_str} {passed:>3}/{count:<3} | {pct:>6.1f}%    | {avg_ms:>6.1f}ms", flush=True)

    print("-" * 85, flush=True)
    bars_total = int(total_pct / 10)
    bar_total_str = "█" * bars_total + "░" * (10 - bars_total)
    print(f"{'GRAND TOTAL BATTERY':<35} | {bar_total_str} {total_passed:>3}/{total_count:<3} | {total_pct:>6.1f}%    | {avg_total_lat:>6.1f}ms", flush=True)
    print("=" * 85, flush=True)
    print(f"⏱️ Total Wall-Clock Test Execution Time: {total_time_s:.1f}s for {total_count} tests", flush=True)

    if master_failures:
        print(f"\n⚠️ DIAGNOSTIC: {len(master_failures)} Edge Cases out of {total_count} tests:", flush=True)
        print("-" * 85, flush=True)
        for i, f in enumerate(master_failures, 1):
            print(f"{i:02d}. [{f['suite']} | {f['category']}] {f['desc']}", flush=True)
            print(f"    Query : {f['query']!r}", flush=True)
            print(f"    Output: {f['output']}", flush=True)
            print()
    else:
        print(f"\n🎉 100.0% FLAWLESS RUN ACROSS ALL {total_count} TESTS!", flush=True)


if __name__ == "__main__":
    main()
