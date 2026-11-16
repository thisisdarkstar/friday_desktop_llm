"""Head-to-Head Benchmark Competition: FRIDAY 0.5B vs FRIDAY 1.5B.

Executes the exhaustive Stress Test Benchmark (225 tests) back-to-back against:
1. friday:latest (0.5B, 525 MB Q8_0)
2. friday:1.5b   (1.5B, 1.64 GB Q8_0)

Generates comparative scorecards, latency percentiles, and category breakdowns.
"""

import json
import statistics
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tests.stress_test_benchmark as stress_mod


def evaluate_model(model_name: str):
    print(f"\n{'='*75}")
    print(f"🥊 TESTING CONTENDER: {model_name}")
    print(f"{'='*75}")

    category_results = {}
    all_latencies = []
    total_passed = 0
    total_tests = 0
    json_valid_count = 0
    failures = []

    start_total = time.perf_counter()

    for cat_name, test_cases in stress_mod.BENCHMARK_SUITE.items():
        cat_p = 0
        cat_tot = len(test_cases)
        cat_lats = []

        print(f"  ▶ Evaluating: {cat_name[:40]} ({cat_tot} tests)...", end="", flush=True)

        for t in test_cases:
            total_tests += 1
            tid = t.get("id", "S")
            query = t["query"]
            desc = t["desc"]

            calls, raw, lat = stress_mod.query_ollama(model_name, query, system=t.get("system"))
            all_latencies.append(lat)
            cat_lats.append(lat)

            # Check JSON syntax validity
            if calls is not None and isinstance(calls, list):
                json_valid_count += 1

            # Validate semantics
            passed = False
            try:
                passed = bool(t["validate"](calls))
            except Exception:
                passed = False

            if passed:
                total_passed += 1
                cat_p += 1
            else:
                failures.append({
                    "id": tid,
                    "cat": cat_name,
                    "desc": desc,
                    "query": query,
                    "output": calls if calls is not None else raw,
                })

        cat_avg_lat = statistics.mean(cat_lats) if cat_lats else 0.0
        cat_pct = (cat_p / cat_tot) * 100
        category_results[cat_name] = {
            "passed": cat_p,
            "total": cat_tot,
            "accuracy": cat_pct,
            "avg_latency": cat_avg_lat,
        }
        print(f" -> {cat_p:>2}/{cat_tot:<2} ({cat_pct:5.1f}%) | {cat_avg_lat:5.1f}ms", flush=True)

    elapsed_total = time.perf_counter() - start_total

    # Compute percentiles
    sorted_lats = sorted(all_latencies)
    p50 = statistics.median(sorted_lats)
    p95 = sorted_lats[int(len(sorted_lats) * 0.95)] if sorted_lats else 0
    p99 = sorted_lats[int(len(sorted_lats) * 0.99)] if sorted_lats else 0
    avg_lat = statistics.mean(sorted_lats) if sorted_lats else 0

    return {
        "model": model_name,
        "total_passed": total_passed,
        "total_tests": total_tests,
        "overall_accuracy": (total_passed / total_tests) * 100,
        "json_validity_rate": (json_valid_count / total_tests) * 100,
        "avg_latency": avg_lat,
        "p50_latency": p50,
        "p95_latency": p95,
        "p99_latency": p99,
        "elapsed_total": elapsed_total,
        "categories": category_results,
        "failures": failures,
    }


def main():
    m1_name = "friday:latest"   # 0.5B
    m2_name = "friday:1.5b"     # 1.5B

    print("\n" + "#" * 85)
    print("🏆 FRIDAY DESKTOP AI: HEAD-TO-HEAD BENCHMARK CHAMPIONSHIP")
    print(f"Contenders: [A] {m1_name} (0.5B Q8_0)  vs  [B] {m2_name} (1.5B Q8_0)")
    print("Total Battery: 225 stress tests across 8 desktop behavioral categories")
    print("#" * 85)

    res1 = evaluate_model(m1_name)
    res2 = evaluate_model(m2_name)

    # Print Summary Comparison Table
    print("\n" + "=" * 85)
    print("🏁 FINAL CHAMPIONSHIP LEADERBOARD")
    print("=" * 85)
    print(f"{'Metric':<32} | {'friday:latest (0.5B)':<22} | {'friday:1.5b (1.5B)':<22} | {'Winner':<10}")
    print("-" * 85)

    # Overall Accuracy
    w_acc = "1.5B 🏆" if res2["overall_accuracy"] > res1["overall_accuracy"] else ("0.5B 🏆" if res1["overall_accuracy"] > res2["overall_accuracy"] else "TIE 🤝")
    print(f"{'Overall Benchmark Accuracy':<32} | {res1['total_passed']}/{res1['total_tests']} ({res1['overall_accuracy']:.1f}%)        | {res2['total_passed']}/{res2['total_tests']} ({res2['overall_accuracy']:.1f}%)        | {w_acc:<10}")

    # JSON Validity
    w_json = "TIE 🤝" if abs(res1["json_validity_rate"] - res2["json_validity_rate"]) < 0.1 else ("1.5B" if res2["json_validity_rate"] > res1["json_validity_rate"] else "0.5B")
    print(f"{'JSON Syntax Strictness':<32} | {res1['json_validity_rate']:.1f}%                 | {res2['json_validity_rate']:.1f}%                 | {w_json:<10}")

    # Latencies
    w_avg = "0.5B ⚡" if res1["avg_latency"] < res2["avg_latency"] else "1.5B"
    print(f"{'Average Latency':<32} | {res1['avg_latency']:.1f} ms               | {res2['avg_latency']:.1f} ms               | {w_avg:<10}")

    w_p50 = "0.5B ⚡" if res1["p50_latency"] < res2["p50_latency"] else "1.5B"
    print(f"{'Median Latency (P50)':<32} | {res1['p50_latency']:.1f} ms               | {res2['p50_latency']:.1f} ms               | {w_p50:<10}")

    w_p95 = "0.5B ⚡" if res1["p95_latency"] < res2["p95_latency"] else "1.5B"
    print(f"{'95th Percentile Latency (P95)':<32} | {res1['p95_latency']:.1f} ms               | {res2['p95_latency']:.1f} ms               | {w_p95:<10}")

    # Size on Disk
    print(f"{'Model Size (VRAM/Disk)':<32} | ~525 MB (Q8_0)         | ~1.64 GB (Q8_0)        | {'0.5B 💾':<10}")

    # Category Breakdown Comparison
    print("\n" + "=" * 85)
    print("📊 CATEGORY-BY-CATEGORY BREAKDOWN")
    print("=" * 85)
    print(f"{'Category':<45} | {'0.5B Accuracy':<15} | {'1.5B Accuracy':<15} | {'Diff':<8}")
    print("-" * 85)

    for cat_name in res1["categories"]:
        c1 = res1["categories"][cat_name]
        c2 = res2["categories"][cat_name]
        diff = c2["accuracy"] - c1["accuracy"]
        diff_str = f"+{diff:.1f}%" if diff > 0 else (f"{diff:.1f}%" if diff < 0 else "0.0%")
        print(f"{cat_name[:45]:<45} | {c1['passed']:>2}/{c1['total']:<2} ({c1['accuracy']:5.1f}%) | {c2['passed']:>2}/{c2['total']:<2} ({c2['accuracy']:5.1f}%) | {diff_str:<8}")

    print("=" * 85)

    # Save results to json for report generation
    out_json = Path("tests/benchmark_competition_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"model_0_5b": res1, "model_1_5b": res2}, f, indent=2)
    print(f"\n[+] Full comparison raw data saved to {out_json}")


if __name__ == "__main__":
    main()
