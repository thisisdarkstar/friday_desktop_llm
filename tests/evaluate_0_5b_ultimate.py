"""Benchmark Evaluation for friday:0.5b-ultimate.

Runs the complete 225-test Stress Benchmark against friday:0.5b-ultimate
and compares against baseline 0.5B (v4) and 1.5B (Kaggle).
"""

import json
import statistics
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tests.stress_test_benchmark as stress_mod
from tests.head_to_head_competition import evaluate_model


def main():
    model_name = "friday:0.5b-ultimate"
    res_ultimate = evaluate_model(model_name)

    # Load baseline head-to-head results
    baseline_path = Path("tests/benchmark_competition_results.json")
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            baselines = json.load(f)
        b_05 = baselines.get("model_0_5b", {})
        b_15 = baselines.get("model_1_5b", {})
    else:
        b_05, b_15 = {}, {}

    print("\n" + "=" * 95)
    print("🏆 THREE-WAY GRAND PRIX: 0.5B (v4) vs 1.5B (Kaggle) vs 0.5B ULTIMATE (Kaggle)")
    print("=" * 95)
    print(f"{'Metric':<30} | {'0.5B v4':<15} | {'1.5B Kaggle':<15} | {'0.5B Ultimate':<18} | {'Advantage'}")
    print("-" * 95)

    u_acc = res_ultimate["overall_accuracy"]
    b05_acc = b_05.get("overall_accuracy", 0.0)
    b15_acc = b_15.get("overall_accuracy", 0.0)
    print(f"{'Total Accuracy':<30} | {b05_acc:.1f}%           | {b15_acc:.1f}%           | {u_acc:.1f}%             | {'0.5B Ult' if u_acc >= b15_acc else '1.5B'}")

    u_lat = res_ultimate["avg_latency"]
    b05_lat = b_05.get("avg_latency", 0.0)
    b15_lat = b_15.get("avg_latency", 0.0)
    print(f"{'Average Latency':<30} | {b05_lat:.1f} ms        | {b15_lat:.1f} ms        | {u_lat:.1f} ms          | {'0.5B Ult ⚡' if u_lat < b15_lat else '1.5B'}")

    u_p50 = res_ultimate["p50_latency"]
    b05_p50 = b_05.get("p50_latency", 0.0)
    b15_p50 = b_15.get("p50_latency", 0.0)
    print(f"{'Median Latency (P50)':<30} | {b05_p50:.1f} ms        | {b15_p50:.1f} ms        | {u_p50:.1f} ms          | {'0.5B Ult ⚡' if u_p50 < b15_p50 else '1.5B'}")

    print(f"{'Model Size (VRAM/Disk)':<30} | 525 MB          | 1.64 GB         | 525 MB             | {'0.5B Ult 💾'}")

    print("\n" + "=" * 95)
    print("📊 CATEGORY-BY-CATEGORY ACCURACY COMPARISON")
    print("=" * 95)
    print(f"{'Category':<38} | {'0.5B v4':<12} | {'1.5B':<12} | {'0.5B Ultimate':<15}")
    print("-" * 95)

    for cat_name, u_cat in res_ultimate["categories"].items():
        c05 = b_05.get("categories", {}).get(cat_name, {}).get("accuracy", 0.0)
        c15 = b_15.get("categories", {}).get(cat_name, {}).get("accuracy", 0.0)
        u_pct = u_cat["accuracy"]
        print(f"{cat_name[:38]:<38} | {c05:5.1f}%      | {c15:5.1f}%      | {u_cat['passed']:>2}/{u_cat['total']:<2} ({u_pct:5.1f}%)")

    print("=" * 95)

    # Save detailed evaluation
    out_file = Path("tests/0_5b_ultimate_evaluation.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res_ultimate, f, indent=2)
    print(f"\n[+] Detailed results saved to {out_file}")


if __name__ == "__main__":
    main()
