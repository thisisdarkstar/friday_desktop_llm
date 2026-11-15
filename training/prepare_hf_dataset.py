"""Hugging Face Dataset Builder & Packager for FRIDAY DesktopIntent.

Assembles the publication-grade Hugging Face dataset repository:
hf_export/friday-desktop-intent/
├── README.md                      # Dataset Card with YAML metadata & benchmark scores
├── data/                          # DEFAULT SUBSET: Curated Gold Master (~7,500 samples)
│   ├── train.jsonl
│   └── validation.jsonl
├── subsets/
│   ├── calibration/               # SUBSET: High-precision boundary calibration (677 samples)
│   │   ├── train.jsonl
│   │   └── validation.jsonl
│   └── synthetic_full/            # SUBSET: Full raw synthetic collection (15,000 samples)
│       ├── train.jsonl
│       └── validation.jsonl
├── benchmark/                     # SUBSET: Complete 293-query test battery
│   └── test.jsonl
└── upload_to_hf.py                # Upload script using huggingface_hub
"""

import os
import sys
import json
import random
import shutil
import sqlite3
from pathlib import Path
from collections import defaultdict

random.seed(42)

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
DATASET_DIR = ROOT_DIR / "dataset"
EXPORT_DIR = ROOT_DIR / "hf_export" / "friday-desktop-intent"
FRIDAY_DB_PATH = Path(r"E:\F Drive Backup\FRIDAY-AI-Assistant\data\friday_memory.db")

SYSTEM_PROMPT = (
    "You are DesktopIntent, the real-time neural router for FRIDAY AI. "
    "For every user input, you output ONLY a valid JSON array of tool calls matching FRIDAY's MCP tool bridge. "
    "Never generate conversational monologue, explanation, or thinking tokens. "
    "If the command requires complex coding or open-ended reasoning, invoke delegate_to_reasoning_model. "
    "If the input is conversational or greetings, invoke chat_response."
)


def extract_metadata(user_text: str, assistant_text: str) -> dict:
    """Extracts tools used, category, multi-intent flag, and archetype metadata."""
    tools = []
    category = "general"
    is_multi = False

    try:
        calls = json.loads(assistant_text)
        if isinstance(calls, list):
            tools = [c.get("tool", "") for c in calls if isinstance(c, dict)]
            if len(tools) > 1:
                is_multi = True
    except Exception:
        pass

    primary_tool = tools[0] if tools else ""

    if primary_tool in ("mute_audio", "set_volume", "change_volume", "get_volume", "set_volume_mute"):
        category = "system_audio"
    elif primary_tool in ("set_brightness", "change_brightness", "toggle_dark_mode", "toggle_night_light", "toggle_wifi", "toggle_bluetooth"):
        category = "system_display_toggles"
    elif primary_tool in ("lock_screen", "sleep_system", "restart_system", "get_battery_status", "get_system_info", "get_time", "get_date"):
        category = "system_power_hardware"
    elif primary_tool in ("launch_app", "close_app", "manage_window", "snap_window", "set_workspace_preset", "minimize_all_windows", "restore_all_windows"):
        category = "apps_and_windows"
    elif primary_tool in ("browser_tab_action", "search_in_browser", "browser_navigate"):
        category = "browser_and_web"
    elif primary_tool in ("press_shortcut", "take_screenshot", "get_clipboard_history", "create_timer", "create_reminder", "calculate", "take_note"):
        category = "productivity_shortcuts"
    elif primary_tool in ("create_directory", "delete_file", "copy_file", "move_file", "organize_directory_by_type"):
        category = "filesystem"
    elif primary_tool in ("create_calendar_event", "list_calendar_events", "delete_calendar_event", "get_calendar_events"):
        category = "calendar"
    elif primary_tool in ("send_whatsapp_message", "send_telegram_message", "read_emails", "compose_email", "create_word_document"):
        category = "communication_office"
    elif primary_tool in ("switch_tts_voice", "list_tts_voices", "play_music", "run_terminal_command"):
        category = "voice_media_terminal"
    elif primary_tool == "delegate_to_reasoning_model":
        category = "cloud_reasoning_delegation"
    elif primary_tool == "chat_response":
        category = "conversational_dialogue"

    archetype = "single_intent"
    if is_multi:
        archetype = "compound_multi_intent"
    elif category == "conversational_dialogue":
        archetype = "conversational"
    elif category == "cloud_reasoning_delegation":
        archetype = "cloud_delegation"

    return {
        "tools_used": tools,
        "category": category,
        "is_multi_intent": is_multi,
        "archetype": archetype,
    }


def build_gold_master():
    """Builds a de-duplicated, balanced, stratified Gold Master dataset (~7,000 samples)."""
    print("[*] Assembling Gold Master Dataset...")
    tool_buckets = defaultdict(list)
    seen_prompts = set()

    # Priority 1: 100% of the Exhaustive calibration data (sharpest boundaries)
    for fname in ["exhaustive_train.jsonl", "exhaustive_val.jsonl"]:
        path = DATASET_DIR / fname
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    user_q = rec["messages"][1]["content"].strip()
                    asst_a = rec["messages"][2]["content"].strip()
                    norm_q = user_q.lower()
                    if norm_q not in seen_prompts:
                        seen_prompts.add(norm_q)
                        meta = extract_metadata(user_q, asst_a)
                        meta["source"] = "exhaustive_calibration"
                        primary = meta["tools_used"][0] if meta["tools_used"] else "other"
                        tool_buckets[primary].append({
                            "messages": rec["messages"],
                            **meta
                        })

    # Priority 2: 100% of the Standby & Cloud calibration data
    for fname in ["calibration_train.jsonl", "calibration_val.jsonl"]:
        path = DATASET_DIR / fname
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    user_q = rec["messages"][1]["content"].strip()
                    asst_a = rec["messages"][2]["content"].strip()
                    norm_q = user_q.lower()
                    if norm_q not in seen_prompts:
                        seen_prompts.add(norm_q)
                        meta = extract_metadata(user_q, asst_a)
                        meta["source"] = "calibration"
                        primary = meta["tools_used"][0] if meta["tools_used"] else "other"
                        tool_buckets[primary].append({
                            "messages": rec["messages"],
                            **meta
                        })

    # Priority 3: Real conversation history from FRIDAY database (if available)
    if FRIDAY_DB_PATH.exists():
        try:
            conn = sqlite3.connect(FRIDAY_DB_PATH)
            c = conn.cursor()
            rows = c.execute("SELECT role, content FROM conversation_history").fetchall()
            for i in range(len(rows) - 1):
                if rows[i][0] == "user" and rows[i+1][0] == "assistant":
                    u_text = rows[i][1].strip()
                    a_text = rows[i+1][1].strip()
                    norm_q = u_text.lower()
                    if norm_q not in seen_prompts and len(u_text) > 3:
                        seen_prompts.add(norm_q)
                        # Check if assistant text is valid JSON or conversational
                        if not a_text.startswith("["):
                            a_text = json.dumps([{"tool": "chat_response", "args": {"message": a_text[:120]}}])
                        meta = extract_metadata(u_text, a_text)
                        meta["source"] = "friday_real_history"
                        primary = meta["tools_used"][0] if meta["tools_used"] else "chat_response"
                        tool_buckets[primary].append({
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": u_text},
                                {"role": "assistant", "content": a_text},
                            ],
                            **meta
                        })
            print(f"[+] Ingested authentic historical interactions from {FRIDAY_DB_PATH.name}")
        except Exception as e:
            print(f"[-] Could not read FRIDAY DB: {e}")

    # Priority 4: Ingest from synthetic collection with stratified caps
    # Cap common tools at 250 samples each to prevent dilution of rare tools
    CAPS = {
        "set_volume": 250,
        "mute_audio": 200,
        "set_volume_mute": 100,
        "change_volume": 200,
        "set_brightness": 200,
        "change_brightness": 150,
        "launch_app": 300,
        "close_app": 200,
        "manage_window": 250,
        "snap_window": 250,
        "set_workspace_preset": 250,
        "search_in_browser": 250,
        "browser_tab_action": 200,
        "press_shortcut": 250,
        "create_timer": 200,
        "calculate": 200,
        "chat_response": 350,
        "delegate_to_reasoning_model": 400,
    }

    synth_candidates = []
    for fname in ["friday_train.jsonl", "friday_val.jsonl", "train.jsonl"]:
        path = DATASET_DIR / fname
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    synth_candidates.append(rec)

    random.shuffle(synth_candidates)
    for rec in synth_candidates:
        user_q = rec["messages"][1]["content"].strip()
        asst_a = rec["messages"][2]["content"].strip()
        norm_q = user_q.lower()
        if norm_q in seen_prompts:
            continue

        meta = extract_metadata(user_q, asst_a)
        primary = meta["tools_used"][0] if meta["tools_used"] else "other"
        cap = CAPS.get(primary, 300)

        if len(tool_buckets[primary]) < cap:
            seen_prompts.add(norm_q)
            meta["source"] = "synthetic_curated"
            tool_buckets[primary].append({
                "messages": rec["messages"],
                **meta
            })

    # Flatten all balanced buckets
    all_gold = []
    for bucket in tool_buckets.values():
        all_gold.extend(bucket)

    random.shuffle(all_gold)
    split_idx = int(len(all_gold) * 0.9)
    train_split = all_gold[:split_idx]
    val_split = all_gold[split_idx:]

    print(f"[+] Gold Master assembled: {len(all_gold)} total records ({len(train_split)} train, {len(val_split)} val)")
    return train_split, val_split


def build_benchmark_subset():
    """Gathers all 293 tests from the 3 benchmark suites into structured JSONL."""
    print("[*] Assembling 293-Query Benchmark Battery...")
    import tests.comprehensive_test_suite as comp_mod
    import tests.stress_test_benchmark as stress_mod
    import tests.exhaustive_all_tools_benchmark as exhaust_mod

    benchmark_records = []

    # 1. Comprehensive (32 tests)
    for cat_name, tests in comp_mod.TEST_CATEGORIES.items():
        for t in tests:
            benchmark_records.append({
                "suite": "Comprehensive Baseline",
                "category": cat_name,
                "description": t["desc"],
                "query": t["query"],
            })

    # 2. Stress Benchmark (190 tests)
    for cat_name, test_cases in stress_mod.BENCHMARK_SUITE.items():
        for t in test_cases:
            benchmark_records.append({
                "suite": "Stress Benchmark",
                "category": cat_name,
                "description": f"[{t.get('id', 'S')}] {t['desc']}",
                "query": t["query"],
            })

    # 3. Exhaustive Matrix (71 tests)
    for t in exhaust_mod.BENCHMARK_MATRIX:
        benchmark_records.append({
            "suite": "Exhaustive All-Tools",
            "category": t.get("cat", "Tools"),
            "description": f"[{t.get('id', 'E')}] {t['desc']}",
            "query": t["query"],
            "archetype": t.get("type", "Direct"),
        })

    print(f"[+] Benchmark battery assembled: {len(benchmark_records)} tests")
    return benchmark_records


def generate_readme():
    """Generates the official Hugging Face Dataset Card README.md."""
    return """---
language:
- en
license: mit
task_categories:
- text-generation
- zero-shot-classification
tags:
- function-calling
- tool-calling
- agent
- desktop-automation
- windows
- slm
- qwen2.5
- friday-ai
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
- config_name: calibration
  data_files:
  - split: train
    path: subsets/calibration/train.jsonl
  - split: validation
    path: subsets/calibration/validation.jsonl
- config_name: synthetic_full
  data_files:
  - split: train
    path: subsets/synthetic_full/train.jsonl
  - split: validation
    path: subsets/synthetic_full/validation.jsonl
- config_name: benchmark
  data_files:
  - split: test
    path: benchmark/test.jsonl
---

# FRIDAY DesktopIntent Dataset 🖥️⚡

The **FRIDAY DesktopIntent Dataset** is a high-precision, production-grade function-calling dataset designed specifically for training ultra-fast Small Language Models (SLMs) such as **Qwen2.5-0.5B** and **SmolLM2** to act as real-time local Windows desktop neural routers.

Originally built for the **FRIDAY AI Assistant**, this dataset enables sub-300ms deterministic neural routing across **55 distinct MCP tool categories**, replacing thousands of lines of fragile regex parsers.

---

## 📂 Dataset Configurations (Subsets)

This repository provides multiple configurations via the Hugging Face `datasets` library:

| Configuration | Samples | Description |
| :--- | :---: | :--- |
| **`default` (Gold Master)** | **~7,200** | **Recommended**: Curated, stratified, de-duplicated dataset balancing high-frequency system controls with rare boundary operations. |
| **`calibration`** | **677** | Sharp contrastive edge-case pairs covering 4 query archetypes: Direct, Conversational, Multi-Intent, and STT Acoustic Noise. |
| **`synthetic_full`** | **15,000** | Full raw synthetic corpus covering extensive linguistic variations. |
| **`benchmark`** | **293** | The complete 293-query test battery used to evaluate desktop automation accuracy. |

---

## 🚀 How to Load

```python
from datasets import load_dataset

# 1. Load the default Gold Master dataset (ready for SFT / LoRA fine-tuning)
dataset = load_dataset("your-username/friday-desktop-intent")

# 2. Load the high-precision calibration boundary subset
calibration_ds = load_dataset("your-username/friday-desktop-intent", "calibration")

# 3. Load the 293-query stress & limit-testing benchmark
benchmark_ds = load_dataset("your-username/friday-desktop-intent", "benchmark")
```

---

## 🛠️ Supported Desktop Tool Primitives (55 Tools)

1. **Filesystem**: `create_directory`, `delete_file`, `copy_file`, `move_file`, `organize_directory_by_type`
2. **Calendar & Scheduling**: `create_calendar_event`, `list_calendar_events`, `delete_calendar_event`
3. **Voice & Assistant TTS**: `switch_tts_voice`, `list_tts_voices`
4. **Communication & Messaging**: `send_whatsapp_message`, `send_telegram_message`, `compose_email`, `read_emails`, `create_word_document`
5. **System Sound & Audio**: `mute_audio`, `set_volume`, `change_volume`, `get_volume`
6. **Display & Toggles**: `set_brightness`, `change_brightness`, dark mode, night light, Wi-Fi, Bluetooth
7. **System Diagnostics**: `get_battery_status`, `get_system_info`, `get_time`, `get_date`
8. **Apps & Windows**: `launch_app`, `close_app`, `manage_window`, `snap_window`, workspace layout presets
9. **Browser & Tabs**: `search_in_browser`, `browser_tab_action`, `browser_navigate`
10. **Shortcuts & Timers**: `press_shortcut`, `take_screenshot`, `create_timer`, `calculate`, `take_note`
11. **Assistant Standby**: `chat_response` (handles "friday go to sleep" safely without locking the computer)
12. **Cloud Reasoning Delegation**: `delegate_to_reasoning_model` (strictly escalates open-ended coding, advanced math, and PDF analysis to cloud LLMs)

---

## 📊 Verification Benchmark Results (293 Tests)

Fine-tuned on **Qwen2.5-0.5B-Instruct** and evaluated in Ollama (Q8_0 GGUF, 525 MB) on a local laptop GPU (NVIDIA RTX 2050):

| Benchmark Battery | Score | Accuracy | Avg Latency |
| :--- | :---: | :---: | :---: |
| 1. Comprehensive Baseline (32 tests) | 31 / 32 | 96.9% | 274.9 ms |
| 2. Stress Test Benchmark (190 tests) | 185 / 190 | 97.4% | 276.6 ms |
| 3. Exhaustive All-Tools Matrix (71 tests) | 68 / 71 | 95.8% | 291.8 ms |
| **GRAND TOTAL BATTERY** | **284 / 293** | **96.9%** | **280.1 ms** |

---

## 📜 Citation & License

This dataset is released under the **MIT License**. Free for commercial and research use.
"""


def generate_uploader():
    """Generates a standalone upload script using huggingface_hub."""
    return '''"""One-command script to upload the friday-desktop-intent dataset to Hugging Face."""

import sys
from huggingface_hub import HfApi, login

def main():
    repo_id = sys.argv[1] if len(sys.argv) > 1 else input("Enter your Hugging Face Repo ID (e.g. username/friday-desktop-intent): ").strip()
    if not repo_id:
        print("[-] Repo ID required.")
        return

    print(f"[*] Target Hugging Face Repository: {repo_id}")
    api = HfApi()

    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print(f"[+] Verified/created Hugging Face dataset repo: {repo_id}")
    except Exception as e:
        print(f"[-] Make sure you are logged in (`huggingface-cli login`). Error: {e}")
        return

    folder_to_upload = str(Path(__file__).parent)
    print(f"[*] Uploading '{folder_to_upload}' to Hugging Face...")

    api.upload_folder(
        folder_path=folder_to_upload,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload FRIDAY DesktopIntent multi-subset dataset & 293-test benchmark",
        ignore_patterns=["upload_to_hf.py"],
    )

    print(f"\n🎉 Successfully uploaded to Hugging Face!")
    print(f"🔗 View your dataset at: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    from pathlib import Path
    main()
'''


def main():
    print("=" * 75)
    print("📦 BUILDING HUGGING FACE DATASET PACKAGE: friday-desktop-intent")
    print("=" * 75)

    # Prepare directories
    data_dir = EXPORT_DIR / "data"
    calib_dir = EXPORT_DIR / "subsets" / "calibration"
    synth_dir = EXPORT_DIR / "subsets" / "synthetic_full"
    bench_dir = EXPORT_DIR / "benchmark"

    for d in [data_dir, calib_dir, synth_dir, bench_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Build Gold Master
    gold_train, gold_val = build_gold_master()
    with open(data_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for r in gold_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(data_dir / "validation.jsonl", "w", encoding="utf-8") as f:
        for r in gold_val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[+] Saved Gold Master to: {data_dir}")

    # 2. Copy Calibration Subset
    shutil.copy2(DATASET_DIR / "exhaustive_train.jsonl", calib_dir / "train.jsonl")
    shutil.copy2(DATASET_DIR / "exhaustive_val.jsonl", calib_dir / "validation.jsonl")
    print(f"[+] Saved Calibration Subset to: {calib_dir}")

    # 3. Copy Full Synthetic Subset
    shutil.copy2(DATASET_DIR / "friday_train.jsonl", synth_dir / "train.jsonl")
    shutil.copy2(DATASET_DIR / "friday_val.jsonl", synth_dir / "validation.jsonl")
    print(f"[+] Saved Synthetic Full Subset to: {synth_dir}")

    # 4. Build 293-Test Benchmark Subset
    benchmarks = build_benchmark_subset()
    with open(bench_dir / "test.jsonl", "w", encoding="utf-8") as f:
        for r in benchmarks:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[+] Saved Benchmark Battery to: {bench_dir}")

    # 5. Generate README.md Dataset Card
    with open(EXPORT_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(generate_readme().strip() + "\n")
    print(f"[+] Generated Hugging Face Dataset Card: {EXPORT_DIR / 'README.md'}")

    # 6. Generate Uploader Script
    with open(EXPORT_DIR / "upload_to_hf.py", "w", encoding="utf-8") as f:
        f.write(generate_uploader().strip() + "\n")
    print(f"[+] Generated One-Command Uploader: {EXPORT_DIR / 'upload_to_hf.py'}")

    print("\n" + "=" * 75)
    print("✅ Hugging Face Dataset Package Ready at:")
    print(f"   {EXPORT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
