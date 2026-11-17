---
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
- config_name: v4_perfect
  data_files:
  - split: train
    path: subsets/v4_perfect/train.jsonl
  - split: validation
    path: subsets/v4_perfect/validation.jsonl
- config_name: v5_precision
  data_files:
  - split: train
    path: subsets/v5_precision/train.jsonl
  - split: validation
    path: subsets/v5_precision/validation.jsonl
- config_name: v6_daily_essentials
  data_files:
  - split: train
    path: subsets/v6_daily_essentials/train.jsonl
  - split: validation
    path: subsets/v6_daily_essentials/validation.jsonl
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
dataset = load_dataset("ThisisDarkstar/friday-desktop-intent")

# 2. Load the high-precision calibration boundary subset
calibration_ds = load_dataset("ThisisDarkstar/friday-desktop-intent", "calibration")

# 3. Load the 293-query stress & limit-testing benchmark
benchmark_ds = load_dataset("ThisisDarkstar/friday-desktop-intent", "benchmark")
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
