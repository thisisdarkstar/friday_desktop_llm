# DesktopIntent-LLM: Neural Intent Router for FRIDAY AI

**DesktopIntent-LLM** is an ultra-fast, lightweight 0.5B parameter neural router fine-tuned specifically for local Windows desktop automation and full-spectrum integration with **FRIDAY AI Assistant** (`E:\F Drive Backup\FRIDAY-AI-Assistant`). It replaces slow, fragile regex parsers (such as FRIDAY's legacy 2,134-line `intent_router.py`) with deterministic, real-time structured JSON tool calls matching FRIDAY's MCP tool bridge.

---

## ⚡ Specifications & Performance

- **Base Architecture**: Qwen2.5-0.5B-Instruct
- **Quantization & Format**: Q8_0 GGUF (525 MB file size)
- **Active Ollama Models**: `friday:latest`, `friday`, `desktop-intent:latest`, `desktop-intent:v2`
- **Preserved Safety Backup**: `friday:v1` (`8f7ef01fbe5e`), `desktop-intent:v1` (`c2fe8c0e7e04`)
- **Inference Latency**: **271.0 ms** average per route on NVIDIA GeForce RTX 2050 (4 GB VRAM)
- **VRAM Footprint**: ~1.1 GB inference / ~2.05 GB training
- **Thinking Overhead**: Zero thinking tokens; outputs pure JSON array immediately.

---

## 📊 Exhaustive All-Tools Benchmark Scorecard

Evaluated across the 71-query exhaustive stress benchmark ([tests/exhaustive_all_tools_benchmark.py](file:///d:/Projects/desktop-llm/tests/exhaustive_all_tools_benchmark.py)) covering all 4 query archetypes (Direct, Conversational, Compound Multi-Intent, STT Noise):

| Tool Category | Pass Rate | Accuracy | Avg Latency | Capabilities Tested |
| :--- | :---: | :---: | :---: | :--- |
| **Filesystem** | **9 / 9** | **100.0%** | 226.3 ms | `create_directory`, `delete_file`, `copy_file`, `move_file`, `organize_directory_by_type` |
| **Calendar & Scheduling** | **6 / 6** | **100.0%** | 279.8 ms | `create_calendar_event`, `list_calendar_events`, `delete_calendar_event` |
| **Voice, Speech & TTS** | **6 / 6** | **100.0%** | 231.6 ms | `switch_tts_voice` (female, Aria, Guy, etc.), `list_tts_voices`, `play_music` |
| **System Audio & Sound** | **4 / 4** | **100.0%** | 185.2 ms | `mute_audio`, `set_volume`, `change_volume`, `get_volume` |
| **System Display & Toggles** | **2 / 2** | **100.0%** | 200.3 ms | `set_brightness`, `change_brightness`, dark mode, night light, Wi-Fi, Bluetooth |
| **System Hardware & Diagnostics**| **4 / 4** | **100.0%** | 178.1 ms | `get_battery_status`, `get_system_info`, `get_time`, `get_date` |
| **Apps & Windows** | **8 / 9** | **88.9%** | 237.0 ms | `launch_app`, `close_app`, `manage_window`, `snap_window`, workspace layouts |
| **Shortcuts & Timers** | **6 / 7** | **85.7%** | 271.8 ms | `press_shortcut`, `take_screenshot`, `create_timer`, `calculate`, `take_note` |
| **Communication & Office** | **5 / 6** | **83.3%** | 256.6 ms | `send_whatsapp_message`, `send_telegram_message`, `compose_email`, `read_emails`, `create_word_document` |
| **Multi-Intent Compound** | **5 / 5** | **100.0%** | 529.3 ms | 2-step & 3-step action chains parsed into ordered sequential JSON calls |
| **Dialogue & Standby** | **5 / 5** | **100.0%** | 253.2 ms | Assistant greetings, identity, status, and standby ("friday go to sleep") |
| **Cloud Reasoning Delegation** | **8 / 8** | **100.0%** | 349.7 ms | Open-ended coding, calculus ODE, business plans, PDF summarization |
| **Overall Score** | **68 / 71** | **95.8%** | **271.0 ms** | **Production Grade** |

---

## 🛠️ Project Directory Structure

```
d:\Projects\desktop-llm\
├── dataset/
│   ├── friday_train.jsonl          # 13,500 synthetic FRIDAY MCP training samples
│   ├── exhaustive_train.jsonl      # 575 all-tools training samples (v3)
│   └── exhaustive_val.jsonl        # 102 validation samples
├── models/
│   ├── desktop-intent-merged-v3/   # Merged v3 HuggingFace safetensors
│   ├── desktop-intent-q8.gguf      # Active Q8_0 GGUF model (525 MB)
│   └── desktop-intent-q8-v1.gguf   # Preserved safety backup
├── runtime/
│   ├── agent.py                    # Standalone interactive CLI agent with native Win32 tools
│   └── friday_bridge.py            # Drop-in FridayNeuralRouter -> SimpleIntentRoute
├── tests/
│   ├── exhaustive_all_tools_benchmark.py # 71-query exhaustive test suite across 12 categories
│   ├── stress_test_benchmark.py          # 190-query stress benchmark
│   └── test_friday_router.py             # Integration test for FRIDAY SimpleIntentRoute mapping
├── tools/
│   ├── dispatcher.py               # Win32 tools: audio, brightness, windows, apps, presets
│   └── keep_awake.py               # Win32 power management watchdog during GPU training
├── training/
│   ├── train_exhaustive.py         # Exhaustive all-tools fine-tuning script
│   ├── generate_exhaustive_dataset.py # Comprehensive 4-archetype dataset generator
│   └── llama_cpp/                  # GGUF conversion scripts
├── Modelfile                       # Ollama configuration
└── requirements.txt                # Python dependencies
```

---

## 🚀 How to Run

### 1. Run Exhaustive All-Tools Benchmark
```powershell
.venv\Scripts\python.exe tests/exhaustive_all_tools_benchmark.py friday
```

### 2. Test FRIDAY Integration Bridge
```powershell
.venv\Scripts\python.exe tests/test_friday_router.py
```

### 3. Interactive Desktop Agent (Standalone)
```powershell
.venv\Scripts\python.exe runtime/agent.py
```

---

## 🔌 Connecting to FRIDAY AI Assistant

In your FRIDAY project (`E:\F Drive Backup\FRIDAY-AI-Assistant`):

```python
from runtime.friday_bridge import FridayNeuralRouter

# Initialize the router targeting your local Ollama model
router = FridayNeuralRouter(model_name="friday")

# Route user speech or text prompt
routes = router.route_intents(user_query)

if not routes:
    # Model recognized a complex query (coding, math, essay, deep analysis)
    # Automatically delegate to FRIDAY's cloud reasoning pipeline!
    execute_cloud_reasoning_pipeline(user_query)
else:
    # Execute deterministic local actions
    for route in routes:
        friday_action_dispatcher.dispatch(route)
```

---

## 🛡️ Safety & Model Rollback

All versions of the model are permanently preserved:
- **`friday:latest` / `friday` (Active)**: Newly calibrated exhaustive v3 model (ID `2193557cab07`).
- **`friday:v1` (Preserved Backup)**: Preserved user backup (ID `8f7ef01fbe5e`).
- **`desktop-intent:v1` (Base Backup)**: Preserved base v1 (ID `c2fe8c0e7e04`).
