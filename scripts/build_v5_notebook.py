import json
import os
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 FRIDAY DesktopIntent-0.5B v5-Precision (Kaggle Cloud Training)\n",
                "\n",
                "Trains the **v5-Precision Neural Router** for FRIDAY AI on Kaggle Tesla T4 GPU.\n",
                "\n",
                "### Direct Fixes for Diagnostic Audit Gaps:\n",
                "1. **Media Control & Track Disambiguation**: (pause/resume vs mute, track navigation)\n",
                "2. **Multi-Monitor & Window Vision**: (monitor indices, window titles, OCR extraction)\n",
                "3. **Canonical Scheduler & Timers**: (set/pause/resume/cancel timer, reminders, alarms)\n",
                "4. **Local Search vs Web Search**: (files, contacts, playlists vs Google web search)\n",
                "5. **Telemetry & Hardware Diagnostics**: (ports, ping, network adapters, browser tabs)\n",
                "6. **Action-Over-Chitchat**: (quoted typing text vs conversational greetings)\n",
                "7. **Core Anchor Replay**: (app launches, snapping, volume, brightness to prevent forgetting)\n",
                "\n",
                "**Export Target**: `friday-0.5b-v5-q8.gguf` (~525 MB)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Install Dependencies & Clone llama.cpp"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q -U transformers peft datasets accelerate trl huggingface_hub gguf sentencepiece protobuf torchao\n",
                "!git clone --depth 1 https://github.com/ggerganov/llama.cpp.git"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Verify GPU Hardware Acceleration"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "print(f\"PyTorch: {torch.__version__}\")\n",
                "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
                "    print(f\"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\")\n",
                "else:\n",
                "    raise SystemError(\"GPU accelerator not detected! Enable GPU in Settings -> Accelerator!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3. Load & Blend Master Dataset from Hugging Face"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from datasets import load_dataset, concatenate_datasets\n",
                "\n",
                "HF_DATASET_ID = \"ThisisDarkstar/friday-desktop-intent\"\n",
                "print(f\"[*] Streaming dataset configurations from: {HF_DATASET_ID}...\")\n",
                "\n",
                "synthetic = load_dataset(HF_DATASET_ID, \"synthetic_full\")\n",
                "calibration = load_dataset(HF_DATASET_ID, \"calibration\")\n",
                "v4_perfect = load_dataset(HF_DATASET_ID, \"v4_perfect\")\n",
                "try:\n",
                "    v5_precision = load_dataset(HF_DATASET_ID, \"v5_precision\")\n",
                "except Exception as e:\n",
                "    print(f\"[*] Falling back to direct JSON URL for v5_precision: {e}\")\n",
                "    v5_precision = load_dataset(\"json\", data_files={\n",
                "        \"train\": f\"https://huggingface.co/datasets/{HF_DATASET_ID}/resolve/main/subsets/v5_precision/train.jsonl\",\n",
                "        \"validation\": f\"https://huggingface.co/datasets/{HF_DATASET_ID}/resolve/main/subsets/v5_precision/validation.jsonl\"\n",
                "    })\n",
                "\n",
                "print(f\"    synthetic_full: {len(synthetic['train']):,} train / {len(synthetic['validation']):,} val\")\n",
                "print(f\"    calibration:    {len(calibration['train']):,} train / {len(calibration['validation']):,} val\")\n",
                "print(f\"    v4_perfect:     {len(v4_perfect['train']):,} train / {len(v4_perfect['validation']):,} val\")\n",
                "print(f\"    v5_precision:   {len(v5_precision['train']):,} train / {len(v5_precision['validation']):,} val\")\n",
                "\n",
                "# Blend with 4x oversampling on v5_precision and 2x on v4 & calibration to maximize audit accuracy\n",
                "train_ds = concatenate_datasets([\n",
                "    synthetic[\"train\"],\n",
                "    calibration[\"train\"],\n",
                "    calibration[\"train\"],\n",
                "    v4_perfect[\"train\"],\n",
                "    v4_perfect[\"train\"],\n",
                "    v5_precision[\"train\"],\n",
                "    v5_precision[\"train\"],\n",
                "    v5_precision[\"train\"],\n",
                "    v5_precision[\"train\"],\n",
                "]).shuffle(seed=42)\n",
                "\n",
                "val_ds = concatenate_datasets([\n",
                "    synthetic[\"validation\"],\n",
                "    calibration[\"validation\"],\n",
                "    v4_perfect[\"validation\"],\n",
                "    v5_precision[\"validation\"],\n",
                "]).shuffle(seed=42)\n",
                "\n",
                "print(f\"[+] Master Blended Training Samples:   {len(train_ds):,}\")\n",
                "print(f\"[+] Master Blended Validation Samples: {len(val_ds):,}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Initialize Qwen2.5-0.5B-Instruct & High-Rank LoRA (r=64)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
                "from peft import LoraConfig, get_peft_model, TaskType\n",
                "\n",
                "BASE_MODEL_ID = \"Qwen/Qwen2.5-0.5B-Instruct\"\n",
                "OUTPUT_DIR = \"./results_0.5b_v5\"\n",
                "MERGED_DIR = \"./friday-0.5b-v5-merged\"\n",
                "\n",
                "print(f\"[*] Loading Tokenizer: {BASE_MODEL_ID}...\")\n",
                "tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)\n",
                "if tokenizer.pad_token is None:\n",
                "    tokenizer.pad_token = tokenizer.eos_token\n",
                "\n",
                "print(f\"[*] Loading Base Model: {BASE_MODEL_ID}...\")\n",
                "model = AutoModelForCausalLM.from_pretrained(\n",
                "    BASE_MODEL_ID,\n",
                "    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,\n",
                "    device_map={\"\": 0},  # Pin directly to GPU 0\n",
                "    trust_remote_code=True,\n",
                ")\n",
                "\n",
                "model.gradient_checkpointing_enable()\n",
                "model.enable_input_require_grads()\n",
                "model.config.use_cache = False\n",
                "\n",
                "# High-capacity LoRA (Rank 64, Alpha 128) covering all attention and MLP projections\n",
                "peft_config = LoraConfig(\n",
                "    task_type=TaskType.CAUSAL_LM,\n",
                "    r=64,\n",
                "    lora_alpha=128,\n",
                "    lora_dropout=0.05,\n",
                "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"],\n",
                "    bias=\"none\",\n",
                ")\n",
                "model = get_peft_model(model, peft_config)\n",
                "model.print_trainable_parameters()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 5. Tokenization & Prompt Loss Masking (512 Context Window)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def format_chatml(batch, max_length=512):\n",
                "    input_ids_list = []\n",
                "    labels_list = []\n",
                "\n",
                "    for messages in batch[\"messages\"]:\n",
                "        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)\n",
                "        prompt_msgs = messages[:-1]\n",
                "        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)\n",
                "\n",
                "        full_tokens = tokenizer(text, truncation=True, max_length=max_length)\n",
                "        prompt_tokens = tokenizer(prompt_text, truncation=True, max_length=max_length)\n",
                "\n",
                "        p_len = len(prompt_tokens[\"input_ids\"])\n",
                "        labels = [-100] * p_len + full_tokens[\"input_ids\"][p_len:]\n",
                "\n",
                "        if len(labels) < len(full_tokens[\"input_ids\"]):\n",
                "            labels.extend([-100] * (len(full_tokens[\"input_ids\"]) - len(labels)))\n",
                "        elif len(labels) > len(full_tokens[\"input_ids\"]):\n",
                "            labels = labels[: len(full_tokens[\"input_ids\"])]\n",
                "\n",
                "        input_ids_list.append(full_tokens[\"input_ids\"])\n",
                "        labels_list.append(labels)\n",
                "\n",
                "    return {\"input_ids\": input_ids_list, \"labels\": labels_list}\n",
                "\n",
                "print(\"[*] Tokenizing datasets...\")\n",
                "tokenized_train = train_ds.map(format_chatml, batched=True, remove_columns=train_ds.column_names)\n",
                "tokenized_val = val_ds.map(format_chatml, batched=True, remove_columns=val_ds.column_names)\n",
                "print(f\"[+] Tokenized {len(tokenized_train):,} training examples successfully!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 6. Execute Fine-Tuning (3 Epochs, Cosine Annealing)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, TrainerCallback\n",
                "\n",
                "# Custom realtime newline callback for Kaggle stdout log streaming\n",
                "class RealtimeLineLogger(TrainerCallback):\n",
                "    def on_log(self, args, state, control, logs=None, **kwargs):\n",
                "        if logs and \"loss\" in logs:\n",
                "            loss = logs.get(\"loss\", 0.0)\n",
                "            lr = logs.get(\"learning_rate\", 0.0)\n",
                "            epoch = logs.get(\"epoch\", state.epoch)\n",
                "            pct = (state.global_step / state.max_steps) * 100 if state.max_steps else 0.0\n",
                "            print(f\"[Step {state.global_step:4d}/{state.max_steps}] ({pct:5.1f}%) | Loss: {loss:.4f} | LR: {lr:.2e} | Epoch: {epoch:.2f}\", flush=True)\n",
                "\n",
                "training_args = TrainingArguments(\n",
                "    output_dir=OUTPUT_DIR,\n",
                "    per_device_train_batch_size=8,\n",
                "    per_device_eval_batch_size=8,\n",
                "    gradient_accumulation_steps=2,       # Effective batch size = 16\n",
                "    learning_rate=2e-4,\n",
                "    lr_scheduler_type=\"cosine\",\n",
                "    warmup_steps=100,\n",
                "    num_train_epochs=3,\n",
                "    fp16=not torch.cuda.is_bf16_supported(),\n",
                "    bf16=torch.cuda.is_bf16_supported(),\n",
                "    logging_steps=25,\n",
                "    disable_tqdm=True,\n",
                "    eval_strategy=\"epoch\",\n",
                "    save_strategy=\"epoch\",\n",
                "    save_total_limit=1,\n",
                "    optim=\"adamw_torch\",\n",
                "    report_to=\"none\",\n",
                ")\n",
                "\n",
                "collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors=\"pt\")\n",
                "\n",
                "trainer = Trainer(\n",
                "    model=model,\n",
                "    args=training_args,\n",
                "    train_dataset=tokenized_train,\n",
                "    eval_dataset=tokenized_val,\n",
                "    data_collator=collator,\n",
                "    callbacks=[RealtimeLineLogger()],\n",
                ")\n",
                "\n",
                "print(f\"[*] Starting DesktopIntent-0.5B-v5-Precision Training on {len(tokenized_train):,} samples...\")\n",
                "trainer.train()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 7. Merge LoRA Adapter & Save Standalone Model"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "print(f\"[*] Merging LoRA adapter into base model and saving to {MERGED_DIR}...\")\n",
                "model.config.use_cache = True\n",
                "merged_model = model.merge_and_unload()\n",
                "os.makedirs(MERGED_DIR, exist_ok=True)\n",
                "merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)\n",
                "tokenizer.save_pretrained(MERGED_DIR)\n",
                "print(f\"[+] Standalone merged model successfully saved to {MERGED_DIR}!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 8. Export to Q8_0 GGUF (~525 MB)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"[*] Exporting to Q8_0 GGUF (~525 MB - ultra-fast inference)...\")\n",
                "!python llama.cpp/convert_hf_to_gguf.py {MERGED_DIR} --outtype q8_0 --outfile ./friday-0.5b-v5-q8.gguf\n",
                "\n",
                "print(\"\\n[+] Generated GGUF Artifacts:\")\n",
                "!ls -lh *.gguf"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

# Write files
target_dir = Path("notebooks/temp_kaggle_v5")
target_dir.mkdir(parents=True, exist_ok=True)

nb_file = target_dir / "friday-0-5b-v5-precision.ipynb"
with open(nb_file, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

meta = {
    "id": "thisisdarkstar/friday-0-5b-v5-precision",
    "title": "friday-0.5b-v5-precision",
    "code_file": "friday-0-5b-v5-precision.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "keywords": [],
    "dataset_sources": [],
    "kernel_sources": [],
    "competition_sources": [],
    "model_sources": []
}

meta_file = target_dir / "kernel-metadata.json"
with open(meta_file, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"[✓] Created notebook at: {nb_file}")
print(f"[✓] Created kernel-metadata.json at: {meta_file}")
