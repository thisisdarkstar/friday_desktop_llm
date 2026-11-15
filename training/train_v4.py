"""DesktopIntent-v4 Training Pipeline (10/10 Perfect Assistant Alignment).

Fine-tunes Qwen2.5-0.5B on:
1. Dynamic Zero-Shot In-Context Tool Schemas (Fast path + Runtime injection)
2. STT Acoustic Noise & Disfluency Hardening
3. Stateful Anaphora & Coreference Resolution
4. Zero-Latency Local Chit-Chat (<15 token witty responses)
5. Calibrated Ambiguity Gating (clarification_required)
"""

import os
import sys
import ctypes
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL_DIR = "D:/Projects/desktop-llm/models/desktop-intent-merged-v3"
OUTPUT_DIR = "D:/Projects/desktop-llm/models/checkpoints_v4"
MERGED_DIR = "D:/Projects/desktop-llm/models/desktop-intent-merged-v4"


def lock_windows_awake():
    """Prevent Windows from sleeping during GPU training."""
    try:
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        res = ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        if res:
            print("[*] Windows power management: Sleep locked for training session.")
    except Exception as e:
        print(f"[-] Could not set Windows execution state: {e}")


def format_chatml(batch, tokenizer, max_length=384):
    input_ids_list = []
    labels_list = []

    for messages in batch["messages"]:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_msgs = messages[:-1]
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )

        full_tokens = tokenizer(text, truncation=True, max_length=max_length)
        prompt_tokens = tokenizer(prompt_text, truncation=True, max_length=max_length)

        p_len = len(prompt_tokens["input_ids"])
        labels = [-100] * p_len + full_tokens["input_ids"][p_len:]

        if len(labels) < len(full_tokens["input_ids"]):
            labels.extend([-100] * (len(full_tokens["input_ids"]) - len(labels)))
        elif len(labels) > len(full_tokens["input_ids"]):
            labels = labels[: len(full_tokens["input_ids"])]

        input_ids_list.append(full_tokens["input_ids"])
        labels_list.append(labels)

    return {"input_ids": input_ids_list, "labels": labels_list}


def main():
    lock_windows_awake()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] DesktopIntent-v4 Training Device: {device}")
    if device == "cuda":
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[*] VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        torch.cuda.empty_cache()

    print(f"[*] Loading Tokenizer and Base Model from: {BASE_MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    print("[*] Applying LoRA Adapter (rank=16, alpha=32)...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("[*] Loading v4 training datasets...")
    dataset_dir = Path("D:/Projects/desktop-llm/dataset")
    train_file = str(dataset_dir / "v4_train.jsonl")
    val_file = str(dataset_dir / "v4_val.jsonl")

    raw_datasets = load_dataset("json", data_files={"train": train_file, "val": val_file})

    train_ds = raw_datasets["train"].map(
        lambda b: format_chatml(b, tokenizer, max_length=384),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    val_ds = raw_datasets["val"].map(
        lambda b: format_chatml(b, tokenizer, max_length=384),
        batched=True,
        remove_columns=raw_datasets["val"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=3,
        fp16=(device == "cuda"),
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        optim="adamw_torch",
        report_to="none",
    )

    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print(f"[*] Starting DesktopIntent-v4 Fine-Tuning ({len(train_ds)} samples)...")
    trainer.train()

    print(f"[*] Merging LoRA weights and saving to {MERGED_DIR}...")
    model.config.use_cache = True
    merged_model = model.merge_and_unload()
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"[+] DesktopIntent-v4 model successfully saved to {MERGED_DIR}!")


if __name__ == "__main__":
    main()
