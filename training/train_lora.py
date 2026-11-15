"""LoRA fine-tuning script for DesktopIntent on Qwen2.5-0.5B-Instruct.
Optimized for NVIDIA GeForce RTX 2050 (4 GB VRAM) with FP16 + Gradient Checkpointing.
"""

import os
import sys
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

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./models/checkpoints"
MERGED_DIR = "./models/desktop-intent-merged"


def format_chatml(batch, tokenizer, max_length=256):
    """Format ChatML messages into tokenized input_ids and labels for causal LM.
    Masks out system and user prompt tokens with -100 so loss is only calculated on assistant tool calls.
    """
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

        # Ensure lengths match
        if len(labels) < len(full_tokens["input_ids"]):
            labels.extend([-100] * (len(full_tokens["input_ids"]) - len(labels)))
        elif len(labels) > len(full_tokens["input_ids"]):
            labels = labels[: len(full_tokens["input_ids"])]

        input_ids_list.append(full_tokens["input_ids"])
        labels_list.append(labels)

    return {"input_ids": input_ids_list, "labels": labels_list}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training Device: {device}")
    if device == "cuda":
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"[*] Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
        )
        torch.cuda.empty_cache()

    print(f"[*] Loading Tokenizer and Base Model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    # Enable gradient checkpointing to save VRAM on RTX 2050 (4GB)
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

    print("[*] Loading datasets...")
    dataset_dir = Path(__file__).parent.parent / "dataset"
    train_file = str(dataset_dir / "friday_train.jsonl")
    if not os.path.exists(train_file):
        train_file = str(dataset_dir / "train.jsonl")

    val_file = str(dataset_dir / "friday_val.jsonl")
    if not os.path.exists(val_file):
        val_file = str(dataset_dir / "val.jsonl")

    print(f"    Train: {train_file}")
    print(f"    Val:   {val_file}")

    raw_datasets = load_dataset("json", data_files={"train": train_file, "val": val_file})

    train_ds = raw_datasets["train"].map(
        lambda b: format_chatml(b, tokenizer, max_length=256),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    val_ds = raw_datasets["val"].map(
        lambda b: format_chatml(b, tokenizer, max_length=256),
        batched=True,
        remove_columns=raw_datasets["val"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=2,
        fp16=(device == "cuda"),
        logging_steps=50,
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

    print("[*] Starting Fine-Tuning on RTX 2050...")
    trainer.train()

    print(f"[*] Merging LoRA weights into base model and saving to {MERGED_DIR}...")
    model.config.use_cache = True
    merged_model = model.merge_and_unload()
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"[+] Model successfully saved to {MERGED_DIR} and ready for Ollama / GGUF conversion!")


if __name__ == "__main__":
    main()
