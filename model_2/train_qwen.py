import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


SEPARATOR = "\n-- SQL QUERY --\n"


def load_pairs(json_path):
    with open(json_path) as f:
        return json.load(f)


def build_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_pair(pair, tokenizer, max_prompt_length, max_output_length):
    prompt_body = pair["input"].rstrip()
    output = pair["output"].strip()
    separator_ids = tokenizer(SEPARATOR, add_special_tokens=False)["input_ids"]
    body_budget = max_prompt_length - len(separator_ids)
    if body_budget <= 0:
        raise ValueError("--max-prompt-length is too small to fit the SQL separator.")

    prompt_ids = tokenizer(
        prompt_body,
        add_special_tokens=False,
        truncation=True,
        max_length=body_budget,
    )["input_ids"] + separator_ids
    output_ids = tokenizer(
        output,
        add_special_tokens=False,
        truncation=True,
        max_length=max_output_length,
    )["input_ids"]

    if tokenizer.eos_token_id is not None:
        output_ids = output_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + output_ids
    labels = [-100] * len(prompt_ids) + output_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class CompletionCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_features = [
            {
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"],
            }
            for f in features
        ]
        batch = self.tokenizer.pad(input_features, return_tensors="pt")

        max_len = batch["input_ids"].shape[1]
        label_rows = []
        for f in features:
            labels = f["labels"]
            pad_len = max_len - len(labels)
            label_rows.append(labels + [-100] * pad_len)
        batch["labels"] = torch.tensor(label_rows, dtype=torch.long)
        return batch


def assert_labels_are_trainable(tokenized_dataset, tokenizer, sample_count=32):
    supervised_counts = []
    first_target = None

    for row in tokenized_dataset.select(range(min(sample_count, len(tokenized_dataset)))):
        labels = row["labels"]
        supervised = [token_id for token_id in labels if token_id != -100]
        supervised_counts.append(len(supervised))
        if first_target is None and supervised:
            first_target = tokenizer.decode(supervised, skip_special_tokens=True)

    if not supervised_counts or max(supervised_counts) == 0:
        raise RuntimeError(
            "No supervised SQL tokens were found. Reduce --max-prompt-length or "
            "increase --max-output-length before training."
        )

    avg_supervised = sum(supervised_counts) / len(supervised_counts)
    print(f"  Supervised SQL tokens/sample: avg={avg_supervised:.1f}, min={min(supervised_counts)}, max={max(supervised_counts)}")
    print(f"  First decoded target: {first_target[:180]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B")
    parser.add_argument("--data-dir", default="processed_data")
    parser.add_argument("--output-dir", default="qwen_sql_model")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--max-output-length", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen2.5-Coder LoRA Training")
    print("=" * 60)
    print(f"  Model:             {args.model}")
    print(f"  Data dir:          {args.data_dir}")
    print(f"  Output dir:        {args.output_dir}")
    print(f"  Epochs:            {args.epochs}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  Grad accumulation: {args.grad_accum}")
    print(f"  Learning rate:     {args.lr}")
    print(f"  Prompt tokens:     {args.max_prompt_length}")
    print(f"  SQL tokens:        {args.max_output_length}")

    train_path = os.path.join(args.data_dir, "train.json")
    train_data = load_pairs(train_path)
    if args.dry_run:
        train_data = train_data[:128]
        args.epochs = 1
        args.output_dir = os.path.join(args.output_dir, "dry_run")
    elif args.max_train_samples:
        train_data = train_data[: args.max_train_samples]

    tokenizer = build_tokenizer(args.model)

    raw_dataset = Dataset.from_list(train_data)
    tokenized_dataset = raw_dataset.map(
        lambda row: tokenize_pair(
            row,
            tokenizer,
            max_prompt_length=args.max_prompt_length,
            max_output_length=args.max_output_length,
        ),
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing train dataset",
    )

    assert_labels_are_trainable(tokenized_dataset, tokenizer)

    print("\nLoading base model...")
    dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        logging_steps=10,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=2,
        optim="adamw_torch",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=CompletionCollator(tokenizer),
    )

    print("\nStarting LoRA training...")
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    print(f"\nSaving tuned adapter to {final_dir}")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Done.")


if __name__ == "__main__":
    main()
