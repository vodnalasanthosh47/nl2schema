"""
train.py
--------
QLoRA fine-tuning of Qwen2.5-Coder-1.5B-Instruct for Natural Language → SQL DDL generation.

Requirements (already installed):
    torch transformers datasets peft accelerate bitsandbytes

Usage:
    python train.py --dataset path/to/dataset.json
"""

import argparse
import json
import os

# Must be set before torch is imported
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import Dataset, Features, Sequence, Value
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_DIR = "./qlora-nl2sql-adapter2"
MAX_LENGTH = 1024   # 2048 OOMs on 6GB; 1024 covers ~p75 of the dataset
DISCARD_THRESHOLD = 1300

SYSTEM_PROMPT = (
    "You are an expert database designer. "
    "Output ONLY SQL DDL. Do not include markdown. Do not include comments."
)

LORA_CONFIG = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

TRAINING_ARGS = dict(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=5,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    #eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    #load_best_model_at_end=True,
    #metric_for_best_model="eval_loss",
    remove_unused_columns=False,
    report_to="none",
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
)


# ---------------------------------------------------------------------------
# 1. Load tokenizer
# ---------------------------------------------------------------------------
def load_tokenizer(model_id: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


# ---------------------------------------------------------------------------
# 2. Load model in 4-bit (QLoRA)
# ---------------------------------------------------------------------------
def load_4bit_model(model_id: str) -> AutoModelForCausalLM:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return model


# ---------------------------------------------------------------------------
# 3. Load & convert dataset
# ---------------------------------------------------------------------------
def load_raw_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "Dataset must be a JSON array."
    assert all("input" in d and "output" in d for d in data), \
        'Each item must have "input" and "output" keys.'
    return data


def build_chat_messages(example: dict) -> list[dict]:
    """Wrap one dataset example in Qwen chat format."""
    return [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": f"Generate ONLY SQL DDL for the following description:\n\n{example['input']}"},
        {"role": "assistant", "content": example["output"]},
    ]


def _extract_ids(template_output) -> list[int]:
    """
    apply_chat_template returns a BatchEncoding (dict-like) when tokenize=True
    on some transformers versions, and a plain list on others.
    Always extract input_ids explicitly to handle both cases.
    """
    if hasattr(template_output, "input_ids"):
        # BatchEncoding with attribute access
        ids = template_output.input_ids
    elif isinstance(template_output, dict):
        # Plain dict
        ids = template_output["input_ids"]
    else:
        # Already a plain list of ints
        ids = template_output
    # Flatten if batched (shape [1, seq_len])
    if isinstance(ids, (list, tuple)) and len(ids) > 0 and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(x) for x in ids]


# ---------------------------------------------------------------------------
# 4. Tokenise — label only the assistant turn (ignore prompt tokens in loss)
# ---------------------------------------------------------------------------
def tokenize_example(example: dict, tokenizer: AutoTokenizer) -> dict:
    assert tokenizer.pad_token_id is not None, \
        "pad_token_id is None — check load_tokenizer()"

    messages = build_chat_messages(example)

    # compute full length WITHOUT truncation to decide whether to discard
    full_ids_no_trunc = _extract_ids(tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=False,
    ))

    if len(full_ids_no_trunc) > DISCARD_THRESHOLD:
        return None
    
    # Full sequence: system + user + assistant
    full_ids = _extract_ids(tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=MAX_LENGTH,
    ))

    # Prompt only: system + user (no assistant reply)
    prompt_ids = _extract_ids(tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ))

    prompt_len = len(prompt_ids)
    seq_len    = len(full_ids)

    # Labels: -100 for prompt tokens (masked from loss), real ids for assistant tokens
    labels         = [-100] * prompt_len + full_ids[prompt_len:]
    attention_mask = [1]    * seq_len

    # Pad / truncate everything to exactly MAX_LENGTH
    pad_id = tokenizer.pad_token_id
    if seq_len < MAX_LENGTH:
        padding        = MAX_LENGTH - seq_len
        full_ids       = full_ids       + [pad_id] * padding
        labels         = labels         + [-100]   * padding
        attention_mask = attention_mask + [0]       * padding
    else:
        full_ids       = full_ids[:MAX_LENGTH]
        labels         = labels[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]

    return {
        "input_ids":      full_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }


# ---------------------------------------------------------------------------
# 4b. Build HuggingFace Dataset with explicit schema
# ---------------------------------------------------------------------------
def build_hf_dataset(raw: list[dict], tokenizer: AutoTokenizer) -> Dataset:
    
    tokenized = [
    t for t in
    (tokenize_example(ex, tokenizer) for ex in raw)
    if t is not None
    ]

    features = Features({
        "input_ids":      Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int32")),
        "labels":         Sequence(Value("int32")),
    })

    return Dataset.from_list(tokenized, features=features)


# ---------------------------------------------------------------------------
# 5. Apply LoRA adapters
# ---------------------------------------------------------------------------
def apply_lora(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_cfg = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------
def train(model, tokenizer, train_dataset: Dataset):
    training_args = TrainingArguments(**TRAINING_ARGS)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        #eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    return trainer


# ---------------------------------------------------------------------------
# 7. Save adapter
# ---------------------------------------------------------------------------
def save_adapter(model, tokenizer):
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅  LoRA adapter saved to: {os.path.abspath(OUTPUT_DIR)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Qwen2.5-Coder for NL→SQL DDL")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON file")
    args = parser.parse_args()

    print("── Step 1/6  Loading tokenizer …")
    tokenizer = load_tokenizer(MODEL_ID)

    print("── Step 2/6  Loading model in 4-bit …")
    model = load_4bit_model(MODEL_ID)

    print("── Step 3/6  Loading & tokenising dataset …")
    raw = load_raw_dataset(args.dataset)

    # 90/10 train/val split
    split = int(len(raw) * 0.9)
    #train_raw, val_raw = raw[:split], raw[split:]

    train_dataset = build_hf_dataset(raw, tokenizer)
    print(f"            {len(train_dataset)} training examples tokenised.")
    #val_dataset   = build_hf_dataset(val_raw,   tokenizer)
    #print(f"            {len(train_dataset)} train / {len(val_dataset)} val examples tokenised.")

    print("── Step 4/6  Attaching LoRA adapters …")
    model = apply_lora(model)

    print("── Step 5/6  Training …")
    trainer = train(model, tokenizer, train_dataset)

    print("── Step 6/6  Saving adapter …")
    save_adapter(trainer.model, tokenizer)


if __name__ == "__main__":
    main()