import argparse     # For parsing command-line flags like --epochs, --lr, --model
import json          # For loading the preprocessed train.json dataset from disk
import os            # For file path operations and checking if directories exist

import torch  # Core PyTorch library for tensor operations and VRAM management
from datasets import Dataset                                          # Hugging Face Dataset wrapper for efficient batch processing
from peft import LoraConfig, TaskType, get_peft_model               # PEFT library for adding LoRA adapters to the base model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments  # Hugging Face transformers for model loading and training loop


# The separator token sequence that tells the model: "everything before this is the schema,
# everything AFTER this is the SQL query you must generate"
SEPARATOR = "\n-- SQL QUERY --\n"


def load_pairs(json_path):
    # Open the preprocessed dataset JSON file (output of preprocess.py)
    with open(json_path) as f:
        return json.load(f)  # Returns a list of {input, output, db_id, augmented} dicts


def build_tokenizer(model_name):
    # Download/load the tokenizer that matches the base model — converts text to token IDs
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # Qwen models don't have a dedicated pad token; reuse EOS so padding doesn't crash
        tokenizer.pad_token = tokenizer.eos_token
    # Pad on the RIGHT so that the label mask alignment stays correct during training
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_pair(pair, tokenizer, max_prompt_length, max_output_length):
    # Strip trailing whitespace from the schema input (the prompt body)
    prompt_body = pair["input"].rstrip()
    # Strip leading/trailing whitespace from the target SQL query
    output      = pair["output"].strip()
    # Pre-tokenize the separator string (e.g. "\n-- SQL QUERY --\n") to know its length
    separator_ids = tokenizer(SEPARATOR, add_special_tokens=False)["input_ids"]
    # Calculate how many tokens remain for the actual schema after reserving space for the separator
    body_budget = max_prompt_length - len(separator_ids)
    if body_budget <= 0:
        # Fail loudly if the config leaves no room at all for the schema — misconfiguration
        raise ValueError("--max-prompt-length is too small to fit the SQL separator.")

    # Tokenize the schema body, truncating to the available budget, then append the separator
    prompt_ids = tokenizer(
        prompt_body,
        add_special_tokens=False,   # Don't add BOS/EOS yet — we control the full sequence
        truncation=True,            # Cut off schema if it exceeds the budget
        max_length=body_budget,     # The maximum token count for the schema portion
    )["input_ids"] + separator_ids  # Concat: schema tokens + separator tokens

    # Tokenize the gold SQL output, also truncating to the output token budget
    output_ids = tokenizer(
        output,
        add_special_tokens=False,  # No BOS — the EOS is added manually below
        truncation=True,
        max_length=max_output_length,
    )["input_ids"]

    if tokenizer.eos_token_id is not None:
        # Append the End-Of-Sequence token so the model learns WHEN to stop generating
        output_ids = output_ids + [tokenizer.eos_token_id]

    # Full input sequence = prompt tokens (schema + separator) + output tokens (SQL)
    input_ids = prompt_ids + output_ids
    # Labels = -100 for prompt tokens (ignored in loss), then the actual output token IDs
    # This is "completion-style" training: the model is only penalized for the SQL part
    labels = [-100] * len(prompt_ids) + output_ids

    return {
        "input_ids":      input_ids,
        "attention_mask": [1] * len(input_ids),  # 1 = real token, attend to it; 0 = pad token, ignore
        "labels":         labels,
    }


class CompletionCollator:
    # Custom collator needed because our sequences have variable lengths after tokenization
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer  # Store tokenizer so we can use its .pad() method

    def __call__(self, features):
        # Extract only the fields the tokenizer's pad() method expects
        input_features = [
            {
                "input_ids":      f["input_ids"],
                "attention_mask": f["attention_mask"],
            }
            for f in features
        ]
        # Pad all sequences in the batch to the same length (the longest sequence in this batch)
        batch = self.tokenizer.pad(input_features, return_tensors="pt")

        # Determine the padded length for aligning labels
        max_len    = batch["input_ids"].shape[1]
        label_rows = []
        for f in features:
            labels  = f["labels"]
            # Calculate how many positions need label padding at the END
            pad_len = max_len - len(labels)
            # Pad with -100 (the "ignore" sentinel) so padding doesn't contribute to loss
            label_rows.append(labels + [-100] * pad_len)
        # Stack label lists into a tensor matching the padded input shape
        batch["labels"] = torch.tensor(label_rows, dtype=torch.long)
        return batch  # Returns a dict with input_ids, attention_mask, labels tensors


def assert_labels_are_trainable(tokenized_dataset, tokenizer, sample_count=32):
    # Sanity check: make sure SOME output tokens have labels != -100 before we waste GPU time
    supervised_counts = []  # Count supervised (non -100) tokens per sample
    first_target      = None  # Store the decoded first target for human inspection

    for row in tokenized_dataset.select(range(min(sample_count, len(tokenized_dataset)))):
        labels     = row["labels"]
        # Filter out all the -100 sentinel values; only keep real output token IDs
        supervised = [token_id for token_id in labels if token_id != -100]
        supervised_counts.append(len(supervised))
        if first_target is None and supervised:
            # Decode the first real output token sequence to verify it's SQL, not garbage
            first_target = tokenizer.decode(supervised, skip_special_tokens=True)

    if not supervised_counts or max(supervised_counts) == 0:
        # If ZERO supervised tokens exist, training will do literally nothing — crash immediately
        raise RuntimeError(
            "No supervised SQL tokens were found. Reduce --max-prompt-length or "
            "increase --max-output-length before training."
        )

    avg_supervised = sum(supervised_counts) / len(supervised_counts)
    # Print diagnostics so the user can verify training will actually update the model
    print(f"  Supervised SQL tokens/sample: avg={avg_supervised:.1f}, min={min(supervised_counts)}, max={max(supervised_counts)}")
    print(f"  First decoded target: {first_target[:180]}")


def main():
    parser = argparse.ArgumentParser()
    # Base model to fine-tune — Qwen2.5-Coder-1.5B is the coding-optimized variant
    parser.add_argument("--model",           default="Qwen/Qwen2.5-Coder-1.5B")
    # Directory containing the preprocessed train.json file (output of preprocess.py)
    parser.add_argument("--data-dir",        default="processed_data")
    # Where to save checkpoints and the final LoRA adapter
    parser.add_argument("--output-dir",      default="qwen_sql_model")
    # Number of full passes over the training dataset
    parser.add_argument("--epochs",          type=float, default=3)
    # VIVA POINT: We use batch_size=1 because a 1.5B model + gradients will crash Colab VRAM.
    parser.add_argument("--batch-size",      type=int, default=1)
    # VIVA POINT: grad-accum=8 mathematically simulates a batch size of 8 (smoother updates)
    parser.add_argument("--grad-accum",      type=int, default=8)
    # VIVA POINT: LoRA uses 1e-4 because it only trains a tiny adapter matrix, needing a larger LR than full-finetuning.
    parser.add_argument("--lr",              type=float, default=1e-4)
    # Maximum number of tokens allowed for the schema portion of each training input
    parser.add_argument("--max-prompt-length", type=int, default=768)
    # Maximum number of tokens allowed for the SQL output portion of each training sample
    parser.add_argument("--max-output-length", type=int, default=256)
    # If set, run a quick 128-sample smoke test to verify the pipeline works end-to-end
    parser.add_argument("--dry-run",         action="store_true")
    # Optional cap on training samples (useful for fast iteration without a full dry-run)
    parser.add_argument("--max-train-samples", type=int, default=None)
    args = parser.parse_args()

    # Print a human-readable training configuration summary
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

    # Build the path to the training pairs file generated by preprocess.py
    train_path = os.path.join(args.data_dir, "train.json")
    train_data = load_pairs(train_path)  # Load all {input, output} pairs into memory
    if args.dry_run:
        train_data   = train_data[:128]      # Tiny subset for a fast smoke-test
        args.epochs  = 1                     # Only one epoch in dry-run mode
        args.output_dir = os.path.join(args.output_dir, "dry_run")  # Separate output folder
    elif args.max_train_samples:
        # Cap the dataset to a specified number of samples for faster debug runs
        train_data = train_data[: args.max_train_samples]

    # Load the tokenizer — must match the model so token IDs are compatible
    tokenizer = build_tokenizer(args.model)

    # Wrap the raw list of dicts in a Hugging Face Dataset for efficient .map() batching
    raw_dataset       = Dataset.from_list(train_data)
    # Apply tokenization to every sample; drop the original string columns we no longer need
    tokenized_dataset = raw_dataset.map(
        lambda row: tokenize_pair(
            row,
            tokenizer,
            max_prompt_length=args.max_prompt_length,
            max_output_length=args.max_output_length,
        ),
        remove_columns=raw_dataset.column_names,  # Drop "input", "output", "db_id", "augmented"
        desc="Tokenizing train dataset",
    )

    # Verify the tokenized dataset actually has supervisable SQL tokens before loading the model
    assert_labels_are_trainable(tokenized_dataset, tokenizer)

    print("\nLoading base model...")
    # Choose the best precision for this GPU — bfloat16 is more numerically stable on Ampere+
    dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16  # Use bfloat16 on A100/H100 for better training stability

    # Load the full 1.5B parameter base model onto available devices (auto shards if multi-GPU)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,            # Load in half-precision to save ~3 GB of VRAM
        device_map="auto",      # Automatically distribute across GPU(s) and CPU if needed
        trust_remote_code=True, # Required for Qwen's custom attention implementation
    )
    # Disable KV-cache during training — it's only useful for inference, wastes VRAM here
    model.config.use_cache = False

    # =================================================================================
    # VIVA POINT (LoRA): "Parameter-Efficient Fine-Tuning"
    # If we trained all 1.5B weights, it would cause OOM and 'Catastrophic Forgetting'.
    # We freeze the base model and only train tiny adapter matrices.
    # =================================================================================
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,   # Tell PEFT this is a causal language model
        # r=16 (Rank): Compresses a 1024x1024 matrix update into 1024x16 * 16x1024.
        # This slashes trainable parameters from 1.5 Billion to just ~10 Million.
        r=16,
        # lora_alpha: Scaling factor (how strongly the adapter overrides base weights)
        lora_alpha=32,
        lora_dropout=0.05,  # Light dropout on LoRA weights to prevent adapter overfitting
        # target_modules: We inject these adapters specifically into the Attention layers
        target_modules=[
            "q_proj",    # Query projection matrix in each attention head
            "k_proj",    # Key projection matrix in each attention head
            "v_proj",    # Value projection matrix in each attention head
            "o_proj",    # Output projection that combines all attention heads
            "gate_proj", # Gate branch of the SwiGLU feed-forward network
            "up_proj",   # Up-projection branch of the SwiGLU feed-forward network
            "down_proj", # Down-projection that reduces dimensionality after SwiGLU
        ],
    )
    # Wrap the frozen base model with the trainable LoRA adapter layers
    model = get_peft_model(model, peft_config)
    # Print a summary showing ~10M trainable vs ~1.5B frozen parameters
    model.print_trainable_parameters()

    if hasattr(model, "enable_input_require_grads"):
        # Required for gradient checkpointing to work correctly with PEFT wrappers
        model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=args.output_dir,                    # Where to save checkpoints
        num_train_epochs=args.epochs,                  # Total training epochs
        per_device_train_batch_size=args.batch_size,   # Samples per GPU per step (1 to avoid OOM)
        gradient_accumulation_steps=args.grad_accum,   # Accumulate gradients over 8 steps → effective batch 8
        learning_rate=args.lr,                         # Starting learning rate for AdamW
        warmup_ratio=0.03,                             # Linearly warm up LR for first 3% of steps
        lr_scheduler_type="cosine",                    # Cosine annealing decays LR smoothly to zero
        fp16=dtype == torch.float16,                   # Enable mixed-precision float16 training
        bf16=dtype == torch.bfloat16,                  # Enable bfloat16 if supported (more stable)
        logging_steps=10,                              # Log loss/LR metrics every 10 optimizer steps
        logging_first_step=True,                       # Also log step 1 to verify training started
        save_strategy="epoch",                         # Save a checkpoint at the end of each epoch
        save_total_limit=2,                            # Keep only the 2 most recent checkpoints
        optim="adamw_torch",                           # Use PyTorch's native AdamW optimizer
        gradient_checkpointing=True,                   # Trade compute for VRAM by recomputing activations
        max_grad_norm=1.0,                             # Clip gradients to prevent exploding gradients
        report_to="none",                              # Disable W&B / TensorBoard logging
        remove_unused_columns=False,                   # Keep all columns; our collator handles them
    )

    # Instantiate the Hugging Face Trainer with our custom data collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=CompletionCollator(tokenizer),  # Handles variable-length padding + label alignment
    )

    print("\nStarting LoRA training...")
    trainer.train()  # Launch the full training loop

    # Save only the LoRA adapter weights (a few MB) — not the full 1.5B frozen base model
    final_dir = os.path.join(args.output_dir, "final")
    print(f"\nSaving tuned adapter to {final_dir}")
    trainer.model.save_pretrained(final_dir)  # Saves adapter_config.json + adapter_model.bin
    tokenizer.save_pretrained(final_dir)      # Save tokenizer so inference doesn't need the original
    print("Done.")


if __name__ == "__main__":
    main()