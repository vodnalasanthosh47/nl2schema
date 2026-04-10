"""
train.py — Training Script for Schema → SQL Query Generation

Fine-tunes CodeT5-base (or T5) on preprocessed Spider data.

Key settings (corrected for fine-tuning, not from-scratch):
  - 5 epochs (15 overfits on 8.6k pairs)
  - lr=5e-5 (3e-4 destabilizes pretrained weights)
  - warmup_ratio=0.1 (not fixed warmup_steps)
  - max_input_length=1024 (512 truncates large schemas)
  - early stopping at patience=3

Usage:
    python train.py                        # From config.yaml
    python train.py --dry-run              # 1 epoch, 100 samples
    python train.py --epochs 3 --lr 3e-5   # Override
"""

import argparse
import json
import os

import numpy as np
import yaml
import torch
# PyTorch 2.6 changed torch.load() to enforce weights_only=True by default.
# HuggingFace's RNG state uses Numpy arrays, which throws a security error now. 
# We explicitly allowcast the numpy constructors to bypass this.
try:
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
    torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
except AttributeError:
    pass

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

from dataset import load_pairs, SchemaToSQLDataset


def load_config(config_path="config.yaml"):
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def compute_metrics(eval_preds, tokenizer):
    """
    Training-time metrics. Only uses eval_loss (primary) and avg prediction length.
    
    NOTE: We do NOT compute exact match or execution accuracy during training.
    This is unconditional generation — there's no single gold query to compare against.
    Real evaluation (SQL validity via SQLite execution, diversity) happens in evaluate.py.
    """
    predictions, labels = eval_preds
    # Replace any out-of-range token IDs (e.g. -100 padding) with pad_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    predictions = np.where(predictions < 0, pad_id, predictions)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    avg_length = np.mean([len(p.split()) for p in decoded_preds]) if decoded_preds else 0

    return {
        "avg_pred_length": round(avg_length, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Schema→SQL model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test: 1 epoch, 100 train + 20 dev samples")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    model_name = args.model or model_cfg.get("name", "Salesforce/codet5-base")
    num_epochs = args.epochs or train_cfg.get("num_epochs", 5)
    batch_size = args.batch_size or train_cfg.get("batch_size", 8)
    lr = args.lr or train_cfg.get("learning_rate", 5e-5)
    output_dir = args.output_dir or train_cfg.get("output_dir", "schema2sql_model")
    processed_dir = args.processed_dir or data_cfg.get("processed_dir", "processed_data")
    max_input_length = model_cfg.get("max_input_length", 1024)
    max_output_length = model_cfg.get("max_output_length", 256)
    warmup_ratio = train_cfg.get("warmup_ratio", 0.1)
    patience = train_cfg.get("early_stopping_patience", 3)

    # Device detection
    if torch.cuda.is_available():
        device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
        use_fp16 = True
        use_bf16 = False
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_info = "Apple MPS"
        use_fp16 = False
        use_bf16 = False  # MPS bf16 support is inconsistent
    else:
        device_info = "CPU"
        use_fp16 = False
        use_bf16 = False

    print("=" * 60)
    print("🚀 Schema → SQL — Training")
    print("=" * 60)
    print(f"  Model:          {model_name}")
    print(f"  Device:         {device_info}")
    print(f"  Epochs:         {num_epochs}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Learning rate:  {lr}")
    print(f"  Warmup ratio:   {warmup_ratio}")
    print(f"  FP16:           {use_fp16}")
    print(f"  Max input len:  {max_input_length}")
    print(f"  Max output len: {max_output_length}")
    print(f"  Early stopping: patience={patience}")
    print(f"  Output dir:     {output_dir}")
    print()

    # ── Load Data ──────────────────────────────────────────────
    print("📖 Loading preprocessed data...")
    train_pairs = load_pairs(os.path.join(processed_dir, "train.json"))
    dev_pairs = load_pairs(os.path.join(processed_dir, "dev.json"))

    if args.dry_run:
        print("⚡ DRY RUN: 100 train, 20 dev, 1 epoch")
        train_pairs = train_pairs[:100]
        dev_pairs = dev_pairs[:20]
        num_epochs = 1

    print(f"  Train: {len(train_pairs)} pairs")
    print(f"  Dev:   {len(dev_pairs)} pairs")

    # ── Load Model ─────────────────────────────────────────────
    print(f"\n📦 Loading model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception as e:
        print(f"  ⚠ Failed: {e}")
        fallback = "t5-small"
        print(f"  Falling back to {fallback}...")
        tokenizer = AutoTokenizer.from_pretrained(fallback)
        model = AutoModelForSeq2SeqLM.from_pretrained(fallback)
        model_name = fallback

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ✓ {model_name} ({num_params:.0f}M parameters)")

    # ── Create Datasets ────────────────────────────────────────
    print("\n🔄 Tokenizing...")
    train_ds = SchemaToSQLDataset(train_pairs, tokenizer, max_input_length, max_output_length)
    dev_ds = SchemaToSQLDataset(dev_pairs, tokenizer, max_input_length, max_output_length)
    print(f"  ✓ Train: {len(train_ds)} | Dev: {len(dev_ds)}")

    # ── Training ───────────────────────────────────────────────
    # Compute warmup_steps from ratio (warmup_ratio deprecated in transformers 5.x)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 2)
    total_steps = (len(train_ds) // (batch_size * grad_accum)) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    print(f"  Warmup steps:   {warmup_steps} (ratio={warmup_ratio} × {total_steps} total steps)")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=grad_accum,

        eval_strategy=train_cfg.get("evaluation_strategy", "epoch"),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=3,

        predict_with_generate=True,
        generation_max_length=max_output_length,

        fp16=use_fp16,
        bf16=use_bf16,

        logging_steps=train_cfg.get("logging_steps", 50),
        logging_first_step=True,
        report_to="none",

        seed=train_cfg.get("seed", 42),
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, label_pad_token_id=-100
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda preds: compute_metrics(preds, tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    print("\n" + "=" * 60)
    print("🏋️ Starting training...")
    print("=" * 60 + "\n")

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # ── Save ───────────────────────────────────────────────────
    final_dir = os.path.join(output_dir, "final")
    print(f"\n💾 Saving to {final_dir}...")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    with open(os.path.join(final_dir, "training_info.json"), "w") as f:
        json.dump({
            "model_name": model_name,
            "num_train_examples": len(train_pairs),
            "num_dev_examples": len(dev_pairs),
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "warmup_ratio": warmup_ratio,
            "max_input_length": max_input_length,
            "max_output_length": max_output_length,
        }, f, indent=2)

    # Final eval
    print("\n📊 Final eval...")
    metrics = trainer.evaluate()
    print(f"  eval_loss: {metrics.get('eval_loss', 'N/A')}")

    with open(os.path.join(final_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Done! Model at: {final_dir}")


if __name__ == "__main__":
    main()
