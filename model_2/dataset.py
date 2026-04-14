"""
dataset.py — PyTorch Dataset for Schema → SQL Query Generation

Loads preprocessed (schema, sql_query) pairs and tokenizes them.

Usage:
    from dataset import SchemaToSQLDataset, load_datasets
    train_ds, dev_ds = load_datasets("processed_data", tokenizer)
"""

import json
import os
import torch
from torch.utils.data import Dataset


class SchemaToSQLDataset(Dataset):
    """
    PyTorch Dataset for Schema → SQL generation.

    Each item:
        input_ids:      Tokenized schema prompt
        attention_mask:  Attention mask
        labels:          Tokenized SQL (padding masked to -100)
    """

    def __init__(self, pairs, tokenizer, max_input_length=1024, max_output_length=256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        input_enc = self.tokenizer(
            pair["input"],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_enc = self.tokenizer(
            pair["output"],
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_enc["input_ids"].squeeze().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels,
        }

    def get_raw_pair(self, idx):
        return self.pairs[idx]


def load_pairs(json_path):
    with open(json_path) as f:
        return json.load(f)


def load_datasets(processed_dir, tokenizer, max_input_length=1024, max_output_length=256):
    """Load train and dev datasets from preprocessed JSON files."""
    train_path = os.path.join(processed_dir, "train.json")
    dev_path = os.path.join(processed_dir, "dev.json")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"No training data at {train_path}. Run preprocess.py first."
        )

    train_pairs = load_pairs(train_path)
    dev_pairs = load_pairs(dev_path) if os.path.exists(dev_path) else []

    print(f"📊 Loaded {len(train_pairs)} train, {len(dev_pairs)} dev pairs")

    return (
        SchemaToSQLDataset(train_pairs, tokenizer, max_input_length, max_output_length),
        SchemaToSQLDataset(dev_pairs, tokenizer, max_input_length, max_output_length),
    )


if __name__ == "__main__":
    from transformers import AutoTokenizer

    print("🧪 Dataset sanity check...")
    processed_dir = "processed_data"

    if not os.path.exists(os.path.join(processed_dir, "train.json")):
        print("⚠ Run `python preprocess.py` first.")
        exit(1)

    try:
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("t5-small")

    train_ds, dev_ds = load_datasets(processed_dir, tokenizer)

    sample = train_ds[0]
    print(f"  input_ids:      {sample['input_ids'].shape}")
    print(f"  attention_mask:  {sample['attention_mask'].shape}")
    print(f"  labels:          {sample['labels'].shape}")

    raw = train_ds.get_raw_pair(0)
    print(f"  DB: {raw['db_id']}")
    print(f"  Input (first 200): {raw['input'][:200]}...")
    print(f"  Output: {raw['output']}")
    print("✅ Passed!")
