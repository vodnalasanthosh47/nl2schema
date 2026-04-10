"""
preprocess.py — Spider Dataset Preprocessing for Schema → SQL Generation

Re-frames the Spider dataset (NL→SQL) into Schema→SQL format:
  Input:  Serialized database schema (CREATE TABLE DDL)
  Output: SQL query

Features:
  - Table-boundary truncation (never truncates mid-table)
  - Name-masking augmentation for generalization to unseen schemas
  - Token-aware serialization using the actual model tokenizer

Usage:
    python preprocess.py
    python preprocess.py --schema-format compact
    python preprocess.py --augment-ratio 0.3
"""

import json
import os
import argparse
import random
import re
import yaml
from collections import defaultdict
from tqdm import tqdm


def load_config(config_path="config.yaml"):
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


# ── Schema Serialization ──────────────────────────────────────


def _build_fk_map(db_schema):
    """Build foreign key lookup: col_idx → (target_table, target_col)."""
    fk_map = {}
    table_names = db_schema["table_names_original"]
    column_names = db_schema["column_names_original"]
    for fk_from, fk_to in db_schema["foreign_keys"]:
        target_tbl_idx = column_names[fk_to][0]
        target_col_name = column_names[fk_to][1]
        target_tbl_name = table_names[target_tbl_idx]
        fk_map[fk_from] = (target_tbl_name, target_col_name)
    return fk_map


def serialize_single_table_ddl(db_schema, table_idx):
    """Serialize one table from the schema as a CREATE TABLE DDL statement."""
    table_names = db_schema["table_names_original"]
    column_names = db_schema["column_names_original"]
    column_types = db_schema["column_types"]
    primary_keys = set(db_schema["primary_keys"])
    fk_map = _build_fk_map(db_schema)
    table_name = table_names[table_idx]

    columns = []
    for col_idx, (tbl_idx, col_name) in enumerate(column_names):
        if tbl_idx != table_idx or col_name == "*":
            continue
        col_type = column_types[col_idx]
        modifiers = []
        if col_idx in primary_keys:
            modifiers.append("PRIMARY KEY")
        if col_idx in fk_map:
            ref_table, ref_col = fk_map[col_idx]
            modifiers.append(f"REFERENCES {ref_table}({ref_col})")
        col_def = f"  {col_name} {col_type}"
        if modifiers:
            col_def += " " + " ".join(modifiers)
        columns.append(col_def)

    if not columns:
        return ""
    return f"CREATE TABLE {table_name} (\n" + ",\n".join(columns) + "\n);"


def serialize_schema_ddl(db_schema):
    """Convert full schema to CREATE TABLE DDL (no truncation)."""
    ddl_parts = []
    for i in range(len(db_schema["table_names_original"])):
        ddl = serialize_single_table_ddl(db_schema, i)
        if ddl:
            ddl_parts.append(ddl)
    return "\n\n".join(ddl_parts)


def serialize_schema_ddl_truncated(db_schema, tokenizer, max_tokens=1024):
    """
    Serialize schema with table-boundary truncation.
    Never truncates mid-table — adds whole tables until the token budget is exhausted.
    Leaves ~50 tokens headroom for the prompt prefix.
    """
    prompt_overhead = 50  # tokens for "Generate a SQL query for this database:\n"
    budget = max_tokens - prompt_overhead

    result = []
    current_len = 0
    for i in range(len(db_schema["table_names_original"])):
        ddl = serialize_single_table_ddl(db_schema, i)
        if not ddl:
            continue
        toks = len(tokenizer.encode(ddl, add_special_tokens=False))
        if current_len + toks > budget:
            break
        result.append(ddl)
        current_len += toks

    return "\n\n".join(result)


def serialize_schema_compact(db_schema):
    """Compact pipe-delimited format: table : col1 (PK) | col2 (FK→ref.col) | ..."""
    lines = []
    table_names = db_schema["table_names_original"]
    column_names = db_schema["column_names_original"]
    primary_keys = set(db_schema["primary_keys"])
    fk_map = _build_fk_map(db_schema)

    for table_idx, table_name in enumerate(table_names):
        cols = []
        for col_idx, (tbl_idx, col_name) in enumerate(column_names):
            if tbl_idx != table_idx or col_name == "*":
                continue
            annotations = []
            if col_idx in primary_keys:
                annotations.append("PK")
            if col_idx in fk_map:
                ref_tbl, ref_col = fk_map[col_idx]
                annotations.append(f"FK→{ref_tbl}.{ref_col}")
            col_str = col_name
            if annotations:
                col_str += f" ({', '.join(annotations)})"
            cols.append(col_str)
        if cols:
            lines.append(f"{table_name} : {' | '.join(cols)}")

    return "\n".join(lines)


# ── Name-Masking Augmentation ─────────────────────────────────


def mask_names_in_pair(schema_text, sql_query, db_schema):
    """
    Replace real table/column names with generic placeholders (table_0, col_0, etc.)
    in both the schema and SQL query. Forces the model to learn SQL structure
    rather than memorize specific names.
    """
    table_names = db_schema["table_names_original"]
    column_names_orig = [
        c[1] for c in db_schema["column_names_original"] if c[1] != "*"
    ]

    # Build mapping: real name → masked name
    # Sort by length descending to avoid partial replacements
    table_map = {}
    for i, t in enumerate(table_names):
        table_map[t] = f"table_{i}"

    col_map = {}
    seen_cols = set()
    col_counter = 0
    for c in column_names_orig:
        if c not in seen_cols:
            col_map[c] = f"col_{col_counter}"
            seen_cols.add(c)
            col_counter += 1

    # Apply replacements — tables first (longer names), then columns
    masked_schema = schema_text
    masked_sql = sql_query

    # Sort by length descending to prevent partial-match issues
    for real, masked in sorted(table_map.items(), key=lambda x: -len(x[0])):
        masked_schema = re.sub(re.escape(real), masked, masked_schema, flags=re.IGNORECASE)
        masked_sql = re.sub(re.escape(real), masked, masked_sql, flags=re.IGNORECASE)

    for real, masked in sorted(col_map.items(), key=lambda x: -len(x[0])):
        masked_schema = re.sub(r'\b' + re.escape(real) + r'\b', masked, masked_schema, flags=re.IGNORECASE)
        masked_sql = re.sub(r'\b' + re.escape(real) + r'\b', masked, masked_sql, flags=re.IGNORECASE)

    return masked_schema, masked_sql


# ── Training Pair Construction ────────────────────────────────


def build_training_pairs(train_data, tables_data, schema_format="ddl",
                         tokenizer=None, max_tokens=1024, augment_ratio=0.0):
    """
    Build (schema_text, sql_query) training pairs.

    Args:
        train_data: Spider training examples
        tables_data: Schema definitions from tables.json
        schema_format: "ddl" or "compact"
        tokenizer: Required for DDL truncation
        max_tokens: Max input tokens (for truncation)
        augment_ratio: Fraction of pairs to duplicate with name masking (0.0-1.0)

    Returns:
        List of dicts: {input, output, db_id, augmented}
    """
    # Build schema cache
    schema_cache = {}
    db_schemas = {s["db_id"]: s for s in tables_data}

    for db_id, schema in db_schemas.items():
        if schema_format == "ddl" and tokenizer:
            schema_cache[db_id] = serialize_schema_ddl_truncated(schema, tokenizer, max_tokens)
        elif schema_format == "ddl":
            schema_cache[db_id] = serialize_schema_ddl(schema)
        else:
            schema_cache[db_id] = serialize_schema_compact(schema)

    # Build base pairs
    pairs = []
    skipped = 0
    for example in tqdm(train_data, desc="Building training pairs"):
        db_id = example["db_id"]
        if db_id not in schema_cache:
            skipped += 1
            continue

        schema_text = schema_cache[db_id]
        sql_query = example["query"].strip()

        pairs.append({
            "input": f"Generate a SQL query for this database:\n{schema_text}",
            "output": sql_query,
            "db_id": db_id,
            "augmented": False,
        })

    if skipped:
        print(f"  ⚠ Skipped {skipped} examples (schema not in tables.json)")

    # Name-masking augmentation
    if augment_ratio > 0:
        num_augment = int(len(pairs) * augment_ratio)
        augment_indices = random.sample(range(len(pairs)), min(num_augment, len(pairs)))
        augmented_pairs = []

        for idx in tqdm(augment_indices, desc="Name-masking augmentation"):
            pair = pairs[idx]
            db_id = pair["db_id"]
            schema = db_schemas[db_id]

            masked_schema, masked_sql = mask_names_in_pair(
                schema_cache[db_id], pair["output"], schema
            )
            augmented_pairs.append({
                "input": f"Generate a SQL query for this database:\n{masked_schema}",
                "output": masked_sql,
                "db_id": db_id,
                "augmented": True,
            })

        pairs.extend(augmented_pairs)
        print(f"  ✓ Added {len(augmented_pairs)} name-masked augmented pairs")

    return pairs


def compute_statistics(pairs):
    """Print dataset statistics."""
    base = [p for p in pairs if not p["augmented"]]
    aug = [p for p in pairs if p["augmented"]]
    db_ids = set(p["db_id"] for p in pairs)

    input_lens = [len(p["input"].split()) for p in pairs]
    output_lens = [len(p["output"].split()) for p in pairs]

    print("\n" + "=" * 60)
    print("📊 Dataset Statistics")
    print("=" * 60)
    print(f"  Base pairs:              {len(base)}")
    print(f"  Augmented pairs:         {len(aug)}")
    print(f"  Total pairs:             {len(pairs)}")
    print(f"  Unique databases:        {len(db_ids)}")
    print(f"  Avg input tokens (ws):   {sum(input_lens)/len(input_lens):.0f}")
    print(f"  Max input tokens (ws):   {max(input_lens)}")
    print(f"  Avg output tokens (ws):  {sum(output_lens)/len(output_lens):.0f}")
    print(f"  Max output tokens (ws):  {max(output_lens)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Preprocess Spider for Schema→SQL training")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--schema-format", choices=["ddl", "compact"], default=None)
    parser.add_argument("--augment-ratio", type=float, default=None)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    schema_format = args.schema_format or data_cfg.get("schema_format", "ddl")
    augment_ratio = args.augment_ratio if args.augment_ratio is not None else data_cfg.get("augment_ratio", 0.2)
    max_tokens = args.max_tokens or model_cfg.get("max_input_length", 1024)
    data_dir = args.data_dir
    processed_dir = os.path.join(data_dir, data_cfg.get("processed_dir", "processed_data"))

    print(f"📁 Data dir:        {data_dir}")
    print(f"📦 Schema format:   {schema_format}")
    print(f"📐 Max tokens:      {max_tokens}")
    print(f"🎭 Augment ratio:   {augment_ratio}")
    print(f"💾 Output dir:      {processed_dir}")

    # Load tokenizer for truncation
    model_name = model_cfg.get("name", "Salesforce/codet5-base")
    print(f"\n📦 Loading tokenizer ({model_name})...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"  ✓ Loaded {model_name} tokenizer")
    except Exception as e:
        print(f"  ⚠ Could not load {model_name}: {e}")
        print("  Falling back to t5-small tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # Load raw data
    print("\n📖 Loading Spider dataset...")
    with open(os.path.join(data_dir, data_cfg.get("train_spider", "train_spider.json"))) as f:
        train_spider = json.load(f)
    print(f"  ✓ train_spider.json: {len(train_spider)} examples")

    with open(os.path.join(data_dir, data_cfg.get("train_others", "train_others.json"))) as f:
        train_others = json.load(f)
    print(f"  ✓ train_others.json: {len(train_others)} examples")

    with open(os.path.join(data_dir, data_cfg.get("dev", "dev.json"))) as f:
        dev_data = json.load(f)
    print(f"  ✓ dev.json: {len(dev_data)} examples")

    with open(os.path.join(data_dir, data_cfg.get("tables", "tables.json"))) as f:
        tables = json.load(f)
    print(f"  ✓ tables.json: {len(tables)} databases")

    all_train = train_spider + train_others
    print(f"\n  Combined training: {len(all_train)} examples")

    # Build pairs
    print("\n🔧 Building training pairs...")
    random.seed(42)
    train_pairs = build_training_pairs(
        all_train, tables, schema_format, tokenizer, max_tokens, augment_ratio
    )

    print("\n🔧 Building dev pairs (no augmentation)...")
    dev_pairs = build_training_pairs(
        dev_data, tables, schema_format, tokenizer, max_tokens, augment_ratio=0.0
    )

    compute_statistics(train_pairs)

    # Save
    os.makedirs(processed_dir, exist_ok=True)
    train_path = os.path.join(processed_dir, "train.json")
    dev_path = os.path.join(processed_dir, "dev.json")

    with open(train_path, "w") as f:
        json.dump(train_pairs, f, indent=2)
    print(f"\n💾 Saved {len(train_pairs)} training pairs → {train_path}")

    with open(dev_path, "w") as f:
        json.dump(dev_pairs, f, indent=2)
    print(f"💾 Saved {len(dev_pairs)} dev pairs → {dev_path}")

    # Show sample
    print("\n📋 Sample pair:")
    s = train_pairs[0]
    print(f"  DB: {s['db_id']} | Augmented: {s['augmented']}")
    print(f"  Input (first 300 chars):\n    {s['input'][:300]}...")
    print(f"  Output: {s['output']}")

    # Show augmented sample if available
    aug = [p for p in train_pairs if p["augmented"]]
    if aug:
        print(f"\n📋 Sample augmented pair:")
        s = aug[0]
        print(f"  DB: {s['db_id']} | Augmented: {s['augmented']}")
        print(f"  Input (first 300 chars):\n    {s['input'][:300]}...")
        print(f"  Output: {s['output']}")

    print("\n✅ Preprocessing complete!")


if __name__ == "__main__":
    main()
