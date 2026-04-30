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

# Standard library imports for file I/O, randomness, regex, and data grouping
import json        # For reading/writing .json files (Spider dataset files)
import os          # For file path operations and checking if files exist
import argparse    # For parsing command-line arguments (--schema-format, --augment-ratio, etc.)
import random      # For randomly selecting pairs to augment
import re          # For regex-based text substitution in name masking
import yaml        # For reading the config.yaml file
from collections import defaultdict  # Unused here but imported for potential grouping utilities
from tqdm import tqdm                # For showing a progress bar when looping over large datasets


def load_config(config_path="config.yaml"):
    # Check if the config file actually exists before attempting to open it
    if os.path.exists(config_path):
        # Open the YAML file and parse it into a Python dictionary
        with open(config_path) as f:
            return yaml.safe_load(f)
    # If no config file is found, return empty dict so downstream .get() calls don't crash
    return {}


# ── Schema Serialization ──────────────────────────────────────


def _build_fk_map(db_schema):
    """Build foreign key lookup: col_idx → (target_table, target_col)."""
    # Initialize an empty dict to store foreign key relationships
    fk_map = {}
    # Pull the list of original table names from the schema JSON
    table_names = db_schema["table_names_original"]
    # Pull the list of (table_index, column_name) pairs for every column
    column_names = db_schema["column_names_original"]
    # Iterate over every foreign key pair in the schema — each is (from_col_idx, to_col_idx)
    for fk_from, fk_to in db_schema["foreign_keys"]:
        # The table index of the target column is stored at position [0]
        target_tbl_idx = column_names[fk_to][0]
        # The column name of the target is stored at position [1]
        target_col_name = column_names[fk_to][1]
        # Resolve the target table index to its actual string name
        target_tbl_name = table_names[target_tbl_idx]
        # Store the mapping: source column index → (target table name, target column name)
        fk_map[fk_from] = (target_tbl_name, target_col_name)
    return fk_map  # e.g. {5: ("orders", "order_id")}


def serialize_single_table_ddl(db_schema, table_idx):
    """Serialize one table from the schema as a CREATE TABLE DDL statement."""
    # Extract all the raw schema arrays we need
    table_names   = db_schema["table_names_original"]   # List of table name strings
    column_names  = db_schema["column_names_original"]  # List of (table_idx, col_name) tuples
    column_types  = db_schema["column_types"]            # List of SQL type strings ("number", "text")
    primary_keys  = set(db_schema["primary_keys"])       # Set of column indices that are PKs
    fk_map        = _build_fk_map(db_schema)             # Dict: col_idx → (ref_table, ref_col)
    # Resolve the integer table index to the actual table name string
    table_name = table_names[table_idx]

    columns = []  # Will hold the formatted column definition strings
    # Enumerate every column in the schema along with its global index
    for col_idx, (tbl_idx, col_name) in enumerate(column_names):
        # Skip columns that belong to a different table, or the wildcard "*" sentinel
        if tbl_idx != table_idx or col_name == "*":
            continue
        col_type  = column_types[col_idx]  # e.g. "number" or "text"
        modifiers = []                      # Will hold "PRIMARY KEY" or "REFERENCES ..." strings
        # Append PRIMARY KEY constraint if this column index is in the PK set
        if col_idx in primary_keys:
            modifiers.append("PRIMARY KEY")
        # Append REFERENCES clause if this column is a foreign key
        if col_idx in fk_map:
            ref_table, ref_col = fk_map[col_idx]  # Unpack the target table and column
            modifiers.append(f"REFERENCES {ref_table}({ref_col})")
        # Start building the column definition: "  col_name TYPE"
        col_def = f"  {col_name} {col_type}"
        # Append any constraints (PRIMARY KEY, REFERENCES) separated by a space
        if modifiers:
            col_def += " " + " ".join(modifiers)
        columns.append(col_def)  # Add this fully-formed column definition to our list

    # If the table has no valid columns (edge case), return empty string
    if not columns:
        return ""
    # Join all column definitions with commas and wrap in a CREATE TABLE block
    return f"CREATE TABLE {table_name} (\n" + ",\n".join(columns) + "\n);"


def serialize_schema_ddl(db_schema):
    """Convert full schema to CREATE TABLE DDL (no truncation)."""
    ddl_parts = []  # Accumulator for each table's DDL string
    # Iterate over every table index using the total count of tables
    for i in range(len(db_schema["table_names_original"])):
        ddl = serialize_single_table_ddl(db_schema, i)  # Serialize this single table
        if ddl:               # Skip tables that returned empty string
            ddl_parts.append(ddl)
    # Join all tables' DDL with a blank line between each for readability
    return "\n\n".join(ddl_parts)


def serialize_schema_ddl_truncated(db_schema, tokenizer, max_tokens=1024):
    """
    ===================================================================================
    VIVA POINT (DATA ENGINEERING): TABLE-BOUNDARY TRUNCATION
    ===================================================================================
    Problem: If a database schema has 50 tables, it will exceed the LLM's 1024 token limit.
    If we just naively cut the string at 1024 tokens, we might slice a CREATE TABLE 
    statement in half (e.g., 'CREATE TABL...'), breaking the SQL syntax completely!
    
    Solution: Instead of cutting the string blindly, we iteratively calculate the token
    length of each FULL table. We keep appending whole tables until we hit our budget.
    This guarantees that the prompt is completely full, but every single table is perfectly
    well-formed with no broken syntax.
    """
    # Reserve 50 tokens for the instruction text prepended to the schema (e.g. "Generate a SQL query...")
    prompt_overhead = 50
    # The remaining token budget available for the schema text itself
    budget = max_tokens - prompt_overhead

    result      = []  # Accumulate fully-formed table DDLs that fit within budget
    current_len = 0   # Running count of tokens consumed so far
    # Iterate over each table in the schema one at a time
    for i in range(len(db_schema["table_names_original"])):
        ddl = serialize_single_table_ddl(db_schema, i)  # Generate this table's DDL
        if not ddl:
            continue  # Skip empty tables (tables with no columns)
        # Count how many tokens this single table's DDL uses
        toks = len(tokenizer.encode(ddl, add_special_tokens=False))
        # If adding this table would exceed the token budget, STOP — don't include it
        if current_len + toks > budget:
            break
        result.append(ddl)      # Safe to include — add to our result list
        current_len += toks     # Update the running token count

    # Join the collected tables with blank lines between them
    return "\n\n".join(result)


def serialize_schema_compact(db_schema):
    """Compact pipe-delimited format: table : col1 (PK) | col2 (FK→ref.col) | ..."""
    lines      = []  # Accumulates one line per table
    table_names  = db_schema["table_names_original"]   # List of table name strings
    column_names = db_schema["column_names_original"]  # List of (table_idx, col_name) tuples
    primary_keys = set(db_schema["primary_keys"])       # Set of PK column indices
    fk_map       = _build_fk_map(db_schema)             # Dict: col_idx → (ref_table, ref_col)

    # Iterate over every table with its index and name
    for table_idx, table_name in enumerate(table_names):
        cols = []  # Accumulate formatted column strings for this table
        for col_idx, (tbl_idx, col_name) in enumerate(column_names):
            # Skip columns from other tables, and the wildcard "*" sentinel
            if tbl_idx != table_idx or col_name == "*":
                continue
            annotations = []  # Collect short annotation tags like "PK" or "FK→..."
            if col_idx in primary_keys:
                annotations.append("PK")  # Mark primary key columns
            if col_idx in fk_map:
                ref_tbl, ref_col = fk_map[col_idx]
                annotations.append(f"FK→{ref_tbl}.{ref_col}")  # Mark foreign key columns
            col_str = col_name
            # Append annotations in parentheses if any exist, e.g. "user_id (PK)"
            if annotations:
                col_str += f" ({', '.join(annotations)})"
            cols.append(col_str)
        if cols:
            # Format as "table_name : col1 | col2 | col3"
            lines.append(f"{table_name} : {' | '.join(cols)}")

    # Join every table onto its own line
    return "\n".join(lines)


# ── Name-Masking Augmentation ─────────────────────────────────


def mask_names_in_pair(schema_text, sql_query, db_schema):
    """
    ===================================================================================
    VIVA POINT (AUGMENTATION): NAME-MASKING
    ===================================================================================
    Problem: Language models often "cheat" by using English semantics instead of 
    learning SQL. For example, if a table is named 'users' and a column is 'user_email', 
    the AI guesses they are joined, instead of actually reading the FOREIGN KEY math.
    
    Solution: We swap real table/column names with generic placeholders (table_0, col_0).
    This strips away all English semantic clues, FORCING the model to learn the actual
    mathematical graph structure of the SQL schema.
    """
    table_names = db_schema["table_names_original"]  # Real table names e.g. ["users", "orders"]
    # Extract only the column name strings (position [1]), excluding the wildcard "*"
    column_names_orig = [
        c[1] for c in db_schema["column_names_original"] if c[1] != "*"
    ]

    # Build mapping: real name → masked name
    # Sort by length descending to avoid partial replacements
    table_map = {}
    # Assign each real table name a generic indexed placeholder
    for i, t in enumerate(table_names):
        table_map[t] = f"table_{i}"  # e.g. "users" → "table_0"

    col_map     = {}         # Will hold real column name → masked placeholder
    seen_cols   = set()      # Track already-processed column names to avoid duplicates
    col_counter = 0          # Monotonically increasing index for placeholder names
    for c in column_names_orig:
        if c not in seen_cols:
            col_map[c] = f"col_{col_counter}"  # e.g. "email" → "col_2"
            seen_cols.add(c)
            col_counter += 1

    # Apply replacements — tables first (longer names), then columns
    masked_schema = schema_text  # Start with the original schema text
    masked_sql    = sql_query    # Start with the original SQL query

    # ===================================================================================
    # VIVA POINT (REGEX ENGINEERING): SUBSTRING COLLISION PREVENTION
    # ===================================================================================
    # If we replace 'user' before 'user_profile', then 'user_profile' gets corrupted 
    # into 'table_0_profile'! 
    # By mathematically sorting the dictionary descending by string length (-len), 
    # we guarantee the longest names ('user_profile') are replaced first.
    # What's left over ('user') is safely replaced afterwards without any collisions!
    for real, masked in sorted(table_map.items(), key=lambda x: -len(x[0])):
        # Replace in schema text — case-insensitive to catch "Users", "USERS", etc.
        masked_schema = re.sub(re.escape(real), masked, masked_schema, flags=re.IGNORECASE)
        # Replace in the SQL query too, so input/output stay consistent
        masked_sql    = re.sub(re.escape(real), masked, masked_sql,    flags=re.IGNORECASE)

    for real, masked in sorted(col_map.items(), key=lambda x: -len(x[0])):
        # Use \b word boundaries to avoid replacing "user_id" when looking for "id"
        masked_schema = re.sub(r'\b' + re.escape(real) + r'\b', masked, masked_schema, flags=re.IGNORECASE)
        masked_sql    = re.sub(r'\b' + re.escape(real) + r'\b', masked, masked_sql,    flags=re.IGNORECASE)

    return masked_schema, masked_sql  # Return both the masked schema and masked SQL


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
    # Build schema cache — pre-serialize every DB's schema so we don't redo it per query
    schema_cache = {}
    # Build a fast lookup dict: db_id string → schema dict
    db_schemas = {s["db_id"]: s for s in tables_data}

    # Pre-serialize every schema into its text form and cache it
    for db_id, schema in db_schemas.items():
        if schema_format == "ddl" and tokenizer:
            # Use token-aware truncation when a tokenizer is provided
            schema_cache[db_id] = serialize_schema_ddl_truncated(schema, tokenizer, max_tokens)
        elif schema_format == "ddl":
            # Fallback: serialize full DDL without any truncation
            schema_cache[db_id] = serialize_schema_ddl(schema)
        else:
            # Use compact pipe-delimited format instead of full DDL
            schema_cache[db_id] = serialize_schema_compact(schema)

    # Build base pairs — one training sample per Spider example
    pairs   = []   # Accumulate all {input, output, db_id, augmented} dicts
    skipped = 0    # Count examples whose db_id wasn't in tables.json
    for example in tqdm(train_data, desc="Building training pairs"):
        db_id = example["db_id"]  # The database ID for this training example
        if db_id not in schema_cache:
            skipped += 1  # This DB has no schema definition — skip it
            continue

        schema_text = schema_cache[db_id]          # Pre-serialized DDL text
        sql_query   = example["query"].strip()     # The gold-standard SQL answer

        # Build the model input as: instruction + schema, and the label as: SQL query
        pairs.append({
            "input":     f"Generate a SQL query for this database:\n{schema_text}",
            "output":    sql_query,
            "db_id":     db_id,
            "augmented": False,  # Flag: this is an original, unmasked pair
        })

    if skipped:
        print(f"  ⚠ Skipped {skipped} examples (schema not in tables.json)")

    # Name-masking augmentation — optionally create additional masked copies of pairs
    if augment_ratio > 0:
        # Calculate how many pairs to augment based on the ratio
        num_augment    = int(len(pairs) * augment_ratio)
        # Randomly select which base pair indices to augment
        augment_indices = random.sample(range(len(pairs)), min(num_augment, len(pairs)))
        augmented_pairs = []  # Store the newly created masked pairs here

        for idx in tqdm(augment_indices, desc="Name-masking augmentation"):
            pair    = pairs[idx]          # The original base pair to augment
            db_id   = pair["db_id"]       # Get its database ID for schema lookup
            schema  = db_schemas[db_id]   # Get the raw schema dict for name replacement

            # Apply name masking to both the schema text and the SQL output
            masked_schema, masked_sql = mask_names_in_pair(
                schema_cache[db_id], pair["output"], schema
            )
            augmented_pairs.append({
                "input":     f"Generate a SQL query for this database:\n{masked_schema}",
                "output":    masked_sql,
                "db_id":     db_id,
                "augmented": True,  # Flag: this is a synthetic name-masked pair
            })

        pairs.extend(augmented_pairs)  # Append augmented pairs after the base pairs
        print(f"  ✓ Added {len(augmented_pairs)} name-masked augmented pairs")

    return pairs  # Return the full list of base + augmented training pairs


def compute_statistics(pairs):
    """Print dataset statistics."""
    # Split pairs into base (original) and augmented (masked) subsets
    base = [p for p in pairs if not p["augmented"]]
    aug  = [p for p in pairs if p["augmented"]]
    # Collect all unique database IDs to count schema coverage
    db_ids = set(p["db_id"] for p in pairs)

    # Approximate token lengths using whitespace splitting (fast, not exact)
    input_lens  = [len(p["input"].split())  for p in pairs]
    output_lens = [len(p["output"].split()) for p in pairs]

    print("\n" + "=" * 60)
    print("📊 Dataset Statistics")
    print("=" * 60)
    print(f"  Base pairs:              {len(base)}")
    print(f"  Augmented pairs:         {len(aug)}")
    print(f"  Total pairs:             {len(pairs)}")
    print(f"  Unique databases:        {len(db_ids)}")
    # Average input length across all pairs
    print(f"  Avg input tokens (ws):   {sum(input_lens)/len(input_lens):.0f}")
    print(f"  Max input tokens (ws):   {max(input_lens)}")
    print(f"  Avg output tokens (ws):  {sum(output_lens)/len(output_lens):.0f}")
    print(f"  Max output tokens (ws):  {max(output_lens)}")
    print("=" * 60)


def main():
    # Set up the command-line argument parser with a description
    parser = argparse.ArgumentParser(description="Preprocess Spider for Schema→SQL training")
    parser.add_argument("--config",         default="config.yaml")                         # Path to YAML config
    parser.add_argument("--schema-format",  choices=["ddl", "compact"], default=None)      # Schema serialization style
    parser.add_argument("--augment-ratio",  type=float, default=None)                      # Fraction of data to augment
    parser.add_argument("--data-dir",       default=".")                                    # Root folder containing Spider JSON files
    parser.add_argument("--max-tokens",     type=int, default=None)                        # Token budget for schema truncation
    args = parser.parse_args()

    # Load the YAML config, then merge with any CLI overrides
    config   = load_config(args.config)
    data_cfg  = config.get("data", {})   # Sub-section of config for data paths
    model_cfg = config.get("model", {})  # Sub-section of config for model parameters

    # CLI flag overrides config file values; config overrides hardcoded defaults
    schema_format = args.schema_format or data_cfg.get("schema_format", "ddl")
    augment_ratio = args.augment_ratio if args.augment_ratio is not None else data_cfg.get("augment_ratio", 0.2)
    max_tokens    = args.max_tokens or model_cfg.get("max_input_length", 1024)
    data_dir      = args.data_dir
    # Build the path where we will save the processed train.json and dev.json
    processed_dir = os.path.join(data_dir, data_cfg.get("processed_dir", "processed_data"))

    # Print a summary of all active settings before starting
    print(f"📁 Data dir:        {data_dir}")
    print(f"📦 Schema format:   {schema_format}")
    print(f"📐 Max tokens:      {max_tokens}")
    print(f"🎭 Augment ratio:   {augment_ratio}")
    print(f"💾 Output dir:      {processed_dir}")

    # Load tokenizer for token-aware truncation — must match the model we'll train
    model_name = model_cfg.get("name", "Salesforce/codet5-base")
    print(f"\n📦 Loading tokenizer ({model_name})...")
    try:
        from transformers import AutoTokenizer
        # Download/cache the tokenizer from Hugging Face Hub
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"  ✓ Loaded {model_name} tokenizer")
    except Exception as e:
        # Fallback if the specified model tokenizer can't be loaded (e.g. no internet)
        print(f"  ⚠ Could not load {model_name}: {e}")
        print("  Falling back to t5-small tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # Load raw Spider dataset files from disk
    print("\n📖 Loading Spider dataset...")
    # Main Spider training file (~7000 NL→SQL examples)
    with open(os.path.join(data_dir, data_cfg.get("train_spider", "train_spider.json"))) as f:
        train_spider = json.load(f)
    print(f"  ✓ train_spider.json: {len(train_spider)} examples")

    # Additional training examples from other datasets bundled with Spider
    with open(os.path.join(data_dir, data_cfg.get("train_others", "train_others.json"))) as f:
        train_others = json.load(f)
    print(f"  ✓ train_others.json: {len(train_others)} examples")

    # Development/validation set used to evaluate generalization
    with open(os.path.join(data_dir, data_cfg.get("dev", "dev.json"))) as f:
        dev_data = json.load(f)
    print(f"  ✓ dev.json: {len(dev_data)} examples")

    # Schema definitions for every database in the dataset
    with open(os.path.join(data_dir, data_cfg.get("tables", "tables.json"))) as f:
        tables = json.load(f)
    print(f"  ✓ tables.json: {len(tables)} databases")

    # Combine both training sources into one unified list
    all_train = train_spider + train_others
    print(f"\n  Combined training: {len(all_train)} examples")

    # Build training pairs with optional name-masking augmentation
    print("\n🔧 Building training pairs...")
    random.seed(42)  # Fix random seed for reproducibility
    train_pairs = build_training_pairs(
        all_train, tables, schema_format, tokenizer, max_tokens, augment_ratio
    )

    # Build dev pairs WITHOUT augmentation — we want clean evaluation data
    print("\n🔧 Building dev pairs (no augmentation)...")
    dev_pairs = build_training_pairs(
        dev_data, tables, schema_format, tokenizer, max_tokens, augment_ratio=0.0
    )

    # Print a statistical summary of the built dataset
    compute_statistics(train_pairs)

    # Create the output directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    train_path = os.path.join(processed_dir, "train.json")
    dev_path   = os.path.join(processed_dir, "dev.json")

    # Serialize and save training pairs to disk
    with open(train_path, "w") as f:
        json.dump(train_pairs, f, indent=2)
    print(f"\n💾 Saved {len(train_pairs)} training pairs → {train_path}")

    # Serialize and save dev pairs to disk
    with open(dev_path, "w") as f:
        json.dump(dev_pairs, f, indent=2)
    print(f"💾 Saved {len(dev_pairs)} dev pairs → {dev_path}")

    # Print the first base pair for a quick sanity check
    print("\n📋 Sample pair:")
    s = train_pairs[0]
    print(f"  DB: {s['db_id']} | Augmented: {s['augmented']}")
    print(f"  Input (first 300 chars):\n    {s['input'][:300]}...")
    print(f"  Output: {s['output']}")

    # Also print the first augmented pair if any were generated
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