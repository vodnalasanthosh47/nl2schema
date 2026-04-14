"""
inference.py — Generate SQL Queries from Database Schemas

Takes a schema (CREATE TABLE DDL) and generates diverse SQL queries
using diverse beam search (diversity_penalty=1.5).

Can be used standalone or imported as a module.

Usage:
    # From a schema file
    python inference.py --model schema2sql_model/final --schema schema.sql

    # From tables.json entry
    python inference.py --model schema2sql_model/final --db-id soccer_3

    # Interactive mode
    python inference.py --model schema2sql_model/final --interactive
"""

import argparse
import json
import os
import sqlite3

import yaml
import torch


def load_config(config_path="config.yaml"):
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def load_model(model_path):
    """Load model, tokenizer, and determine device."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_queries(schema_text, model, tokenizer, device,
                     num_queries=10, max_input_length=1024, max_output_length=256,
                     method="diverse_beam"):
    """
    Generate diverse SQL queries for a schema.

    Args:
        schema_text: CREATE TABLE DDL string
        model: Trained seq2seq model
        tokenizer: Model tokenizer
        device: torch device
        num_queries: How many queries to return
        max_input_length: Max input tokens
        max_output_length: Max output tokens
        method: "diverse_beam" or "sampling"

    Returns:
        List of SQL query strings (deduplicated)
    """
    prompt = f"Generate a SQL query for this database:\n{schema_text}"

    inputs = tokenizer(
        prompt,
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        if method == "diverse_beam":
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_output_length,
                num_return_sequences=num_queries,
                num_beams=num_queries * 2,
                num_beam_groups=num_queries,
                diversity_penalty=1.5,
                early_stopping=True,
                no_repeat_ngram_size=3,
                trust_remote_code=True,
                custom_generate="transformers-community/group-beam-search",
            )
        elif method == "sampling":
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_output_length,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                num_return_sequences=num_queries,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    # Decode and deduplicate
    seen = set()
    queries = []
    for output in outputs:
        sql = tokenizer.decode(output, skip_special_tokens=True).strip()
        sql_normalized = sql.lower().strip()
        if sql and sql_normalized not in seen:
            queries.append(sql)
            seen.add(sql_normalized)

    return queries


def validate_queries(queries, db_path):
    """
    Validate queries by executing against the actual SQLite database.
    Returns (valid_queries, invalid_queries).
    """
    valid = []
    invalid = []
    for q in queries:
        try:
            conn = sqlite3.connect(db_path)
            conn.execute(q)
            conn.close()
            valid.append(q)
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            invalid.append(q)
    return valid, invalid


def categorize_queries(queries):
    """Categorize queries by SQL construct type for better UX."""
    categories = {
        "basic_select": [],
        "filtering": [],
        "aggregation": [],
        "join": [],
        "grouping": [],
        "ordering": [],
        "subquery": [],
        "set_operations": [],
    }

    for q in queries:
        q_upper = q.upper()
        if "JOIN" in q_upper:
            categories["join"].append(q)
        elif any(op in q_upper for op in ["INTERSECT", "UNION", "EXCEPT"]):
            categories["set_operations"].append(q)
        elif "GROUP BY" in q_upper:
            categories["grouping"].append(q)
        elif any(agg in q_upper for agg in ["COUNT", "SUM", "AVG", "MAX", "MIN"]):
            categories["aggregation"].append(q)
        elif "ORDER BY" in q_upper:
            categories["ordering"].append(q)
        elif "WHERE" in q_upper:
            categories["filtering"].append(q)
        elif "SELECT" in q_upper and ("IN" in q_upper or "EXISTS" in q_upper):
            categories["subquery"].append(q)
        else:
            categories["basic_select"].append(q)

    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def serialize_schema_from_tables_json(db_schema):
    """Serialize a tables.json entry as DDL."""
    table_names = db_schema["table_names_original"]
    column_names = db_schema["column_names_original"]
    column_types = db_schema["column_types"]
    primary_keys = set(db_schema["primary_keys"])

    fk_map = {}
    for fk_from, fk_to in db_schema["foreign_keys"]:
        tbl_idx = column_names[fk_to][0]
        fk_map[fk_from] = (table_names[tbl_idx], column_names[fk_to][1])

    parts = []
    for table_idx, table_name in enumerate(table_names):
        cols = []
        for col_idx, (tbl_idx, col_name) in enumerate(column_names):
            if tbl_idx != table_idx or col_name == "*":
                continue
            col_def = f"  {col_name} {column_types[col_idx]}"
            mods = []
            if col_idx in primary_keys:
                mods.append("PRIMARY KEY")
            if col_idx in fk_map:
                ref_tbl, ref_col = fk_map[col_idx]
                mods.append(f"REFERENCES {ref_tbl}({ref_col})")
            if mods:
                col_def += " " + " ".join(mods)
            cols.append(col_def)
        if cols:
            parts.append(f"CREATE TABLE {table_name} (\n" + ",\n".join(cols) + "\n);")

    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Generate SQL queries from schemas")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--schema", default=None, help="Path to a .sql schema file")
    parser.add_argument("--db-id", default=None, help="db_id from tables.json")
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--method", choices=["diverse_beam", "sampling"], default="diverse_beam")
    parser.add_argument("--validate", action="store_true", help="Validate against SQLite DB")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    max_input = model_cfg.get("max_input_length", 1024)
    max_output = model_cfg.get("max_output_length", 256)

    print(f"📦 Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    print(f"  Device: {device}\n")

    if args.interactive:
        # Interactive mode
        print("🔮 Interactive mode — paste a schema and press Enter twice to generate.\n")
        print("Type 'quit' to exit.\n")
        while True:
            print("-" * 40)
            lines = []
            print("Schema (paste DDL, empty line to finish):")
            while True:
                line = input()
                if line.strip() == "" and lines:
                    break
                if line.strip().lower() == "quit":
                    print("Bye!")
                    return
                lines.append(line)

            schema_text = "\n".join(lines)
            queries = generate_queries(
                schema_text, model, tokenizer, device,
                num_queries=args.num_queries, method=args.method,
                max_input_length=max_input, max_output_length=max_output,
            )

            categorized = categorize_queries(queries)
            print(f"\n🔮 Generated {len(queries)} queries:\n")
            for cat, cat_queries in categorized.items():
                print(f"  [{cat}]")
                for q in cat_queries:
                    print(f"    → {q}")
            print()

    elif args.db_id:
        # Generate from tables.json db_id
        tables_path = data_cfg.get("tables", "tables.json")
        with open(tables_path) as f:
            all_schemas = json.load(f)

        schema = next((s for s in all_schemas if s["db_id"] == args.db_id), None)
        if not schema:
            # Try test_tables
            test_path = data_cfg.get("test_tables", "test_tables.json")
            if os.path.exists(test_path):
                with open(test_path) as f:
                    all_schemas = json.load(f)
                schema = next((s for s in all_schemas if s["db_id"] == args.db_id), None)

        if not schema:
            print(f"❌ db_id '{args.db_id}' not found")
            return

        schema_text = serialize_schema_from_tables_json(schema)
        print(f"📋 Schema for '{args.db_id}':")
        print(schema_text)
        print()

        queries = generate_queries(
            schema_text, model, tokenizer, device,
            num_queries=args.num_queries, method=args.method,
            max_input_length=max_input, max_output_length=max_output,
        )

        # Optionally validate
        if args.validate:
            db_dir = data_cfg.get("database_dir", "database")
            db_path = os.path.join(db_dir, args.db_id, f"{args.db_id}.sqlite")
            if not os.path.exists(db_path):
                db_dir = data_cfg.get("test_database_dir", "test_database")
                db_path = os.path.join(db_dir, args.db_id, f"{args.db_id}.sqlite")

            if os.path.exists(db_path):
                valid, invalid = validate_queries(queries, db_path)
                print(f"🔮 Generated {len(queries)} queries "
                      f"({len(valid)} valid, {len(invalid)} invalid):\n")
                for q in valid:
                    print(f"  ✓ {q}")
                for q in invalid:
                    print(f"  ✗ {q}")
            else:
                print(f"  ⚠ No SQLite DB found at {db_path}, showing without validation\n")
                for q in queries:
                    print(f"  → {q}")
        else:
            categorized = categorize_queries(queries)
            print(f"🔮 Generated {len(queries)} queries:\n")
            for cat, cat_queries in categorized.items():
                print(f"  [{cat}]")
                for q in cat_queries:
                    print(f"    → {q}")

    elif args.schema:
        with open(args.schema) as f:
            schema_text = f.read()

        queries = generate_queries(
            schema_text, model, tokenizer, device,
            num_queries=args.num_queries, method=args.method,
            max_input_length=max_input, max_output_length=max_output,
        )

        categorized = categorize_queries(queries)
        print(f"🔮 Generated {len(queries)} queries:\n")
        for cat, cat_queries in categorized.items():
            print(f"  [{cat}]")
            for q in cat_queries:
                print(f"    → {q}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
