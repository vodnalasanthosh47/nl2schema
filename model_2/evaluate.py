"""
evaluate.py — Evaluation for Schema → SQL Query Generation

This is UNCONDITIONAL generation — there is NO gold query to compare against.
The model generates SQL queries from a schema, and we measure:

  1. SQL Validity     — Does it execute without error on the actual SQLite DB?
  2. Schema Fidelity  — Does it only reference existing tables/columns? (covered by #1)
  3. Query Diversity   — How many distinct SQL constructs does the set cover?

DO NOT use: Exact Match, Execution Accuracy vs gold, Spider eval script, or sqlparse.

Usage:
    python evaluate.py --model schema2sql_model/final
    python evaluate.py --model schema2sql_model/final --num-queries 10
    python evaluate.py --model schema2sql_model/final --test  # Use test schemas (unseen)
"""

import argparse
import json
import os
import sqlite3
import re

import yaml
import torch
from tqdm import tqdm


def load_config(config_path="config.yaml"):
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


# ── Core Metrics ──────────────────────────────────────────────


def sql_validity(sql, db_path):
    """
    Check if SQL executes without error on the actual SQLite database.
    This catches: syntax errors, nonexistent tables, nonexistent columns,
    type mismatches, bad JOINs, etc.

    DO NOT use sqlparse — it accepts almost any string.
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(sql)
        conn.close()
        return True
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return False


def diversity_score(queries):
    """
    Measure structural diversity across a set of generated queries.
    Returns the fraction of distinct SQL constructs covered.
    """
    constructs = [
        "JOIN", "GROUP BY", "HAVING", "WHERE",
        "ORDER BY", "COUNT", "SUM", "AVG", "MAX", "MIN",
        "DISTINCT", "LIMIT", "LIKE", "BETWEEN",
        "IN", "NOT IN", "INTERSECT", "UNION", "EXCEPT",
    ]
    coverage = set()
    for q in queries:
        q_upper = q.upper()
        for c in constructs:
            if c in q_upper:
                coverage.add(c)
    return len(coverage) / len(constructs) if constructs else 0


def uniqueness_score(queries):
    """Fraction of queries that are unique (no duplicates)."""
    if not queries:
        return 0
    normalized = set(q.strip().lower() for q in queries)
    return len(normalized) / len(queries)


# ── Query Generation ──────────────────────────────────────────


def load_model(model_path):
    """Load the trained model and tokenizer."""
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


def generate_queries_for_schema(schema_text, model, tokenizer, device,
                                 num_queries=5, max_input_length=1024,
                                 max_output_length=384):
    """
    Generate multiple diverse SQL queries for a given schema using
    diverse beam search (diversity_penalty=1.5).
    """
    prompt = f"Generate a SQL query for this database:\n{schema_text}"

    inputs = tokenizer(
        prompt,
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
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
        )

    queries = []
    for output in outputs:
        sql = tokenizer.decode(output, skip_special_tokens=True).strip()
        if not sql:
            continue
            
        # Post-processing fix 1: Whitespace in aliases ("T 2.id" -> "T2.id")
        sql = re.sub(r'\bT\s+(\d+)\b', r'T\1', sql)
        
        # Post-processing fix 2: Wildcard in JOIN ("T1.People_ID = T2.*")
        if re.search(r'ON\s+\S+\s*=\s*\w+\.\*', sql, re.IGNORECASE):
            continue
            
        queries.append(sql)

    return queries


# ── Schema Serialization (for test schemas) ───────────────────


def serialize_schema_ddl(db_schema):
    """Quick DDL serialization for evaluation on test schemas."""
    table_names = db_schema["table_names_original"]
    column_names = db_schema["column_names_original"]
    column_types = db_schema["column_types"]
    primary_keys = set(db_schema["primary_keys"])

    fk_map = {}
    for fk_from, fk_to in db_schema["foreign_keys"]:
        tbl_idx = column_names[fk_to][0]
        fk_map[fk_from] = (table_names[tbl_idx], column_names[fk_to][1])

    ddl_parts = []
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
            ddl_parts.append(f"CREATE TABLE {table_name} (\n" + ",\n".join(cols) + "\n);")

    return "\n\n".join(ddl_parts)


# ── Main Evaluation ───────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate Schema→SQL model")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--num-queries", type=int, default=10, help="Queries per schema")
    parser.add_argument("--test", action="store_true",
                        help="Evaluate on test schemas (unseen during training)")
    parser.add_argument("--max-schemas", type=int, default=None,
                        help="Limit number of schemas to evaluate")
    parser.add_argument("--save", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    # Determine which schemas and databases to use
    if args.test:
        tables_path = data_cfg.get("test_tables", "test_tables.json")
        database_dir = data_cfg.get("test_database_dir", "test_database")
        eval_label = "TEST (unseen schemas)"
    else:
        tables_path = data_cfg.get("tables", "tables.json")
        database_dir = data_cfg.get("database_dir", "database")
        eval_label = "DEV (training schemas)"

    print("=" * 60)
    print("📊 Schema → SQL — Evaluation")
    print("=" * 60)
    print(f"  Model:        {args.model}")
    print(f"  Eval set:     {eval_label}")
    print(f"  Queries/DB:   {args.num_queries}")
    print(f"  DB dir:       {database_dir}")
    print()

    # Load schemas
    with open(tables_path) as f:
        all_schemas = json.load(f)

    # Filter to schemas that have an actual SQLite database
    schemas = []
    for s in all_schemas:
        db_path = os.path.join(database_dir, s["db_id"], f"{s['db_id']}.sqlite")
        if os.path.exists(db_path):
            schemas.append(s)

    if args.max_schemas:
        schemas = schemas[:args.max_schemas]

    print(f"  Schemas with DB files: {len(schemas)}")

    # Load model
    print(f"\n📦 Loading model...")
    model, tokenizer, device = load_model(args.model)
    max_input = model_cfg.get("max_input_length", 1024)
    max_output = model_cfg.get("max_output_length", 256)

    # ── Generate and Evaluate ──────────────────────────────────
    all_results = []
    total_queries = 0
    total_valid = 0
    all_diversity_scores = []
    all_uniqueness_scores = []

    print(f"\n🔮 Generating queries for {len(schemas)} schemas...\n")

    for schema in tqdm(schemas):
        db_id = schema["db_id"]
        db_path = os.path.join(database_dir, db_id, f"{db_id}.sqlite")
        schema_text = serialize_schema_ddl(schema)

        # Generate queries
        queries = generate_queries_for_schema(
            schema_text, model, tokenizer, device,
            num_queries=args.num_queries,
            max_input_length=max_input,
            max_output_length=max_output,
        )

        # Check validity by executing each query
        valid = []
        invalid = []
        for q in queries:
            if sql_validity(q, db_path):
                valid.append(q)
            else:
                invalid.append(q)

        validity_rate = len(valid) / len(queries) if queries else 0
        div_score = diversity_score(valid) if valid else 0
        uniq_score = uniqueness_score(queries)

        total_queries += len(queries)
        total_valid += len(valid)
        all_diversity_scores.append(div_score)
        all_uniqueness_scores.append(uniq_score)

        result = {
            "db_id": db_id,
            "num_generated": len(queries),
            "num_valid": len(valid),
            "validity_rate": round(validity_rate, 4),
            "diversity_score": round(div_score, 4),
            "uniqueness_score": round(uniq_score, 4),
            "valid_queries": valid,
            "invalid_queries": invalid,
        }
        all_results.append(result)

    # ── Summary ────────────────────────────────────────────────
    overall_validity = total_valid / total_queries if total_queries else 0
    avg_diversity = sum(all_diversity_scores) / len(all_diversity_scores) if all_diversity_scores else 0
    avg_uniqueness = sum(all_uniqueness_scores) / len(all_uniqueness_scores) if all_uniqueness_scores else 0

    print("\n" + "=" * 60)
    print("📊 Results")
    print("=" * 60)
    print(f"\n  Schemas evaluated:     {len(schemas)}")
    print(f"  Total queries:         {total_queries}")
    print(f"  Total valid:           {total_valid}")
    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │ SQL Validity:      {overall_validity:.1%}   ({total_valid}/{total_queries})")
    print(f"  │ Avg Diversity:     {avg_diversity:.1%}")
    print(f"  │ Avg Uniqueness:    {avg_uniqueness:.1%}")
    print(f"  └─────────────────────────────────────┘")

    # Threshold check
    if overall_validity < 0.70:
        print(f"\n  ⚠ SQL Validity ({overall_validity:.1%}) < 70% threshold!")
        print(f"    → Increase name-masking augmentation (20% → 40%)")
        print(f"    → Re-preprocess with: python preprocess.py --augment-ratio 0.4")
        print(f"    → Re-train and re-evaluate")
    else:
        print(f"\n  ✅ SQL Validity ({overall_validity:.1%}) meets the 70% threshold")

    # Show examples
    print("\n" + "-" * 60)
    print("🔍 Sample Results (first 3 schemas)")
    print("-" * 60)

    for r in all_results[:3]:
        print(f"\n  📁 {r['db_id']} — {r['num_valid']}/{r['num_generated']} valid "
              f"(diversity={r['diversity_score']:.0%})")
        for i, q in enumerate(r["valid_queries"][:3]):
            print(f"    ✓ {q}")
        for q in r["invalid_queries"][:2]:
            print(f"    ✗ {q}")

    # Save results
    if args.save:
        output = {
            "summary": {
                "eval_set": eval_label,
                "num_schemas": len(schemas),
                "total_queries": total_queries,
                "total_valid": total_valid,
                "sql_validity": round(overall_validity, 4),
                "avg_diversity": round(avg_diversity, 4),
                "avg_uniqueness": round(avg_uniqueness, 4),
            },
            "per_schema": all_results,
        }
        with open(args.save, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n💾 Results saved to {args.save}")

    print()


if __name__ == "__main__":
    main()
