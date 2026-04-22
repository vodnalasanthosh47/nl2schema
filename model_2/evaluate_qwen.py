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

def sql_validity(sql, db_path):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(sql)
        return True
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return False
    finally:
        if conn:
            conn.close()

def diversity_score(queries):
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
    if not queries: return 0
    normalized = set(q.strip().lower() for q in queries)
    return len(normalized) / len(queries)

def load_model(model_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        from peft import AutoPeftModelForCausalLM
        has_peft = True
    except ImportError:
        has_peft = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Auto-detect if it's a LoRA adapter or merged model
    if os.path.exists(os.path.join(model_path, "adapter_config.json")) and has_peft:
        print("  Detected LoRA Adapter! Loading with peft...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )

    model.eval()
    return model, tokenizer, model.device

def encode_prompt_keeping_separator(prompt_body, tokenizer, device, max_input):
    separator = "\n-- SQL QUERY --\n"
    separator_ids = tokenizer(separator, add_special_tokens=False)["input_ids"]
    body_budget = max_input - len(separator_ids)
    if body_budget <= 0:
        raise ValueError("max_input is too small to fit the SQL separator.")

    body_ids = tokenizer(
        prompt_body,
        truncation=True,
        max_length=body_budget,
        add_special_tokens=False,
    )["input_ids"]
    input_ids = torch.tensor([body_ids + separator_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

def clean_generated_sql(text):
    sql = text.strip()
    sql = re.split(r"\n\s*(?:--|#|Generate a SQL query|CREATE TABLE)\b", sql, maxsplit=1)[0].strip()
    if ";" in sql:
        sql = sql.split(";", 1)[0].strip() + ";"
    else:
        sql = sql.split("\n")[0].strip()
    sql = re.sub(r'\bT\s+(\d+)\b', r'T\1', sql)
    return sql

def generate_queries_for_schema(
    schema_text,
    model,
    tokenizer,
    device,
    num_queries=5,
    max_input=768,
    max_output=256,
):
    # CRITICAL: Prompt perfectly matches the formatting separator from train_qwen.py
    prompt = f"Generate a SQL query for this database:\n{schema_text}"
    inputs = encode_prompt_keeping_separator(prompt, tokenizer, device, max_input)

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_output,
            num_return_sequences=num_queries,
            do_sample=True, # Diverse generation
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    queries = []
    for output in outputs:
        # CRITICAL: Slice off the prompt!
        generated_tokens = output[input_length:]
        sql = clean_generated_sql(tokenizer.decode(generated_tokens, skip_special_tokens=True))
        if not sql: continue
        if re.search(r'ON\s+\S+\s*=\s*\w+\.\*', sql, re.IGNORECASE):
            continue
            
        queries.append(sql)

    return queries

def serialize_schema_ddl(db_schema):
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
            if tbl_idx != table_idx or col_name == "*": continue
            col_def = f"  {col_name} {column_types[col_idx]}"
            mods = []
            if col_idx in primary_keys: mods.append("PRIMARY KEY")
            if col_idx in fk_map:
                ref_tbl, ref_col = fk_map[col_idx]
                mods.append(f"REFERENCES {ref_tbl}({ref_col})")
            if mods: col_def += " " + " ".join(mods)
            cols.append(col_def)
        if cols:
            ddl_parts.append(f"CREATE TABLE {table_name} (\n" + ",\n".join(cols) + "\n);")
    return "\n\n".join(ddl_parts)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen Schema→SQL model")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--num-queries", type=int, default=10, help="Queries per schema")
    parser.add_argument("--max-input", type=int, default=768)
    parser.add_argument("--max-output", type=int, default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--max-schemas", type=int, default=None)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    if args.test:
        tables_path = data_cfg.get("test_tables", "test_tables.json")
        database_dir = data_cfg.get("test_database_dir", "test_database")
    else:
        tables_path = data_cfg.get("tables", "tables.json")
        database_dir = data_cfg.get("database_dir", "database")

    print("\n📦 Loading Causal Model...")
    model, tokenizer, device = load_model(args.model)
    max_input = args.max_input
    max_output = args.max_output or model_cfg.get("max_output_length", 256)

    with open(tables_path) as f:
        all_schemas = json.load(f)

    schemas = []
    for s in all_schemas:
        if os.path.exists(os.path.join(database_dir, s["db_id"], f"{s['db_id']}.sqlite")):
            schemas.append(s)
            
    if args.max_schemas: schemas = schemas[:args.max_schemas]

    all_results = []
    total_queries = total_valid = 0
    all_diversity_scores = []
    all_uniqueness_scores = []

    print(f"\n🔮 Generating Causal queries for {len(schemas)} schemas...\n")
    for schema in tqdm(schemas):
        db_id = schema["db_id"]
        db_path = os.path.join(database_dir, db_id, f"{db_id}.sqlite")
        schema_text = serialize_schema_ddl(schema)

        queries = generate_queries_for_schema(
            schema_text,
            model,
            tokenizer,
            device,
            args.num_queries,
            max_input,
            max_output,
        )

        valid, invalid = [], []
        for q in queries:
            if sql_validity(q, db_path): valid.append(q)
            else: invalid.append(q)

        validity_rate = len(valid) / len(queries) if queries else 0
        div_score = diversity_score(valid) if valid else 0
        uniq_score = uniqueness_score(queries)

        total_queries += len(queries)
        total_valid += len(valid)
        all_diversity_scores.append(div_score)
        all_uniqueness_scores.append(uniq_score)

        all_results.append({
            "db_id": db_id,
            "num_generated": len(queries),
            "num_valid": len(valid),
            "validity_rate": round(validity_rate, 4),
            "diversity_score": round(div_score, 4),
            "uniqueness_score": round(uniq_score, 4),
            "valid_queries": valid,
            "invalid_queries": invalid,
        })

    overall_validity = total_valid / total_queries if total_queries else 0
    avg_diversity = sum(all_diversity_scores) / len(all_diversity_scores) if all_diversity_scores else 0
    avg_uniqueness = sum(all_uniqueness_scores) / len(all_uniqueness_scores) if all_uniqueness_scores else 0

    print("\n" + "=" * 60)
    print("📊 Causal Generative Results")
    print("=" * 60)
    print(f"  SQL Validity:      {overall_validity:.1%}   ({total_valid}/{total_queries})")
    print(f"  Avg Diversity:     {avg_diversity:.1%}")
    print(f"  Avg Uniqueness:    {avg_uniqueness:.1%}\n")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"💾 Results saved to {args.save}")

if __name__ == "__main__":
    main()
