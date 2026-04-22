import argparse
import json
import os
import re
import sqlite3
import yaml
import torch

def load_config(config_path="config.yaml"):
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}

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

    if os.path.exists(os.path.join(model_path, "adapter_config.json")) and has_peft:
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
        add_special_tokens=False,
        truncation=True,
        max_length=body_budget,
    )["input_ids"]
    input_ids = torch.tensor([body_ids + separator_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

def clean_generated_sql(text):
    sql = text.strip()
    sql = re.sub(r"^```(?:sql)?", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"```$", "", sql).strip()
    sql = re.split(r"\n\s*(?:--|#|Generate a SQL query|CREATE TABLE)\b", sql, maxsplit=1)[0].strip()
    if ";" in sql:
        sql = sql.split(";", 1)[0].strip() + ";"
    else:
        sql = sql.split("\n")[0].strip()
    return re.sub(r'\bT\s+(\d+)\b', r'T\1', sql)

def sql_compile_error(sql, db_path):
    if not sql:
        return "empty query"
    if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        return "not a SELECT/WITH query"

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("EXPLAIN QUERY PLAN " + sql)
        return None
    except sqlite3.Error as exc:
        return str(exc)
    finally:
        if conn:
            conn.close()

def generate_queries(schema_text, model, tokenizer, device, num_queries=10, max_input=768, max_output=256, temperature=0.7):
    prompt = (
        "Generate one SQLite SELECT query for this database.\n"
        "Use only tables and columns present in the schema.\n"
        "Return SQL only, with no explanation or markdown.\n"
        f"{schema_text}"
    )
    inputs = encode_prompt_keeping_separator(prompt, tokenizer, device, max_input)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_output,
            num_return_sequences=num_queries,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    seen = set()
    queries = []
    for output in outputs:
        generated_tokens = output[input_length:]
        sql = clean_generated_sql(tokenizer.decode(generated_tokens, skip_special_tokens=True))
        if re.search(r'ON\s+\S+\s*=\s*\w+\.\*', sql, re.IGNORECASE):
            continue
        sql_norm = sql.lower()
        if sql and sql_norm not in seen:
            queries.append(sql)
            seen.add(sql_norm)
    return queries

def generate_compilable_queries(
    schema_text,
    db_path,
    model,
    tokenizer,
    device,
    num_queries=10,
    max_input=768,
    max_output=256,
    max_attempts=80,
):
    seen = set()
    valid = []
    invalid_samples = []
    attempts = 0

    while len(valid) < num_queries and attempts < max_attempts:
        remaining = num_queries - len(valid)
        batch_size = min(max(remaining * 3, 6), max_attempts - attempts)
        temperature = 0.45 if attempts > 0 else 0.7
        candidates = generate_queries(
            schema_text,
            model,
            tokenizer,
            device,
            num_queries=batch_size,
            max_input=max_input,
            max_output=max_output,
            temperature=temperature,
        )
        attempts += batch_size

        for sql in candidates:
            norm = sql.strip().lower()
            if not norm or norm in seen:
                continue
            seen.add(norm)

            error = sql_compile_error(sql, db_path)
            if error is None:
                valid.append(sql)
                if len(valid) == num_queries:
                    break
            elif len(invalid_samples) < 20:
                invalid_samples.append({"sql": sql, "error": error})

    return {
        "queries": valid,
        "num_valid": len(valid),
        "attempt_budget": max_attempts,
        "invalid_samples": invalid_samples,
    }

def serialize_schema_from_tables_json(db_schema):
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
            if tbl_idx != table_idx or col_name == "*": continue
            cdef = f"  {col_name} {column_types[col_idx]}"
            mods = []
            if col_idx in primary_keys: mods.append("PRIMARY KEY")
            if col_idx in fk_map: mods.append(f"REFERENCES {fk_map[col_idx][0]}({fk_map[col_idx][1]})")
            if mods: cdef += " " + " ".join(mods)
            cols.append(cdef)
        if cols: parts.append(f"CREATE TABLE {table_name} (\n" + ",\n".join(cols) + "\n);")
    return "\n\n".join(parts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--db-id", default=None)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--ddl-file", default=None)
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--max-input", type=int, default=768)
    parser.add_argument("--max-output", type=int, default=256)
    parser.add_argument("--valid-only", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=80)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)

    schema_text = None
    db_path = args.db_path

    if args.ddl_file:
        with open(args.ddl_file) as f:
            schema_text = f.read()
    elif args.db_id:
        with open("tables.json") as f:
            all_schemas = json.load(f)
        schema = next((s for s in all_schemas if s["db_id"] == args.db_id), None)
        if schema is None:
            raise ValueError(f"Unknown db_id: {args.db_id}")
        schema_text = serialize_schema_from_tables_json(schema)
        if db_path is None:
            db_path = os.path.join("database", args.db_id, f"{args.db_id}.sqlite")

    if schema_text is None:
        raise ValueError("Pass either --db-id or --ddl-file.")

    if args.valid_only:
        if db_path is None:
            raise ValueError("--valid-only requires --db-path, or --db-id with database/<db_id>/<db_id>.sqlite present.")
        result = generate_compilable_queries(
            schema_text,
            db_path,
            model,
            tokenizer,
            device,
            num_queries=args.num_queries,
            max_input=args.max_input,
            max_output=args.max_output,
            max_attempts=args.max_attempts,
        )
        queries = result["queries"]
        print(f"\nGenerated {len(queries)} compile-valid queries")
        print(f"Attempt budget: {result['attempt_budget']}")
        if result["invalid_samples"]:
            print(f"Invalid samples kept for debugging: {len(result['invalid_samples'])}")
    else:
        queries = generate_queries(
            schema_text,
            model,
            tokenizer,
            device,
            args.num_queries,
            max_input=args.max_input,
            max_output=args.max_output,
        )
        result = {"queries": queries}
        print(f"\nGenerated {len(queries)} raw queries")

    for q in queries:
        print(f"  -> {q}")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
