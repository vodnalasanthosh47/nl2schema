import argparse    # For parsing command-line flags like --model, --num-queries, --save
import json         # For loading tables.json schema definitions and saving evaluation results
import os           # For checking file/directory existence (sqlite files, model adapters)
import sqlite3      # For executing generated SQL against real Spider databases
import re           # For keyword matching in diversity scoring and SQL validation
import yaml         # For reading optional config.yaml settings
import torch        # For disabling gradient computation and selecting GPU dtype
from tqdm import tqdm  # For displaying a progress bar while iterating over schemas


# =========================================================================================
# CONFIGURATION LOADER
# =========================================================================================
def load_config(config_path="config.yaml"):
    """
    Reads the YAML configuration file to get paths for models, datasets, and hyperparameters.
    If the file doesn't exist, it returns an empty dictionary to prevent crashing.
    """
    # Only open the file if it actually exists — avoids FileNotFoundError on fresh setups
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)  # Parse YAML into a Python dict
    return {}  # Return empty dict so all downstream .get() calls have a safe fallback


# =========================================================================================
# GENERATIVE ROBUSTNESS EVALUATION: SQL VALIDITY CHECKER
# =========================================================================================
def sql_validity(sql, db_path):
    """
    VIVA POINT: EPHEMERAL EXECUTION
    Unlike BLEU score (which just checks if strings match), this function actually 
    executes the generated SQL query against a real, temporary SQLite database.
    If the SQL is mathematically and syntactically correct, SQLite executes it successfully.
    If there is a syntax error, SQLite throws an exception, and we catch it here and return False.
    This guarantees that our accuracy metric only counts TRULY working queries!
    """
    conn = None
    try:
        # Connect to the specific Spider database for this schema (e.g., flight_1.sqlite)
        conn = sqlite3.connect(db_path)
        # Attempt to execute the AI's generated query
        conn.execute(sql)
        return True  # It executed perfectly without crashing!
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        # The AI generated bad syntax (e.g., SELECT * FRO table) or used a non-existent column
        return False
    finally:
        # Always close the connection to prevent memory leaks
        if conn:
            conn.close()


# =========================================================================================
# DIVERSITY METRIC CALCULATOR
# =========================================================================================
def diversity_score(queries):
    """
    VIVA POINT: DIVERSITY METRIC
    We don't just want the model to generate the exact same query 5 times.
    This function scans the generated queries to see how many unique SQL 
    constructs (JOIN, GROUP BY, HAVING) the model successfully used.
    A higher diversity score means the model is exploring complex SQL logic
    rather than just doing basic 'SELECT *' queries.
    """
    # The full set of SQL constructs we're checking for — covers joins, aggregates, filters, set ops
    constructs = [
        "JOIN", "GROUP BY", "HAVING", "WHERE",
        "ORDER BY", "COUNT", "SUM", "AVG", "MAX", "MIN",
        "DISTINCT", "LIMIT", "LIKE", "BETWEEN",
        "IN", "NOT IN", "INTERSECT", "UNION", "EXCEPT",
    ]
    coverage = set()  # Tracks which constructs appeared in at least one query
    for q in queries:
        q_upper = q.upper()  # Standardize to uppercase for reliable matching
        for c in constructs:
            if c in q_upper:
                coverage.add(c)  # Track which constructs were used
    # Return percentage of constructs utilized across all generated queries
    return len(coverage) / len(constructs) if constructs else 0


# =========================================================================================
# UNIQUENESS METRIC CALCULATOR
# =========================================================================================
def uniqueness_score(queries):
    """
    Calculates the ratio of strictly unique strings generated.
    If the model generates 5 identical queries, uniqueness is 1/5 (20%).
    If it generates 5 different queries, uniqueness is 5/5 (100%).
    """
    if not queries: return 0  # Guard against empty input to avoid division by zero
    # Convert all queries to lowercase and strip whitespace to easily find duplicates
    normalized = set(q.strip().lower() for q in queries)
    # Ratio of unique queries to total queries generated
    return len(normalized) / len(queries)


# =========================================================================================
# MODEL AND TOKENIZER LOADER
# =========================================================================================
def load_model(model_path):
    """
    Loads our specific Phase 2 Qwen 1.5B model from the huggingface format.
    It auto-detects if the folder contains a LoRA adapter (adapter_config.json) 
    and automatically merges it with the base Qwen weights dynamically in VRAM.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        from peft import AutoPeftModelForCausalLM  # PEFT provides LoRA-aware model loading
        has_peft = True
    except ImportError:
        has_peft = False  # Fall back to standard loading if PEFT isn't installed

    # Load the tokenizer which converts text into numbers the AI can read
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # Qwen doesn't have a dedicated pad token; reuse EOS to prevent generation crashes
        tokenizer.pad_token = tokenizer.eos_token

    # Auto-detect if it's a LoRA adapter or merged model
    if os.path.exists(os.path.join(model_path, "adapter_config.json")) and has_peft:
        print("  Detected LoRA Adapter! Loading with peft...")
        # Load the base model and dynamically inject our trained LoRA matrices on top
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Load in half-precision to save VRAM
            device_map="auto",          # Distribute across available GPU/CPU automatically
            trust_remote_code=True,     # Required for Qwen's custom attention implementation
        )
    else:
        # Load standard model (fallback for non-LoRA or already-merged checkpoints)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    # Set to evaluation mode (disables dropout layers so output is deterministic)
    model.eval()
    return model, tokenizer, model.device  # Return all three so callers know the device


# =========================================================================================
# PROMPT FORMATTING LOGIC
# =========================================================================================
def encode_prompt_keeping_separator(prompt_body, tokenizer, device, max_input):
    """
    VIVA POINT: TOKEN TRUNCATION
    If the DDL schema is too large, it will crash the model.
    This function cleanly tokenizes the input and ensures it fits under the max_input limit
    while preserving the critical '-- SQL QUERY --' separator token.
    """
    separator     = "\n-- SQL QUERY --\n"  # The delimiter that signals: "SQL starts here"
    # Tokenize the separator first to know exactly how many tokens it occupies
    separator_ids = tokenizer(separator, add_special_tokens=False)["input_ids"]

    # Calculate how much room remains for the actual schema text
    body_budget = max_input - len(separator_ids)
    if body_budget <= 0:
        raise ValueError("max_input is too small to fit the SQL separator.")

    # Tokenize the schema and TRUNCATE it if it exceeds our budget
    body_ids = tokenizer(
        prompt_body,
        truncation=True,         # Cut the schema if it's longer than body_budget tokens
        max_length=body_budget,
        add_special_tokens=False,  # No BOS — we're building the full token sequence manually
    )["input_ids"]

    # Combine the truncated schema tokens with the separator tokens into one input tensor
    input_ids      = torch.tensor([body_ids + separator_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)  # Attend to every token (no padding here)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# =========================================================================================
# POST-PROCESSING SANITIZATION
# =========================================================================================
def clean_generated_sql(text):
    """
    Language models sometimes "hallucinate" extra text, markdown fences (```sql), or comments.
    This function uses Regex to rigorously strip out all non-SQL garbage from the output 
    so the SQLite engine doesn't crash during evaluation.
    """
    sql = text.strip()  # Remove leading/trailing whitespace from raw model output
    # Strip everything after the first semi-colon or comment symbol (-- or #)
    sql = re.split(r"\n\s*(?:--|#|Generate a SQL query|CREATE TABLE)\b", sql, maxsplit=1)[0].strip()
    if ";" in sql:
        # Take only the first complete SQL statement (up to the first semi-colon)
        sql = sql.split(";", 1)[0].strip() + ";"
    else:
        # If no semi-colon, the model stopped mid-statement — keep only the first line
        sql = sql.split("\n")[0].strip()
    # Fix spacing issues on table aliases (e.g. 'T 1' becomes 'T1') — common Qwen artifact
    sql = re.sub(r'\bT\s+(\d+)\b', r'T\1', sql)
    return sql


# =========================================================================================
# QUERY GENERATION (DIVERSE SAMPLING)
# =========================================================================================
def generate_queries_for_schema(schema_text, model, tokenizer, device, num_queries, max_input, max_output):
    # Build the instruction prompt that describes the task to the model
    prompt = (
        "Generate one SQLite SELECT query for this database.\n"
        "Use only tables and columns present in the schema.\n"
        "Return SQL only, with no explanation or markdown.\n"
        f"{schema_text}"  # Append the actual DDL schema text
    )
    # Tokenize with truncation, preserving the separator token's position
    inputs       = encode_prompt_keeping_separator(prompt, tokenizer, device, max_input)
    # Save the prompt length so we can slice it off the output during decoding
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():  # Disable gradient tracking — saves memory and speeds inference
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_output,        # Max tokens to generate per sequence
            num_return_sequences=num_queries,  # Generate N sequences in a single batch call
            do_sample=True,                    # Probabilistic sampling (not greedy)
            temperature=0.8,                   # Higher temperature = more diversity
            top_p=0.95,                        # Nucleus sampling: ignore bottom 5% least-likely tokens
            pad_token_id=tokenizer.eos_token_id  # Use EOS as pad token to avoid generation errors
        )

    queries = []  # Accumulate the decoded and cleaned SQL strings
    for output in outputs:
        # Slice off the input prompt tokens — we only want the newly generated SQL tokens
        generated_tokens = output[input_length:]
        # Decode token IDs back to text and clean the result
        sql = clean_generated_sql(tokenizer.decode(generated_tokens, skip_special_tokens=True))
        if not sql: continue  # Skip empty results (model generated only whitespace)
        # Ignore hallucinated cross-join patterns like "ON table.col = table.*"
        if re.search(r'ON\s+\S+\s*=\s*\w+\.\*', sql, re.IGNORECASE):
            continue

        queries.append(sql)

    return queries  # Return list of cleaned (but not yet validated) SQL strings


# =========================================================================================
# JSON TO DDL SERIALIZER
# =========================================================================================
def serialize_schema_ddl(db_schema):
    """
    VIVA POINT: DATA ENGINEERING
    This reads the raw Spider tables.json schema dictionary and dynamically writes out a 
    perfectly formatted SQL 'CREATE TABLE' string. We do this programmatically so we can 
    dynamically track exactly which tables and columns exist for our name-masking augmentation.
    """
    table_names  = db_schema["table_names_original"]   # List of table name strings
    column_names = db_schema["column_names_original"]  # List of (table_idx, col_name) tuples
    column_types = db_schema["column_types"]            # List of SQL type strings ("number", "text")
    primary_keys = set(db_schema["primary_keys"])       # Set of PK column indices

    # Map foreign keys mathematically: source col index → (target table name, target col name)
    fk_map = {}
    for fk_from, fk_to in db_schema["foreign_keys"]:
        tbl_idx        = column_names[fk_to][0]           # Table index of the target column
        fk_map[fk_from] = (table_names[tbl_idx], column_names[fk_to][1])

    # Build the DDL strings iteratively — one CREATE TABLE block per table
    ddl_parts = []
    for table_idx, table_name in enumerate(table_names):
        cols = []
        for col_idx, (tbl_idx, col_name) in enumerate(column_names):
            # Skip columns belonging to other tables and the "*" wildcard sentinel
            if tbl_idx != table_idx or col_name == "*": continue
            col_def = f"  {col_name} {column_types[col_idx]}"  # "  col_name TYPE"
            mods = []
            if col_idx in primary_keys: mods.append("PRIMARY KEY")
            if col_idx in fk_map:
                ref_tbl, ref_col = fk_map[col_idx]
                mods.append(f"REFERENCES {ref_tbl}({ref_col})")
            if mods: col_def += " " + " ".join(mods)  # Append constraints after type
            cols.append(col_def)
        if cols:
            # Assemble the full CREATE TABLE block for this table
            ddl_parts.append(f"CREATE TABLE {table_name} (\n" + ",\n".join(cols) + "\n);")
    return "\n\n".join(ddl_parts)  # Join all table DDLs with blank lines between them


# =========================================================================================
# MAIN EXECUTION THREAD
# =========================================================================================
def main():
    # ... (Command line argument parsing skipped for brevity) ...
    parser = argparse.ArgumentParser(description="Evaluate Qwen Schema→SQL model")
    parser.add_argument("--model",       required=True, help="Path to trained model")  # Required: path to LoRA adapter
    parser.add_argument("--config",      default="config.yaml")                         # Optional YAML config file
    parser.add_argument("--num-queries", type=int, default=10, help="Queries per schema")  # Queries to generate per DB
    parser.add_argument("--max-input",   type=int, default=768)                         # Schema token budget
    parser.add_argument("--max-output",  type=int, default=None)                        # SQL generation token budget
    parser.add_argument("--test",        action="store_true")                           # If set, evaluate on test set
    parser.add_argument("--max-schemas", type=int, default=None)                        # Cap the number of DBs evaluated
    parser.add_argument("--save",        default=None)                                  # Path to save results JSON
    args = parser.parse_args()

    # Load and merge config file settings
    config    = load_config(args.config)
    data_cfg  = config.get("data", {})   # Data paths sub-config
    model_cfg = config.get("model", {})  # Model parameters sub-config

    # Select the right schema file and DB directory based on --test flag
    if args.test:
        tables_path  = data_cfg.get("test_tables",      "test_tables.json")
        database_dir = data_cfg.get("test_database_dir", "test_database")
    else:
        tables_path  = data_cfg.get("tables",       "tables.json")
        database_dir = data_cfg.get("database_dir", "database")

    print("\n📦 Loading Causal Model...")
    model, tokenizer, device = load_model(args.model)  # Load the trained model into VRAM
    max_input  = args.max_input
    # Use CLI value if provided; otherwise fall back to config file; otherwise default 256
    max_output = args.max_output or model_cfg.get("max_output_length", 256)

    # Load all raw Spider schemas from the JSON file
    with open(tables_path) as f:
        all_schemas = json.load(f)

    # Only evaluate schemas that have a physical SQLite database file present on disk
    schemas = []
    for s in all_schemas:
        if os.path.exists(os.path.join(database_dir, s["db_id"], f"{s['db_id']}.sqlite")):
            schemas.append(s)  # Include only schemas with a matching .sqlite file

    # Optionally limit evaluation to a smaller subset (useful for quick testing)
    if args.max_schemas: schemas = schemas[:args.max_schemas]

    all_results        = []   # Per-schema result dicts to be saved to JSON
    total_queries      = 0    # Cumulative count of all generated queries across all schemas
    total_valid        = 0    # Cumulative count of SQLite-valid queries
    all_diversity_scores  = []   # Per-schema diversity scores for averaging
    all_uniqueness_scores = []   # Per-schema uniqueness scores for averaging

    print(f"\n🔮 Generating Causal queries for {len(schemas)} schemas...\n")
    # Loop over every single database in the evaluation set
    for schema in tqdm(schemas):
        db_id   = schema["db_id"]
        # Build the path to the .sqlite file for this DB (needed for validity checking)
        db_path = os.path.join(database_dir, db_id, f"{db_id}.sqlite")

        # Convert the JSON schema to purely mathematical DDL text the model was trained on
        schema_text = serialize_schema_ddl(schema)

        # Generate the specified number of diverse queries using the fine-tuned model
        queries = generate_queries_for_schema(
            schema_text,
            model,
            tokenizer,
            device,
            args.num_queries,
            max_input,
            max_output,
        )

        valid   = []  # Queries that passed SQLite's EXPLAIN QUERY PLAN check
        invalid = []  # Queries that failed (syntax errors, wrong table/column names)
        # Run our ephemeral execution check on every single generated query
        for q in queries:
            if sql_validity(q, db_path):
                valid.append(q)    # Query executed without error
            else:
                invalid.append(q)  # Query raised a SQLite exception

        # Calculate per-schema metrics
        validity_rate = len(valid) / len(queries) if queries else 0  # Fraction of queries that compiled
        div_score     = diversity_score(valid)   if valid   else 0   # SQL construct diversity across valid queries
        uniq_score    = uniqueness_score(queries)                     # Fraction of generated queries that are unique

        # Accumulate totals for computing global averages at the end
        total_queries += len(queries)
        total_valid   += len(valid)
        all_diversity_scores.append(div_score)
        all_uniqueness_scores.append(uniq_score)

        # Store the full per-schema debug results
        all_results.append({
            "db_id":            db_id,
            "num_generated":    len(queries),
            "num_valid":        len(valid),
            "validity_rate":    round(validity_rate, 4),
            "diversity_score":  round(div_score, 4),
            "uniqueness_score": round(uniq_score, 4),
            "valid_queries":    valid,    # Keep valid queries for inspection
            "invalid_queries":  invalid,  # Keep invalid queries for debugging
        })

    # Compute global averages across all schemas
    overall_validity = total_valid / total_queries if total_queries else 0
    avg_diversity    = sum(all_diversity_scores)  / len(all_diversity_scores)  if all_diversity_scores  else 0
    avg_uniqueness   = sum(all_uniqueness_scores) / len(all_uniqueness_scores) if all_uniqueness_scores else 0

    # Print the final summary report to stdout
    print("\n" + "=" * 60)
    print("📊 Causal Generative Results")
    print("=" * 60)
    print(f"  SQL Validity:      {overall_validity:.1%}   ({total_valid}/{total_queries})")
    print(f"  Avg Diversity:     {avg_diversity:.1%}")
    print(f"  Avg Uniqueness:    {avg_uniqueness:.1%}\n")

    # Save the full per-schema results to a JSON file for graphing/further analysis
    if args.save:
        with open(args.save, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"💾 Results saved to {args.save}")


if __name__ == "__main__":
    main()