import argparse   # For parsing CLI flags like --model, --db-id, --num-queries
import json        # For reading tables.json schema definitions and writing output results
import os          # For path checks (adapter_config.json, database files)
import re          # For regex-based SQL post-processing and validation
import sqlite3     # For actually executing generated SQL to verify it compiles
import yaml        # For reading the optional config.yaml settings file
import torch       # For tensor creation and disabling gradient computation during inference


def load_config(config_path="config.yaml"):
    # Only attempt to open the file if it actually exists on disk
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)  # Parse YAML into a Python dictionary
    return {}  # Return empty dict if no config file — callers use .get() safely


def load_model(model_path):
    # Import transformers here (lazy import keeps the module usable even without GPU)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        from peft import AutoPeftModelForCausalLM  # PEFT provides LoRA model loading
        has_peft = True
    except ImportError:
        has_peft = False  # If PEFT isn't installed, fall back to a standard model load

    # Load the tokenizer that converts text strings ↔ integer token IDs
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # Qwen lacks a dedicated pad token; reusing EOS prevents crashes in generation
        tokenizer.pad_token = tokenizer.eos_token

    # If an adapter_config.json file exists, this is a LoRA checkpoint — load accordingly
    if os.path.exists(os.path.join(model_path, "adapter_config.json")) and has_peft:
        # AutoPeftModelForCausalLM automatically merges frozen base model + LoRA adapter
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Half-precision to fit in GPU VRAM
            device_map="auto",          # Automatically distribute across available devices
            trust_remote_code=True,     # Required for Qwen's custom attention code
        )
    else:
        # Fallback: load as a standard (non-LoRA) causal language model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    # Switch to eval mode — disables dropout layers so output is deterministic
    model.eval()
    return model, tokenizer, model.device  # Return model, tokenizer, and the device it's on


def encode_prompt_keeping_separator(prompt_body, tokenizer, device, max_input):
    # The separator string that signals to the model: "start generating SQL here"
    separator     = "\n-- SQL QUERY --\n"
    # Tokenize the separator alone (without BOS) to find how many tokens it occupies
    separator_ids = tokenizer(separator, add_special_tokens=False)["input_ids"]
    # Reserve separator space from the total budget so the separator is never truncated
    body_budget   = max_input - len(separator_ids)
    if body_budget <= 0:
        # This config is too restrictive — raise immediately rather than silently break
        raise ValueError("max_input is too small to fit the SQL separator.")
    # Tokenize the schema text and TRUNCATE it if it exceeds the remaining budget
    body_ids = tokenizer(
        prompt_body,
        add_special_tokens=False,  # No BOS here — we'll build the full sequence manually
        truncation=True,           # Cut schema at body_budget tokens if needed
        max_length=body_budget,
    )["input_ids"]
    # Concatenate schema tokens + separator tokens into one flat list, then wrap in a batch tensor
    input_ids      = torch.tensor([body_ids + separator_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)  # Attend to every token — no padding here
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def clean_generated_sql(text):
    # Start with raw decoded model output
    sql = text.strip()
    # Strip any opening markdown code fences the model may have hallucinated (```sql or ```)
    sql = re.sub(r"^```(?:sql)?", "", sql, flags=re.IGNORECASE).strip()
    # Strip any closing markdown code fences
    sql = re.sub(r"```$", "", sql).strip()
    # Cut off anything after a SQL comment, a new "Generate..." instruction, or a CREATE TABLE
    # (the model sometimes starts generating the NEXT prompt when given too many tokens)
    sql = re.split(r"\n\s*(?:--|#|Generate a SQL query|CREATE TABLE)\b", sql, maxsplit=1)[0].strip()
    if ";" in sql:
        # If a semi-colon exists, take only the FIRST complete statement and re-add the semi-colon
        sql = sql.split(";", 1)[0].strip() + ";"
    else:
        # If no semi-colon, take only the first line (the model stopped mid-statement)
        sql = sql.split("\n")[0].strip()
    # Fix a common generation artifact where table aliases are split: "T 1" → "T1"
    return re.sub(r'\bT\s+(\d+)\b', r'T\1', sql)


def sql_compile_error(sql, db_path):
    # Reject completely empty strings before even trying to connect to SQLite
    if not sql:
        return "empty query"
    # Only SELECT and WITH (CTE) queries are valid for our schema exploration task
    if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        return "not a SELECT/WITH query"

    conn = None
    try:
        # Connect to the specific Spider .sqlite file for this schema
        conn = sqlite3.connect(db_path)
        # EXPLAIN QUERY PLAN parses and validates the SQL without actually running it
        # This catches syntax errors and references to non-existent tables/columns
        conn.execute("EXPLAIN QUERY PLAN " + sql)
        return None  # No error — the query is syntactically and structurally valid
    except sqlite3.Error as exc:
        return str(exc)  # Return the error message string for debugging
    finally:
        if conn:
            conn.close()  # Always release the DB connection even if an exception occurred


def generate_queries(schema_text, model, tokenizer, device, num_queries=10, max_input=768, max_output=256, temperature=0.7):
    # Build the instruction prompt that tells the model what task to perform
    prompt = (
        "Generate one SQLite SELECT query for this database.\n"
        "Use only tables and columns present in the schema.\n"
        "Return SQL only, with no explanation or markdown.\n"
        f"{schema_text}"  # Append the actual DDL schema text to the instruction
    )
    # Tokenize the prompt with length-aware truncation, keeping the separator intact
    inputs       = encode_prompt_keeping_separator(prompt, tokenizer, device, max_input)
    # Record the prompt length so we can slice it off the output later
    input_length = inputs["input_ids"].shape[1]

    # ==============================================================================
    # VIVA POINT (DIVERSE SAMPLING & TEMPERATURE):
    # If do_sample=False (Greedy Search), the AI picks the #1 math probability token.
    # By setting do_sample=True & temperature=0.7, we flatten the probability curve.
    # This forces the AI to explore different tokens, generating structurally DIFFERENT queries!
    # ==============================================================================
    with torch.no_grad():  # Disable gradient tracking — saves VRAM and speeds up inference
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_output,        # Maximum SQL tokens to generate per sequence
            num_return_sequences=num_queries,  # Generate N independent sequences in one call
            do_sample=True,                    # Probabilistic (non-greedy) token selection
            temperature=temperature,           # Higher temp → flatter distribution → more variety
            top_p=0.9,                         # Nucleus sampling: ignore tokens below cumulative p=0.9
            pad_token_id=tokenizer.eos_token_id  # Use EOS as pad to avoid generation errors
        )

    seen    = set()    # Track normalized SQL strings to de-duplicate identical queries
    queries = []       # Accumulate unique, cleaned SQL strings
    for output in outputs:
        # Slice off the prompt token IDs — we only want the newly generated SQL tokens
        generated_tokens = output[input_length:]
        # Decode token IDs back to a text string, then clean/normalize it
        sql      = clean_generated_sql(tokenizer.decode(generated_tokens, skip_special_tokens=True))
        # Reject hallucinated cross-join patterns like "ON table1.col = table2.*"
        if re.search(r'ON\s+\S+\s*=\s*\w+\.\*', sql, re.IGNORECASE):
            continue
        sql_norm = sql.lower()  # Normalize to lowercase for duplicate detection
        if sql and sql_norm not in seen:
            queries.append(sql)    # Keep this unique query
            seen.add(sql_norm)     # Mark as seen to skip future duplicates
    return queries  # Return list of unique, cleaned SQL strings


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
    seen            = set()   # Global set of normalized SQL strings seen across all batches
    valid           = []      # Accumulates SQL queries that passed SQLite compilation check
    invalid_samples = []      # Stores up to 20 failed queries + their errors for debugging
    attempts        = 0       # Total number of candidate queries generated so far

    # Keep generating batches until we have enough valid queries or exhaust the attempt budget
    while len(valid) < num_queries and attempts < max_attempts:
        remaining  = num_queries - len(valid)            # How many more valid queries we need
        # Generate 3× as many as needed (to account for invalid ones), capped at budget
        batch_size = min(max(remaining * 3, 6), max_attempts - attempts)
        # Use lower temperature on retry passes — reduces wild hallucinations
        temperature = 0.45 if attempts > 0 else 0.7
        # Generate a batch of candidate SQL queries
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
        attempts += batch_size  # Count all attempts, including invalid ones

        for sql in candidates:
            norm = sql.strip().lower()  # Normalize for duplicate detection
            if not norm or norm in seen:
                continue    # Skip empty or already-seen queries
            seen.add(norm)

            # Run the SQLite compile check against the actual database file
            error = sql_compile_error(sql, db_path)
            if error is None:
                valid.append(sql)          # Query is valid — add to our kept set
                if len(valid) == num_queries:
                    break                  # We have enough valid queries — stop early
            elif len(invalid_samples) < 20:
                # Store up to 20 invalid samples for debugging/inspection
                invalid_samples.append({"sql": sql, "error": error})

    return {
        "queries":        valid,            # Final list of compile-valid SQL queries
        "num_valid":      len(valid),       # How many valid queries we found
        "attempt_budget": max_attempts,     # What our total attempt ceiling was
        "invalid_samples": invalid_samples, # Debugging info: failed queries and their errors
    }


def serialize_schema_from_tables_json(db_schema):
    # Extract the raw arrays from the Spider tables.json schema dictionary
    table_names  = db_schema["table_names_original"]   # List of table name strings
    column_names = db_schema["column_names_original"]  # List of (table_idx, col_name) tuples
    column_types = db_schema["column_types"]            # List of SQL type strings ("number", "text")
    primary_keys = set(db_schema["primary_keys"])       # Set of column indices that are PKs
    fk_map       = {}
    # Build the foreign key mapping: source col index → (target table name, target col name)
    for fk_from, fk_to in db_schema["foreign_keys"]:
        tbl_idx        = column_names[fk_to][0]    # The table index of the FK target column
        fk_map[fk_from] = (table_names[tbl_idx], column_names[fk_to][1])

    parts = []  # Accumulate one CREATE TABLE string per table
    for table_idx, table_name in enumerate(table_names):
        cols = []
        for col_idx, (tbl_idx, col_name) in enumerate(column_names):
            # Skip columns from other tables and the special wildcard "*" sentinel
            if tbl_idx != table_idx or col_name == "*": continue
            # Start with "  column_name TYPE"
            cdef = f"  {col_name} {column_types[col_idx]}"
            mods = []
            if col_idx in primary_keys: mods.append("PRIMARY KEY")          # Mark PK
            if col_idx in fk_map:       mods.append(f"REFERENCES {fk_map[col_idx][0]}({fk_map[col_idx][1]})")  # Mark FK
            if mods: cdef += " " + " ".join(mods)  # Append constraints after type
            cols.append(cdef)
        if cols: parts.append(f"CREATE TABLE {table_name} (\n" + ",\n".join(cols) + "\n);")
    # Join all tables' DDL with blank lines between them
    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        required=True)               # Path to trained LoRA adapter or merged model
    parser.add_argument("--db-id",        default=None)                # Spider database ID (e.g. "flight_1")
    parser.add_argument("--db-path",      default=None)                # Direct path to .sqlite file
    parser.add_argument("--ddl-file",     default=None)                # Path to a raw DDL text file to use as schema
    parser.add_argument("--num-queries",  type=int, default=10)        # How many SQL queries to generate
    parser.add_argument("--max-input",    type=int, default=768)       # Max token budget for prompt
    parser.add_argument("--max-output",   type=int, default=256)       # Max tokens to generate per query
    parser.add_argument("--valid-only",   action="store_true")         # If set, only output SQLite-valid queries
    parser.add_argument("--max-attempts", type=int, default=80)        # Total generation attempts for valid-only mode
    parser.add_argument("--save",         default=None)                # Optional path to save results as JSON
    args = parser.parse_args()

    # Load the model, tokenizer, and determine the device (CPU/GPU/MPS)
    model, tokenizer, device = load_model(args.model)

    schema_text = None          # Will hold the final DDL text string
    db_path     = args.db_path  # Will hold the path to the .sqlite file (for validation)

    if args.ddl_file:
        # Load schema directly from a user-provided DDL text file
        with open(args.ddl_file) as f:
            schema_text = f.read()
    elif args.db_id:
        # Look up the schema from Spider's tables.json by database ID
        with open("tables.json") as f:
            all_schemas = json.load(f)
        schema = next((s for s in all_schemas if s["db_id"] == args.db_id), None)
        if schema is None:
            raise ValueError(f"Unknown db_id: {args.db_id}")  # Fail clearly if ID not found
        # Convert the JSON schema dict to a DDL string
        schema_text = serialize_schema_from_tables_json(schema)
        if db_path is None:
            # Auto-infer the .sqlite file path from the standard Spider directory structure
            db_path = os.path.join("database", args.db_id, f"{args.db_id}.sqlite")

    if schema_text is None:
        # Neither --ddl-file nor --db-id was provided — can't proceed without a schema
        raise ValueError("Pass either --db-id or --ddl-file.")

    if args.valid_only:
        if db_path is None:
            raise ValueError("--valid-only requires --db-path, or --db-id with database/<db_id>/<db_id>.sqlite present.")
        # Generate queries and keep only the ones that pass SQLite's EXPLAIN QUERY PLAN
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
        queries = result["queries"]  # Only the compile-valid subset
        print(f"\nGenerated {len(queries)} compile-valid queries")
        print(f"Attempt budget: {result['attempt_budget']}")
        if result["invalid_samples"]:
            print(f"Invalid samples kept for debugging: {len(result['invalid_samples'])}")
    else:
        # Generate raw queries without any validation filtering
        queries = generate_queries(
            schema_text,
            model,
            tokenizer,
            device,
            args.num_queries,
            max_input=args.max_input,
            max_output=args.max_output,
        )
        result = {"queries": queries}  # Wrap in dict for consistent save format
        print(f"\nGenerated {len(queries)} raw queries")

    # Print each generated query to stdout
    for q in queries:
        print(f"  -> {q}")

    # Optionally persist the full result dict (including invalid samples) to a JSON file
    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()