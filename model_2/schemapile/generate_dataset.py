import json
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ─── Custom Exception for Daily Limit ─────────────────────────────────────────
class DailyLimitExceeded(Exception):
    pass

# ─── Load SchemaPile ───────────────────────────────────────────────────────────
def load_schemapile(path="../../data/schemapile.json"):
    print("Loading SchemaPile...")
    with open(path) as f:
        return json.load(f)

# ─── Convert SchemaPile JSON → SQL DDL ────────────────────────────────────────
def convert_to_ddl(schema_name, schema_data):
    ddl = ""
    tables = schema_data.get("TABLES", {})
    for table_name, table_data in tables.items():
        ddl += f"CREATE TABLE {table_name} (\n"
        columns = table_data.get("COLUMNS", {})
        col_defs = []
        for col_name, col_data in columns.items():
            col_type = col_data.get("TYPE", "VARCHAR(255)")
            nullable = "" if col_data.get("NULLABLE", True) else " NOT NULL"
            unique = " UNIQUE" if col_data.get("UNIQUE", False) else ""
            pk = " PRIMARY KEY" if col_data.get("PRIMARY_KEY", False) else ""
            col_defs.append(f"  {col_name} {col_type}{pk}{nullable}{unique}")
        ddl += ",\n".join(col_defs)
        ddl += "\n);\n\n"
    return ddl.strip()

# ─── Filter Complex Schemas ────────────────────────────────────────────────────
def is_complex(schema_data):
    tables = schema_data.get("TABLES", {})

    if len(tables) < 3:
        return False

    for table_name, table_data in tables.items():
        columns = table_data.get("COLUMNS", {})
        if len(columns) < 3:
            return False

    fk_count = 0
    for table_name, table_data in tables.items():
        fk_count += len(table_data.get("FOREIGN_KEYS", {}))
    if fk_count < 2:
        return False

    return True

# ─── Generate Queries via Gemini ───────────────────────────────────────────────
def generate_queries(ddl, max_retries=5):
    prompt = f"""
You are an expert SQL query generator.

Given the following SQL schema, generate queries at three complexity levels:
- 3 BASIC queries: simple SELECT with WHERE, single table
- 5 INTERMEDIATE queries: JOINs across 2 tables, GROUP BY, HAVING, aggregations
- 4 ADVANCED queries: multi-table JOINs (3+ tables), subqueries, nested queries

For each query provide:
- intent: natural language description of what the query does
- sql: the SQL query
- level: basic / intermediate / advanced

Schema:
{ddl}

Respond ONLY in this JSON format, no extra text, no markdown backticks:
{{
  "queries": [
    {{
      "intent": "...",
      "sql": "...",
      "level": "basic"
    }}
  ]
}}
"""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=prompt
            )
            text = response.text.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)

        except errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                # Check if daily limit is hit
                if "GenerateRequestsPerDayPerProjectPerModel" in str(e):
                    raise DailyLimitExceeded("Daily limit exceeded. Stopping.")
                # Per minute limit — wait 60s and retry
                print(f"  ⚠ Rate limit hit. Waiting 60s (attempt {attempt+1}/{max_retries})...")
                time.sleep(60)
            else:
                raise e

    raise Exception(f"Failed after {max_retries} retries")

# ─── Main Pipeline ─────────────────────────────────────────────────────────────
def generate_dataset(
    schemapile_path="../../data/schemapile.json",
    output_path="../../data/dataset.json",
    failed_path="failed.json",
    resume_path="resume.json",
    delay=5,
    start_from = 17943
):
    data = load_schemapile(schemapile_path)

    # ── Load existing dataset if resuming ──
    if os.path.exists(output_path):
        with open(output_path) as f:
            dataset = json.load(f)
        print(f"  📂 Resuming — loaded {len(dataset)} existing schemas")
    else:
        dataset = []

    # ── Load resume point if exists ──
    resume_from = None
    if os.path.exists(resume_path):
        with open(resume_path) as f:
            resume_from = json.load(f)["last_processed"]
        print(f"  ⏩ Will skip until: {resume_from}")

    # ── Load existing failed schemas if any ──
    if os.path.exists(failed_path):
        with open(failed_path) as f:
            failed_schemas = json.load(f)
        print(f"  📋 Loaded {len(failed_schemas)} previously failed schemas")
    else:
        failed_schemas = []

    skipped = 0
    processed = 0
    failed = 0
    skipping = resume_from is not None

    for idx, (schema_name, schema_data) in enumerate(data.items()):
        if idx < start_from:
            continue
        
        # ── Skip until resume point ──
        if skipping:
            if schema_name == resume_from:
                skipping = False
                print(f"  ✅ Found resume point, continuing from here...")
            else:
                continue

        print(f"[{idx+1}] Processing: {schema_name}")

        # ── Filter simple schemas ──
        if not is_complex(schema_data):
            skipped += 1
            print(f"  ⤸ Skipped (too simple)")
            continue

        # ── Convert to DDL ──
        ddl = convert_to_ddl(schema_name, schema_data)

        # ── Generate queries ──
        try:
            result = generate_queries(ddl)

            dataset.append({
                "id": str(len(dataset) + 1).zfill(5),
                "schema_name": schema_name,
                "schema": ddl,
                "queries": result["queries"]
            })

            processed += 1
            print(f"  ✓ Generated {len(result['queries'])} queries")

        except DailyLimitExceeded as e:
            print(f"\n🛑 {e}")
            with open(output_path, "w") as f:
                json.dump(dataset, f, indent=2)
            with open(failed_path, "w") as f:
                json.dump(failed_schemas, f, indent=2)
            with open(resume_path, "w") as f:
                json.dump({"last_processed": schema_name}, f, indent=2)
            print(f"  💾 Progress saved to    : {output_path}")
            print(f"  📋 Failed saved to      : {failed_path}")
            print(f"  ⏩ Resume point saved to : {resume_path}")
            print(f"  Processed today         : {processed}")
            print(f"  Resume tomorrow!")
            return

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
            failed_schemas.append({
                "schema_name": schema_name,
                "schema_data": schema_data,
                "error": str(e)
            })

        # ── Proactive delay ──
        print(f"  ⏳ Waiting {delay}s...")
        time.sleep(delay)

        # ── Checkpoint every 30 schemas ──
        if processed % 30 == 0 and processed > 0:
            with open(output_path, "w") as f:
                json.dump(dataset, f, indent=2)
            with open(resume_path, "w") as f:
                json.dump({"last_processed": schema_name}, f, indent=2)
            print(f"  💾 Checkpoint saved at {processed} schemas")

    # ── Final save ──
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    with open(failed_path, "w") as f:
        json.dump(failed_schemas, f, indent=2)

    # ── Delete resume.json since run is complete ──
    if os.path.exists(resume_path):
        os.remove(resume_path)
        print(f"  🗑 resume.json deleted (run complete)")

    print(f"\n✅ Done!")
    print(f"  Processed : {processed}")
    print(f"  Skipped   : {skipped}")
    print(f"  Failed    : {failed}")
    print(f"  Saved to  : {output_path}")

if __name__ == "__main__":
    generate_dataset()