import sys
import json
import os
import re

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

# Import Phase 2 logic
from inference_qwen import load_model, generate_queries

def normalize_ddl_types(ddl_text):
    """Normalize standard SQL types to Spider's format for model consistency."""
    replacements = {
        'VARCHAR': 'text', 'CHAR': 'text', 'STRING': 'text',
        'INTEGER': 'number', 'INT': 'number', 'BIGINT': 'number',
        'FLOAT': 'number', 'DOUBLE': 'number', 'DECIMAL': 'number',
        'BOOLEAN': 'text', 'BOOL': 'text',
        'DATETIME': 'time', 'TIMESTAMP': 'time', 'DATE': 'time'
    }
    for sql_type, spider_type in replacements.items():
        ddl_text = re.sub(rf'\b{sql_type}\b', spider_type, ddl_text, flags=re.IGNORECASE)
    # Strip size annotations like VARCHAR(255) -> text
    ddl_text = re.sub(r'\b(text|number|time)\s*\(\d+(?:,\d+)?\)', r'\1', ddl_text)
    return ddl_text

def main():
    print("="*70)
    print("🚀 END-TO-END DEMO: PHASE 1 (Schemapile) ➔ PHASE 2 (Qwen Model)")
    print("="*70)
    
    # 1. Simulate Phase 1 Input
    sample_schemapile_json = {
        "TABLES": {
            "ecommerce_users": {
                "COLUMNS": {
                    "user_id": {"TYPE": "INTEGER", "PRIMARY_KEY": True},
                    "email": {"TYPE": "VARCHAR(255)", "UNIQUE": True},
                    "signup_date": {"TYPE": "DATETIME"}
                }
            },
            "ecommerce_orders": {
                "COLUMNS": {
                    "order_id": {"TYPE": "INTEGER", "PRIMARY_KEY": True},
                    "user_id": {"TYPE": "INTEGER", "NULLABLE": False},
                    "total_amount": {"TYPE": "DECIMAL(10,2)"},
                    "status": {"TYPE": "VARCHAR(50)"}
                },
                "FOREIGN_KEYS": {"user_id": "ecommerce_users.user_id"}
            }
        }
    }
    
    print("\n[1/4] PHASE 1: Raw JSON Dictionary from Data Engineering Team")
    print(json.dumps(sample_schemapile_json, indent=2))
    
    print("\n[2/4] PHASE 1 ➔ 2 TRANSITION: Executing `convert_to_ddl`...")
    raw_ddl = convert_to_ddl("ecommerce_db", sample_schemapile_json)
    
    # Apply type normalization to bridge Phase 1 to Phase 2 safely
    final_ddl = normalize_ddl_types(raw_ddl)
    
    print("\nGenerated DDL (Normalized to Spider Vocabulary):")
    print(final_ddl)
    
    # Phase 2
    model_path = "final_qwen"
    print(f"\n[3/4] PHASE 2: Loading Qwen2.5-Coder Model from '{model_path}'...")
    
    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: Model path '{model_path}' not found!")
        print("Please ensure your fine-tuned weights are in 'final_qwen' or update the path in this script.")
        return
        
    model, tokenizer, device = load_model(model_path)
    
    print("\n[4/4] PHASE 2: Generating 5 Autonomous Queries using Diverse Sampling...")
    queries = generate_queries(
        schema_text=final_ddl,
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_queries=5,
        max_input=768,
        max_output=256,
        temperature=0.8
    )
    
    print("\n" + "="*70)
    print("🎯 FINAL OUTPUT: Synthesized SQL Queries")
    print("="*70)
    for i, q in enumerate(queries, 1):
        print(f" {i}. {q}")
    print("="*70)

if __name__ == "__main__":
    main()
