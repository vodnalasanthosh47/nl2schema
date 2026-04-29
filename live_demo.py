import os
import re
import sys
import json

# ==============================================================================
# PHASE 1: YOUR FRIENDS' TEXT-TO-SCHEMA MODEL
# ==============================================================================

def run_phase1_model(nl_description):
    """
    TODO: Plug in your friends' Phase 1 Model inference code here!
    
    This function should take the Natural Language text and return the JSON schema
    dictionary exactly as their model outputs it.
    
    For now, I am simulating their model's output using a mock JSON dictionary.
    """
    print(f"\n[Phase 1 Model] Processing Natural Language Input...")
    
    # ⬇⬇⬇ REPLACE THIS MOCK WITH THEIR ACTUAL INFERENCE CODE ⬇⬇⬇
    simulated_schema_json = {
        "ecommerce_users": {
            "user_id": {"TYPE": "INTEGER", "PRIMARY_KEY": True},
            "email": {"TYPE": "VARCHAR(255)", "UNIQUE": True},
            "signup_date": {"TYPE": "DATETIME"}
        },
        "ecommerce_orders": {
            "order_id": {"TYPE": "INTEGER", "PRIMARY_KEY": True},
            "user_id": {"TYPE": "INTEGER", "NULLABLE": False},
            "total_amount": {"TYPE": "DECIMAL(10,2)"},
            "status": {"TYPE": "VARCHAR(50)"}
        }
    }
    # ⬆⬆⬆ REPLACE THIS MOCK WITH THEIR ACTUAL INFERENCE CODE ⬆⬆⬆
    
    return simulated_schema_json


def convert_phase1_to_ddl(tables_dict):
    """
    Converts Phase 1 JSON output into standard SQL DDL.
    Handles the structure where tables map directly to columns.
    """
    ddl = ""
    for table_name, columns in tables_dict.items():
        ddl += f"CREATE TABLE {table_name} (\n"
        col_defs = []
        for col_name, col_data in columns.items():
            # If col_data is a string (just the type) or a dictionary (with constraints)
            if isinstance(col_data, str):
                col_defs.append(f"  {col_name} {col_data}")
            else:
                col_type = col_data.get("TYPE", "VARCHAR(255)")
                nullable = "" if col_data.get("NULLABLE", True) else " NOT NULL"
                unique = " UNIQUE" if col_data.get("UNIQUE", False) else ""
                pk = " PRIMARY KEY" if col_data.get("PRIMARY_KEY", False) else ""
                col_defs.append(f"  {col_name} {col_type}{pk}{nullable}{unique}")
                
        ddl += ",\n".join(col_defs)
        ddl += "\n);\n\n"
    return ddl.strip()


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

# ==============================================================================
# PHASE 2: YOUR SCHEMA-TO-SQL MODEL (QWEN)
# ==============================================================================

def run_phase2_model(final_ddl):
    """Loads your Qwen model and generates the queries."""
    from inference_qwen import load_model, generate_queries
    
    model_path = "final_qwen" # Ensure this points to your Qwen model!
    print(f"\n[Phase 2 Model] Loading Qwen2.5-Coder Model from '{model_path}'...")
    
    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: Model path '{model_path}' not found!")
        print("Please ensure your fine-tuned weights are present or update the path.")
        return []
        
    model, tokenizer, device = load_model(model_path)
    
    print("\n[Phase 2 Model] Synthesizing 5 Autonomous Queries using Diverse Sampling...")
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
    return queries


# ==============================================================================
# MAIN DEMO PIPELINE
# ==============================================================================

def main():
    print("="*80)
    print("🚀 LIVE E2E DEMO: NL → Phase 1 Model (Schema) → Phase 2 Model (SQL Queries)")
    print("="*80)
    
    # 1. Ask TA for Natural Language Input
    print("\n[Input] Please enter a business requirement description:")
    print("(Example: 'I need an e-commerce database to track users, their emails, and the orders they place.')")
    
    nl_input = input("\nYour Description > ")
    if not nl_input.strip():
        nl_input = "I need an e-commerce database to track users, their emails, and the orders they place."
        print(f"Using default description: {nl_input}")
    
    # 2. Run Phase 1
    phase1_json = run_phase1_model(nl_input)
    print("\n[Phase 1 Output] Generated JSON Schema:")
    print(json.dumps(phase1_json, indent=2))
    
    # 3. Transition (Bridge)
    print("\n[Transition] Converting JSON to DDL and Normalizing Types...")
    raw_ddl = convert_phase1_to_ddl(phase1_json)
    final_ddl = normalize_ddl_types(raw_ddl)
    
    print("\n[Transition Output] Final DDL Passed to Phase 2:")
    print(final_ddl)
    
    # 4. Run Phase 2
    queries = run_phase2_model(final_ddl)
    
    # 5. Output Results
    print("\n" + "="*80)
    print("🎯 FINAL OUTPUT: Synthesized SQL Queries for the Business Domain")
    print("="*80)
    for i, q in enumerate(queries, 1):
        print(f" {i}. {q}")
    print("="*80)

if __name__ == "__main__":
    main()
