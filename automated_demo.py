import os
import re
import sys
import json
import time

def slow_print(text, delay=0.03):
    """Prints text character by character for a nice terminal effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def step_header(step_num, title):
    print("\n" + "="*80)
    print(f"🔹 STEP {step_num}: {title}")
    print("="*80)

# ==============================================================================
# PHASE 1: TEXT TO SCHEMA (Simulating Friends' Model)
# ==============================================================================

def run_phase1_text_to_schema(nl_description):
    """
    Executes the true Phase 1 logic live.
    It calls the LLM (acting as the Phase 1 model) to translate the 
    natural language business description into the standardized JSON schema.
    """
    try:
        from google import genai
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
            
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        You are a Phase 1 text-to-schema model. 
        Given this business requirement: "{nl_description}"
        Generate a relational database schema.
        
        You MUST output ONLY a valid JSON dictionary matching this exact structure:
        {{
            "table_name": {{
                "column_name": {{"TYPE": "VARCHAR(255)", "PRIMARY_KEY": true}}
            }}
        }}
        Do not include markdown backticks or any other text.
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
        
    except Exception as e:
        print(f"\n⚠️ Phase 1 Model Execution Failed: {e}")
        print("Falling back to pre-computed schema for demonstration purposes...")
        # Fallback if Colab doesn't have API keys set up yet
        schema_json = {
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
        time.sleep(2)
        return schema_json

def convert_phase1_to_ddl(tables_dict):
    """Converts Phase 1 JSON output into standard SQL DDL."""
    ddl = ""
    for table_name, columns in tables_dict.items():
        ddl += f"CREATE TABLE {table_name} (\n"
        col_defs = []
        for col_name, col_data in columns.items():
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
    """Normalize standard SQL types to Spider's format for Qwen model consistency."""
    replacements = {
        'VARCHAR': 'text', 'CHAR': 'text', 'STRING': 'text',
        'INTEGER': 'number', 'INT': 'number', 'BIGINT': 'number',
        'FLOAT': 'number', 'DOUBLE': 'number', 'DECIMAL': 'number',
        'BOOLEAN': 'text', 'BOOL': 'text',
        'DATETIME': 'time', 'TIMESTAMP': 'time', 'DATE': 'time'
    }
    for sql_type, spider_type in replacements.items():
        ddl_text = re.sub(rf'\b{sql_type}\b', spider_type, ddl_text, flags=re.IGNORECASE)
    ddl_text = re.sub(r'\b(text|number|time)\s*\(\d+(?:,\d+)?\)', r'\1', ddl_text)
    return ddl_text

# ==============================================================================
# PHASE 2: SCHEMA TO SQL (Qwen Model)
# ==============================================================================

def run_phase2_model(final_ddl):
    """Loads Qwen model and unconditionally generates the queries."""
    from inference_qwen import load_model, generate_queries
    
    # Auto-detect the hyper-tuned model folder
    possible_paths = ["qwen_sql_model_v2/final", "qwen_sql_model/final", "final_qwen", "qwen_sql_model", "qwen_sql_model_v2"]
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
            
    if model_path is None:
        print("\n❌ ERROR: Could not find your fine-tuned model folder!")
        print("Please ensure your trained Qwen weights are saved in 'qwen_sql_model/final' or 'final_qwen'.")
        return []
        
    print(f"Loading your Hyper-Tuned Qwen2.5-Coder Model from '{model_path}'...")
    model, tokenizer, device = load_model(model_path)
    
    print("Model loaded successfully! Synthesizing 5 Autonomous Queries...")
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
# AUTOMATED DEMO
# ==============================================================================

def main():
    print("\n" + "★"*80)
    print("   END-TO-END PIPELINE DEMO: Natural Language → JSON Schema → SQL Queries")
    print("★"*80)
    
    # STEP 1
    step_header(1, "Natural Language Input (Business Requirement)")
    business_requirement = "I need an e-commerce database to track my users, their emails, and the orders they place."
    slow_print(f'Input: "{business_requirement}"')
    
    # STEP 2
    step_header(2, "Phase 1 Execution: Text-to-Schema Model")
    slow_print("Running friends' Phase 1 Model...")
    phase1_json = run_phase1_text_to_schema(business_requirement)
    print("\n[Phase 1 Output] JSON Schema Generated:")
    print(json.dumps(phase1_json, indent=2))
    
    # STEP 3
    step_header(3, "Bridge: Converting JSON to Normalized DDL")
    slow_print("Transforming JSON dictionary to structured SQL DDL...")
    raw_ddl = convert_phase1_to_ddl(phase1_json)
    final_ddl = normalize_ddl_types(raw_ddl)
    print("\n[Bridge Output] Normalized DDL Ready for Phase 2:")
    print(final_ddl)
    
    # STEP 4
    step_header(4, "Phase 2 Execution: Schema-to-SQL (Qwen Model)")
    queries = run_phase2_model(final_ddl)
    
    # STEP 5
    step_header(5, "Final Output: Autonomous Query Synthesis")
    if queries:
        for i, q in enumerate(queries, 1):
            slow_print(f" {i}. {q}", delay=0.01)
    print("="*80)
    print("✅ Pipeline Execution Complete.")

if __name__ == "__main__":
    main()
