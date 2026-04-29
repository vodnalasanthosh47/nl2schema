import os
import sys
import gc
import torch
import time

def slow_print(text, delay=0.03):
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
# PHASE 1: NL -> DDL (Parth's Model)
# ==============================================================================
def run_phase1_nl_to_ddl(nl_description, adapter_dir):
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nl2schema_repo", "schema_model"))
    try:
        from inferenceTuned import load_tokenizer, load_model, generate_sql
    except ImportError:
        print("❌ ERROR: Could not import Parth's inferenceTuned.py. Ensure you have the latest git pull.")
        return None

    MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    
    if not os.path.exists(adapter_dir):
        print(f"❌ ERROR: Phase 1 Adapter not found at '{adapter_dir}'!")
        print("Please upload Parth's adapter to this folder.")
        return None
        
    print(f"Loading Phase 1 Base Model ({MODEL_ID}) & LoRA Adapter ({adapter_dir})...")
    tokenizer = load_tokenizer(MODEL_ID, adapter_dir)
    model = load_model(MODEL_ID, adapter_dir)
    
    print("Generating DDL from Natural Language...")
    ddl_output = generate_sql(nl_description, model, tokenizer)
    
    # 🧹 CRITICAL: Clear VRAM so Phase 2 doesn't crash Google Colab!
    print("Unloading Phase 1 Model to free VRAM for Phase 2...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return ddl_output

# ==============================================================================
# PHASE 2: DDL -> SQL Queries (Your Model)
# ==============================================================================
def run_phase2_ddl_to_queries(ddl_text, model_path):
    from inference_qwen import load_model as load_p2_model, generate_queries
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Phase 2 Model not found at '{model_path}'!")
        return []
        
    print(f"Loading Phase 2 Base Model & LoRA Adapter ({model_path})...")
    model, tokenizer, device = load_p2_model(model_path)
    
    print("Synthesizing 5 Autonomous Queries...")
    queries = generate_queries(
        schema_text=ddl_text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_queries=5,
        max_input=768,
        max_output=256,
        temperature=0.8
    )
    
    # Clean up
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return queries

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    print("\n" + "★"*80)
    print("   TRUE END-TO-END DEMO: NL → Parth's Model (DDL) → Your Model (SQL)")
    print("★"*80)
    
    # Auto-detect Phase 2 model path
    p2_paths = ["qwen_sql_model_v2/final", "qwen_sql_model/final", "final_qwen", "qwen_sql_model", "qwen_sql_model_v2"]
    p2_model_path = next((p for p in p2_paths if os.path.exists(p)), None)
    
    # Auto-detect Phase 1 (Parth's) adapter path
    p1_paths = [
        "nl2schema_repo/schema_model/qlora-nl2sql",
        "schema_model/qlora-nl2sql",
        "nl2schema_repo/schema_model/qlora-nl2sql-adapter2",
        "qlora-nl2sql"
    ]
    p1_adapter_path = next((p for p in p1_paths if os.path.exists(p)), None)
    
    if not p2_model_path:
        print("\n❌ Missing Phase 2 Model.")
        return
        
    if not p1_adapter_path:
        print("\n❌ Missing Phase 1 (Parth's) Adapter.")
        print("Please upload Parth's 'qlora-nl2sql' folder into your Drive!")
        return

    # STEP 1
    step_header(1, "Natural Language Input (Business Requirement)")
    business_requirement = "A blog platform where users write posts. Users have an ID and a unique username. Posts have an ID, a title, and a text body."
    print("Auto-supplying test requirement:")
    slow_print(f'Input: "{business_requirement}"')
    nl_input = business_requirement
        
    # STEP 2
    step_header(2, "Phase 1: Text-to-DDL (Parth's Qwen Model)")
    generated_ddl = run_phase1_nl_to_ddl(nl_input, p1_adapter_path)
    if not generated_ddl:
        return
        
    print("\n[Phase 1 Output] Generated SQL DDL:")
    print(generated_ddl)
    
    # STEP 3
    step_header(3, "Phase 2: DDL-to-SQL (Your Qwen Model)")
    queries = run_phase2_ddl_to_queries(generated_ddl, p2_model_path)
    
    # STEP 4
    step_header(4, "Final Output: Synthesized Queries")
    if queries:
        for i, q in enumerate(queries, 1):
            slow_print(f" {i}. {q}", delay=0.01)
    print("\n✅ Pipeline Execution Complete.")

if __name__ == "__main__":
    main()
