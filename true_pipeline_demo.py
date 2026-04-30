import os     # For path checks (checking if adapter/model folders exist)
import sys    # For sys.path manipulation (adding Phase 1 repo to import path) and stdout writing
import gc     # For explicitly running Python's garbage collector to free VRAM between phases
import torch  # For emptying CUDA's GPU memory cache between Phase 1 and Phase 2
import time   # For the character-by-character typing animation delay


def slow_print(text, delay=0.03):
    # Print each character one at a time to create a "typing" animation effect
    for char in text:
        sys.stdout.write(char)    # Write a single character without a newline
        sys.stdout.flush()        # Force the character to appear on screen immediately (no buffering)
        time.sleep(delay)         # Pause for `delay` seconds between characters
    print()  # Emit a final newline after the full string has been printed


def step_header(step_num, title):
    # Print a visually distinct section header to make pipeline stages easy to follow
    print("\n" + "="*80)                  # Top border line of 80 "=" characters
    print(f"🔹 STEP {step_num}: {title}") # Step number and title with a blue diamond icon
    print("="*80)                          # Bottom border line of 80 "=" characters


# ==============================================================================
# PHASE 1: NL -> DDL (Parth's Model)
# ==============================================================================
def run_phase1_nl_to_ddl(nl_description, adapter_dir):
    # Dynamically add Parth's schema_model sub-directory to Python's import search path
    # This lets us import inferenceTuned.py without installing it as a package
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nl2schema_repo", "schema_model"))
    try:
        # Import the three functions we need from Parth's NL→DDL inference script
        from inferenceTuned import load_tokenizer, load_model, generate_sql
    except ImportError:
        # If the import fails, Parth's repo wasn't cloned or isn't in the expected path
        print("❌ ERROR: Could not import Parth's inferenceTuned.py. Ensure you have the latest git pull.")
        return None  # Return None so the main pipeline can detect the failure and abort

    # The Hugging Face model ID for the base model Phase 1 was fine-tuned from
    MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    if not os.path.exists(adapter_dir):
        # The LoRA adapter folder is missing — inform the user exactly what to upload
        print(f"❌ ERROR: Phase 1 Adapter not found at '{adapter_dir}'!")
        print("Please upload Parth's adapter to this folder.")
        return None  # Return None — can't proceed without the adapter

    print(f"Loading Phase 1 Base Model ({MODEL_ID}) & LoRA Adapter ({adapter_dir})...")
    # Load Parth's tokenizer, which converts natural language text to token IDs
    tokenizer = load_tokenizer(MODEL_ID, adapter_dir)
    # Load Parth's base model with the LoRA adapter merged on top
    model     = load_model(MODEL_ID, adapter_dir)

    print("Generating DDL from Natural Language...")
    # Run Phase 1 inference: convert the user's natural language description to a DDL schema
    ddl_output = generate_sql(nl_description, model, tokenizer)

    # 🧹 CRITICAL: Clear VRAM so Phase 2 doesn't crash Google Colab!
    print("Unloading Phase 1 Model to free VRAM for Phase 2...")
    del model      # Remove the model object reference — marks it for Python's garbage collector
    del tokenizer  # Remove the tokenizer object reference as well
    gc.collect()            # Immediately run Python's garbage collector to free CPU memory
    torch.cuda.empty_cache()  # Tell CUDA to release all GPU memory it's holding for reuse

    return ddl_output  # Return the generated DDL string for Phase 2 to consume


# ==============================================================================
# PHASE 2: DDL -> SQL Queries (Your Model)
# ==============================================================================
def run_phase2_ddl_to_queries(ddl_text, model_path):
    # Import our own model loading and query generation functions from inference_qwen.py
    from inference_qwen import load_model as load_p2_model, generate_queries

    if not os.path.exists(model_path):
        # The Phase 2 model folder is missing — tell the user which path was expected
        print(f"❌ ERROR: Phase 2 Model not found at '{model_path}'!")
        return []  # Return empty list — caller will detect no queries and handle gracefully

    print(f"Loading Phase 2 Base Model & LoRA Adapter ({model_path})...")
    # Load our trained Qwen DDL→SQL model; returns (model, tokenizer, device)
    model, tokenizer, device = load_p2_model(model_path)

    print("Synthesizing 5 Autonomous Queries...")
    # Generate 5 diverse SQL SELECT queries from the DDL schema produced by Phase 1
    queries = generate_queries(
        schema_text=ddl_text,  # The DDL string output from Phase 1 (Parth's model)
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_queries=5,         # Generate exactly 5 unique SQL queries
        max_input=768,         # Maximum token budget for the DDL schema prompt
        max_output=256,        # Maximum token budget for the generated SQL
        temperature=0.8        # Slightly higher temperature → more varied query structures
    )

    # Clean up the Phase 2 model from VRAM (symmetric to Phase 1 cleanup)
    del model
    del tokenizer
    gc.collect()             # Free Python objects from CPU memory
    torch.cuda.empty_cache() # Free CUDA GPU memory so the system returns to a clean state

    return queries  # Return the list of generated SQL strings


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    # Print a prominent banner to signal the start of the full end-to-end demo
    print("\n" + "★"*80)
    print("   TRUE END-TO-END DEMO: NL → Parth's Model (DDL) → Your Model (SQL)")
    print("★"*80)

    # Auto-detect Phase 2 model path — try several common folder names in order
    p2_paths = ["qwen_sql_model_v2/final", "qwen_sql_model/final", "final_qwen", "qwen_sql_model", "qwen_sql_model_v2"]
    # Pick the first path that actually exists on disk (returns None if none exist)
    p2_model_path = next((p for p in p2_paths if os.path.exists(p)), None)

    # Auto-detect Phase 1 (Parth's) adapter path — try several common locations
    p1_paths = [
        "nl2schema_repo/schema_model/qlora-nl2sql",           # Standard repo location
        "schema_model/qlora-nl2sql",                           # Flattened structure
        "nl2schema_repo/schema_model/qlora-nl2sql-adapter2",  # Alternative adapter name
        "qlora-nl2sql"                                         # Root-level fallback
    ]
    # Pick the first path that actually exists on disk (returns None if none exist)
    p1_adapter_path = next((p for p in p1_paths if os.path.exists(p)), None)

    if not p2_model_path:
        # Phase 2 model is essential — abort with a clear error message
        print("\n❌ Missing Phase 2 Model.")
        return

    if not p1_adapter_path:
        # Phase 1 adapter is essential — tell the user exactly what to upload
        print("\n❌ Missing Phase 1 (Parth's) Adapter.")
        print("Please upload Parth's 'qlora-nl2sql' folder into your Drive!")
        return

    # STEP 1: Collect the user's natural language database description
    step_header(1, "Natural Language Input (Business Requirement)")
    print("Welcome! Please describe the database you want to build in Natural Language.")
    print("(Example: 'I need an e-commerce database to track users, their emails, and the orders they place.')\n")

    # Prompt the user for their natural language input via stdin
    nl_input = input("Your Description > ")

    if not nl_input.strip():
        # If the user just pressed Enter with no input, fall back to a built-in default
        nl_input = "A blog platform where users write posts. Users have an ID and a unique username. Posts have an ID, a title, and a text body."
        print(f"\n[No input detected. Using default description: '{nl_input}']")

    # STEP 2: Run Phase 1 — convert the natural language description to a DDL schema
    step_header(2, "Phase 1: Text-to-DDL (Parth's Qwen Model)")
    generated_ddl = run_phase1_nl_to_ddl(nl_input, p1_adapter_path)
    if not generated_ddl:
        # Phase 1 returned None — either import failed or adapter was missing
        return

    # Print the raw DDL schema generated by Phase 1 for user inspection
    print("\n[Phase 1 Output] Generated SQL DDL:")
    print(generated_ddl)

    # STEP 3: Run Phase 2 — generate diverse SQL queries from the DDL schema
    step_header(3, "Phase 2: DDL-to-SQL (Your Qwen Model)")
    queries = run_phase2_ddl_to_queries(generated_ddl, p2_model_path)

    # STEP 4: Display the final synthesized SQL queries with a slow-print animation
    step_header(4, "Final Output: Synthesized Queries")
    if queries:
        # Print each generated SQL query with its index, character by character
        for i, q in enumerate(queries, 1):
            slow_print(f" {i}. {q}", delay=0.01)  # Short delay for a fast but animated reveal
    print("\n✅ Pipeline Execution Complete.")  # Signal successful end of the demo


if __name__ == "__main__":
    main()  # Entry point — only runs when the script is called directly, not imported