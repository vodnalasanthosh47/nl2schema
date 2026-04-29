import json
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import signal
import sys

from eval_utils import *


############################################
# PATHS (EDIT THESE)
############################################

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
TEST_DATA_PATH = "../data/schemapile/test/fixed-sample100.json"

SAVE_PATH = "partial_results_base.json"
SAVE_EVERY = 5


############################################
# LOAD MODEL
############################################

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

model.eval()

print("MODEL DEVICE:", model.device)


############################################
# GENERATION
############################################

def generate_sql(prompt):

    messages = [
        {
            "role": "system",
            "content": "You are an expert database designer. Output ONLY SQL DDL. Do not include markdown. Do not include comments or explanation."
        },
        {
            "role": "user",
            "content": "Generate ONLY SQL DDL for the following description:\n\n" + prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1000,   # FIXED
        do_sample=False
    )

    generated_ids = outputs[:, inputs.input_ids.shape[1]:]

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


############################################
# EVALUATION
############################################

def evaluate(detailed=False):

    global results

    with open(TEST_DATA_PATH) as f:
        data = json.load(f)

    ############################################
    # PRE-FILTER VALID REFERENCES
    ############################################

    valid_data = []

    """for i, ex in enumerate(data[:5]):
        ref = clean_sql(ex["output"])
        print("\n--- SAMPLE", i, "---")
        print(ref)
        print("EXECUTES:", executes_successfully_mysql(ref))"""

    for ex in data:
        ref = clean_sql(ex["output"])
        if executes_successfully_mysql(ref):
            valid_data.append(ex)
        else:
            print("\n--- INVALID REFERENCE ---")
            print(ref)


    print(f"Valid references: {len(valid_data)} / {len(data)}")

    ############################################

    results = []
    valid_refs = len(valid_data)

    ############################################
    # SIGNAL HANDLER
    ############################################

    def save_and_exit(signum, frame):
        print("\nSaving partial results before exit...")
        with open(SAVE_PATH, "w") as f:
            json.dump(results, f, indent=2)
        sys.exit(0)

    signal.signal(signal.SIGINT, save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)

    ############################################
    # MAIN LOOP
    ############################################

    for i, ex in enumerate(tqdm(valid_data)):

        try:
            ref = clean_sql(ex["output"])

            gen = generate_sql(ex["input"])
            gen = clean_sql(gen)

            if not gen.strip():
                parse_ok = False
                exec_ok = False
                match = False
            else:
                parse_ok = is_valid_sql(gen)
                exec_ok  = executes_successfully_mysql(gen)
                match    = schema_match(ref, gen)

            result = {
                "input": ex["input"],
                "reference": ref,
                "generated": gen,
                "parse": parse_ok,
                "exec": exec_ok,
                "schema_match": match
            }

            results.append(result)

        except Exception as e:
            print(f"Error at index {i}: {e}")
            continue

        ############################################
        # PERIODIC SAVE
        ############################################

        if i % SAVE_EVERY == 0 and results:
            with open(SAVE_PATH, "w") as f:
                json.dump(results, f, indent=2)

    ############################################
    # FINAL SAVE
    ############################################

    with open("final_results_base.json", "w") as f:
        json.dump(results, f, indent=2)

    ############################################
    # METRICS
    ############################################

    n = len(results)

    summary = {
        "valid_reference_count": valid_refs,
        "evaluated": n,
        "parse_rate": sum(r["parse"] for r in results) / n if n else 0,
        "execution_rate": sum(r["exec"] for r in results) / n if n else 0,
        "schema_match_rate": sum(r["schema_match"] for r in results) / n if n else 0
    }

    print("\n==== RESULTS ====")
    print(f"Valid references   : {summary['valid_reference_count']}")
    print(f"Evaluated          : {summary['evaluated']}")
    print(f"Parse rate         : {summary['parse_rate']:.2%}")
    print(f"Execution rate     : {summary['execution_rate']:.2%}")
    print(f"Schema match rate  : {summary['schema_match_rate']:.2%}")

    if detailed:
        with open("detailed_results_base.json", "w") as f:
            json.dump(results, f, indent=2)

    return summary


############################################
# RUN
############################################

if __name__ == "__main__":
    evaluate(detailed=True)