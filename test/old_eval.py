import json
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from eval_utils import *


############################################
# PATHS (EDIT THESE)
############################################

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_PATH = "../schema_model/qlora-nl2sql-adapter2"
TEST_DATA_PATH = "../data/schemapile/ddl_filtered/schemapile_200_filtered-ddl-filtered.json"


############################################
# LOAD MODEL + ADAPTER
############################################

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()


############################################
# GENERATION (same)
############################################

def generate_sql(prompt):

    messages = [
        {
            "role": "system",
            "content": "You are an expert database designer. Output ONLY SQL DDL. Do not include markdown. Do not include comments."
        },
        {
            "role": "user",
            "content": "Generate SQL DDL for the following description:\n\n" + prompt
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
        max_new_tokens=800,
        do_sample=False
    )

    generated_ids = outputs[:, inputs.input_ids.shape[1]:]

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


############################################
# EVALUATION (same)
############################################

def evaluate(detailed=False):

    with open(TEST_DATA_PATH) as f:
        data = json.load(f)

    results = []

    valid_refs = 0

    for ex in tqdm(data):

        ref = clean_sql(ex["output"])

        if not executes_successfully_mysql(ref):
            continue

        valid_refs += 1

        gen = generate_sql(ex["input"])
        gen = clean_sql(gen)

        parse_ok = is_valid_sql(gen)
        exec_ok  = executes_successfully_mysql(gen)
        match    = schema_match(ref, gen)

        results.append({
            "input": ex["input"],
            "reference": ref,
            "generated": gen,
            "parse": parse_ok,
            "exec": exec_ok,
            "schema_match": match
        })

    n = len(results)

    summary = {
        "valid_reference_count": valid_refs,
        "evaluated": n,
        "parse_rate": sum(r["parse"] for r in results) / n if n else 0,
        "execution_rate": sum(r["exec"] for r in results) / n if n else 0,
        "schema_match_rate": sum(r["schema_match"] for r in results) / n if n else 0
    }

    print("\n==== RESULTS ====")
    for k, v in summary.items():
        print(k, ":", v)

    if detailed:
        with open("detailed_results_lora.json", "w") as f:
            json.dump(results, f, indent=2)

    return summary


if __name__ == "__main__":
    evaluate(detailed=True)