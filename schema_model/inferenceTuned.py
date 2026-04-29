"""
inference.py
------------
Load the base Qwen2.5-Coder-1.5B-Instruct model, attach the saved LoRA adapter,
and generate SQL DDL from a natural language schema description.

Usage:
    python inference.py --prompt "A blog platform where users write posts. Users have id and username. Posts have id, title, and body."
    python inference.py --prompt "..." --adapter ./qlora-nl2sql-adapter
"""

import argparse
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_ID    = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_DIR = "./qlora-nl2sql-adapter2"
MAX_NEW_TOKENS = 512

SYSTEM_PROMPT = (
    "You are an expert database designer. "
    "Output ONLY SQL DDL. Do not include markdown. Do not include comments."
)


# ---------------------------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------------------------
def load_tokenizer(model_id: str, adapter_dir: str) -> AutoTokenizer:
    # Load from the adapter directory so any saved tokenizer config is honoured
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-padding is correct for generation
    return tokenizer


# ---------------------------------------------------------------------------
# Load base model in 4-bit + attach LoRA adapter
# ---------------------------------------------------------------------------
def load_model(model_id: str, adapter_dir: str) -> PeftModel:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()   # disable dropout for deterministic output
    return model


# ---------------------------------------------------------------------------
# Build prompt and run generation
# ---------------------------------------------------------------------------
"""def generate_sql(prompt: str, model: PeftModel, tokenizer: AutoTokenizer) -> str:
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": f"Generate SQL DDL for the following description:\n\n{prompt}"},
    ]

    # apply_chat_template with add_generation_prompt=True appends the
    # assistant-start token so the model knows it should start replying
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,           # greedy — deterministic, best for structured output
            temperature=1.0,           # irrelevant when do_sample=False but avoids warnings
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Slice off the input tokens so we only decode the generated part
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return clean_output(raw_output)
"""

# ---------------------------------------------------------------------------
# Post-process: strip any accidental markdown fences
# ---------------------------------------------------------------------------
def clean_output(text: str) -> str:
    # Remove ```sql ... ``` or ``` ... ``` wrappers if the model produced them
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = text.replace("```", "")
    return text.strip()


def generate_sql(prompt: str, model, tokenizer: AutoTokenizer) -> str:
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": f"Generate SQL DDL for the following description:\n\n{prompt}"},
    ]

    # apply_chat_template returns BatchEncoding on some transformers versions
    # Always extract input_ids explicitly and convert to tensor manually
    raw = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if isinstance(raw, dict) or hasattr(raw, "input_ids"):
        input_ids = raw["input_ids"].to(model.device)
    else:
        input_ids = raw.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return clean_output(raw_output)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NL → SQL DDL inference with fine-tuned QLoRA adapter")
    parser.add_argument("--prompt",  required=True,      help="Natural language schema description")
    parser.add_argument("--adapter", default=ADAPTER_DIR, help="Path to saved LoRA adapter directory")
    args = parser.parse_args()

    print("Loading tokenizer …")
    tokenizer = load_tokenizer(MODEL_ID, args.adapter)

    print("Loading model + LoRA adapter …")
    model = load_model(MODEL_ID, args.adapter)

    print("\nGenerating SQL DDL …\n")
    sql = generate_sql(args.prompt, model, tokenizer)

    print("=" * 60)
    print(sql)
    print("=" * 60)


if __name__ == "__main__":
    main()
