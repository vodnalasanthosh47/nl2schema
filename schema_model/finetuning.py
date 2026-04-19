import json
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)

from peft import LoraConfig, get_peft_model


############################################
# configuration
############################################

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

DATA_PATH = "dataset.json"


############################################
# load tokenizer
############################################

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


############################################
# load model in 4-bit mode
############################################

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_quant_type="nf4"

)


model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    device_map="auto"

)


############################################
# LoRA configuration
############################################

lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    bias="none",

    target_modules=["q_proj","v_proj"],

    task_type="CAUSAL_LM"

)


model = get_peft_model(model, lora_config)


############################################
# load dataset
############################################

with open(DATA_PATH) as f:

    raw_data = json.load(f)


dataset = Dataset.from_list(raw_data)


############################################
# convert each example into chat format
############################################

def format_example(example):

    messages = [

        {
            "role": "system",
            "content":
            "You are an expert database designer. Output ONLY SQL DDL. Do not include markdown. Do not include comments."
        },

        {
            "role": "user",
            "content":
            "Generate SQL DDL for the following description:\n\n"
            + example["input"]
        },

        {
            "role": "assistant",
            "content": example["output"]
        }

    ]


    text = tokenizer.apply_chat_template(

        messages,

        tokenize=False

    )


    return {"text": text}


dataset = dataset.map(format_example)


############################################
# tokenize
############################################

def tokenize(example):

    tokens = tokenizer(

        example["text"],

        truncation=True,

        padding="max_length",

        max_length=512

    )


    tokens["labels"] = tokens["input_ids"].copy()

    return tokens


dataset = dataset.map(tokenize, batched=True)


############################################
# training configuration
############################################

training_args = TrainingArguments(

    output_dir="qwen_ddl_adapter",

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    num_train_epochs=3,

    learning_rate=2e-4,

    fp16=True,

    logging_steps=10,

    save_strategy="epoch",

    report_to="none"

)


############################################
# trainer
############################################

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=dataset

)


############################################
# train
############################################

trainer.train()


############################################
# save adapter weights
############################################

model.save_pretrained("qwen_ddl_adapter")

tokenizer.save_pretrained("qwen_ddl_adapter")