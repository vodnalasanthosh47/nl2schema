import json
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


############################################
# model
############################################

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    torch_dtype="auto",

    device_map="auto"

)

model.eval()


############################################
# load dataset
############################################

with open("../data/schemapile/generated/schemapile-pruned-sample5-with-nl.json", "r") as f:

    dataset = json.load(f)


############################################
# generation function
############################################

def generate_sql(description):

    messages = [

        {

            "role": "system",

            "content":

            "You are an expert database designer. Output ONLY a relational database schema in json format. Do not include explanations. Do NOT include markdown formatting.Do NOT include ``` or code fences.Do NOT include comments."

            #"You are an expert database designer. Output ONLY SQL DDL. Do not include explanations. Do NOT include markdown formatting.Do NOT include ``` or code fences.Do NOT include comments."

        },

        {

            "role": "user",

            "content":

            f"Generate a relational database schema in json for the following description:\n\n{description}"
            #f"Generate SQL DDL for the following description:\n\n{description}"

        }

    ]


    text = tokenizer.apply_chat_template(

        messages,

        tokenize=False,

        add_generation_prompt=True

    )


    inputs = tokenizer(

        [text],

        return_tensors="pt"

    ).to(model.device)


    outputs = model.generate(

        **inputs,

        max_new_tokens=300,

        temperature=0.1

    )


    generated_ids = [

        output_ids[len(input_ids):]

        for input_ids, output_ids in zip(

            inputs.input_ids,

            outputs

        )

    ]


    response = tokenizer.batch_decode(

        generated_ids,

        skip_special_tokens=True

    )[0]


    return response


############################################
# run inference
############################################

for i, example in enumerate(dataset):

    description = example["input"]

    prediction = generate_sql(description)


    print("\n====================================")

    print(f"Example {i+1}")

    print("\nINPUT:\n")

    print(description)

    print("\nPREDICTED DDL:\n")

    print(prediction)


    if "output" in example:

        print("\nGROUND TRUTH:\n")

        print(example["output"])