import json
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import time
from tqdm import tqdm


PATH_TO_DATA_FOLDER = "../../data/schemapile"

# Load environment variables
load_dotenv("../../.env")

def get_gemini_client():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return client

def test_gemini_api(client = get_gemini_client()):
    working_context = {
        "user_data": {"name": "Alex", "membership": "Gold"},
        "rules": ["Be concise", "Tone: Professional"],
        "topic": "Welcome message for the annual gala"
    }

    # Convert JSON to a string
    json_context_str = json.dumps(working_context, indent=2)

    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=f"Context: {json_context_str}\n\nTask: Based on the context above, write a welcome paragraph.",
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a text processing utility. "
                "Output ONLY the final paragraph. "
                "Do not include any headers, footers, introductions, or conversational filler."
            ),
            temperature=0.3  # Lower temperature for more consistent, focused output
        )
    )

    print(response.text)

PROMPT = "Task: Given this database schema, write a detailed and specific natural language business description that a non-technical \
        stakeholder might give to describe the system they need. The description would be fed to a text to database schema model to generate \
        the given schema. Include any business rules that are implied by the schema and is worth mentioning."

def get_natural_desc_of_schema(schema, client = get_gemini_client()):
    # only required fields for the prompt are tables
    response = client.models.generate_content(
        model='gemini-3.1-flash-lite-preview',
        contents=f"Schema: {json.dumps(schema['tables'], indent=2)}\n\n{PROMPT}",
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a expert database analyst. You can look at a database schema and understand the underlying business domain and use cases perfectly."
                "Output ONLY the final description. Output must be a single paragraph."
                "Do not include any headers, footers, introductions, or conversational filler."
            ),
            temperature=0.7  # Lower temperature for more consistent, focused output
        )
    )

    return response.text

def get_json_from_file(file_path):
    with open(file_path, 'r') as f:
        return json.loads(f.read())


def generate_dataset(schemas):
    dataset = []
    client = get_gemini_client()
    i = 1
    for schema in schemas:
        natural_language_description = get_natural_desc_of_schema(schema, client)
        dataset.append({
            "input": natural_language_description,
            "output": json.dumps(schema['tables'], ensure_ascii=False) + '\n'
        })
        #  sleep for 4s
        print(f"Completed {i}/{len(schemas)}:")
        print(natural_language_description)
        print("\n\n")
        i += 1
        time.sleep(3)
    return dataset
    


# print("Getting natural language description for schema: ",)
# print(json.dumps(schemas[0]['tables'], indent=2))
# print("\n\n")
# start_time = time.time()
# print("\nNatural language description: ", get_natural_desc_of_schema(schemas[2]), "\n\n")
# end_time = time.time()
# print("Time taken: ", end_time - start_time)

schemas = get_json_from_file(f"{PATH_TO_DATA_FOLDER}/processed/schemapile-pruned-sample200.json")["schemas"]
dataset = generate_dataset(schemas)

with open(f"{PATH_TO_DATA_FOLDER}/generated/schemapile-pruned-sample200-with-nl.json", 'w') as f:
    json.dump(dataset, f)