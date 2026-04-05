import json
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv


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

PROMPT = "Task: Given this database schema, write a natural language business description that a non-technical stakeholder might give to describe the system they need."

def get_natural_desc_of_schema(schema, client = get_gemini_client()):
    # only required fields for the prompt are tables
    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=f"Schema: {json.dumps(schema['tables'], indent=2)}\n\n{PROMPT}",
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a expert database analyst. You can look at a database schema and understand the underlying business domain and use cases perfectly."
                "Output ONLY the final description. "
                "Do not include any headers, footers, introductions, or conversational filler."
            ),
            temperature=0.3  # Lower temperature for more consistent, focused output
        )
    )

    return response.text

def get_json_from_file(file_path):
    with open(file_path, 'r') as f:
        return json.loads(f.read())


schemas = get_json_from_file(f"{PATH_TO_DATA_FOLDER}/processed/schemapile-pruned-sample5.json")["schemas"]

print("Getting natural language description for schema: ",)
print(json.dumps(schemas[0]['tables'], indent=2))
print("\n\n")
print("Natural language description: ", get_natural_desc_of_schema(schemas[0]))
