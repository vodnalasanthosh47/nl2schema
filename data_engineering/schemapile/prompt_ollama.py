import json
from ollama import chat
from ollama import ChatResponse


PATH_TO_DATA_FOLDER = "../../data/schemapile"

"""
from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='gemma3', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)
"""

PROMPT = "Task: Given this database schema, write a detailed and specific natural language business description that a non-technical \
        stakeholder might give to describe the system they need. The description would be fed to a text to database schema model to generate \
        the given schema. Include any business rules that are implied by the schema and is worth mentioning."

def get_natural_desc_of_schema(schema):
    response: ChatResponse = chat(model='gemma4:e4b', messages=[
        {
            'role': 'user',
            'content': f"Schema: {json.dumps(schema['tables'], indent=2)}\n\n{PROMPT}",
        }
    ])
    return response['message']['content']

def get_json_from_file(file_path):
    with open(file_path, 'r') as f:
        return json.loads(f.read())

data = get_json_from_file(f"{PATH_TO_DATA_FOLDER}/processed/schemapile-pruned-sample5.json")["schemas"]

print(get_natural_desc_of_schema(data[0]))
