from prompt_gemini import PATH_TO_DATA_FOLDER, get_json_from_file
from langdetect import detect, LangDetectException
import json


def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def is_english_schema(schema):
    for table_name, table_data in schema['tables'].items():
        if not is_english(table_name):
            return False
        for col_name in table_data['COLUMNS'].keys():
            if not is_english(col_name):
                return False
    
    return True


def get_english_schemas(json_data_path):
    data = get_json_from_file(json_data_path)
    cleaned = []

    for schema in data["schemas"]:
        if is_english_schema(schema):
            cleaned.append(schema)

    print(f"Kept:    {len(cleaned)}")
    return {"schemas": cleaned}


cleaned_201_to_end = get_english_schemas(f"{PATH_TO_DATA_FOLDER}/processed/schemapile-pruned-sample201_end.json")
with open(f"{PATH_TO_DATA_FOLDER}/processed/schemapile-en-sample201_end.json", 'w') as f:
    json.dump(cleaned_201_to_end, f)