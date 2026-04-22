import json
from prompt_gemini import get_json_from_file
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.converter import convert_json_to_sql_ddl

PATH_TO_DATA_FOLDER = "../../data/schemapile/processed/"


def save_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def main(file_name):
    schemas = get_json_from_file(f"{PATH_TO_DATA_FOLDER}{file_name}")["schemas"]

    filtered_schemas = []
    for i, schema in enumerate(schemas):
        try:
            sql_output = convert_json_to_sql_ddl(schema)
        except Exception as e:
            print(f"Error converting schema {i + 1} to SQL DDL: {e}")
            continue
        else:
            if sql_output is None:
                print(f"Error converting schema {i + 1} to SQL DDL. Skipping.")
                continue
            
            print(f"SQL DDL for Schema {i + 1}:\n")
            print(sql_output)
            print("\n\n")

            filtered_schemas.append({
                "sql_ddl": sql_output
            })
    
    print(f"Total schemas processed: {len(schemas)}")
    print(f"Total schemas successfully converted to SQL DDL: {len(filtered_schemas)}")
    save_json_file(filtered_schemas, f"{PATH_TO_DATA_FOLDER}/{file_name.replace('.json', '-ddl-filtered.json')}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
        main(file_name=file_name)
    else:
        print("No file name provided. Note: the file is expected to be in the 'processed' subfolder of the data folder.\n \
              The file must be a list of JSON schemas.\n Example usage: \n\npython convert_generated_into_ddl.py schemapile-pruned-sample5.json")
