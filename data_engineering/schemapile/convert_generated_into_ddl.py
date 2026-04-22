import json
from prompt_gemini import get_json_from_file
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.converter import build_schema_ddl_from_tables

PATH_TO_DATA_FOLDER = "../../data/schemapile/generated/"


def save_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main(file_name):
    input_output_pairs = get_json_from_file(f"{PATH_TO_DATA_FOLDER}{file_name}")

    filtered_schemas = []
    for i, pair in enumerate(input_output_pairs):
        nl_description = pair["input"]
        tables_json = json.loads(pair["output"])
        print(f"Processing schema {i + 1} with schema JSON:\n{tables_json}, {type(tables_json)}\n")
        try:
            sql_output = build_schema_ddl_from_tables(tables_json)
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
                "input": nl_description,
                "output": sql_output
            })
    
    print(f"Total schemas processed: {len(input_output_pairs)}")
    print(f"Total schemas successfully converted to SQL DDL: {len(filtered_schemas)}")
    save_json_file(filtered_schemas, f"{PATH_TO_DATA_FOLDER}/{file_name.replace('.json', '-ddl-filtered.json')}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
        main(file_name=file_name)
    else:
        print("No file name provided. Note: the file is expected to be in the 'generated' subfolder of the data folder.\n \
              The file must be a list of input-output pairs of nl descriptions and JSON schemas.\n Example usage: \n\npython convert_generated_into_ddl.py schemapile-pruned-sample5-with-nl.json")
