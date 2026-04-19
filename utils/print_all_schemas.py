import json
import time

PATH_TO_DATA_FOLDER = "../data/schemapile/processed"

def get_json_from_file(file_path):
    with open(file_path, 'r') as f:
        return json.loads(f.read())
    
def print_all_schema_names(json_data_path):
    data = get_json_from_file(json_data_path)
    for i, schema in enumerate(data["schemas"]):
        print(f"Schema {i + 1}:")
        for table_name in schema["tables"].keys():
            print(table_name)
            for column_name in schema["tables"][table_name]["COLUMNS"].keys():
                print(f"  - {column_name}")
        print("\n\n")

        time.sleep(2)


if __name__ == "__main__":
    print_all_schema_names(f"{PATH_TO_DATA_FOLDER}/schemapile-pruned-sample200.json")
