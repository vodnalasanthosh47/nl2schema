import json
from langdetect import detect

PATH_TO_DATA_FOLDER = "../../data/schemapile"

def get_json_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.loads(f.read())


def filter_schemas_by_language(schemas, language="en"):
    filtered_schemas = []
    failed_schemas = []
    for i, schema in enumerate(schemas):
        print(f"Detecting language for schema {i+1}...")

        identifiers = []
        for table_name, table_data in schema["tables"].items():
            identifiers.append(table_name)
            for column_name in table_data["COLUMNS"].keys():
                identifiers.append(column_name)
        
        # Join into one string — gives fasttext much more context
        text = ' '.join(identifiers).replace('_', ' ').replace("id", "").replace("dept", "department").replace("addr", "address").replace("loc", "location").replace("desc", "description").replace("info", "information").replace("num", "number").replace("amt", "amount").replace("qty", "quantity").replace("emp", "employee").replace("cust", "customer").replace("prod", "product")
        
        try:
            detected_language = detect(text)
            if detected_language == language:
                filtered_schemas.append(i+1)  # Store the index of the schema
            else:
                failed_schemas.append(i+1)  # Store the index of the failed schema
                print(f"Text for language detection: {text}...")  # Print the first 100 characters for debugging
        except Exception as e:
            print(f"Error detecting language for schema {i+1}: {e}")
            failed_schemas.append(i+1)  # Store the index of the failed schema
            print(f"Text for language detection: {text[:150]}...")  # Print the first 100 characters for debugging

    return filtered_schemas, failed_schemas


def main():
    schemapile = get_json_from_file(f"{PATH_TO_DATA_FOLDER}/processed/schemapile-pruned-sample200.json")
    schemas = schemapile["schemas"]
    filtered_schema_indices, failed_schema_indices = filter_schemas_by_language(schemas, language="en")
    
    print(f"Schemas with descriptions in English: \n{filtered_schema_indices}")
    print(f"Schemas that failed the language check: \n{failed_schema_indices}")


if __name__ == "__main__":
    main()
