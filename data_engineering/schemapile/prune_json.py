import json


PATH_TO_DATA_FOLDER = "../../data/schemapile"

def prune_schemapile():
    with open(f"{PATH_TO_DATA_FOLDER}/raw/schemapile-perm.json", 'r') as f:
        schemapile = json.loads(f.read())


    pruned_schemapile = {"schemas": []}
    total_schemas = len(schemapile)
    current_schema = 1

    for (schema_name, schema) in schemapile.items():
        print(f"Processing schema {current_schema}/{total_schemas}: {schema_name}")
        pruned_schema = {
            "name": schema_name,
            "url": schema["INFO"]["URL"],
            "license": schema["INFO"]["LICENSE"],
            "permissive": schema["INFO"]["PERMISSIVE"],
            "tables": {}
        }

        for (table_name, table) in schema["TABLES"].items():
            pruned_table = {
                "COLUMNS": {}
            }
            for (attribute_name, attribute_value) in table["COLUMNS"].items():
                pruned_table["COLUMNS"][attribute_name] = {
                    "TYPE": attribute_value["TYPE"],
                    "NULLABLE": attribute_value["NULLABLE"],
                    "UNIQUE": attribute_value["UNIQUE"],
                    "DEFAULT": attribute_value["DEFAULT"],
                    "CHECKS": attribute_value["CHECKS"],
                    "IS_PRIMARY": attribute_value["IS_PRIMARY"],
                }

            pruned_table["PRIMARY_KEYS"] = table["PRIMARY_KEYS"]
            pruned_table["FOREIGN_KEYS"] = table["FOREIGN_KEYS"]
            pruned_table["CHECKS"] = table["CHECKS"]

            pruned_schema["tables"][table_name] = pruned_table
        
        pruned_schemapile["schemas"].append(pruned_schema)
        current_schema += 1
            

    with open(f"{PATH_TO_DATA_FOLDER}/processed/schemapile-pruned.json", 'w') as f:
        f.write(json.dumps(pruned_schemapile))


def sample_firstN_from_pruned_schemapile(N: int = 500):
    if N <= 0:
        print("N should be a positive integer.")
        return

    with open(f"{PATH_TO_DATA_FOLDER}/processed/schemapile-pruned.json", 'r') as f:
        schemapile = json.loads(f.read())
    
    if N > len(schemapile["schemas"]):
        print(f"N is greater than the total number of schemas ({len(schemapile['schemas'])}). ")
        return
        

    sampled_schemapile = {"schemas": schemapile["schemas"][:N]}

    with open(f"{PATH_TO_DATA_FOLDER}/processed/schemapile-pruned-sample{N}.json", 'w') as f:
        f.write(json.dumps(sampled_schemapile, indent=4))

# sample_firstN_from_pruned_schemapile(5)