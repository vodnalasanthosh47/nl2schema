import json
import csv


PATH_TO_DATA_FOLDER = "../../data/schemapile"

with open(f"{PATH_TO_DATA_FOLDER}/processed/schemapile-pruned.json", 'r') as f:
    schemapile = json.loads(f.read())

with open(f"{PATH_TO_DATA_FOLDER}/processed/schemapile.csv", 'w', newline='') as csvfile:
    fieldnames = ["schema_name", "url", "license", "permissive"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for schema in schemapile["schemas"]:
        writer.writerow({
            "schema_name": schema["name"],
            "url": schema["url"],
            "license": schema["license"],
            "permissive": schema["permissive"]
        })
