import json


PATH_TO_DATA_FOLDER = "../../data/schemapile"

# with open(f"{PATH_TO_DATA_FOLDER}/generated/schemapile-pruned-sample200-with-nl.json", 'r') as f:
#     dataset = json.loads(f.read())

# with open(f"{PATH_TO_DATA_FOLDER}/generated/schemapile-pruned-sample5-with-nl.json", 'w') as f:
#     json.dump(dataset[:5], f, indent=2)

with open(f"{PATH_TO_DATA_FOLDER}/processed/schemapile-pruned.json", 'r') as f:
    dataset = json.loads(f.read())

new_json = {
    "schemas": []
}
new_json["schemas"] = dataset["schemas"][201:]

with open(f"{PATH_TO_DATA_FOLDER}/processed/schemapile-pruned-sample201_end.json", 'w') as f:
    json.dump(new_json, f)