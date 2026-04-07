import json


PATH_TO_DATA_FOLDER = "../../data/schemapile"

with open(f"{PATH_TO_DATA_FOLDER}/generated/schemapile-pruned-sample200-with-nl.json", 'r') as f:
    dataset = json.loads(f.read())

with open(f"{PATH_TO_DATA_FOLDER}/generated/schemapile-pruned-sample5-with-nl.json", 'w') as f:
    json.dump(dataset[:5], f, indent=2)