import json


PATH_TO_DATA_FOLDER = "../../data/schemapile"

# take input json of format [{},{},...] and sample first 100 entries, and save it to a new json file of type [{},{},...]

with open(f"{PATH_TO_DATA_FOLDER}/test/schemapile-pruned-sample882_to_2382-ddl-filtered-with-nl.json", 'r') as f:
    dataset = json.loads(f.read())

new_json = []

#sample first 100 schemas, and save it to a new json file

for i in range(100):
    new_json.append(dataset[i])



with open(f"{PATH_TO_DATA_FOLDER}/test/test-sample100_2.json", 'w') as f:
    json.dump(new_json, f, indent =2)