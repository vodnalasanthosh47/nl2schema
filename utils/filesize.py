import json

def get_json_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.loads(f.read()) 

file = '../data/schemapile/test/schemapile-pruned-sample882_to_2382-ddl-filtered-with-nl.json'

# count number of schemas in the file
schemas = get_json_from_file(file)
print(len(schemas))