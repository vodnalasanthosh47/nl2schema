import json

with open('schemapile-perm.json') as f:
    data = json.load(f)

print(f"Number of schemas: {len(data)}")

# Look at first entry
first_key = list(data.keys())[0]
print(json.dumps(data[first_key], indent=2))