import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")

with open("../data/schemapile/ddl_filtered/ddl-filtered-combined.json") as f:
    data = json.load(f)

lengths = []
for ex in data:
    text = ex["input"] + ex["output"]
    lengths.append(len(tokenizer.encode(text)))

print(f"Max tokens: {max(lengths)}")
print(f"Mean tokens: {sum(lengths)//len(lengths)}")
print(f"Examples exceeding 512: {sum(1 for l in lengths if l > 512)}")


lengths = sorted(len(tokenizer.encode(ex["input"] + ex["output"])) for ex in data)
total = len(lengths)

for pct in [50, 75, 90, 95, 99]:
    idx = int(pct / 100 * total)
    print(f"p{pct}: {lengths[idx]} tokens")

print(f"Total examples: {total}")