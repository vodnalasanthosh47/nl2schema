import json
from pathlib import Path


INPUT_PATH = "../data/schemapile/test/test-sample100.json"     # your current dataset
OUTPUT_PATH = "../data/schemapile/test/fixed-sample100.json"   # cleaned dataset


def fix_output_string(s: str) -> str:
    """
    Fix double-escaped SQL strings like:
    "\"CREATE TABLE ...\\n...\""
    → "CREATE TABLE ...\n..."
    """

    if not s:
        return s

    s = s.strip()

    # Remove wrapping quotes if present
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]

    # Convert escaped sequences
    s = s.encode("utf-8").decode("unicode_escape")

    return s.strip()


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixed_data = []

    for obj in data:
        new_obj = obj.copy()

        if "output" in new_obj:
            new_obj["output"] = fix_output_string(new_obj["output"])

        fixed_data.append(new_obj)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(fixed_data, f, indent=2)

    print(f"Fixed dataset written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()