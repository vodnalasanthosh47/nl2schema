"""
json_to_mermaid.py
==================
Parses a SchemaPile-format JSON file and emits Mermaid ER diagrams
for every schema it contains.

Usage:
    python json_to_mermaid.py input.json [-o output.md] [-i schema_index]

If no output path is given the Mermaid diagrams are printed to stdout.
"""

import json
import sys
import argparse
from pathlib import Path


def build_mermaid_ddl(schema: dict) -> str:
    name = schema.get("name", "unnamed")
    tables = schema.get("tables", schema)  # Fallback: schema might just be the tables dict
    
    lines = []
    lines.append(f"## Schema: {name}")
    lines.append("```mermaid")
    lines.append("erDiagram")
    
    relationships = []
    
    for table_name, table_def in tables.items():
        # table_def could be a string if the schema is malformed, so check
        if not isinstance(table_def, dict):
            continue
            
        lines.append(f"    {table_name} {{")
        columns = table_def.get("COLUMNS", {})
        pk_cols = table_def.get("PRIMARY_KEYS", [])
        fk_list = table_def.get("FOREIGN_KEYS", [])
        
        fk_col_set = set()
        for fk in fk_list:
            for col in fk.get("COLUMNS", []):
                fk_col_set.add(col)
                
            ref_table = fk.get("FOREIGN_TABLE")
            if ref_table:
                # Add relationship: ref_table ||--o{ table_name
                relationships.append(f'    {ref_table} ||--o{{ {table_name} : "foreign_key"')
        
        for col_name, col in columns.items():
            if not isinstance(col, dict):
                continue
                
            col_type = col.get("TYPE", "Unknown")
            # Mermaid requires data types to be continuous strings without spaces/parentheses
            col_type = col_type.replace(" ", "_").replace("(", "_").replace(")", "_")
            if not col_type:
                col_type = "Unknown"
                
            keys = []
            if col_name in pk_cols:
                keys.append("PK")
            if col_name in fk_col_set:
                keys.append("FK")
                
            key_str = ", ".join(keys)
            if key_str:
                lines.append(f"        {col_type} {col_name} {key_str}")
            else:
                lines.append(f"        {col_type} {col_name}")
                
        lines.append("    }")
        
    lines.append("")
    # Add relationships (removing duplicates)
    for rel in sorted(list(set(relationships))):
        lines.append(rel)
        
    lines.append("```\n")
    return "\n".join(lines)


def convert(input_path: str, output_path: str | None = None, schema_index: int | None = None) -> None:
    try:
        with open(input_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        print(f"Error reading {input_path}: {e}", file=sys.stderr)
        return

    # Handle if data is the full wrapper or just the schemas array
    if isinstance(data, dict):
        schemas = data.get("schemas", [])
        if not schemas:
            # Fallback if the json is just a single un-wrapped schema definition
            if "tables" in data or "COLUMNS" in next(iter(data.values())):
                 schemas = [{"name": "unnamed", "tables": data.get("tables", data)}]
            else:
                print("Warning: no 'schemas' key found. Proceeding with single dict.", file=sys.stderr)
                schemas = [data]
    elif isinstance(data, list):
        schemas = data
    else:
        print("Invalid JSON structure.", file=sys.stderr)
        return

    if schema_index is not None:
        if 0 <= schema_index < len(schemas):
            schemas = [schemas[schema_index]]
        else:
            print(f"Error: Index {schema_index} is out of bounds for {len(schemas)} schemas.", file=sys.stderr)
            return

    blocks = [
        "<!-- Auto-generated Mermaid ER Diagrams -->",
        f"<!-- Source file : {Path(input_path).name} -->",
        f"<!-- Schemas     : {len(schemas)} -->\n"
    ]

    for schema in schemas:
        blocks.append(build_mermaid_ddl(schema))
        blocks.append("")  # Blank line between schemas

    md_output = "\n".join(blocks)

    if output_path:
        Path(output_path).write_text(md_output, encoding="utf-8")
        print(f"Written to {output_path}")
    else:
        print(md_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parses a SchemaPile-format JSON file and emits Mermaid ER diagrams.")
    parser.add_argument("input", help="Input JSON file path")
    parser.add_argument("-o", "--output", help="Optional output Markdown file path")
    parser.add_argument("-i", "--index", type=int, help="Optional index of the specific schema to extract (0-indexed)")
    
    args = parser.parse_args()
    convert(args.input, args.output, args.index)
