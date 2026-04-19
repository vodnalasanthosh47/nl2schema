"""
nl_schema_to_ddl.py
===================
Reads a fine-tuning JSON file whose entries have the shape:

    { "input":  "<natural-language description>",
      "output": "<JSON-encoded table map>" }

The output value is a JSON string whose parsed form is:

    {
      "<table_name>": {
          "COLUMNS":      { "<col>": { "TYPE", "NULLABLE", "UNIQUE",
                                       "DEFAULT", "CHECKS", "IS_PRIMARY" }, … },
          "PRIMARY_KEYS": [ … ],
          "FOREIGN_KEYS": [ { "COLUMNS", "FOREIGN_TABLE",
                               "REFERRED_COLUMNS",
                               "ON_DELETE", "ON_UPDATE" }, … ],
          "CHECKS":       [ … ]
      }, …
    }

The script rewrites every entry, keeping "input" unchanged and replacing
"output" with the equivalent SQL DDL string.

Usage:
    python nl_schema_to_ddl.py input.json [output.json]

If no output path is given the result is printed to stdout.
"""

import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------

TYPE_MAP: dict[str, str] = {
    "Int":         "INT",
    "UnsignedInt": "INT UNSIGNED",
    "BigInt":      "BIGINT",
    "SmallInt":    "SMALLINT",
    "Varchar":     "VARCHAR(255)",
    "Text":        "TEXT",
    "Date":        "DATE",
    "Timestamp":   "TIMESTAMP",
}


def sql_type(abstract_type: str) -> str:
    mapped = TYPE_MAP.get(abstract_type)
    if mapped is None:
        return f"{abstract_type} /* unknown type – review manually */"
    return mapped


# ---------------------------------------------------------------------------
# Referential-action mapping
# ---------------------------------------------------------------------------

ACTION_MAP: dict[str, str] = {
    "Cascade":    "CASCADE",
    "SetNull":    "SET NULL",
    "SetDefault": "SET DEFAULT",
    "NoAction":   "NO ACTION",
    "Restrict":   "RESTRICT",
}


def sql_action(action: str | None) -> str | None:
    if action is None:
        return None
    return ACTION_MAP.get(action, action.upper())


# ---------------------------------------------------------------------------
# Column & table builders
# ---------------------------------------------------------------------------

def build_column_def(col_name: str, col: dict) -> str:
    """Return a single column definition line (no trailing comma)."""
    parts = [f"    {col_name}", sql_type(col["TYPE"])]

    if col.get("NULLABLE") is False:
        parts.append("NOT NULL")

    default = col.get("DEFAULT")
    if default is not None:
        if isinstance(default, str) and default.upper() not in (
            "NULL", "CURRENT_TIMESTAMP", "TRUE", "FALSE"
        ):
            parts.append(f"DEFAULT '{default}'")
        else:
            parts.append(f"DEFAULT {default}")

    if col.get("UNIQUE") is True:
        parts.append("UNIQUE")

    # Inline CHECK constraints on the column
    for check in col.get("CHECKS", []):
        parts.append(f"CHECK ({check})")

    return " ".join(parts)


def build_table_ddl(table_name: str, table: dict) -> str:
    """Return a complete CREATE TABLE … ; statement."""
    columns  = table.get("COLUMNS", {})
    pk_cols  = table.get("PRIMARY_KEYS", [])
    fk_list  = table.get("FOREIGN_KEYS", [])
    tbl_chks = table.get("CHECKS", [])

    col_defs: list[str] = []

    # Column definitions
    for col_name, col in columns.items():
        col_defs.append(build_column_def(col_name, col))

    # PRIMARY KEY constraint
    if pk_cols:
        col_defs.append(f"    PRIMARY KEY ({', '.join(pk_cols)})")

    # FOREIGN KEY constraints
    for fk in fk_list:
        fk_cols   = ", ".join(fk["COLUMNS"])
        ref_table = fk["FOREIGN_TABLE"]
        ref_cols  = ", ".join(fk["REFERRED_COLUMNS"])
        fk_line   = f"    FOREIGN KEY ({fk_cols}) REFERENCES {ref_table} ({ref_cols})"

        on_delete = sql_action(fk.get("ON_DELETE"))
        on_update = sql_action(fk.get("ON_UPDATE"))
        if on_delete:
            fk_line += f" ON DELETE {on_delete}"
        if on_update:
            fk_line += f" ON UPDATE {on_update}"

        col_defs.append(fk_line)

    # Table-level CHECK constraints
    for check in tbl_chks:
        col_defs.append(f"    CHECK ({check})")

    body = ",\n".join(col_defs)
    return f"CREATE TABLE {table_name} (\n{body}\n);"


def schema_map_to_ddl(table_map: dict) -> str:
    """Convert a {table_name: table_def, …} dict to a full DDL string."""
    blocks = [build_table_ddl(name, defn) for name, defn in table_map.items()]
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def convert(input_path: str, output_path: str | None = None) -> None:
    with open(input_path, encoding="utf-8") as fh:
        records = json.load(fh)

    if not isinstance(records, list):
        print("Error: expected a JSON array at the top level.", file=sys.stderr)
        sys.exit(1)

    result = []
    for i, record in enumerate(records):
        nl_input = record.get("input", "")

        raw_output = record.get("output", "{}")
        try:
            table_map = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            print(f"Warning: item {i} has invalid JSON in 'output': {exc}",
                  file=sys.stderr)
            table_map = {}

        ddl = schema_map_to_ddl(table_map)
        result.append({"input": nl_input, "output": ddl})

    json_out = json.dumps(result, indent=2, ensure_ascii=False)

    if output_path:
        Path(output_path).write_text(json_out, encoding="utf-8")
        print(f"Written to {output_path}")
    else:
        print(json_out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    in_path  = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    convert(in_path, out_path)
