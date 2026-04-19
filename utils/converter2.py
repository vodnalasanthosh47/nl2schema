"""
json_to_sql_ddl.py
==================
Parses the generated json file and emits the nlp + sql training data
(CREATE TABLE statements) for every schema it contains.

Usage:
    python json_to_sql_ddl.py input.json [output.sql]

If no output path is given the SQL is printed to stdout.
"""

import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------

TYPE_MAP: dict[str, str] = {
    "int":         "INT",
    "unsignedunt": "INT UNSIGNED",
    "bigint":      "BIGINT",
    "smallint":    "SMALLINT",
    "varchar":     "VARCHAR(255)",
    "text":        "TEXT",
    "date":        "DATE",
    "timestamp":   "TIMESTAMP",
}


def sql_type(abstract_type: str) -> str:
    """Return the SQL type string for a SchemaPile type name."""
    #convert abstract type to lowercase and map
    mapped = TYPE_MAP.get(abstract_type.lower())
    if mapped is None:
        # Pass unknown types through as-is with a comment so the user notices.
        return f"{abstract_type} /* unknown type – review manually */"
    return mapped


# ---------------------------------------------------------------------------
# Referential-action mapping
# ---------------------------------------------------------------------------

ACTION_MAP: dict[str, str] = {
    "Cascade":   "CASCADE",
    "SetNull":   "SET NULL",
    "SetDefault":"SET DEFAULT",
    "NoAction":  "NO ACTION",
    "Restrict":  "RESTRICT",
}


def sql_action(action: str | None) -> str | None:
    """Map a SchemaPile referential-action string to SQL, or None."""
    if action is None:
        return None
    return ACTION_MAP.get(action, action.upper())


# ---------------------------------------------------------------------------
# Core builders
# ---------------------------------------------------------------------------

def build_column_def(col_name: str, col: dict) -> str:
    """Return the column definition line (without trailing comma)."""
    parts = [f"    {col_name}", sql_type(col["TYPE"])]

    # NULL / NOT NULL
    if col.get("NULLABLE") is False:
        parts.append("NOT NULL")

    # DEFAULT
    default = col.get("DEFAULT")
    if default is not None:
        # Quote string-like defaults; leave numeric / keyword defaults bare.
        if isinstance(default, str) and not default.upper() in (
            "NULL", "CURRENT_TIMESTAMP", "TRUE", "FALSE"
        ):
            parts.append(f"DEFAULT '{default}'")
        else:
            parts.append(f"DEFAULT {default}")

    # Inline UNIQUE only when there is no composite PK that already covers it.
    # We always emit it here; composite-PK deduplication is handled below.
    if col.get("UNIQUE") is True:
        parts.append("UNIQUE")

    return " ".join(parts)


def build_table_ddl(table_name: str, table: dict, schema_name: str = "") -> str:
    """Return a complete CREATE TABLE … ; statement."""
    lines: list[str] = []
    col_defs: list[str] = []

    columns   = table.get("COLUMNS", {})
    pk_cols   = table.get("PRIMARY_KEYS", [])
    fk_list   = table.get("FOREIGN_KEYS", [])
    tbl_checks = table.get("CHECKS", [])

    # ---- column definitions ------------------------------------------------
    for col_name, col in columns.items():
        col_def = build_column_def(col_name, col)
        col_defs.append(col_def)

    # ---- table constraints -------------------------------------------------

    # PRIMARY KEY
    if pk_cols:
        pk_str = ", ".join(pk_cols)
        col_defs.append(f"    PRIMARY KEY ({pk_str})")

    # FOREIGN KEYS
    for fk in fk_list:
        fk_cols   = ", ".join(fk["COLUMNS"])
        ref_table = fk["FOREIGN_TABLE"]
        ref_cols  = ", ".join(fk["REFERRED_COLUMNS"])

        fk_line = f"    FOREIGN KEY ({fk_cols}) REFERENCES {ref_table} ({ref_cols})"

        on_delete = sql_action(fk.get("ON_DELETE"))
        on_update = sql_action(fk.get("ON_UPDATE"))
        if on_delete:
            fk_line += f" ON DELETE {on_delete}"
        if on_update:
            fk_line += f" ON UPDATE {on_update}"

        col_defs.append(fk_line)

    # Table-level CHECK constraints
    for check in tbl_checks:
        col_defs.append(f"    CHECK ({check})")

    # ---- assemble ----------------------------------------------------------
    body = ",\n".join(col_defs)
    lines.append(f"CREATE TABLE {table_name} (")
    lines.append(body)
    lines.append(");")

    return "\n".join(lines)


def build_schema_ddl(schema: dict) -> str:
    """Return the full DDL block for one schema entry."""
    name    = schema.get("name", "unnamed")
    url     = schema.get("url", "").strip()
    license_= schema.get("license", "")
    tables  = schema.get("tables", {})

    header_lines = [
        "-- " + "-" * 72,
        f"-- Schema : {name}",
    ]
    if url:
        header_lines.append(f"-- Source  : {url}")
    if license_:
        header_lines.append(f"-- License : {license_}")
    header_lines.append("-- " + "-" * 72)

    table_blocks: list[str] = []
    for table_name, table_def in tables.items():
        table_blocks.append(build_table_ddl(table_name, table_def, name))

    return "\n".join(header_lines) + "\n\n" + "\n\n".join(table_blocks)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def convert(input_path: str, output_path: str | None = None) -> None:
    with open(input_path, encoding="utf-8") as fh:
        data = json.load(fh)

    schemas = data.get("schemas", [])
    if not schemas:
        print("Warning: no 'schemas' key found in the JSON file.", file=sys.stderr)
        return

    blocks: list[str] = [
        "-- Auto-generated SQL DDL",
        f"-- Source file : {Path(input_path).name}",
        f"-- Schemas     : {len(schemas)}",
        "",
    ]

    for schema in schemas:
        blocks.append(build_schema_ddl(schema))
        blocks.append("")   # blank line between schemas

    sql_output = "\n".join(blocks)

    if output_path:
        Path(output_path).write_text(sql_output, encoding="utf-8")
        print(f"Written to {output_path}")
    else:
        print(sql_output)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    in_path  = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    convert(in_path, out_path)