"""
json_to_sql_ddl.py
==================
Parses a SchemaPile-format JSON file and emits a MySQL-standard SQL DDL script
(CREATE TABLE statements) for every schema it contains.

Usage:
    python json_to_sql_ddl.py input.json [output.sql]

If no output path is given the SQL is printed to stdout.
"""

import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Type mapping — comprehensive MySQL-standard mapping
# ---------------------------------------------------------------------------

TYPE_MAP: dict[str, str] = {
    # --- Integer types ---
    "Int":                  "INT",
    "int":                  "INT",
    "INT":                  "INT",
    "INT4":                 "INT",
    "int4":                 "INT",
    "INT2":                 "SMALLINT",
    "int2":                 "SMALLINT",
    "INT8":                 "BIGINT",
    "int8":                 "BIGINT",
    "Integer":              "INT",
    "INTEGER":              "INT",
    "integer":              "INT",
    "interger":             "INT",       # common typo in the wild
    "UnsignedInt":          "INT UNSIGNED",
    "UnsignedInteger":      "INT UNSIGNED",
    "UNSIGNED":             "INT UNSIGNED",
    "UnsignedBigInt":       "BIGINT UNSIGNED",
    "UnsignedSmallInt":     "SMALLINT UNSIGNED",
    "UnsignedMediumInt":    "MEDIUMINT UNSIGNED",
    "UnsignedTinyInt":      "TINYINT UNSIGNED",
    "SmallInt":             "SMALLINT",
    "smallint":             "SMALLINT",
    "SMALLINT":             "SMALLINT",
    "BigInt":               "BIGINT",
    "bigint":               "BIGINT",
    "BIGINT":               "BIGINT",
    "bitInt":               "BIGINT",    # typo
    "bint":                 "BIGINT",    # typo
    "TinyInt":              "TINYINT",
    "tinyint":              "TINYINT",
    "TINYINT":              "TINYINT",
    "TYNYINT":              "TINYINT",   # typo
    "MediumInt":            "MEDIUMINT",
    "LONGINTEGER":          "BIGINT",
    "SHORTINTEGER":         "SMALLINT",
    "SHORT":                "SMALLINT",
    "PosInteger":           "INT UNSIGNED",
    "positive_int":         "INT UNSIGNED",

    # --- Auto-increment / serial types (map to INT AUTO_INCREMENT) ---
    "serial":               "INT AUTO_INCREMENT",
    "Serial":               "INT AUTO_INCREMENT",
    "SERIAL":               "INT AUTO_INCREMENT",
    "SERIAL2":              "SMALLINT AUTO_INCREMENT",
    "SERIAL4":              "INT AUTO_INCREMENT",
    "serial4":              "INT AUTO_INCREMENT",
    "SERIAL8":              "BIGINT AUTO_INCREMENT",
    "serial8":              "BIGINT AUTO_INCREMENT",
    "BIGSERIAL":            "BIGINT AUTO_INCREMENT",
    "bigserial":            "BIGINT AUTO_INCREMENT",
    "BiGSERIAL":            "BIGINT AUTO_INCREMENT",
    "smallserial":          "SMALLINT AUTO_INCREMENT",
    "SMALLSERIAL":          "SMALLINT AUTO_INCREMENT",
    "IDENTITY":             "INT AUTO_INCREMENT",
    "identity":             "INT AUTO_INCREMENT",
    "auto_increment":       "INT AUTO_INCREMENT",

    # --- Floating point ---
    "Float":                "FLOAT",
    "float":                "FLOAT",  
    "FLOAT":                "FLOAT",
    "FLOAT4":               "FLOAT",
    "float4":               "FLOAT",
    "FLOAT8":               "DOUBLE",
    "float8":               "DOUBLE",
    "Double":               "DOUBLE",
    "DoublePrecision":      "DOUBLE",
    "Real":                 "DOUBLE",
    "REAL":                 "DOUBLE",
    "real":                 "DOUBLE",
    "Dec":                  "DECIMAL",
    "Decimal":              "DECIMAL(10,2)",
    "DECIMAL":              "DECIMAL(10,2)",
    "Numeric":              "DECIMAL(10,2)",
    "NUMERIC":              "DECIMAL(10,2)",
    "NUMBER":               "DECIMAL(10,2)",
    "number":               "DECIMAL(10,2)",
    "FIXED":                "DECIMAL(10,2)",
    "money":                "DECIMAL(15,2)",
    "MONEY":                "DECIMAL(15,2)",
    "smallmoney":           "DECIMAL(6,2)",
    "SMALLMONEY":           "DECIMAL(6,2)",
    "monetary":             "DECIMAL(15,2)",

    # --- String types ---
    "Varchar":              "VARCHAR(255)",
    "VARCHAR":              "VARCHAR(255)",
    "varchar":              "VARCHAR(255)",
    "VARCHAR2":             "VARCHAR(255)",       # Oracle → MySQL
    "Varchar2":             "VARCHAR(255)",
    "varchar2":             "VARCHAR(255)",
    "NVARCHAR2":            "VARCHAR(255)",
    "nvarchar2":            "VARCHAR(255)",
    "Nvarchar":             "VARCHAR(255)",
    "NVARCHAR":             "VARCHAR(255)",        # also handles Nvarchar → VARCHAR
    "nvarchar":             "VARCHAR(255)",
    "CharacterVarying":     "VARCHAR(255)",
    "CharVarying":          "VARCHAR(255)",
    "LONGVARCHAR":          "TEXT",
    "longvarchar":          "TEXT",
    "LVARCHAR":             "TEXT",
    "varcahr":              "VARCHAR(255)",        # typo
    "varcgar":              "VARCHAR(255)",        # typo
    "VARHCAR":              "VARCHAR(255)",        # typo
    "VARINTEGER":           "INT",                # misclassified in source
    "Char":                 "CHAR(1)",
    "CHAR":                 "CHAR(1)",
    "char":                 "CHAR(1)",
    "NCHAR":                "CHAR(1)",
    "nchar":                "CHAR(1)",
    "bpchar":               "CHAR(1)",            # PostgreSQL blank-padded char
    "Character":            "CHAR(1)",
    "String":               "VARCHAR(255)",
    "string":               "VARCHAR(255)",
    "NAME":                 "VARCHAR(255)",
    "name":                 "VARCHAR(255)",
    "sysname":              "VARCHAR(128)",
    "Text":                 "TEXT",
    "TEXT":                 "TEXT",
    "text":                 "TEXT",
    "Clob":                 "LONGTEXT",            # Oracle CLOB → MySQL LONGTEXT
    "NCLOB":                "LONGTEXT",
    "nclob":                "LONGTEXT",
    "mediumtext":           "MEDIUMTEXT",
    "MEDIUMTEXT":           "MEDIUMTEXT",
    "MediumText":           "MEDIUMTEXT",
    "longtext":             "LONGTEXT",
    "LONGTEXT":             "LONGTEXT",
    "Longtext":             "LONGTEXT",
    "tinytext":             "TINYTEXT",
    "TINYTEXT":             "TINYTEXT",
    "NTEXT":                "TEXT",
    "ntext":                "TEXT",
    "LONG":                 "LONGTEXT",            # Oracle LONG → MySQL LONGTEXT
    "long":                 "LONGTEXT",
    "citext":               "TEXT",                # PostgreSQL case-insensitive text → TEXT
    "CITEXT":               "TEXT",
    "xml":                  "TEXT",                # XML stored as TEXT in MySQL
    "XML":                  "TEXT",
    "xmltype":              "TEXT",
    "JSON":                 "JSON",
    "json":                 "JSON",
    "jsonb":                "JSON",                # PostgreSQL JSONB → MySQL JSON
    "JSONB":                "JSON",
    "JSONb":                "JSON",
    "JSON1":                "JSON",

    # --- Date / time ---
    "Date":                 "DATE",
    "DATE":                 "DATE",
    "date":                 "DATE",
    "Timestamp":            "TIMESTAMP",
    "TIMESTAMP":            "TIMESTAMP",
    "timestamp_ntz":        "TIMESTAMP",
    "Datetime":             "DATETIME",
    "DATETIME":             "DATETIME",
    "datetime":             "DATETIME",
    "DATETIME2":            "DATETIME",
    "DateTime2":            "DATETIME",
    "datetime2":            "DATETIME",
    "datetimeoffset":       "DATETIME",
    "DATETIMEOFFSET":       "DATETIME",
    "Time":                 "TIME",
    "TIME":                 "TIME",
    "ddatetime":            "DATETIME",           # typo
    "datatime":             "DATETIME",           # typo
    "smalldatetime":        "DATETIME",
    "SMALLDATETIME":        "DATETIME",
    "year":                 "YEAR",
    "YEAR":                 "YEAR",
    "Interval":             "TIME",                # no direct MySQL equiv, closest is TIME

    # --- Binary types ---
    "Blob":                 "BLOB",
    "BLOB":                 "BLOB",
    "Binary":               "BINARY(255)",
    "BINARY":               "BINARY(255)",
    "Bytea":                "BLOB",               # PostgreSQL BYTEA → BLOB
    "Varbinary":            "VARBINARY(255)",
    "longblob":             "LONGBLOB",
    "LONGBLOB":             "LONGBLOB",
    "mediumblob":           "MEDIUMBLOB",
    "MEDIUMBLOB":           "MEDIUMBLOB",
    "mediunblob":           "MEDIUMBLOB",         # typo
    "tinyblob":             "TINYBLOB",
    "TINYBLOB":             "TINYBLOB",
    "IMAGE":                "LONGBLOB",           # SQL Server IMAGE → LONGBLOB
    "image":                "LONGBLOB",
    "RAW":                  "VARBINARY(255)",      # Oracle RAW → VARBINARY
    "raw":                  "VARBINARY(255)",
    "longvarbinary":        "LONGBLOB",
    "LONGVARBINARY":        "LONGBLOB",
    "BIT":                  "BIT(1)",
    "bit":                  "BIT(1)",
    "Bit":                  "BIT(1)",

    # --- Boolean ---
    "Boolean":              "TINYINT(1)",          # MySQL has no native BOOL, uses TINYINT(1)
    "BOOLEAN":              "TINYINT(1)",
    "bool":                 "TINYINT(1)",
    "BOOL":                 "TINYINT(1)",
    "Bool":                 "TINYINT(1)",
    "BOOl":                 "TINYINT(1)",          # typo

    # --- UUID ---
    "Uuid":                 "CHAR(36)",
    "uuid":                 "CHAR(36)",
    "UUID":                 "CHAR(36)",
    "UNIQUEIDENTIFIER":     "CHAR(36)",            # SQL Server → CHAR(36)
    "uniqueidentifier":     "CHAR(36)",
    "uuid_b64":             "CHAR(36)",

    # --- Enum (treat as VARCHAR; actual ENUM values not available in schema) ---
    "Enum":                 "VARCHAR(64)",
    "ENUM":                 "VARCHAR(64)",
    "Set":                  "VARCHAR(255)",         # MySQL SET → VARCHAR fallback

    # --- PostgreSQL-specific geometric / network types → closest MySQL equiv ---
    "point":                "POINT",
    "POINT":                "POINT",
    "Point":                "POINT",
    "geometry":             "GEOMETRY",
    "GEOMETRY":             "GEOMETRY",
    "GEOGRAPHY":            "GEOMETRY",
    "geography":            "GEOMETRY",
    "MULTIPOLYGON":         "MULTIPOLYGON",
    "multipolygon":         "MULTIPOLYGON",
    "polygon":              "POLYGON",
    "POLYGON":              "POLYGON",
    "linestring":           "LINESTRING",
    "LINE":                 "LINESTRING",
    "line":                 "LINESTRING",
    "LSEG":                 "LINESTRING",
    "lseg":                 "LINESTRING",
    "BOX":                  "POLYGON",
    "box":                  "POLYGON",
    "CIRCLE":               "POLYGON",
    "circle":               "POLYGON",
    "PATH":                 "LINESTRING",
    "path":                 "LINESTRING",
    "inet":                 "VARCHAR(45)",          # IP address → VARCHAR(45) covers IPv6
    "INET":                 "VARCHAR(45)",
    "cidr":                 "VARCHAR(45)",
    "CIDR":                 "VARCHAR(45)",
    "macaddr":              "CHAR(17)",
    "MACADDR":              "CHAR(17)",
    "tsvector":             "TEXT",                 # full-text search vector → TEXT
    "TSVECTOR":             "TEXT",
    "tsquery":              "TEXT",
    "hstore":               "JSON",                 # key-value → JSON
    "HSTORE":               "JSON",
    "Array":                "JSON",                 # arrays → JSON
    "oid":                  "BIGINT UNSIGNED",
    "OID":                  "BIGINT UNSIGNED",
    "int2vector":           "JSON",
    "oidvector":            "JSON",

    # --- Oracle-specific ---
    "ROWID":                "VARCHAR(18)",
    "ANYDATA":              "LONGTEXT",
    "SDO_GEOMETRY":         "GEOMETRY",
    "MDSYS":                "GEOMETRY",
    "ORDSYS":               "LONGBLOB",
    "BINARY_FLOAT":         "FLOAT",
    "BINARY_DOUBLE":        "DOUBLE",
    "ABSTIME":              "DATETIME",
    "DESCRIPTION_2000":     "VARCHAR(2000)",
    "DESCRIPTION_1000":     "VARCHAR(1000)",
    "DESCRIPTION_250":      "VARCHAR(250)",
    "DESCRIPTION_80":       "VARCHAR(80)",
    "TIME_STAMP_DFL":       "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    "TIME_STAMP":           "TIMESTAMP",
    "generic_timestamp":    "TIMESTAMP",
    "generic_timestamp_now":"TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    "generic_timestamp_null":"TIMESTAMP",
    "BOOLEAN_CHAR":         "CHAR(1)",
    "BOOLEAN_CHAR_OR_UNKNOWN": "CHAR(1)",
    "OBJECT_NAME":          "VARCHAR(255)",
    "COLUMN_LABEL":         "VARCHAR(255)",
    "ORDINAL_INT":          "INT",
    "TEXT_VALUE":           "TEXT",
    "FILE_NAME":            "VARCHAR(255)",
    "FILE_PATH":            "VARCHAR(1024)",
    "FILE":                 "VARCHAR(255)",
    "TITLE_100":            "VARCHAR(100)",
    "LONG_LABEL":           "VARCHAR(255)",
    "SHORT_LABEL":          "VARCHAR(100)",
    "TECH_ID":              "BIGINT",
    "IDENTIFIER":           "VARCHAR(255)",
    "CODE":                 "VARCHAR(50)",
    "REAL_VALUE":           "DOUBLE",
    "INTEGER_NUMBER":       "INT",
    "REAL_NUMBER":          "DOUBLE",

    # --- Domain/custom types seen frequently — best-effort mapping ---
    "Locale":               "VARCHAR(10)",
    "Currency":             "VARCHAR(3)",
    "Number":               "DECIMAL(10,2)",
    "Val":                  "VARCHAR(255)",
    "val":                  "VARCHAR(255)",
    "LocalDate":            "DATE",
    "datetimetz":           "DATETIME",
    "datetimeltz":          "DATETIME",
    "tstzrange":            "JSON",                 # range type → JSON
    "int4range":            "JSON",
    "citext":               "TEXT",
}

# Domains/custom types that look like application-level enums.
# These appear as column types in some schemas (e.g. "job_status", "gender_type").
# We map them all to VARCHAR(64) as a safe fallback.
ENUM_LIKE_SUFFIXES = (
    "_status", "_type", "_state", "_role", "_enum", "_kind",
    "_mode", "_action", "_level", "_format", "_scope",
)


def sql_type(abstract_type: str) -> str:
    """Return the MySQL SQL type string for a SchemaPile type name."""
    # Direct lookup first
    mapped = TYPE_MAP.get(abstract_type)
    if mapped:
        return mapped

    # Types that already contain size info e.g. "VARCHAR(150)", "CHAR(6)"
    upper = abstract_type.upper()
    if upper.startswith(("VARCHAR(", "CHAR(", "NVARCHAR(")):
        return abstract_type.upper().replace("NVARCHAR", "VARCHAR")

    # Application-level enum-like types → VARCHAR(64)
    lower = abstract_type.lower()
    if any(lower.endswith(s) for s in ENUM_LIKE_SUFFIXES):
        return f"VARCHAR(64) /* was: {abstract_type} */"

    # Anything still unknown → VARCHAR(255) with a comment so it's visible
    return f"VARCHAR(255) /* unknown type: {abstract_type} */"


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
# Core builders
# ---------------------------------------------------------------------------

def build_column_def(col_name: str, col: dict) -> str:
    """Return the column definition line (without trailing comma)."""
    # Backtick-quote column names to handle reserved words and special chars
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

    # Only add inline UNIQUE if this column is NOT part of a composite PK.
    # (Composite PK columns should not have UNIQUE individually.)
    if col.get("UNIQUE") is True and not col.get("IS_PRIMARY"):
        parts.append("UNIQUE")

    return " ".join(parts)


def build_table_ddl(table_name: str, table: dict) -> str:
    """Return a complete CREATE TABLE … ; statement."""
    col_defs: list[str] = []

    columns    = table.get("COLUMNS", {})
    pk_cols    = table.get("PRIMARY_KEYS", [])
    fk_list    = table.get("FOREIGN_KEYS", [])
    tbl_checks = table.get("CHECKS", [])

    # Strip PostgreSQL schema prefix (e.g. "public.tablename" → "tablename")
    clean_table_name = table_name.split(".")[-1]

    # ---- column definitions ------------------------------------------------
    for col_name, col in columns.items():
        col_defs.append(build_column_def(col_name, col))

    # ---- PRIMARY KEY -------------------------------------------------------
    if pk_cols:
        pk_str = ", ".join(f"{c}" for c in pk_cols)
        col_defs.append(f"    PRIMARY KEY ({pk_str})")

    # ---- FOREIGN KEYS ------------------------------------------------------
    for fk in fk_list:
        fk_cols   = ", ".join(f"{c}" for c in fk["COLUMNS"])
        ref_table = fk["FOREIGN_TABLE"].split(".")[-1]   # strip schema prefix
        ref_cols  = ", ".join(f"{c}" for c in fk["REFERRED_COLUMNS"])

        fk_line = f"    FOREIGN KEY ({fk_cols}) REFERENCES {ref_table} ({ref_cols})"

        on_delete = sql_action(fk.get("ON_DELETE"))
        on_update = sql_action(fk.get("ON_UPDATE"))
        if on_delete:
            fk_line += f" ON DELETE {on_delete}"
        if on_update:
            fk_line += f" ON UPDATE {on_update}"

        col_defs.append(fk_line)

    # ---- CHECK constraints -------------------------------------------------
    for check in tbl_checks:
        col_defs.append(f"    CHECK ({check})")

    # ---- assemble ----------------------------------------------------------
    body = ",\n".join(col_defs)
    return f"CREATE TABLE {clean_table_name} (\n{body}\n);"


def build_schema_ddl(schema: dict) -> str:
    """Return the full DDL block for one schema entry."""
    name     = schema.get("name", "unnamed")
    url      = schema.get("url", "").strip()
    license_ = schema.get("license", "")
    tables   = schema.get("tables", {})

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
        table_blocks.append(build_table_ddl(table_name, table_def))

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
        "-- Auto-generated MySQL DDL",
        f"-- Source file : {Path(input_path).name}",
        f"-- Schemas     : {len(schemas)}",
        "",
    ]

    for schema in schemas:
        blocks.append(build_schema_ddl(schema))
        blocks.append("")

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