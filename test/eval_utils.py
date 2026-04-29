import sqlglot
import mysql.connector
import uuid
import re


############################################
# CONFIG
############################################

MYSQL_CONFIG = {
    "host": "localhost",
    "user": "eval_user",
    "password": "password"
}


############################################
# CLEANING
############################################

import re

def clean_sql(output: str) -> str:
    if not output:
        return ""

    # Remove markdown fences
    output = output.replace("```sql", "").replace("```", "")

    # Find all CREATE / ALTER statements ending with ;
    pattern = r"(CREATE\s+TABLE.*?;|ALTER\s+TABLE.*?;)"
    matches = re.findall(pattern, output, flags=re.IGNORECASE | re.DOTALL)

    if matches:
        return "\n\n".join(matches).strip()

    # fallback: cut everything after last semicolon
    if ";" in output:
        return output[:output.rfind(";") + 1].strip()

    return output.strip()


############################################
# PARSE CHECK
############################################

def is_valid_sql(ddl: str) -> bool:
    try:
        sqlglot.parse(ddl, dialect="mysql")
        return True
    except:
        return False


############################################
# MYSQL EXECUTION
############################################

def executes_successfully_mysql(ddl: str) -> bool:
    db_name = f"eval_{uuid.uuid4().hex[:8]}"

    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()

        cursor.execute(f"CREATE DATABASE {db_name}")
        cursor.execute(f"USE {db_name}")

        statements = [s.strip() for s in ddl.split(";") if s.strip()]

        for stmt in statements:
            cursor.execute(stmt)

        conn.commit()

        cursor.execute(f"DROP DATABASE {db_name}")
        cursor.close()
        conn.close()

        return True

    except:
        try:
            cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
        except:
            pass
        return False


############################################
# SCHEMA MATCH (your definition)
############################################

def extract_schema_counts(ddl: str):
    """
    Returns:
    {
        table_name: column_count
    }
    """
    result = {}

    try:
        parsed = sqlglot.parse(ddl, dialect="mysql")

        for stmt in parsed:
            if stmt and stmt.key == "CREATE":
                table = stmt.find("Table")
                if not table:
                    continue

                table_name = table.name.lower()

                columns = list(stmt.find_all("ColumnDef"))
                result[table_name] = len(columns)

    except:
        return None

    return result


def schema_match(reference: str, generated: str) -> bool:
    ref = extract_schema_counts(reference)
    gen = extract_schema_counts(generated)

    if ref is None or gen is None:
        return False

    if len(ref) != len(gen):
        return False

    ref_counts = sorted(ref.values())
    gen_counts = sorted(gen.values())

    return ref_counts == gen_counts