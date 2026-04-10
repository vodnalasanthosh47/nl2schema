"""
pipeline.py — End-to-End Schema → SQL Query Pipeline

Receives CREATE TABLE DDL from Model 1 (business logic → schema)
and generates diverse example SQL queries using the trained Model 2.

Usage:
    from pipeline import SchemaToQueryPipeline

    pipe = SchemaToQueryPipeline("schema2sql_model/final")
    queries = pipe.generate("CREATE TABLE users (...); CREATE TABLE orders (...);")
"""

import os
import re
import sqlite3
import tempfile
from inference import (
    load_model,
    generate_queries,
    validate_queries,
    categorize_queries,
)


class SchemaToQueryPipeline:
    """
    Production-ready pipeline that takes DDL from Model 1
    and returns categorized, validated SQL queries.
    """

    @staticmethod
    def normalize_ddl_types(ddl_text):
        """Normalize standard SQL types to Spider's format for model consistency.
        Model 1 may output VARCHAR, INT, DATETIME etc. but our model was trained
        on Spider's type vocabulary (text, number, time). This prevents the model
        from seeing unknown type tokens at inference time."""
        replacements = {
            'VARCHAR': 'text', 'CHAR': 'text', 'STRING': 'text',
            'INTEGER': 'number', 'INT': 'number', 'BIGINT': 'number',
            'FLOAT': 'number', 'DOUBLE': 'number', 'DECIMAL': 'number',
            'BOOLEAN': 'text', 'BOOL': 'text',
            'DATETIME': 'time', 'TIMESTAMP': 'time',
        }
        for sql_type, spider_type in replacements.items():
            ddl_text = re.sub(rf'\b{sql_type}\b', spider_type, ddl_text, flags=re.IGNORECASE)
        # Strip size annotations like VARCHAR(255) → text
        ddl_text = re.sub(r'\b(text|number|time)\s*\(\d+\)', r'\1', ddl_text)
        return ddl_text

    def __init__(self, model_path, max_input_length=1024, max_output_length=256):
        self.model, self.tokenizer, self.device = load_model(model_path)
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def generate(self, ddl_text, num_queries=10, method="diverse_beam",
                 validate=True, categorize=True):
        """
        Generate SQL queries from DDL text.

        Args:
            ddl_text: CREATE TABLE statements (from Model 1)
            num_queries: How many queries to generate
            method: "diverse_beam" or "sampling"
            validate: If True, validate queries by executing against in-memory DB
            categorize: If True, return queries grouped by SQL construct type

        Returns:
            If categorize=True:  dict of {category: [queries]}
            If categorize=False: list of query strings
        """
        # Normalize standard SQL types to Spider's vocabulary
        ddl_text = self.normalize_ddl_types(ddl_text)

        queries = generate_queries(
            ddl_text, self.model, self.tokenizer, self.device,
            num_queries=num_queries,
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
            method=method,
        )

        if validate:
            queries = self._validate_against_schema(queries, ddl_text)

        if categorize:
            return categorize_queries(queries)
        return queries

    def _validate_against_schema(self, queries, ddl_text):
        """
        Create a temporary in-memory SQLite DB from the DDL,
        then validate each query by executing it.
        Only returns queries that execute without error.
        """
        try:
            conn = sqlite3.connect(":memory:")
            # Execute DDL to create the schema
            for statement in ddl_text.split(";"):
                statement = statement.strip()
                if statement:
                    try:
                        conn.execute(statement + ";")
                    except sqlite3.OperationalError:
                        pass  # Some DDL features may not be supported

            # Test each query
            valid = []
            for q in queries:
                try:
                    conn.execute(q)
                    valid.append(q)
                except (sqlite3.OperationalError, sqlite3.DatabaseError):
                    pass

            conn.close()
            return valid
        except Exception:
            return queries  # If schema creation fails, return all


def main():
    """Demo usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Schema→SQL Pipeline Demo")
    parser.add_argument("--model", required=True)
    parser.add_argument("--ddl", default=None, help="Path to DDL file")
    args = parser.parse_args()

    pipe = SchemaToQueryPipeline(args.model)

    if args.ddl:
        with open(args.ddl) as f:
            ddl = f.read()
    else:
        # Demo DDL
        ddl = """
CREATE TABLE students (
  student_id INTEGER PRIMARY KEY,
  name TEXT,
  age INTEGER,
  major TEXT,
  gpa REAL
);

CREATE TABLE courses (
  course_id INTEGER PRIMARY KEY,
  course_name TEXT,
  department TEXT,
  credits INTEGER
);

CREATE TABLE enrollments (
  enrollment_id INTEGER PRIMARY KEY,
  student_id INTEGER REFERENCES students(student_id),
  course_id INTEGER REFERENCES courses(course_id),
  grade TEXT,
  semester TEXT
);
"""

    print("📋 Schema:")
    print(ddl)

    print("\n🔮 Generating queries...\n")
    result = pipe.generate(ddl, num_queries=10, validate=True, categorize=True)

    for category, queries in result.items():
        print(f"  [{category}]")
        for q in queries:
            print(f"    → {q}")
        print()

    # Also show raw list
    flat = pipe.generate(ddl, num_queries=10, validate=True, categorize=False)
    print(f"Total valid queries: {len(flat)}")


if __name__ == "__main__":
    main()
