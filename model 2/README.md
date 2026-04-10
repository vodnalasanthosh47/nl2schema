# Model 2: Unconditional Schema-to-SQL Generation Database Pipeline

Welcome to Model 2! This project re-frames the traditional Spider NLP task (translating natural language to SQL) into a purely **Unconditional Schema-to-SQL Generation** task. 

This model's sole objective is to autonomously output a batch of diverse, structurally coherent, and executable SQL queries given *only* the structural definition of a database, essentially acting as an autonomous SQL generation engine.

---

## 1. Data Hijacking & Preprocessing (`preprocess.py`)

The original Spider dataset was constructed for NLP tasks and formatted as: `Natural Language Question -> SQL Query`. Furthermore, historical schemas were stored in complex JSON graph files (`tables.json`).

**How we construct the Data:**
We completely discard all natural language questions. Instead, our pipeline dynamically reconstructs `CREATE TABLE` DDL fragments out of `tables.json` and forcefully maps them directly to the original SQL queries. We essentially repurposed the entire dataset into a blueprint generation system.

### Preprocessing Safeguards:
- **Table-Boundary Truncation:** Deep learning models, specifically CodeT5, possess a strict 1024 token limit. Naïve string slicing corrupts the schema. Our algorithm iteratively adds whole tables and stops strictly prior to breaching the ~1024 budget. Only pristine, contiguous schemas hit the model.
- **Name-Masking Augmentation:** A massive caveat of pre-trained language models is that they aggressively leverage semantic patterns (e.g. knowing that `salaries` associates with `employees`). To force structural topological learning, we implemented a 40% augmentation routine that obfuscates real schema components with placeholders (`table_0`, `col_1`). Consequently, the model must understand explicit connections (`FOREIGN KEY`) instead of guessing via English.

---

## 2. Input/Output Structure in Training (`dataset.py`)

During training, each instance fundamentally looks like this:

**Input (DDL Schema with prompt constraints):**
```text
Generate a SQL query for this database:
CREATE TABLE departmental_head (
  head_id number PRIMARY KEY,
  name text,
  born_state text,
  age number
);

CREATE TABLE department (
  department_id number PRIMARY KEY,
  name text,
  creation text,
  ranking number,
  budget_in_billions number,
  num_employees number
);
```

**Target Output (Synthesized SQL):**
```sql
SELECT name, age FROM departmental_head WHERE age > 50
```

---

## 3. Model Architecture & Hyperparameters (`train.py` & `config.yaml`)

We chose **Salesforce/codet5-base** due to its specific *Identifier-Aware Denoising* pre-training, giving it superior mathematical tracking of `table.column_id` paths across deep queries.

**Optimized Hyperparameters:**
- **Epochs:** `5` (To combat fast overfitting inherent with schema abstraction sets).
- **Learning Rate:** `5e-5` (To prevent Catastrophic Forgetting mapping geometric logic). 
- **Warmup Ratio:** `0.1` (Prevents extreme gradient turbulence from shattering learned syntax during early batches).
- **Batching Strategy:** `Batch Size 8, Gradient Accumulation 2` (Smoothly manages memory trajectories while effectively hitting larger theoretical batches). 

---

## 4. Evaluation via Generative Robustness (`evaluate.py`)

Because the model unconditionally imagines its own queries, there is definitively no pre-determined "right answer" to exact-match against. Attempting to grade the model against a static SQL string from the original NLP dataset is mathematically incompatible. 

We threw out traditional evaluation tracking in favor of compiling the actual queries via **Execution Validation**.

**How Evaluation works in this pipeline:**
1. **Virtual Sandbox Generation:** The Python pipeline spins up an isolated, ephemerally hosted `sqlite3` DBMS entirely stored in memory space.
2. **Schema Injection:** It literally executes the underlying database schemas, injecting `CREATE TABLE` rules into the SQLite instance.
3. **Execution Trial:** The freshly synthesized/hallucinated SQL query string is force-executed against the empty framework.
4. **Pass/Fail Logging:** 
    - The model earns a **Pass** if the query is structurally and syntactically flawless (e.g. types match, joins refer to valid edges, and syntax compiles correctly—even if resulting rows are 0).
    - If SQLite traps an `OperationalError` (e.g. hallucinating a non-existent column, utilizing a broken syntax syntax tree), the generation fails.

Additionally, to prevent the architecture from exclusively generating cheap `SELECT * FROM x` queries to farm easy validity points, we split generations utilizing a Diverse Beam Search and score the system against mathematical Relational Algebra limits (specifically checking what fraction of 19 distinct operations like `MIN`/`MAX`/`JOIN` the model attempts to invoke per database).
