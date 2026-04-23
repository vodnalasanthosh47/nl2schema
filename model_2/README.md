# 🧠 Model 2: Unconditional Schema-to-SQL Generation (Qwen2.5-Coder)

## Problem Formulation

Most NLP models treat SQL generation as a simple translation task: *Natural Language Question in → SQL out*.

This repository **re-frames the problem** into **Unconditional Schema-to-SQL Generation**. Given nothing but the structural blueprints of a database (`CREATE TABLE` DDL statements), this model autonomously generates batches of diverse, executable SQL queries — no natural language question is provided.

The system leverages the code-reasoning capabilities of **Qwen2.5-Coder-1.5B**, fine-tuned with **LoRA** on the Spider dataset.

---

## 1. Data Preprocessing (`preprocess.py`)

The [Spider Dataset](https://yale-lily.github.io/spider) is a large-scale NL-to-SQL benchmark. To create an unconditional generator, we discard all NL questions entirely, keeping only the schema → SQL query pairs. The model learns to map structural relationships (`FOREIGN KEY`, `PRIMARY KEY`) directly into relational algebra (`JOIN`, `GROUP BY`).

### Schema Serialization

Spider stores schemas as JSON graphs with cross-referencing arrays (`table_names_original`, `column_names_original`, `foreign_keys`). The `preprocess.py` pipeline serializes these into `CREATE TABLE` DDL statements, explicitly preserving `PRIMARY KEY` and `REFERENCES` annotations so the transformer's attention mechanism can learn relational edges.

### Name-Masking Augmentation (40%)

Pre-trained Language Models exploit semantic clues — if they see tables named `employees` and `salaries`, they guess the `JOIN` from English meaning rather than reading the explicit `REFERENCES` syntax.

**Solution:** 40% of training pairs have all identifiers algorithmically replaced with generic placeholders (`table_0`, `col_1`). Replacements are sorted by descending string length to prevent partial-match corruption (e.g., replacing "id" inside "student_id"). This forces the model to trace structural relationships rather than relying on name semantics, improving zero-shot generalization on unseen schemas.

### Token-Aware Table-Boundary Truncation

Transformers have rigid maximum sequence lengths. Rather than naively truncating text (which slices `CREATE TABLE` blocks mid-statement, corrupting SQL grammar), we use an iterative truncation algorithm:

1. Measure each table's token length using the tokenizer's `.encode()` function
2. Add whole tables until the cumulative length approaches the budget (max tokens − 50 prompt overhead)
3. Stop before exceeding the limit — the model only ever trains on syntactically complete DDL

---

## 2. Training Pipeline (`train_qwen.py`)

### Input/Output Construction

The full training sequence is assembled as:
```
prompt_ids + separator_ids (\n-- SQL QUERY --\n) + output_ids + eos_token
```

Every pair begins with the prefix: `Generate a SQL query for this database:\n{schema_text}`

### Causal Loss Masking

Standard next-token-prediction computes loss over the entire sequence. This wastes model capacity on memorizing input schemas. Instead, labels for the prompt and separator tokens are set to `-100` (PyTorch ignore index):

```python
labels = [-100] * len(prompt_ids) + output_ids
```

Backpropagation gradients update **only** on the SQL output tokens. The schema DDL acts as static conditioning context.

### Model Architecture & LoRA

We fine-tune **`Qwen/Qwen2.5-Coder-1.5B`** using LoRA (Low-Rank Adaptation):

| Parameter | Value | Rationale |
|---|---|---|
| LoRA Rank (r) | 16 | Sufficient capacity without overfitting |
| Alpha (α) | 32 | Standard 2× rank scaling |
| Dropout | 0.05 | Light regularization |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | All dense layers in attention + MLP |

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | `adamw_torch` + Cosine LR schedule |
| Learning Rate | `1e-4` with warmup ratio `0.03` |
| Batch Size | 1 (with 8 gradient accumulation steps → effective batch size 8) |
| Precision | `bfloat16` (CUDA) or `float16` fallback |
| Gradient Checkpointing | Enabled |
| Max Prompt Length | 768 tokens |
| Max Output Length | 256 tokens |

A custom `CompletionCollator` dynamically pads each micro-batch to the longest sequence, applying `-100` to all padding positions.

---

## 3. Evaluation Framework (`evaluate_qwen.py`)

Since the model generates queries unconditionally (no ground-truth query to compare against), traditional Exact Match scoring is inapplicable. We built a custom **Generative Robustness** framework:

### Execution Validity
Each generated query is executed against the corresponding Spider SQLite database file (which contains the full schema and sample data rows). If `sqlite3` raises an `OperationalError` or `DatabaseError` (e.g., hallucinated column, broken syntax, illegal join), the query fails. Otherwise it passes.

### Diversity Score
Indexes 19 distinct SQL constructs (`JOIN`, `GROUP BY`, `HAVING`, `UNION`, `INTERSECT`, `BETWEEN`, `LIMIT`, etc.) and computes fractional coverage per schema. This prevents trivially inflating validity by spamming `SELECT * FROM table_1`.

### Uniqueness Rate
Measures the ratio of distinct queries (case-normalized deduplication) to total queries generated, ensuring the model doesn't suffer from mode collapse.

---

## 4. Inference (`inference_qwen.py`)

At inference time, the model uses nucleus sampling (`temperature=0.7`, `top_p=0.9`) for diverse generation. A retry loop (`generate_compilable_queries`) iteratively:
1. Generates batches of candidate queries
2. Validates each against the target SQLite database using `EXPLAIN QUERY PLAN`
3. Collects compilable queries up to the requested count or attempt budget

---

## 5. Results

Evaluated on **40 unseen schemas** from the Spider test set (verified zero overlap with training data). 10 queries generated per schema (400 total).

| Model | Execution Validity | Diversity | Uniqueness |
|---|---|---|---|
| Base Qwen2.5-Coder-1.5B (no fine-tuning) | 65.5% (262/400) | 26.4% | 91.7% |
| **Fine-tuned (LoRA)** | **90.5% (362/400)** | **45.9%** | **96.8%** |
| **Improvement** | **+25.0%** | **+19.5%** | **+5.1%** |

The prior CodeT5-base prototype achieved 55.9% validity. The migration to Qwen2.5-Coder represents a **+34.6% absolute improvement**.

---

## File Structure

```
model_2/
├── preprocess.py          # Spider → Schema-SQL pairs (DDL serialization, name-masking, truncation)
├── train_qwen.py          # LoRA fine-tuning of Qwen2.5-Coder-1.5B
├── evaluate_qwen.py       # Generative robustness evaluation (validity, diversity, uniqueness)
├── inference_qwen.py      # Interactive inference with retry-based compilation validation
├── dataset.py             # PyTorch Dataset class (used by CodeT5 baseline)
├── train.py               # CodeT5 baseline training script
├── evaluate.py            # CodeT5 baseline evaluation script
├── inference.py           # CodeT5 baseline inference script
├── pipeline.py            # End-to-end Model1→Model2 pipeline connector
├── config.yaml            # Configuration (paths, hyperparameters)
├── requirements.txt       # Python dependencies
├── technical_report.md    # Detailed technical report
└── README.md              # This file
```
