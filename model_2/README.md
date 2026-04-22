# 🧠 Pipeline Architecture: Unconditional Schema-to-SQL Generation (Qwen2.5-Coder)

Welcome to the cutting-edge of automated database engineering. Most NLP models treat SQL generation as a simple translation task: *Language in ➔ SQL out*.

This repository **re-frames the problem.** We built an **Unconditional SQL Synthesis Engine**. Given absolutely nothing but the structural blueprints of a database (the `CREATE TABLE` schemas), this model autonomously hallucinates batches of diverse, perfectly executable SQL queries. 

In Part 2 of this pipeline, we migrated our architecture entirely to causal LLMs, leveraging the deep reasoning capabilities of **Qwen2.5-Coder-1.5B**. 

Below is an exhaustive, technical breakdown of the pipeline's preprocessing, token mapping, single-epoch training dynamics, and execution-validation protocol.

---

## 1. Data Hijacking & Preprocessing (`preprocess.py`)

The original [Spider Dataset](https://yale-lily.github.io/spider) is massive but deeply coupled to Natural Language (NL) questions. To create an unconditional generator, we fundamentally "hijacked" the dataset, discarding all NL constraints. 

Instead of teaching the model to map English nouns to SQL tables, we force it to mathematically map **topological edges** (`FOREIGN KEY`, `PRIMARY KEY`) directly into Relational Algebra (`JOIN`, `GROUP BY`).

### The Regularization Problem (Name-Masking Augmentation)

Pre-trained Language Models (PLMs) are notoriously lazy. If they see a table named `employees` and a table named `salaries`, they will guess how to `JOIN` them based on pre-trained English semantic context rather than reading the explicit `REFERENCES` syntax in the schema definition.

**Our Solution: 40% Name-Masking Ablation.**
We randomly select 40% of our training batches and programmatically obliterate all English semantics before they enter the tokenizer.
*   **The Algorithm:** We programmatically map realistic table names (`employees`) to `table_0`, `table_1`, etc., and column names to `col_0`, `col_1`. 
*   **Length-Descending Safeguard:** We sort the string replacement map by descending length. This is a critical engineering safeguard to prevent catastrophic partial-match bugs (e.g., replacing "id" inside "hidden_id" and breaking the string).
*   **Mathematical Impact:** The model is no longer allowed to cheat by knowing what an "employee" is. It is forced to mathematically trace `col_0` through the database to figure out valid query syntax. This vastly improves its zero-shot performance on unseen enterprise schemas.

### Token-Aware Table-Boundary Truncation

Transformers have rigid maximum memory limits.
*   **The Naïve Approach:** Chop the string at a fixed max length. This violently slices `CREATE TABLE` blocks in half, inherently corrupting the Abstract Syntax Tree (AST) grammar that the model needs to learn.
*   **Our Approach:** We deploy an iterative **Table-Boundary Truncation algorithm**. It measures token lengths using the actual HuggingFace tokenizer `.encode()`, tracking cumulative length. If appending the next `CREATE TABLE` block breaches the token budget (leaving exact headroom for the prompt overhead), the script halts and outputs the finalized schema subset. The model guarantees it only ever trains on syntactically pristine data.

---

## 2. Input/Output Pair Construction (`dataset.py` & `train_qwen.py`)

To instruct the Causal LLM to act as an unconditional query synthesizer, we rigorously construct the training matrices:

1.  **The Input Schema Prompt:** Every single query pair begins with the exact prefix: `Generate a SQL query for this database:\n{schema_text}`
2.  **The Structural Separator:** Between the schema definition and the target SQL output, we inject a rigid separator: `\n-- SQL QUERY --\n`.
3.  **The Autoregressive Assembly:** The full sequence is assembled as: `prompt_ids` + `separator_ids` + `output_ids` + `eos_token_id`.

### The Optimization Key: Causal Loss Masking

When fine-tuning a causal language model, standard next-token-prediction computes the gradient loss over the *entire* text sequence. If we did this, the model would waste massive parameter capacity learning how to recite `CREATE TABLE` schemas.

Instead, we initialize our sequence `labels` array with the `-100` ignore index for the exact length of the prompt and separator:
```python
labels = [-100] * len(prompt_ids) + output_ids
```
**Result:** Backpropagation gradients are calculated **exclusively** on the hallucinated SQL tokens. The model treats the schema DDL as static contextual conditioning and learns strictly how to synthesize complex Relational Algebra from it.

---

## 3. Model Architecture & LoRA Training (`train_qwen.py`)

To achieve state-of-the-art structural generation, we selected **`Qwen/Qwen2.5-Coder-1.5B`**. This model possesses profound reasoning over code blocks and variable tracking across massive contexts.

### The Training Execution: 1 Epoch

We executed the training pipeline for **exactly 1 Epoch**. 
*   **Why only 1 Epoch?** PLMs applied to highly constrained, synthetic datasets (like abstracted topological SQL trees) overfit incredibly fast. Fine-tuning for multiple epochs immediately triggers Catastrophic Forgetting, completely shattering the massive geometric representation of code syntax the model spent millions of computation hours building. One epoch acts as the optimal domain adaptation layer.

### Parameter-Efficient Fine-Tuning (PEFT)
Full-parameter fine-tuning of a 1.5 Billion parameter model is computationally intractable on consumer hardware. We utilized **LoRA (Low-Rank Adaptation)**:
*   **Rank (r):** `16`
*   **Alpha:** `32`
*   **Dropout:** `0.05`
*   **Target Modules:** We injected LoRA adapters comprehensively across *all* dense projection layers in the attention and MLP manifolds (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

### Hyperparameters & System Constraints
*   **Optimizer:** `adamw_torch` mapped with a Cosine Learning Rate Scheduler.
*   **Learning Rate:** `1e-4` with a Warmup Ratio of `0.03` to protect pre-trained weights from errant, aggressive gradient updates on step 1.
*   **Batch Physics:** We use `Batch Size = 1` combined with `Gradient Accumulation Steps = 8` to simulate an effective batch size of 8, allowing continuous smooth gradient descent on memory-constrained GPUs without triggering Out-Of-Memory (OOM) halts.
*   **Collator Padding:** We deployed a custom `CompletionCollator` that pads dynamically to the longest sequence in the micro-batch, enforcing the `-100` ignore index on all padding tokens dynamically.

---

## 4. Evaluation: The Virtual Sandbox (`evaluate_qwen.py`)

In traditional NLP, a model is scored using **Exact Match**. If it guesses exactly what the human wrote, it gets a point.

**The Paradigm Shift:** Since our model is unconditionally dreaming up queries out of thin air, there is no "ground truth" to compare against. A model might hallucinate a brilliant, highly complex 3-level `JOIN` query that technically works perfectly, but would fail standard NLP tests because it didn't "match the answer key".

We built a custom **Generative Robustness Paradigm** that evaluates organic execution compiling:

### The Ephemeral Test Protocol

1.  **RAM Sandboxing:** For every hallucinated query, `evaluate_qwen.py` spawns a clean `sqlite3` database engine entirely inside system RAM.
2.  **State Injection:** The serialized DDL blueprints (from our schema generation logic) are forcefully executed into the sandbox to instantiate the table constraints.
3.  **The Generation Phase:** The model utilizes autoregressive causal decoding to generate multiple queries down its sampling beam (`temperature=0.8`, `top_p=0.95`, `do_sample=True`). 
4.  **Token Stripping:** The code dynamically searches for the exact separator `\n-- SQL QUERY --\n` to slice the prompt tokens off and isolates the raw SQL string.
5.  **The Compilation Trial:** The raw SQL query is piped through the SQLite instance.
    *   **Fail Condition (0):** If it crashes (e.g., hallucinated columns, `sqlite3.OperationalError`, fractured syntax strings, illegal grouping semantics), the model organically fails.
    *   **Pass Condition (1):** If the query safely compiles and executes—returning a dataset or even an empty array (since there are no data rows yet)—the AST syntax and structural graph joins are proven mathematically viable. 

### Advanced Structural Metrics

To prevent the model from cheating the compilation protocol by spamming identically cheap `SELECT * FROM table_1` queries, the evaluator tracks advanced structural physics:
*   **Diversity Scoring:** The engine indexes over 19 distinct Relational Algebra commands (`MIN`, `MAX`, `INTERSECT`, `UNION`, `LIMIT`, `HAVING`, etc.). It computes the exact fractional coverage of the total syntax dictionary the model exercises per schema.
*   **Uniqueness Rate:** Calculates a mathematical collision score on deduplicated vs. total queries generated down the beam to ensure highly variance structural synthesis.

---

### 💡 The End Result
By architecting a seamless pipeline merging strict topological truncation, aggressive 40% semantic name-masking ablations, Causal sequence loss-masking (`-100`), single-epoch LoRA adaptation, and rigorous compile-time execution sandboxing, we successfully transformed an ordinary NLP translation system into a fully autonomous, highly dynamic relational database intelligence.
