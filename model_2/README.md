# 🧠 Pipeline Architecture: Unconditional Schema-to-SQL Generation

Welcome to the cutting-edge of automated database engineering. Most NLP models treat SQL generation as a simple translation task: *Language in ➔ SQL out*.

This repository **re-frames the problem.** We built an **Unconditional SQL Synthesis Engine**. Given absolutely nothing but the structural blueprints of a database (the `CREATE TABLE` schemas), this model autonomously hallucinates batches of diverse, perfectly executable SQL queries.

It doesn’t translate. It *invents*.

---

## 🏗️ High-Level System Architecture

How do you teach a machine to dream in SQL? You rip out its English constraints, feed it raw database scaffolding, and rigorously score it in an ephemeral sandbox.

```mermaid
graph TD
    A[Spider JSON Database] -->|Discard NL Questions| B(Extraction Engine)
    B --> C{Name-Masking Augmentation}
    C -->|40% Obfuscated| D[Tokenizer Truncation]
    C -->|60% Standard| D
    D --> E((Causal / Seq2Seq Synthesizer))
    
    E -->|Tokens| F[Autoregressive Decoder]
    F -->|Generated SQL Query| G[SQLite Ephemeral Sandbox]
    
    H[(In-Memory Database)] -. Schema Injection .-> G
    G -->|Execution Trial| I{Compilation Status}
    
    I -->|DatabaseError| J[❌ Invalid]
    I -->|Successful Return| K[✅ Valid Query]
```

---

## 1. Data Hijacking & Preprocessing (`preprocess.py`)

The original [Spider Dataset](https://yale-lily.github.io/spider) is massive but deeply coupled to Natural Language questions. To create an unconditional generator, we had to "hijack" the dataset. 

Instead of teaching the model to map English nouns to SQL tables, we fundamentally force it to map **topological edges** (`FOREIGN KEY`, `PRIMARY KEY`) into Relational Algebra (`JOIN`, `GROUP BY`).

### The Regularization Problem (Name-Masking)

Pre-trained Language Models (PLMs) are notoriously lazy. If they see a table named `employees` and a table named `salaries`, they will guess how to `JOIN` them based on pre-trained English context rather than reading the explicit `REFERENCES` syntax in your schema.

**Our Solution: 40% Name-Masking Ablation.**
We randomly select 40% of our training batches and programmatically obliterate all English semantics before passing them to the model.

```diff
- CREATE TABLE departmental_head (head_id PRIMARY KEY, name text)
+ CREATE TABLE table_0 (col_0 PRIMARY KEY, col_1 text)
```

**Why is this brilliant?** The model isn't allowed to cheat by knowing what an "employee" is. It is forced to mathematically trace `col_0` through the database to figure out valid query syntax. This vastly improves its zero-shot performance on entirely unseen database concepts.

### Token-Aware Truncation
Transformers have rigid memory limits. Many enterprise databases far exceed token limits.
*   **The Naïve Way:** Chop the string at a fixed character length. This violently slices `CREATE TABLE` blocks in half, corrupting the Abstract Syntax Tree (AST).
*   **Our Way:** We deploy an iterative **Table-Boundary Truncation algorithm**. We use the actual Huggingface model tokenizer to count subsets of schemas, appending whole tables until we are just under the budget. Only pristine, contiguous schema chunks are fed to the model.

---

## 2. Model Architecture & Training 

We progressed our architecture from Seq2Seq (CodeT5) to state-of-the-art Causal LLMs (**Qwen2.5-Coder-1.5B**).

### Legacy: CodeT5 Synthesizer (`train.py`)
Initially, we used **`Salesforce/codet5-base`** (220M parameters), pre-trained on Identifier-Aware Denoising. We optimized it using a learning rate of `5e-5` to avoid Catastrophic Forgetting, with an effective batch size of 16 (Gradient Accumulation = 2, Batch Size = 8).

### Current State-of-the-Art: Qwen2.5-Coder (`train_qwen.py`)
To dramatically enhance the structural coherence and complexity of generated SQL, we migrated to **`Qwen/Qwen2.5-Coder-1.5B`**. This model possesses profound reasoning over code blocks and deep context tracking.

**Qwen Training Details:**
*   **Task Formulation (Causal Masking):** We treat schema-to-SQL as a causal language modeling task. The input schema prompt is separated from the target SQL via `\n-- SQL QUERY --\n`. Crucially, we apply **Loss Masking** (`-100`) to the prompt tokens, meaning backpropagation only penalizes the model for the generated SQL query, preventing it from wasting capacity on reciting the schema.
*   **LoRA Fine-Tuning:** Full-parameter fine-tuning of a 1.5B model is computationally intractable. We utilized **Parameter-Efficient Fine-Tuning (PEFT)** via LoRA:
    *   **Rank (r):** `16`
    *   **Alpha:** `32`
    *   **Target Modules:** Comprehensive injection across all attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
*   **Hyperparameters:** 
    *   **Epochs:** `3` (Early stopping or low epochs prevent overfitting on the synthetic structure).
    *   **Learning Rate:** `1e-4` using `adamw_torch` optimizer and Cosine LR Scheduler with a `0.03` Warmup Ratio.
    *   **Batching & Accumulation:** Executed with `Batch Size = 1` and `Gradient Accumulation = 8` to simulate larger batch dynamics on memory-constrained GPUs.
    *   **Precision:** Mixed precision training (`bf16` or `fp16`) with Gradient Checkpointing enabled.

---

## 3. Evaluation: The Virtual Sandbox (`evaluate_qwen.py`)

In traditional NLP, a model is scored using **Exact Match**. If it guesses exactly what the human wrote, it gets a point.

**The Paradigm Shift:** Since our model is unconditionally dreaming up queries out of thin air, there is no "ground truth" to compare against. The model might hallucinate a brilliant, highly complex 3-level `JOIN` query that technically works perfectly, but would fail standard NLP tests because it didn't "match the answer key".

We built a custom **Generative Robustness Paradigm** that evaluates execution compiling instead of semantic string matching:

```mermaid
sequenceDiagram
    participant Model
    participant Evaluator
    participant RAM_Sandbox
    
    Model->>Evaluator: "SELECT T1.col_1 FROM table_0 T1 JOIN..."
    Evaluator->>RAM_Sandbox: Initialize ephemeral sqlite3 engine
    Evaluator->>RAM_Sandbox: Force-Push Schema DDL Definitions
    RAM_Sandbox-->>Evaluator: Database Blueprint Ready
    Evaluator->>RAM_Sandbox: Execute Hallucinated Query
    
    alt Operational Error
        RAM_Sandbox-->>Evaluator: sqlite3.OperationalError (e.g. no such column)
        Evaluator-->>Model: Grade: 0 (Failed Compilation)
    else Successful Compile
        RAM_Sandbox-->>Evaluator: Empty Vector [ ] or Valid Data Array
        Evaluator-->>Model: Grade: 1 (Execution Validated)
    end
```

### The Test Protocol
1.  **Ephemeral Sandboxing:** A clean `sqlite3` database engine is spawned entirely in your system RAM.
2.  **State Injection:** The exact structure of the database (DDL blueprints) is forcefully executed into the sandbox.
3.  **The Compilation Trial:** The model's hallucinated query is piped through the SQLite instance. 
4.  **Grading Logistics:** If it crashes (e.g., hallucinated columns, fractured syntax string, illegal grouping semantics), the model organically fails. If the query safely executes and returns data (even an empty array, since there are no rows yet), the syntax and structural joins are proven completely viable. **Pass.**

### Advanced Structural Metrics
To prevent the model from spamming trivial `SELECT *` queries:
*   **Temperature Sampling:** We generate queries with `temperature=0.8` and `top_p=0.95` for high variance.
*   **Diversity & Uniqueness Scoring:** `evaluate_qwen.py` dynamically penalizes duplicate structural outputs and evaluates the presence of over 19 Relational Algebra constructs (`JOIN`, `GROUP BY`, `INTERSECT`, `HAVING`, etc.) to verify grammatical mastery.

---

### 💡 The End Result
By architecting a synergy between strict topological truncation, aggressive semantic name-masking ablations, Causal LoRA fine-tuning, and rigorous compile-time execution scoring, we bridge the gap between static semantic parsers and highly dynamic, fully autonomous database intelligence systems.
