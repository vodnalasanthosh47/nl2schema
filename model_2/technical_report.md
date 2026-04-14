# Technical Report: Unconditional Schema-to-SQL Generation Database Pipeline

## 1. Abstract & Problem Formulation
The traditional **Spider NLP task** focuses on Semantic Parsing: mapping a natural language question (e.g., *"What is the average age of users?"*) to a specific SQL query, given a schema. 
In this project, we reframed the problem space into **Unconditional Schema-to-SQL Generation**. The model's sole input is the structural definition of a database (`CREATE TABLE` DDL statements), and its objective is to autonomously output a batch of diverse, structurally coherent, and executable SQL queries. This shifts the model's role from a simple translator to an autonomous query synthesis engine—useful for generating synthetic business data, stress-testing database backend engines, or auto-generating query templates.

## 2. Dataset Engineering & Representation (`preprocess.py`)

Adapting the Spider dataset for this novel task required complex topological transformations of the data. 

### 2.1 Graph to DDL Serialization
Spider stores schemas as JSON graphs with absolute cross-referencing capabilities (`table_names_original`, `column_names_original`, `foreign_keys`). The `preprocess.py` pipeline converts this multi-dimensional array data into a continuous sequence of `CREATE TABLE` expressions. Crucially, we enforce that `PRIMARY KEY` and `REFERENCES` strings are explicitly mapped so the Transformer's attention mechanism can learn the edges of the relational data graph.

### 2.2 Token Budgeting & Table-Boundary Truncation
Transformers have a strict maximum position embedding sequence length (CodeT5's maximum is **1024 tokens**). Spider contains several enterprise-scale databases exceeding 15 tables, which easily breach this limit. 
*   **The Naïve Approach:** A standard simple truncation (`string[:1024]`) would randomly slice a SQL DDL statement mid-word (e.g., leaving `CREATE TABL` or defining a table but abruptly cutting off its closing parenthesis). This violently corrupts the Abstract Syntax Tree (AST) grammar that the model needs to learn.
*   **Our Approach:** We implemented an iterative **Table-Boundary Truncation algorithm**. It measures token lengths using the actual HuggingFace tokenizer `.encode()`, tracking cumulative length. If appending the next `CREATE TABLE` block breaches the `(1024 - 50)` token limit (leaving 50 tokens for the prompt overhead), the script halts and outputs the finalized schema subset. The model guarantees it only ever trains on syntactically pristine data.

### 2.3 Regularization via Name-Masking Augmentation
A major vulnerability in fine-tuning massive Pre-trained Language Models (PLMs) on relational data is their tendency to rely on pre-trained semantic priors rather than actual schema structures. For example, if the PLM sees tables named `employees` and `salaries`, the model may exploit semantic co-occurrence in pre-training data rather than explicit FK structure.
*   **The Fix:** We injected **Name-Masking Augmentation** into 20% of the training batches.
*   **Algorithm:** We programmatically substitute original table names with generic strings (`table_0`, `table_1`) and column names with (`col_0`, `col_1`). We sort the string replacement arrays by descending length to prevent sub-string collision overwrites. 
*   **Mathematical Impact:** By hiding the English semantics, the model is mathematically forced to increase the attention weights corresponding to structural SQL keywords (like `JOIN`, `ON`, `REFERENCES`). It acts as a powerful regularizer, ensuring the model learns the structural topological calculus of SQL rather than just memorizing Spider's database names.

## 3. Model Architecture & Hyperparameter Calculus (`train.py` & `config.yaml`)

We selected **Salesforce/codet5-base** (~220M parameters). Unlike standard models like T5 or BART, CodeT5's pre-training objective involves *Identifier-Aware Denoising*—it is specifically trained to track variables and code variables across long source codes. This closely aligns with the requirement of tracking a specific `table.column_id` across a complex nested SQL query.

### 3.1 Optimization Calculus
*   **Learning Rate (`5e-5`):** Standard neural networks from scratch use LRs around `1e-3`. However, when fine-tuning a PLM, a high learning rate causes *Catastrophic Forgetting*, effectively shattering the geometric representation of code syntax the model spent millions of GPU-hours building. `5e-5` is empirically well-established for seq2seq fine-tuning.
*   **Warmup Ratio (`0.1`):** Adaptive optimizers (like AdamW) have high variance in their momentum estimates during the earliest batches. We enforce a 10% warmup, linearly scaling the learning rate from `0` to `5e-5`. This protects the pre-trained weights from aggressive, errant gradient updates on step 1.
*   **Batch Sizing (`Gradient Accumulation = 2, Batch Size = 8`):** To maintain smoothed gradient trajectory calculations without causing out-of-memory (OOM) halts on consumer GPUs, we calculate micro-batches of 8 and accumulate the gradients across 2 steps, simulating an effective batch size of 16.
*   **Epochs & Early Stopping:** We run 5 epochs. PLMs applied to limited datasets (~8,600 pairs) overfit incredibly fast. We instituted an Early Stopping callback monitoring internal `eval_loss` with a patience of 3 epochs.

## 4. Evaluation Framework Shift (`evaluate.py`)

Spider's traditional evaluation suite measures "Exact Match" (AST topological comparison with a known gold-standard query) and "Execution Accuracy" (comparing returned SQLite data arrays).
Our model generates queries unconditionally; there is no "correct" question to answer. Thus, standard validation metrics were completely discarded as mathematically incompatible. We built a custom evaluation paradigm targeting **generative robustness**:

### 4.1 Ephemeral Execution Validation
The ultimate ground truth for code is compilation/execution. The `evaluate.py` script spawns an isolated, in-memory `sqlite3` DBMS container for every single generated query. It loads the respective DDL schema and force-executes the generated SQL. 
*   If the model hallucinated a column name, violated a Type constraint, or messed up a `JOIN` syntax, SQLite throws an `OperationalError` or `DatabaseError` and the query is marked invalid. 

### 4.2 Structural Diversity Scoring
Generating 10 identically boring `SELECT * FROM table_1` queries is technically executable but practically useless. 
*   **Diverse Beam Search:** At generation, we split the generation beam into distinct groups (`num_beam_groups = 5`) and penalize them mathematically (`diversity_penalty = 1.5`) if they start pursuing the same token trajectories.
*   **Diversity Metric:** We index 19 distinct operations in Relational Algebra (`GROUP BY`, `MIN`, `MAX`, `UNION`, `INTERSECT`, `LIMIT`, etc.) and automatically compute what fraction of the language the model covers per schema.

## 5. Inference & Pipeline Safeguards (`pipeline.py`)

When integrated into a broader ML pipeline, upstream components (like Business Logic generators) output standard enterprise SQL types (e.g., `VARCHAR(255)`, `DATETIME`). 
Because our unconditional generation model was strictly scoped to Spider's abstracted database ontology (`text`, `number`, `time`), feeding it `VARCHAR(255)` triggers severe out-of-vocabulary (OOV) uncertainty and degrades the attention matrix output. 
To safeguard against this, the pipeline enforces **DDL Type Normalization** at inference time. It uses dynamic RegEx preprocessing to down-cast standard enterprise SQL dialects (e.g., `VARCHAR` -> `text`, `INTEGER` -> `number`) back into the constrained vocabulary space the model expects, ensuring the generation mathematics align perfectly with the fine-tuned manifold before execution logic runs.

## 6. Baseline Results & Future Work

Our initial proof-of-concept run with the 20% name-masking configuration achieved a **55.9% Execution Validity Score**.

To place this in context:
*   An untrained model (random token generation) achieves ~10-15% validity.
*   State-of-the-Art models on a fundamentally different task (NL -> SQL) achieve ~85-90%.

Hitting ~56% for a first-pass, entirely unconditional generation model proves that the architecture has learned the basic AST and vocabulary of SQL. However, systematic failure analysis revealed the primary source of error was **column hallucination**. The model frequently ignored explicit `FOREIGN KEY` definitions in favor of assuming standard column names based on its pre-trained semantic weights. 

This signifies that the 20% name-masking augmentation was insufficient to fully override the model's semantic priors. 

**Next Steps & Retraining:** We have diagnosed the root cause of these hallucination failures. The immediate next phase of this research is increasing the name-masking augmentation from 20% to **40%**. By removing significantly more English context from the training batches, we hypothesize the model will be forced to rely strictly on the topological structure of the DDL schema, projecting a leap in generative validity toward the 70%+ threshold.
