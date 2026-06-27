"""Prompt templates for Text-to-SQL fine-tuning.

Two variants, selected via configs/train_config.yaml: data.prompt_variant
  - "with_schema": includes CREATE TABLE statements (with foreign keys) for the
    target database. This is the default condition.
  - "no_schema": question only, no schema block. Weaker condition used for the
    schema-ablation comparison.

EOS handling: the caller is responsible for appending tokenizer.eos_token after
the gold SQL when building training examples (see build_training_example below).
We never hardcode a literal EOS string here so the same code works if the base
model is swapped (e.g. to Llama-3.2-3B-Instruct).
"""

from __future__ import annotations

WITH_SCHEMA_TEMPLATE = """### Instruction:
You are a SQL expert. Given a database schema and a question, write the SQL query that answers the question.

### Schema:
{schema_block}

### Question:
{question}

### SQL:
"""

NO_SCHEMA_TEMPLATE = """### Instruction:
You are a SQL expert. Write the SQL query that answers the question.

### Question:
{question}

### SQL:
"""


def build_schema_block(db_id: str, tables: dict) -> str:
    """Build a compact CREATE TABLE block (with foreign keys) for one Spider DB.

    `tables` is the entry from tables.json for this db_id, in Spider's native
    format (table_names_original, column_names_original, column_types,
    primary_keys, foreign_keys).
    """
    table_names = tables["table_names_original"]
    column_names = tables["column_names_original"]  # list of [table_idx, col_name]
    column_types = tables["column_types"]
    primary_keys = set(tables.get("primary_keys", []))
    foreign_keys = tables.get("foreign_keys", [])  # list of [col_idx, ref_col_idx]

    # Group columns by owning table (skip the synthetic "*" column at idx 0).
    cols_by_table: dict[int, list[tuple[int, str, str]]] = {i: [] for i in range(len(table_names))}
    for col_idx, (tbl_idx, col_name) in enumerate(column_names):
        if tbl_idx == -1:
            continue
        cols_by_table[tbl_idx].append((col_idx, col_name, column_types[col_idx]))

    fk_by_col = {fk[0]: fk[1] for fk in foreign_keys}

    statements = []
    for tbl_idx, tbl_name in enumerate(table_names):
        lines = []
        for col_idx, col_name, col_type in cols_by_table[tbl_idx]:
            sql_type = col_type.upper() if col_type else "TEXT"
            suffix = " PRIMARY KEY" if col_idx in primary_keys else ""
            lines.append(f"  {col_name} {sql_type}{suffix}")
        for col_idx, _, _ in cols_by_table[tbl_idx]:
            if col_idx in fk_by_col:
                ref_idx = fk_by_col[col_idx]
                ref_tbl_idx, ref_col_name = column_names[ref_idx]
                col_name = column_names[col_idx][1]
                ref_tbl_name = table_names[ref_tbl_idx]
                lines.append(f"  FOREIGN KEY ({col_name}) REFERENCES {ref_tbl_name}({ref_col_name})")
        statements.append(f"CREATE TABLE {tbl_name} (\n" + ",\n".join(lines) + "\n)")

    return "\n".join(statements)


def build_prompt(question: str, schema_block: str | None, variant: str) -> str:
    if variant == "with_schema":
        if schema_block is None:
            raise ValueError("with_schema variant requires a schema_block")
        return WITH_SCHEMA_TEMPLATE.format(schema_block=schema_block, question=question)
    elif variant == "no_schema":
        return NO_SCHEMA_TEMPLATE.format(question=question)
    else:
        raise ValueError(f"Unknown prompt variant: {variant}")


def build_training_example(question: str, schema_block: str | None, sql: str, variant: str, eos_token: str) -> str:
    """Full example used for training: prompt + gold SQL + EOS.

    Loss masking (applied in train.py) covers only the `sql + eos_token` span;
    everything up to and including '### SQL:\\n' is masked out of the loss.
    """
    prompt = build_prompt(question, schema_block, variant)
    return prompt + sql.strip() + eos_token
