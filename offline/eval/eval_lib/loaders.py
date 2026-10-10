# offline/eval/eval_lib/loaders.py
# ===================
# Data loading — YAML dataset files and DB schemas.

import yaml


def load_questions(path: str, difficulty: str = None) -> list[dict]:
    """Load benchmark questions from YAML with optional difficulty filter."""
    with open(path) as f:
        questions = yaml.safe_load(f)
    if difficulty:
        questions = [q for q in questions if q["difficulty"] == difficulty]
    return questions


def load_schema(db_name: str) -> str:
    """
    Load full DDL schema for judge context.
    1. Try schema_store (enriched DDLs from Chroma)
    2. Fallback: introspect from SQLDatabase
    """
    from resources import schema_store, db_registry  # type: ignore[import-not-found]

    if db_name in schema_store and schema_store[db_name]:
        return "\n\n".join(schema_store[db_name].values())
    return db_registry[db_name].get_table_info()