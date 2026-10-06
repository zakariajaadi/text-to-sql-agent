# src/lib/categorical_values.py

"""
Categorical values — the distinct values of the `categorical_columns` declared in
config/semantics/<db>.yaml, read from the database at index time (never stored in config).
Used by: schema_retrieval/schema_indexer.py (build_documents)
"""

from langchain_community.utilities import SQLDatabase
from loguru import logger
from sqlalchemy import text

# The values of a categorical column are inlined in the prompt only when there are
# at most this many of them...
MAX_CATEGORICAL_VALUES = 25
# ...all of them short
MAX_VALUE_LENGTH       = 50


def fetch_categorical_values(db: SQLDatabase, db_name: str, table: str,
                             columns: list[str]) -> dict[str, list[str]]:
    """
    Sorted distinct values of each categorical column of a table, one GROUP BY per column.
    Columns whose values do not fit in the prompt are left out, with a warning.

    Run tables one after the other: each GROUP BY runs a Hive job, and concurrent
    jobs fail on a local Tez.
    """
    values = {}
    for column in columns:
        try:
            counts = _value_counts(db, table, column)
        except Exception as e:
            # Values are an optional enrichment: one failing column must not abort the run
            logger.warning(f"[{db_name}] No values for {table}.{column}: query failed — {str(e)[:200]}")
            continue

        if _fits_in_prompt(counts):
            values[column] = sorted(counts)
        else:
            logger.warning(f"[{db_name}] No values for {table}.{column}: none, more than "
                           f"{MAX_CATEGORICAL_VALUES}, or longer than {MAX_VALUE_LENGTH} chars")
    return values


def _value_counts(db: SQLDatabase, table: str, column: str) -> dict[str, int]:
    """Row count of each distinct non-null value, capped at MAX_CATEGORICAL_VALUES + 1 values."""
    quote    = db._engine.dialect.identifier_preparer.quote
    tbl, col = quote(table), quote(column)
    query    = (f"SELECT {col}, COUNT(*) FROM {tbl} WHERE {col} IS NOT NULL "
                f"GROUP BY {col} LIMIT {MAX_CATEGORICAL_VALUES + 1}")

    with db._engine.connect() as conn:
        return {str(value): n for value, n in conn.execute(text(query))}


def _fits_in_prompt(counts: dict[str, int]) -> bool:
    """A value list is inlined in the prompt only when it is short and made of short values."""
    return (
        0 < len(counts) <= MAX_CATEGORICAL_VALUES
        and all(len(v) <= MAX_VALUE_LENGTH for v in counts)
    )
