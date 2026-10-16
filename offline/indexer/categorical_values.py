# offline/indexer/categorical_values.py

"""
Categorical values — the distinct values of the `categorical_columns` declared in
semantic_layer/<db>/tables.yaml, read from the database at index time (never stored in the semantic layer).
Used by: offline/indexer/schema_indexer.py (fetch_values)

Every value list goes to the value index (lib/value_index.py), searched at query time.
The short ones are also inlined in the table documents (fits_in_prompt).
"""

from collections.abc import Iterable

from langchain_community.utilities import SQLDatabase
from loguru import logger
from sqlalchemy import text

from config import cfg

# The values of a categorical column are inlined in the prompt only when there are
# at most this many of them...
MAX_CATEGORICAL_VALUES = 25
# ...all of them short
MAX_VALUE_LENGTH       = 50


def fetch_categorical_columns_values(db: SQLDatabase, db_name: str, table: str,
                                     columns: list[str]) -> dict[str, dict[str, int]]:
    """
    Row count of each distinct value of each categorical column of a table, one GROUP BY per column:
    {column: {value: row count}}.

    Columns with no value, or with more than value_search.max_values (an identifier or free
    text rather than labels), are left out, with a warning.

    Run tables one after the other: each GROUP BY runs a Hive job, and concurrent
    jobs fail on a local Tez.
    """
    max_values = cfg.value_search.max_values
    values     = {}
    for column in columns:
        try:
            counts = _column_value_counts(db, table, column, limit=max_values + 1)
        except Exception as e:
            # Values are an optional enrichment: one failing column must not abort the run
            logger.warning(f"[{db_name}] No values for {table}.{column}: query failed — {str(e)[:200]}")
            continue

        if not counts or len(counts) > max_values:
            logger.warning(f"[{db_name}] No values for {table}.{column}: none, or more than {max_values}")
            continue

        values[column] = counts
        if not fits_in_prompt(counts):
            logger.info(f"[{db_name}] {table}.{column}: {len(counts)} values, searched at query time only")
    return values


def fits_in_prompt(values: Iterable[str]) -> bool:
    """A value list is inlined in the prompt only when it is short and made of short values."""
    values = list(values)
    return (
        0 < len(values) <= MAX_CATEGORICAL_VALUES
        and all(len(v) <= MAX_VALUE_LENGTH for v in values)
    )


def _column_value_counts(db: SQLDatabase, table: str, column: str, limit: int) -> dict[str, int]:
    """Row count of each distinct non-null value, capped at `limit` values."""
    quote    = db._engine.dialect.identifier_preparer.quote
    tbl, col = quote(table), quote(column)
    query    = (f"SELECT {col}, COUNT(*) FROM {tbl} WHERE {col} IS NOT NULL "
                f"GROUP BY {col} LIMIT {limit}")

    with db._engine.connect() as conn:
        return {str(value): n for value, n in conn.execute(text(query))}
