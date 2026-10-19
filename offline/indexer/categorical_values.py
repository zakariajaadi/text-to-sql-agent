# offline/indexer/categorical_values.py

"""
Categorical values — the distinct values of the `categorical_columns` declared in
semantic_layer/<db>/tables.yaml, read from the database at index time (never stored in the semantic layer).
Used by: offline/indexer/schema_indexer.py (fetch_values)

Every value list goes to the value index (lib/value_index.py), searched at query time.
The short ones are also inlined in the table documents (fits_in_prompt).
"""

import time
from collections.abc import Callable, Iterable
from typing import TypeVar

from langchain_community.utilities import SQLDatabase
from loguru import logger
from sqlalchemy import text

from config import cfg

T = TypeVar("T")
# The values of a categorical column are inlined in the prompt only when there are
# at most this many of them...
MAX_CATEGORICAL_VALUES = 25
# ...all of them short
MAX_VALUE_LENGTH       = 50

# A failing GROUP BY is run again this many times...
QUERY_RETRIES          = 2
# ...after this many seconds, doubled at each retry
RETRY_DELAY_SECONDS    = 5


class CategoricalValuesError(RuntimeError):
    """
    The values of some categorical columns could not be read, retries included.

    Attributes:
        db_name: Name of the database.
        columns: The failed columns, as "<table>.<column>".
    """

    def __init__(self, db_name: str, columns: list[str]):
        self.db_name = db_name
        self.columns = columns
        super().__init__(
            f"[{db_name}] Values could not be read for {len(columns)} categorical column(s): "
            f"{', '.join(columns)}. Nothing was indexed: rerun the index, or fix or remove "
            f"these columns from semantic_layer/{db_name}/tables.yaml (categorical_columns)."
        )


def fetch_categorical_columns_values(db: SQLDatabase, db_name: str, table: str,
                                     columns: list[str]) -> dict[str, dict[str, int]]:
    """
    Row count of each distinct value of each categorical column of a table, one GROUP BY per column:
    {column: {value: row count}}.

    Columns with no value, or with more than value_search.max_values (an identifier or free
    text rather than labels), are left out, with a warning.

    A failing query is retried QUERY_RETRIES times (a Hive / Tez job can fail transiently).
    Raises CategoricalValuesError, once every column is tried, when some still failed: their
    value lists would otherwise be missing from the documents and the value index unnoticed.

    Run tables one after the other: each GROUP BY runs a Hive job, and concurrent
    jobs fail on a local Tez.
    """
    max_values = cfg.value_search.max_values
    values     = {}
    failed     = []
    for column in columns:
        try:
            counts = _with_retries(lambda: _column_value_counts(db, table, column, limit=max_values + 1),
                                   label=f"[{db_name}] {table}.{column}")
        except Exception as e:
            # Keep reading the other columns: every failed column is reported at once
            logger.error(f"[{db_name}] No values for {table}.{column}: query failed — {str(e)[:200]}")
            failed.append(f"{table}.{column}")
            continue

        if not counts or len(counts) > max_values:
            logger.warning(f"[{db_name}] No values for {table}.{column}: none, or more than {max_values}")
            continue

        values[column] = counts
        if not fits_in_prompt(counts):
            logger.info(f"[{db_name}] {table}.{column}: {len(counts)} values, searched at query time only")

    if failed:
        raise CategoricalValuesError(db_name, failed)
    return values


def fits_in_prompt(values: Iterable[str]) -> bool:
    """A value list is inlined in the prompt only when it is short and made of short values."""
    values = list(values)
    return (
        0 < len(values) <= MAX_CATEGORICAL_VALUES
        and all(len(v) <= MAX_VALUE_LENGTH for v in values)
    )


def _with_retries(query: Callable[[], T], label: str) -> T:
    """Run `query`, again up to QUERY_RETRIES times on failure; the last failure is raised."""
    for attempt in range(QUERY_RETRIES + 1):
        try:
            return query()
        except Exception as e:
            if attempt == QUERY_RETRIES:
                raise
            delay = RETRY_DELAY_SECONDS * 2 ** attempt
            logger.warning(f"{label}: query failed, retry {attempt + 1}/{QUERY_RETRIES} "
                           f"in {delay}s — {str(e)[:200]}")
            time.sleep(delay)


def _column_value_counts(db: SQLDatabase, table: str, column: str, limit: int) -> dict[str, int]:
    """Row count of each distinct non-null value, capped at `limit` values."""
    quote    = db._engine.dialect.identifier_preparer.quote
    tbl, col = quote(table), quote(column)
    query    = (f"SELECT {col}, COUNT(*) FROM {tbl} WHERE {col} IS NOT NULL "
                f"GROUP BY {col} LIMIT {limit}")

    with db._engine.connect() as conn:
        return {str(value): n for value, n in conn.execute(text(query))}
