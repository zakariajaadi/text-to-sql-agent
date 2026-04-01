# eval/eval_lib/execution.py
# =====================
# Execution accuracy — ground-truth comparison of SQL result sets.
#
# Runs both the gold query (accepted_queries) and the generated query
# against the real database, then compares result sets.
# This is the gold standard metric in Text-to-SQL evaluation.

from langchain_community.utilities import SQLDatabase
from loguru import logger

def execution_accuracy(db: SQLDatabase, accepted_queries: list[str],
                       generated_sql: str | None) -> float | None:
    """
    Compute execution accuracy for a single benchmark question.

    Compares the generated SQL result set against each accepted (gold) query.
    Takes the best match if multiple gold queries exist — a question can
    have several valid SQL formulations.

    Returns:
        1.0  — result sets match (at least one gold query)
        0.0  — no gold query matches
        None — generated_sql is None (OOS refusal or agent error)
    """
    # OOS or agent crash — not scorable
    if not generated_sql:
        return None

    # No gold queries defined — skip
    if not accepted_queries:
        logger.warning("[execution] No accepted_queries defined, skipping")
        return None

    # Execute the generated query
    gen_rows = _execute_query(db, generated_sql)
    if gen_rows is None:
        # Generated SQL crashed — that's a 0, not a None
        return 0.0

    # Compare against each accepted query, take best match
    for gold_sql in accepted_queries:
        gold_rows = _execute_query(db, gold_sql)
        if gold_rows is None:
            logger.error(f"[execution] Gold query failed: {gold_sql}")
            continue

        order_sensitive = _has_order_dependency(gold_sql) or _has_order_dependency(generated_sql)
        score = _compare_result_sets(gold_rows, gen_rows, order_sensitive)

        if score == 1.0:
            return 1.0

    return 0.0

# ── Helpers ──────────────────────────────────────────────────────────

def _execute_query(db: SQLDatabase, sql: str) -> list[tuple] | None:
    """
    Execute a SQL query and return rows as a list of tuples.
    Returns None on execution error (syntax, missing table, etc.).
    """
    try:
        from sqlalchemy import text
        with db._engine.connect() as conn:
            rows = conn.execute(text(sql)).fetchall()
            return [tuple(row) for row in rows]
    except Exception as e:
        logger.debug(f"[execution] Query failed: {e}")
        return None

def _has_order_dependency(sql: str) -> bool:
    """
    Determine if row order matters for this query.
    ORDER BY alone is cosmetic — the result set is the same regardless of order.
    ORDER BY + LIMIT/OFFSET is semantic — different order = different rows.
    """
    sql_upper = sql.upper()
    if "ORDER BY" not in sql_upper:
        return False
    return "LIMIT" in sql_upper or "OFFSET" in sql_upper

def _compare_result_sets(gold_rows: list, gen_rows: list, order_sensitive: bool) -> float:
    """
    Compare two result sets. Returns 1.0 on exact match, 0.0 otherwise.
    When order doesn't matter, both sets are sorted before comparison.
    """
    gold = [_normalize_row(tuple(r)) for r in gold_rows]
    gen = [_normalize_row(tuple(r)) for r in gen_rows]

    if not order_sensitive:
        gold = sorted(gold)
        gen = sorted(gen)

    return 1.0 if gold == gen else 0.0

def _normalize_row(row: tuple) -> tuple:
    """
    Normalize a result row for reliable comparison.
    SQLite is loose with types — the same value can come back as
    int, float, or Decimal depending on the query structure.
    Rounds floats to 4 decimals to avoid false negatives.
    """
    normalized = []
    for v in row:
        if v is None:
            normalized.append(None)
        elif isinstance(v, float):
            normalized.append(round(v, 4))
        elif isinstance(v, int):
            normalized.append(v)
        elif isinstance(v, str):
            normalized.append(v.strip())
        else:
            # Catch-all: Decimal, bool, etc.
            try:
                normalized.append(round(float(v), 4))
            except (ValueError, TypeError):
                normalized.append(str(v))
    return tuple(normalized)