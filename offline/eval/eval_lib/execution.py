# offline/eval/eval_lib/execution.py
# =====================
# Execution accuracy — ground-truth comparison of SQL result sets.
#
# Runs both the gold query (accepted_queries) and the generated query
# against the real database, then compares result sets.
# This is the gold standard metric in Text-to-SQL evaluation.

from itertools import permutations

from langchain_community.utilities import SQLDatabase
from loguru import logger


def execution_accuracy(db: SQLDatabase, accepted_queries: list[str],
                       generated_sql: str | None) -> dict | None:
    """
    Compute execution accuracy for a single benchmark question.

    Compares the generated SQL result set against each accepted (gold) query.
    Takes the best score if multiple gold queries exist — a question can
    have several valid SQL formulations.

    Returns:
        {"exact": 0.0|1.0} — score for the best-matching gold query
        None — generated_sql is None (OOS refusal or agent error)
    """
    if not generated_sql:
        return None

    if not accepted_queries:
        logger.warning("[execution] No accepted_queries defined, skipping")
        return None


    # Syntax error or runtime failure, score 0 rather than crashing
    gen_rows = _execute_query(db, generated_sql)
    if gen_rows is None:
        return {"exact": 0.0}

    # An ordered result can be compared on its first rows (full ranking vs expected top N)
    is_ordered = "order by" in generated_sql.lower()

    best_exact = 0.0

    for gold_sql in accepted_queries:
        gold_rows = _execute_query(db, gold_sql)
        if gold_rows is None:
            # Broken gold query — skip rather than penalise the generated SQL
            logger.error(f"[execution] Gold query failed: {gold_sql}")
            continue

        exact = _exact_execution_score(gold_rows, gen_rows, is_ordered)
        if exact > best_exact:
            best_exact = exact

        # Short-circuit: perfect match found, no need to check remaining gold queries
        if best_exact == 1.0:
            return {"exact": 1.0}

    return {"exact": best_exact}

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

    
def _exact_execution_score(gold_rows: list, gen_rows: list, is_ordered: bool = False) -> float:
    """
    Binary execution score — 1.0 if generated rows are a valid subset of gold, 0.0 otherwise.
    Checks that every row returned by the agent exists in the gold result set.

    Extra generated columns are allowed (an id or an email next to the expected name):
    the score holds if some choice of generated columns, one per gold column and in any
    order, gives rows that all exist in gold.

    An ordered result with more rows than gold (a full ranking when gold expects a top N)
    is compared on its first len(gold_rows) rows: a well-ordered ranking starts with the
    expected top, a badly ordered one does not.
    """
    if not gold_rows and not gen_rows:
        return 1.0
    if not gold_rows or not gen_rows:
        return 0.0
    if len(gen_rows[0]) < len(gold_rows[0]):
        return 0.0

    if is_ordered:
        gen_rows = gen_rows[:len(gold_rows)]

    gold = set([_normalize_row(tuple(r)) for r in gold_rows])
    gen = [_normalize_row(tuple(r)) for r in gen_rows]

    # Every generated row must exist in gold (avoid LIMIT clause overhead)
    for columns in permutations(range(len(gen[0])), len(gold_rows[0])):
        if all(tuple(r[i] for i in columns) in gold for r in gen):
            return 1.0
    return 0.0

def _normalize_row(row: tuple) -> tuple:
    """
    Normalize a result row for reliable comparison.
    Drivers return numbers as int, float or Decimal depending on the query
    (PyHive: DECIMAL columns come back as Decimal).
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