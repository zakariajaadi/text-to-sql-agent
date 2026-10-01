"""
Deterministic fixes applied to every generated query, for Hive and Impala.
Used by: nodes.py (sql_coder)
"""

import sqlglot
from sqlglot import exp
from sqlglot.dialects.hive import Hive
from sqlglot.errors import ParseError


class Impala(Hive):
    """sqlglot has no Impala dialect: Hive syntax, but NULL sorts as the largest value."""
    NULL_ORDERING = "nulls_are_large"


_SQLGLOT_DIALECTS = {"hive": Hive, "impala": Impala}


def normalize_sql(query: str, dialect: str) -> str:
    """
    - Backtick every identifier: reserved-word columns (e.g. `time`) no longer break parsing.
    - NULLs always sort last: a "top N" never starts with NULL rows (Impala sorts them first in DESC).
    Leading "--" comment lines (assumptions) are kept as is.
    Returns the query unchanged for other dialects, or when sqlglot can't parse it.
    """
    target = _SQLGLOT_DIALECTS.get(dialect)
    if target is None:
        return query

    # sqlglot would rewrite "-- Assumption:" lines as /* */ comments: set them aside
    lines    = query.strip().splitlines()
    comments = [line for line in lines if line.lstrip().startswith("--")]
    sql      = "\n".join(line for line in lines if not line.lstrip().startswith("--"))

    try:
        tree = sqlglot.parse_one(sql, read=target)
    except ParseError:
        return query

    for ordered in tree.find_all(exp.Ordered):
        ordered.set("nulls_first", False)

    return "\n".join(comments + [tree.sql(dialect=target, identify=True, pretty=True)])