"""
SQL helpers built on sqlglot: deterministic fixes applied to every generated query (Hive, Impala),
tables a query reads, and whether a query computes a given expression.
Used by: agent/nodes.py (sql_coder), ui/streaming.py (result card), lib/metrics.py,
offline/semantic_gen/metrics.py
"""

import sqlglot
from sqlglot import exp
from sqlglot.dialects.hive import Hive
from sqlglot.errors import ParseError, TokenError


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
    except (ParseError, TokenError):  # unparseable SQL, e.g. an unclosed quote
        return query

    for ordered in tree.find_all(exp.Ordered):
        ordered.set("nulls_first", False)

    return "\n".join(comments + [tree.sql(dialect=target, identify=True, pretty=True)])


def sqlglot_dialect(dialect: str):
    """sqlglot dialect of a configured database dialect (Impala has its own class, see above)."""
    return _SQLGLOT_DIALECTS.get(dialect, dialect)


def get_query_tables(query: str, dialect: str) -> list[str]:
    """
    Names of the tables the query reads, without CTE names
    (joins written with commas, subqueries and CTE bodies included).
    Empty when sqlglot can't parse the query.
    """
    try:
        tree = sqlglot.parse_one(query, read=sqlglot_dialect(dialect))
    except (ParseError, TokenError):  # unparseable SQL, e.g. an unclosed quote
        return []

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    names     = (table.name for table in tree.find_all(exp.Table))
    return list(dict.fromkeys(name for name in names if name and name.lower() not in cte_names))


def contains_expression(query: str, expression: str, dialect: str) -> bool:
    """
    True when the query computes the expression, whatever the table aliases and the quoting
    (e.g. SUM(od.unitprice * od.quantity) contains SUM(order_details.unitprice * order_details.quantity)).
    False when either side can't be parsed.
    """
    read = sqlglot_dialect(dialect)
    try:
        tree   = sqlglot.parse_one(query, read=read)
        target = sqlglot.parse_one(expression, read=read)
    except (ParseError, TokenError):  # unparseable SQL, e.g. an unclosed quote
        return False

    def unqualified(node: exp.Expression) -> str:
        node = node.copy()
        for column in node.find_all(exp.Column):
            column.set("table", None)
        return node.sql(dialect=read, identify=True)

    wanted = unqualified(target)
    return any(unqualified(node) == wanted for node in tree.find_all(type(target)))