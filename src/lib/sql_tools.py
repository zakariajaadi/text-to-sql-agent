"""
SQL helpers built on sqlglot: deterministic fixes applied to every generated query (Hive, Impala),
tables a query reads, the text values it filters on, the excluded columns it reads, whether it is
a refusal rather than a query, whether a query computes a given expression, and the join condition
of a declared relationship.
Used by: agent/nodes.py (sql_coder, sql_validator), ui/streaming.py (result card), lib/metrics.py,
lib/semantics.py, offline/semantic_gen/metrics.py
"""

import re

import sqlglot
from sqlglot import exp
from sqlglot.dialects.hive import Hive
from sqlglot.errors import ParseError, SqlglotError, TokenError
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.scope import traverse_scope


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


def get_value_filters(query: str, dialect: str) -> list[tuple[str, str, str]]:
    """
    (table, column, value) of each text value the query filters a column on:
    col = 'x', col <> 'x', col IN ('x', 'y'), col NOT IN (...). Aliases are resolved to their
    table; table is '' for an unqualified column. Filters on an expression (LOWER(col) = 'x')
    or a LIKE are left out. Empty when sqlglot can't parse the query.
    """
    try:
        tree = sqlglot.parse_one(query, read=sqlglot_dialect(dialect))
    except (ParseError, TokenError):  # unparseable SQL, e.g. an unclosed quote
        return []

    # Alias (or name) → table name; a CTE or subquery alias maps to no table of the database
    tables = {}
    for table in tree.find_all(exp.Table):
        tables.setdefault(table.alias_or_name.lower(), table.name.lower())

    def text_values(nodes) -> list[str]:
        return [node.this for node in nodes if isinstance(node, exp.Literal) and node.is_string]

    filters = []
    for node in tree.find_all(exp.EQ, exp.NEQ, exp.In):
        if isinstance(node, exp.In):
            column, values = node.this, text_values(node.expressions)
        elif isinstance(node.this, exp.Column):
            column, values = node.this, text_values([node.expression])
        else:   # 'x' = col
            column, values = node.expression, text_values([node.this])

        if not isinstance(column, exp.Column) or not values:
            continue
        table = tables.get(column.table.lower(), column.table.lower()) if column.table else ""
        filters += [(table, column.name.lower(), value) for value in values]
    return filters


def find_excluded_columns(query: str, dialect: str, excluded: dict[str, set[str]]) -> list[str]:
    """
    The excluded columns the query reads, as table.column, and "table.*" for a SELECT * (or t.*)
    over a table that has some. excluded: {table: {column, ...}}, the `exclude_columns` of
    tables.yaml; names compare case-insensitively.

    Each column is resolved to its table in its own scope (subqueries, CTEs, correlated references
    to an outer query, table aliases); an unqualified one counts for every table of its FROM that
    excludes a column of that name. COUNT(*) reads no column. When sqlglot can't parse the query,
    every word naming an excluded column of a table the query names counts.
    """
    excluded = {t.lower(): {c.lower() for c in columns} for t, columns in excluded.items() if columns}
    if not excluded:
        return []
    try:
        scopes = traverse_scope(sqlglot.parse_one(query, read=sqlglot_dialect(dialect)))
    except SqlglotError:   # unparseable SQL: words naming an excluded column of a named table
        words = {word.lower() for word in re.findall(r"\w+", query)}
        return [f"{t}.{c}" for t, columns in excluded.items() if t in words for c in sorted(columns) if c in words]

    def tables_of(scope) -> dict[str, str]:
        """Alias (or name) → table of the database, for the tables read directly by the scope."""
        return {alias.lower(): source.name.lower() for alias, source in scope.sources.items()
                if isinstance(source, exp.Table)}

    def table_of(scope, alias: str) -> str | None:
        """Table an alias stands for, in the scope or an outer one (correlated subquery)."""
        while scope is not None:
            if alias.lower() in {a.lower() for a in scope.sources}:
                return tables_of(scope).get(alias.lower())   # None: a subquery or a CTE, checked on its own
            scope = scope.parent
        return None

    found = []
    for scope in scopes:
        for column in scope.columns:
            if isinstance(column.this, exp.Star):
                continue
            tables = [table_of(scope, column.table)] if column.table else tables_of(scope).values()
            found += [f"{t}.{column.name.lower()}" for t in tables if column.name.lower() in excluded.get(t, ())]
        # SELECT * and t.* read every column of their tables, the excluded ones too
        for projection in scope.expression.expressions if isinstance(scope.expression, exp.Select) else []:
            if isinstance(projection, exp.Star):
                tables = tables_of(scope).values()
            elif isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star):
                tables = [table_of(scope, projection.table)]
            else:
                continue
            found += [f"{t}.*" for t in tables if t in excluded]
    return list(dict.fromkeys(found))


def is_refusal(query: str, dialect: str) -> bool:
    """
    True when sql_coder wrote no query to run: only comments (the "-- Not available: ..." line of
    agent/prompts.py), or a SELECT of text literals from no table (a message dressed as a query).
    """
    sql = "\n".join(line for line in query.splitlines() if not line.lstrip().startswith("--")).strip()
    if not sql:
        return True
    try:
        tree = sqlglot.parse_one(sql, read=sqlglot_dialect(dialect))
    except SqlglotError:
        return False
    return (isinstance(tree, exp.Select) and tree.find(exp.From) is None
            and all(isinstance(p.unalias(), exp.Literal) and p.unalias().is_string for p in tree.expressions))


def contains_expression(query: str, expression: str, dialect: str,
                        sources: dict[str, str | None] | None = None) -> bool:
    """
    True when the query computes the expression on the same tables, whatever the aliases and the
    quoting: SUM(od.unitprice * od.quantity) FROM order_details od contains
    SUM(order_details.unitprice * order_details.quantity), COUNT(DISTINCT eden.numero_contrat)
    never contains COUNT(DISTINCT wafasalaf.numero_contrat).
    In a query, an alias stands for its table. In an expression alone (a metric, written table.column),
    the qualifier is the table, or the one `sources` names ({alias: table, None for a step of the
    query: a CTE, a subquery}). A column whose table can't be told (read from a subquery, unqualified
    among several tables) matches the column of any table.
    False when either side can't be parsed.
    """
    read = sqlglot_dialect(dialect)
    try:
        tree   = _with_tables(sqlglot.parse_one(query, read=read), read, sources)
        target = _with_tables(sqlglot.parse_one(expression, read=read), read, None)
    except (ParseError, TokenError):  # unparseable SQL, e.g. an unclosed quote
        return False

    def untold(node: exp.Expression) -> set[str]:
        return {column.name for column in node.find_all(exp.Column) if not column.table}

    def text(node: exp.Expression, untold: set[str]) -> str:
        """The SQL of the node, the columns of an untold table compared by their name only."""
        node = node.copy()
        for column in node.find_all(exp.Column):
            if column.name in untold:
                column.set("table", None)
        return node.sql(dialect=read, identify=True)

    for node in tree.find_all(type(target)):
        names = untold(node) | untold(target)
        if text(node, names) == text(target, names):
            return True
    return False


def _with_tables(tree: exp.Expression, read, sources: dict[str, str | None] | None) -> exp.Expression:
    """
    The tree, identifiers normalized as the dialect reads them, each column qualified by the name of its
    table, or by nothing when its table can't be told (see contains_expression).
    """
    tree = normalize_identifiers(tree, dialect=read)
    if tree.find(exp.Select):
        try:
            scopes = traverse_scope(tree)
        except SqlglotError:
            scopes = []
            for column in tree.find_all(exp.Column):
                _qualify(column, {})
        for scope in scopes:
            tables = {alias: source.name if isinstance(source, exp.Table) else None
                      for alias, source in scope.sources.items()}
            for column in scope.columns:
                _qualify(column, tables)
    elif sources is not None:
        tables = {alias.lower(): name.lower() if name else None for alias, name in sources.items()}
        for column in tree.find_all(exp.Column):
            _qualify(column, tables)
    return tree


def _qualify(column: exp.Column, tables: dict[str, str | None]) -> None:
    """The column qualified by the table its alias stands for; unqualified, by the only table there is."""
    if column.table:
        table = tables.get(column.table)
    else:
        table = next(iter(tables.values())) if len(tables) == 1 else None
    column.set("table", exp.to_identifier(table) if table else None)
    column.set("db", None)
    column.set("catalog", None)


# Type families of the column types read from a DDL, by their base name ("DECIMAL(8, 0)" -> decimal)
_INTEGER_TYPES = {"tinyint", "smallint", "int", "integer", "bigint"}
_NUMBER_TYPES  = _INTEGER_TYPES | {"decimal", "numeric", "number", "float", "double", "real"}
_TEXT_TYPES    = {"string", "varchar", "char", "text", "nvarchar", "nchar"}


def join_condition(source: str, target: str, column_types: dict[str, str], dialect: str) -> str:
    """
    Join condition of a declared relationship, each side written table.column:
    "invoice.customerid = customer.customerid".

    When one side is stored as text and the other as a number (column_types: {"table.column": type}),
    the text side is cast, written by sqlglot in the database dialect:
    "CAST(review.customerid AS BIGINT) = customer.customerid". Impala refuses to compare text with
    a number, Hive compares both as DOUBLE (inexact on long identifiers). The cast is to BIGINT
    (join keys are identifiers), or to the number type itself when it has decimals.
    No cast when a type is unknown or of another family (int vs decimal compare fine).
    """
    def base(side: str) -> str:
        return re.match(r"\w*", column_types.get(side, "").strip().lower()).group()

    if base(source) in _TEXT_TYPES and base(target) in _NUMBER_TYPES:
        text_side, number_side = source, target
    elif base(target) in _TEXT_TYPES and base(source) in _NUMBER_TYPES:
        text_side, number_side = target, source
    else:
        return f"{source} = {target}"

    # Decimals kept only when the number type has some: DECIMAL(10, 2), DOUBLE...
    number_type = column_types[number_side]
    has_decimals = (base(number_side) in {"float", "double", "real"}
                    or re.search(r",\s*[1-9]", number_type) is not None)
    read  = sqlglot_dialect(dialect)
    table, _, column = text_side.partition(".")
    try:
        cast = exp.cast(exp.column(column, table=table),
                        exp.DataType.build(number_type if has_decimals else "BIGINT", dialect=read))
    except (ParseError, TokenError, ValueError):   # a type sqlglot can't parse: no cast
        return f"{source} = {target}"

    cast = cast.sql(dialect=read)
    return f"{cast} = {target}" if text_side == source else f"{source} = {cast}"