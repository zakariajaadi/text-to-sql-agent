"""
Lineage of a SQL query, read by sqlglot without running it: the steps of the query (CTEs, derived
tables, subqueries, set operation branches, then the final SELECT), and for each step the tables it
reads with the columns it reads from them, its joins, filters and grouping, and for each output
column the columns of the database it is computed from.
Physical names only: the business labels of the semantic layer are added on top of it.
Used by: ui/streaming.py (Calcul tab of the result card)
"""

import re
from datetime import date, timedelta

import sqlglot
from sqlglot import exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import Scope, ScopeType, traverse_scope

from lib.sql_tools import sqlglot_dialect

# Clause of a SELECT a column is read in, by the arg name sqlglot gives it
_CLAUSES = {"expressions": "select", "joins": "join", "where": "where", "group": "group_by",
            "having": "having", "order": "order_by", "qualify": "qualify"}


def build_query_lineage(query: str, dialect: str, columns: dict[str, list[str]] | None = None) -> dict | None:
    """
    {"steps": [step, ...]}, the steps a step reads coming before it, the final query last
    (id "result"). Each step:
      id, kind      "result", "cte", "derived", "subquery", "branch" (of a set operation)
      operation     "SELECT", or "UNION ALL", "INTERSECT"... for a set operation
      sources       [{alias, name, kind: "table" | "step", columns: {column: [clause, ...]}}]
                    a "step" source names the step it reads; clause: select, join, where, group_by...
      joins         [{type, condition, sources: [alias, ...], tables, keys, pairs, implicit}], the
                    joined source first; tables: the names of those sources; keys: [[column, column],
                    ...] the column equalities of the condition, pairs: the same with their source
                    ([{source: alias, column}, {source: alias, column}]); implicit for a comma join
                    whose condition is written in the WHERE
      filters       [{condition, steps}], the WHERE split on AND; steps: the subqueries it reads
      group_by, having, order_by (SQL texts), limit (SQL text or None), distinct
      outputs       [{name, expression, formula, grain, aggregated, window, aggregate, reads, origins}]
                    formula: the expression without table qualifiers, * written ×
                    grain: {grain, column} when it truncates a date to a period, else None
                    aggregate: {function, distinct, operand, round} when it is one aggregation
                    (under a ROUND or a CAST), else None; see _Lineage._aggregate
                    reads: [{source: alias, column}], the columns of the step's own sources it reads
                    origins: [{table, column}], the columns of the database it is computed from,
                    through every step
                    a COUNT(*) reads the rows of every source (column "*")
      summary       what the clauses do, short, for the graph of the result card:
                    filters   [{kind, column, sql, ...}], see _Lineage._filter; a lower and an
                              upper bound of the same column make one range, or one period
                    group_by  [{column, grain, sql}]
                    having    [sql, ...]
                    order_by  [{label, descending}], label: the output it sorts on when it is one
                    base      alias of the source whose rows the step aggregates (the one its
                              aggregations read most), else its first source

    columns ({table: [column, ...]}) resolves the unqualified columns of a query reading several
    tables, and expands SELECT *; without it, an unqualified column is only resolved when its
    step reads a single source. None when sqlglot can't parse the query or it is not a query.
    """
    read = sqlglot_dialect(dialect)
    try:
        tree = sqlglot.parse_one(query, read=read)
        scopes = traverse_scope(tree) if isinstance(tree, exp.Query) else []
    except SqlglotError:
        return None
    if not scopes:
        return None
    return {"steps": _Lineage(scopes, read, columns or {}).steps}


class _Lineage:
    def __init__(self, scopes: list[Scope], read, columns: dict[str, list[str]]):
        self.read     = read
        self.reserved = {word for keyword in Dialect.get_or_raise(read).tokenizer_class.KEYWORDS for word in keyword.split()}
        self.schema   = {table.lower(): {c.lower(): c for c in cols} for table, cols in columns.items()}
        self.ids      = {}   # id(scope.expression) → step id
        self.outputs  = {}   # step id → outputs
        self.tables   = {}   # step id → tables of the database it reads, through the steps it reads
        self.counts   = {}
        self.steps    = [self._step(scope) for scope in scopes]

    def _sql(self, node: exp.Expression) -> str:
        """
        SQL text of a clause, to be read: identifiers unquoted (the generated query quotes them all,
        see normalize_sql), except the ones that need it (reserved word, space...).
        """
        node = node.copy()
        for identifier in node.find_all(exp.Identifier):
            if re.fullmatch(r"[A-Za-z_]\w*", identifier.name) and identifier.name.upper() not in self.reserved:
                identifier.set("quoted", False)
        return node.sql(dialect=self.read)

    # ── Steps ────────────────────────────────────────────────────────────────

    def _step_id(self, scope: Scope) -> str:
        if scope.is_root:
            return "result"
        parent = scope.expression.parent
        if scope.is_cte and isinstance(parent, exp.CTE):
            return parent.alias
        if scope.is_derived_table and isinstance(parent, exp.Subquery) and parent.alias:
            return parent.alias
        kind = self._kind(scope)
        self.counts[kind] = self.counts.get(kind, 0) + 1
        return f"{kind}_{self.counts[kind]}"

    @staticmethod
    def _kind(scope: Scope) -> str:
        return {ScopeType.ROOT: "result", ScopeType.CTE: "cte", ScopeType.DERIVED_TABLE: "derived",
                ScopeType.SET_OPERATION: "branch"}.get(scope.scope_type, "subquery")

    def _step(self, scope: Scope) -> dict:
        step_id = self._step_id(scope)
        self.ids[id(scope.expression)] = step_id
        step = {"id": step_id, "kind": self._kind(scope)}
        if isinstance(scope.expression, exp.SetOperation):
            step.update(self._set_operation(scope))
        else:
            step.update(self._select(scope))
        self.outputs[step_id] = step["outputs"]
        return step

    def _set_operation(self, scope: Scope) -> dict:
        """A UNION (INTERSECT, EXCEPT) reads its branches: output i comes from output i of each branch."""
        union    = scope.expression
        branches = [self.ids[id(b)] for b in (union.left, union.right) if id(b) in self.ids]
        all_rows = isinstance(union, exp.Union) and not union.args.get("distinct")
        first    = self.outputs[branches[0]] if branches else []
        outputs  = []
        for i, output in enumerate(first):
            same    = [(b, self.outputs[b][i]) for b in branches if i < len(self.outputs[b])]
            reads   = [{"source": b, "column": out["name"]} for b, out in same]
            origins = [o for _, out in same for o in out["origins"]]
            outputs.append({**output, "expression": output["name"], "formula": output["name"], "aggregate": None,
                            "reads": reads, "origins": _unique(origins)})
        self.tables[self.ids[id(union)]] = _unique([t for b in branches for t in self.tables[b]])
        return {
            "operation": type(union).__name__.upper() + (" ALL" if all_rows else ""),
            "sources":   [{"alias": b, "name": b, "kind": "step", "columns": {}} for b in branches],
            "joins": [], "filters": [], "group_by": [], "having": [], "order_by": self._order(union),
            "limit": self._limit(union), "distinct": False, "outputs": outputs,
            "summary": {"filters": [], "group_by": [], "having": [], "base": None,
                        "order_by": self._order_summary(union, [], outputs)},
        }

    def _select(self, scope: Scope) -> dict:
        select  = scope.expression
        sources = {alias: {"alias": alias, **self._source(source)} for alias, (_, source) in scope.selected_sources.items()}

        for column in scope.columns:
            # sqlglot lists the unqualified columns of a subquery as outer ones too: the subquery resolves them
            if not column.table and not _owned_by(column, select):
                continue
            found = self._resolve(scope, column)
            if found and found[0] is scope:   # a correlated reference is listed by the outer step
                _, alias, name = found
                roles = sources[alias]["columns"].setdefault(name, [])
                if (role := self._clause(column, select)) not in roles:
                    roles.append(role)

        self.tables[self.ids[id(select)]] = _unique(
            [t for s in sources.values() for t in ([s["name"]] if s["kind"] == "table" else self.tables.get(s["name"], []))])
        joins, filters, conditions = self._joins_and_filters(scope)
        group   = (select.args.get("group") or exp.Group()).expressions
        having  = _conjuncts(select.args.get("having"))
        outputs = self._outputs(scope)
        bodies  = [p.unalias() for p in select.expressions]
        return {
            "operation": "SELECT",
            "sources":   list(sources.values()),
            "joins":     joins,
            "filters":   filters,
            "group_by":  [self._sql(e) for e in group],
            "having":    [self._sql(c) for c in having],
            "order_by":  self._order(select),
            "limit":     self._limit(select),
            "distinct":  bool(select.args.get("distinct")),
            "outputs":   outputs,
            "summary": {
                "filters":  _merge_bounds([self._filter(c, f["steps"]) for c, f in zip(conditions, filters)]),
                "group_by": [self._group(e, bodies) for e in group],
                "having":   [self._plain(c) for c in having],
                "order_by": self._order_summary(select, bodies, outputs),
                "base":     _base(list(sources.values()), outputs),
            },
        }

    def _source(self, source) -> dict:
        if isinstance(source, Scope):
            return {"name": self.ids.get(id(source.expression), "?"), "kind": "step", "columns": {}}
        return {"name": source.name, "kind": "table", "columns": {}}

    def _order(self, query: exp.Query) -> list[str]:
        return [self._sql(o) for o in (query.args.get("order") or exp.Order()).expressions]

    def _limit(self, query: exp.Query) -> str | None:
        limit = query.args.get("limit")
        return self._sql(limit.expression) if limit and limit.expression else None

    @staticmethod
    def _clause(column: exp.Column, select: exp.Select) -> str:
        """Clause of the select the column is read in (a subquery's correlated column: the clause holding it)."""
        node = column
        while node.parent is not None and node.parent is not select:
            node = node.parent
        return _CLAUSES.get(node.arg_key, node.arg_key)

    # ── Joins and filters ────────────────────────────────────────────────────

    def _joins_and_filters(self, scope: Scope) -> tuple[list[dict], list[dict], list[exp.Expression]]:
        """
        The joins of the FROM, the WHERE split on AND, and the conditions of those filters.
        A WHERE equality between columns of two sources is the condition of a comma join: it moves
        to the joins, marked implicit.
        """
        select = scope.expression
        names  = {alias: self._source(source)["name"] for alias, (_, source) in scope.selected_sources.items()}

        def join(kind: str, condition: str | None, aliases: list[str], keys: list[list[str]],
                 pairs: list[list[dict]], implicit: bool) -> dict:
            return {"type": kind, "condition": condition, "sources": aliases, "tables": [names.get(a, a) for a in aliases],
                    "keys": keys, "pairs": pairs, "implicit": implicit}

        joins, filters, conditions, implicit = [], [], [], []
        for condition in _conjuncts(select.args.get("where")):
            aliases = self._aliases(scope, condition)
            if (isinstance(condition, exp.EQ) and len(aliases) == 2
                    and all(isinstance(side, exp.Column) for side in (condition.this, condition.expression))):
                implicit.append(join("INNER", self._sql(condition), aliases, _keys(condition), self._pairs(scope, condition), True))
            else:
                steps = [self.ids[id(q)] for q in condition.find_all(exp.Query) if id(q) in self.ids]
                filters.append({"condition": self._sql(condition), "steps": _unique(steps)})
                conditions.append(condition)

        joined_implicitly = {alias for j in implicit for alias in j["sources"]}
        for node in select.args.get("joins") or []:
            alias = node.this.alias_or_name
            kind  = " ".join(part for part in (node.side, node.kind) if part) or "INNER"
            if on := node.args.get("on"):
                joins.append(join(kind, self._sql(on), _unique([alias, *self._aliases(scope, on)]), _keys(on),
                                  self._pairs(scope, on), False))
            elif using := node.args.get("using"):
                columns = [u.name for u in using]
                joins.append(join(kind, f"USING ({', '.join(self._sql(u) for u in using)})", [alias],
                                  [[c, c] for c in columns], self._using_pairs(scope, alias, columns), False))
            elif alias not in joined_implicitly:
                joins.append(join(kind, None, [alias], [], [], False))
        return joins + implicit, filters, conditions

    def _pairs(self, scope: Scope, condition: exp.Expression) -> list[list[dict]]:
        """[{source, column}, {source, column}] of each column equality of a join condition."""
        pairs = []
        for eq in condition.find_all(exp.EQ):
            if not (isinstance(eq.this, exp.Column) and isinstance(eq.expression, exp.Column)):
                continue
            sides = [self._resolve(scope, column) for column in (eq.this, eq.expression)]
            if all(side and side[0] is scope for side in sides):
                pairs.append([{"source": alias, "column": name} for _, alias, name in sides])
        return pairs

    def _using_pairs(self, scope: Scope, alias: str, columns: list[str]) -> list[list[dict]]:
        """JOIN ... USING (c): the joined source and the source before it that has the column (else the one just before)."""
        sources = scope.selected_sources
        before  = list(sources)[:list(sources).index(alias)] if alias in sources else []
        pairs   = []
        for name in columns:
            other = next((a for a in before if self._has_column(sources[a][1], name) is True), before[-1] if before else None)
            if other is not None:
                pairs.append([{"source": alias, "column": self._column_name(sources[alias][1], name)},
                              {"source": other, "column": self._column_name(sources[other][1], name)}])
        return pairs

    def _aliases(self, scope: Scope, node: exp.Expression) -> list[str]:
        """Sources of the scope whose columns the expression reads (its own columns, not a subquery's)."""
        found = (self._resolve(scope, c) for c in node.find_all(exp.Column) if _owned_by(c, scope.expression))
        return _unique([f[1] for f in found if f and f[0] is scope])

    # ── Columns ──────────────────────────────────────────────────────────────

    def _resolve(self, scope: Scope, column: exp.Column) -> tuple[Scope, str, str] | None:
        """
        (scope, alias, column name) of the source a column reads: the scope or an outer one for a
        correlated reference. None when it can't be told (several sources and no schema), or when
        the name is an output alias of the select (ORDER BY revenue).
        """
        name = column.name
        if column.table:
            current = scope
            while current is not None:
                alias = _find(current.selected_sources, column.table) or _find(current.sources, column.table)
                if alias is not None:
                    return current, alias, self._column_name(current.sources[alias], name)
                current = current.parent
            return None

        current = scope
        while current is not None:
            sources = current.selected_sources
            having  = [a for a, (_, s) in sources.items() if self._has_column(s, name) is True]
            unknown = [a for a, (_, s) in sources.items() if self._has_column(s, name) is None]
            if len(having) == 1:
                alias = having[0]
            elif not having and len(sources) == 1 and unknown:
                alias = unknown[0]
            elif not having and len(unknown) == 1:
                alias = unknown[0]
            else:
                alias = None
            if alias is not None:
                return current, alias, self._column_name(sources[alias][1], name)
            if having or unknown:   # ambiguous between several sources
                return None
            # No source of a subquery has it: a correlated reference to the outer query
            current = current.parent if current.scope_type == ScopeType.SUBQUERY else None
        return None

    def _has_column(self, source, name: str) -> bool | None:
        """Whether a source has the column: None when unknown (table not in the schema, step with a *)."""
        if isinstance(source, Scope):
            outputs = self.outputs.get(self.ids.get(id(source.expression)), None)
            if outputs is None or any(o["name"] == "*" for o in outputs):
                return None
            return any(o["name"].lower() == name.lower() for o in outputs)
        table = self.schema.get(source.name.lower())
        return None if table is None else name.lower() in table

    def _column_name(self, source, name: str) -> str:
        """Column name as the schema writes it, as the query writes it otherwise."""
        if isinstance(source, exp.Table):
            return self.schema.get(source.name.lower(), {}).get(name.lower(), name)
        return name

    def _origins(self, scope: Scope, expression: exp.Expression) -> list[dict]:
        """Columns of the database an expression of the scope is computed from, through every step."""
        origins = []
        for column in expression.find_all(exp.Column):
            if not _owned_by(column, scope.expression) or isinstance(column.this, exp.Star):
                continue
            found = self._resolve(scope, column)
            if not found:
                continue
            owner, alias, name = found
            source = owner.sources[alias]
            if isinstance(source, exp.Table):
                origins.append({"table": source.name, "column": name})
            else:
                outputs = self.outputs.get(self.ids.get(id(source.expression)), [])
                origins += [o for out in outputs if out["name"].lower() == name.lower() for o in out["origins"]]
        # COUNT(*) reads no column: the rows of the tables it counts
        if _counts_rows(expression, scope):
            origins += [{"table": t, "column": "*"} for t in self.tables[self.ids[id(scope.expression)]]]
        # A scalar subquery in the expression: the columns its outputs come from
        for query in expression.find_all(exp.Query):
            if id(query) in self.ids:
                origins += [o for out in self.outputs[self.ids[id(query)]] for o in out["origins"]]
        return _unique(origins)

    def _reads(self, scope: Scope, expression: exp.Expression) -> list[dict]:
        """Columns of the scope's own sources an expression reads, the rows ("*") of each for a COUNT(*)."""
        found = (self._resolve(scope, c) for c in expression.find_all(exp.Column)
                 if _owned_by(c, scope.expression) and not isinstance(c.this, exp.Star))
        reads = [{"source": f[1], "column": f[2]} for f in found if f and f[0] is scope]
        if _counts_rows(expression, scope):
            reads += [{"source": alias, "column": "*"} for alias in scope.selected_sources]
        return _unique(reads)

    def _outputs(self, scope: Scope) -> list[dict]:
        outputs = []
        for projection in scope.expression.expressions:
            star = projection if isinstance(projection, exp.Star) else (
                projection if isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star) else None)
            if star is not None:
                outputs += self._star_outputs(scope, star)
                continue
            body  = projection.unalias()
            grain = _grain(body)
            # SUM(x) OVER (...) is a window, SUM(x) in its ORDER BY aggregates the GROUP BY
            aggregated = any(not (isinstance(agg.parent, exp.Window) and agg.arg_key == "this")
                             for agg in body.find_all(exp.AggFunc) if _owned_by(agg, scope.expression))
            window = body.find(exp.Window) is not None
            outputs.append({
                "name":       projection.alias_or_name or self._sql(projection),
                "expression": self._sql(body),
                "formula":    self._plain(body),
                "grain":      {"grain": grain[0], "column": grain[1].name} if grain else None,
                "aggregated": aggregated,
                "window":     window,
                "aggregate":  self._aggregate(body) if aggregated and not window else None,
                "reads":      self._reads(scope, body),
                "origins":    self._origins(scope, body),
            })
        return outputs

    def _star_outputs(self, scope: Scope, star: exp.Expression) -> list[dict]:
        """SELECT * (or t.*): the columns of its sources when the schema or their step knows them, a "*" otherwise."""
        sources = scope.selected_sources
        if isinstance(star, exp.Column) and star.table:
            alias   = _find(sources, star.table)
            sources = {alias: sources[alias]} if alias else {}
        outputs = []
        for alias, (_, source) in sources.items():
            if isinstance(source, Scope):
                known = self.outputs.get(self.ids.get(id(source.expression)))
                known = None if known is None or any(o["name"] == "*" for o in known) else known
                columns = [(o["name"], o["origins"]) for o in known] if known is not None else None
            else:
                table   = self.schema.get(source.name.lower())
                columns = [(c, [{"table": source.name, "column": c}]) for c in table.values()] if table else None
            if columns is None:
                return [{"name": "*", "expression": self._sql(star), "formula": "*", "grain": None,
                         "aggregated": False, "window": False, "aggregate": None,
                         "reads": [{"source": a, "column": "*"} for a in sources], "origins": []}]
            outputs += [{"name": name, "expression": f"{alias}.{name}", "formula": name, "grain": None,
                         "aggregated": False, "window": False, "aggregate": None,
                         "reads": [{"source": alias, "column": name}], "origins": origins}
                        for name, origins in columns]
        return outputs

    # ── Summaries ────────────────────────────────────────────────────────────
    # What a clause does, short: the graph of the result card phrases it, the SQL stays the reference

    def _plain(self, node: exp.Expression) -> str:
        """SQL text without table qualifiers, multiplications written ×: SUM(il.unitprice * il.quantity) → SUM(unitprice × quantity)."""
        node = node.copy()
        for column in node.find_all(exp.Column):
            column.set("table", None)
        return self._sql(node).replace(" * ", " × ")

    def _aggregate(self, body: exp.Expression) -> dict | None:
        """
        An output that is one aggregation, under a ROUND or a CAST: {function, distinct, operand, round}.
        ROUND(SUM(od.unitprice * od.quantity), 2) → {"sum", False, "unitprice × quantity", 2};
        COUNT(*) → {"count", False, "*", None}. None for anything else: a ratio of two sums, a CASE...
        """
        node, rounding = body, None
        while isinstance(node, (exp.Round, exp.Cast, exp.TryCast, exp.Paren)):
            if isinstance(node, exp.Round):
                decimals = _value(node.args.get("decimals"))
                rounding = int(decimals) if decimals and decimals.isdigit() else 0
            node = node.this
        function = _AGGREGATES.get(type(node))
        if function is None:
            return None
        operand  = node.this
        distinct = isinstance(operand, exp.Distinct)
        if distinct:
            operand = operand.expressions[0] if operand.expressions else None
        return {"function": function, "distinct": distinct, "round": rounding,
                "operand": "*" if operand is None or isinstance(operand, exp.Star) else self._plain(operand)}

    def _filter(self, condition: exp.Expression, steps: list[str]) -> dict:
        """
        One condition of the WHERE, by kind:
          period    {grain, value}: YEAR(d) = 2023, strftime('%Y-%m', d) = '2023-03'; merged from
                    a range, with its low, high and last (the last day it keeps)
          compare   {op, value, grain}: d >= '2023-01-01' (dates without a midnight time), YEAR(d) > 2020
          range     {low, low_op, high, high_op, grain}: BETWEEN, or two bounds merged (_merge_bounds)
          in        {values, negated}
          subquery  {steps, negated}: IN (SELECT ...), EXISTS (...)
          like      {mode: starts | ends | contains | pattern, value, negated}
          null      {negated}: IS NULL, IS NOT NULL (negated)
          sql       anything else
        Each with column (the column it tests, None when it can't be told) and sql (without qualifiers).
        """
        negated = isinstance(condition, exp.Not)
        node    = condition.this.unnest() if negated else condition
        columns = {c.name for c in condition.find_all(exp.Column)}
        item    = {"kind": "sql", "column": next(iter(columns)) if len(columns) == 1 else None, "sql": self._plain(condition)}

        def subject(expression: exp.Expression) -> dict | None:
            """Column (and grain) a condition tests: d, CAST(d AS DATE), YEAR(d)..."""
            if grain := _grain(expression):
                return {"column": grain[1].name, "grain": grain[0]}
            base = _unwrap(expression)
            return {"column": base.name, "grain": None} if isinstance(base, exp.Column) else None

        if type(node) in _OPERATORS:
            left, right, op = node.this, node.expression, _OPERATORS[type(node)]
            if _value(left) is not None and _value(right) is None:
                left, right, op = right, left, _FLIPPED[op]
            value, tested = _value(right), subject(left)
            if value is None or tested is None:
                return item
            op = _NEGATED[op] if negated else op
            if op == "=" and tested["grain"]:
                return {**item, "kind": "period", "column": tested["column"], "grain": tested["grain"], "value": value}
            return {**item, "kind": "compare", **tested, "op": op, "value": _day(value)}

        if isinstance(node, exp.Between) and not negated and (tested := subject(node.this)):
            low, high = _value(node.args["low"]), _value(node.args["high"])
            if low is not None and high is not None:
                return {**item, "kind": "range", **tested, "low": _day(low), "low_op": ">=", "high": _day(high), "high_op": "<="}
        if isinstance(node, exp.In) and (tested := subject(node.this)):
            if node.args.get("query"):
                return {**item, "kind": "subquery", "column": tested["column"], "steps": steps, "negated": negated}
            values = [_value(v) for v in node.expressions]
            if values and None not in values:
                return {**item, "kind": "in", "column": tested["column"], "values": values, "negated": negated}
        if isinstance(node, exp.Exists):
            return {**item, "kind": "subquery", "column": None, "steps": steps, "negated": negated}
        if isinstance(node, (exp.Like, exp.ILike)) and (tested := subject(node.this)):
            pattern = _value(node.expression)
            if pattern is not None:
                core = pattern.strip("%")
                mode = ("pattern" if re.search(r"[%_]", core) or not core else
                        "contains" if pattern.startswith("%") and pattern.endswith("%") else
                        "ends" if pattern.startswith("%") else "starts" if pattern.endswith("%") else "pattern")
                return {**item, "kind": "like", "column": tested["column"], "mode": mode,
                        "value": core if mode != "pattern" else pattern, "negated": negated}
        if isinstance(node, exp.Is) and isinstance(node.expression, exp.Null) and (tested := subject(node.this)):
            return {**item, "kind": "null", "column": tested["column"], "negated": negated}
        return item

    def _group(self, expression: exp.Expression, bodies: list[exp.Expression]) -> dict:
        """A GROUP BY expression: {column, grain, sql}; GROUP BY 2 stands for the second output."""
        if isinstance(expression, exp.Literal) and expression.is_int and 0 < int(expression.this) <= len(bodies):
            expression = bodies[int(expression.this) - 1]
        grain = _grain(expression)
        base  = grain[1] if grain else _unwrap(expression)
        return {"column": base.name if isinstance(base, exp.Column) else None,
                "grain": grain[0] if grain else None, "sql": self._plain(expression)}

    def _order_summary(self, query: exp.Query, bodies: list[exp.Expression], outputs: list[dict]) -> list[dict]:
        """ORDER BY: {label, descending}, label the output sorted on (its name, its expression or its position)."""
        names, summary = [o["name"] for o in outputs], []
        for ordered in (query.args.get("order") or exp.Order()).expressions:
            expression, label = ordered.this, None
            if isinstance(expression, exp.Literal) and expression.is_int and 0 < int(expression.this) <= len(names):
                label = names[int(expression.this) - 1]
            elif isinstance(expression, exp.Column) and not expression.table:
                label = next((n for n in names if n.lower() == expression.name.lower()), None)
            if label is None:
                plain = self._plain(expression)
                label = next((n for n, b in zip(names, bodies) if self._plain(b) == plain), plain)
            summary.append({"label": label, "descending": bool(ordered.args.get("desc"))})
        return summary


# Aggregations the graph phrases in words; any other one shows as its formula
_AGGREGATES = {exp.Sum: "sum", exp.Avg: "avg", exp.Count: "count", exp.Min: "min", exp.Max: "max"}

# Comparison operators, as written once the column is on the left
_OPERATORS = {exp.EQ: "=", exp.NEQ: "<>", exp.GT: ">", exp.GTE: ">=", exp.LT: "<", exp.LTE: "<="}
_FLIPPED   = {"=": "=", "<>": "<>", ">": "<", ">=": "<=", "<": ">", "<=": ">="}
_NEGATED   = {"=": "<>", "<>": "=", ">": "<=", ">=": "<", "<": ">=", "<=": ">"}

# Conversions around a date column or a date literal: CAST(d AS DATE), TO_DATE(d), DATE('2023-01-01')...
_CONVERSIONS = (exp.Cast, exp.TryCast, exp.TsOrDsToDate, exp.TsOrDsToTimestamp, exp.TimeStrToTime,
                exp.StrToTime, exp.StrToDate, exp.Date, exp.Paren)
# Parsing functions sqlglot keeps by name (Impala TO_TIMESTAMP(d, 'dd/MM/yyyy')): the value is the first argument
_PARSERS = {"TO_TIMESTAMP", "TO_DATE", "STR_TO_DATE", "PARSE_DATE", "PARSE_TIMESTAMP", "DATE", "DATETIME"}
# Formatting functions of a date, by name, when sqlglot doesn't map them in the dialect
_FORMATTERS = {"DATE_FORMAT", "FORMAT_DATE", "TO_CHAR", "STRFTIME"}
# Units of DATE_TRUNC, TRUNC...
_UNITS = {"year": "year", "yyyy": "year", "yy": "year", "y": "year", "quarter": "quarter", "q": "quarter",
          "month": "month", "mm": "month", "mon": "month", "week": "week", "wk": "week", "ww": "week",
          "day": "day", "dd": "day", "d": "day", "ddd": "day"}


def _unwrap(node: exp.Expression) -> exp.Expression:
    """The value under its date conversions: CAST(d AS DATE), TO_TIMESTAMP(d, 'dd/MM/yyyy')... → d."""
    while True:
        if isinstance(node, _CONVERSIONS):
            node = node.this
        elif isinstance(node, exp.Anonymous) and node.name.upper() in _PARSERS and node.expressions:
            node = node.expressions[0]
        else:
            return node


def _grain(node: exp.Expression) -> tuple[str, exp.Column] | None:
    """
    (grain, column) of a date cut to a period: year, quarter, month, week, day, or month_of_year
    for a month number. YEAR(d), DATE_FORMAT(d, 'yyyy-MM'), strftime('%Y-%m', d), DATE_TRUNC('month', d),
    TRUNC(d, 'MM'), SUBSTR(d, 1, 7), CONCAT(YEAR(d), '-', MONTH(d))... None for anything else.
    """
    node, grain, base = _unwrap(node), None, None
    if isinstance(node, exp.Year):
        grain = "year"
    elif isinstance(node, exp.Quarter):
        grain = "quarter"
    elif isinstance(node, exp.Month):
        grain = "month_of_year"
    elif isinstance(node, (exp.WeekOfYear, exp.Week)):
        grain = "week"
    elif isinstance(node, exp.TimeToStr):
        grain = _format_grain(node.args["format"].name if node.args.get("format") else "")
    elif isinstance(node, exp.Anonymous) and node.name.upper() in _FORMATTERS and len(node.expressions) == 2:
        # A formatting function sqlglot doesn't know in this dialect: DATE_FORMAT(d, 'yyyy-MM') read as SQLite
        node  = exp.TimeToStr(this=node.expressions[0], format=node.expressions[1])
        grain = _format_grain(_value(node.args["format"]) or "")
    elif isinstance(node, (exp.DateTrunc, exp.TimestampTrunc, exp.DatetimeTrunc, exp.TimeTrunc)):
        unit  = node.args.get("unit")
        grain = _UNITS.get(unit.name.lower()) if unit is not None else None
    elif isinstance(node, exp.Substring):
        start, length = _value(node.args.get("start") or exp.Literal.number(0)), _value(node.args.get("length") or exp.Null())
        grain = {"4": "year", "7": "month", "10": "day"}.get(length) if start == "1" else None
    elif isinstance(node, exp.Concat):
        parts = [_grain(part) for part in node.expressions]
        found = {g[0]: g[1] for g in parts if g}
        if "year" in found and "month_of_year" in found and found["year"].name == found["month_of_year"].name:
            return "month", found["year"]
        return None
    if grain is None:
        return None
    base = _unwrap(node.this)
    return (grain, base) if isinstance(base, exp.Column) else None


def _format_grain(text: str) -> str | None:
    """Period a date format keeps, strftime ('%Y-%m') or Java and Oracle ('yyyy-MM', 'YYYY-MM') patterns."""
    year  = re.search(r"%Y|yyyy|YYYY", text)
    grain = ("day" if re.search(r"%[dej]|dd|DD", text) else "week" if re.search(r"%[WUV]|ww|IW", text) else
             ("month" if year else "month_of_year") if re.search(r"%[mbB]|MM|MON", text) else
             "year" if year else None)
    return grain


def _value(node: exp.Expression | None) -> str | None:
    """The literal a node holds, under its conversions: '2023', 2023, DATE '2023-01-01', -3. None for anything else."""
    node = _unwrap(node) if node is not None else None
    if isinstance(node, exp.Literal):
        return str(node.this)
    if isinstance(node, exp.Neg) and isinstance(node.this, exp.Literal):
        return f"-{node.this.this}"
    if isinstance(node, exp.Boolean):
        return "TRUE" if node.this else "FALSE"
    return None


def _day(value: str) -> str:
    """A date bound without its time when it is a day's start or end: '2023-12-31 23:59:59' → '2023-12-31'."""
    return re.sub(r"^(\d{4}-\d{2}-\d{2})[ T](00:00:00|23:59:59)(\.0+|\.9+)?$", r"\1", value)


def _merge_bounds(filters: list[dict]) -> list[dict]:
    """
    A lower and an upper bound of the same column (and grain) make one range, in place of the first;
    a range covering exactly one year or one month of dates is that period.
    """
    merged, used = [], set()
    for i, item in enumerate(filters):
        if i in used:
            continue
        if item["kind"] == "compare" and item["op"] in (">", ">=", "<", "<="):
            lower = item["op"].startswith(">")
            pair  = next((j for j, other in enumerate(filters) if j > i and j not in used and other["kind"] == "compare"
                          and (other["column"], other["grain"]) == (item["column"], item["grain"])
                          and other["op"].startswith("<" if lower else ">") and other["op"] != "<>"), None)
            if pair is not None:
                used.add(pair)
                low, high = (item, filters[pair]) if lower else (filters[pair], item)
                item = {"kind": "range", "column": item["column"], "grain": item["grain"], "sql": f"{low['sql']}\n{high['sql']}",
                        "low": low["value"], "low_op": low["op"], "high": high["value"], "high_op": high["op"]}
        if item["kind"] == "range" and not item["grain"] and (period := _period(item)):
            item = {**item, "kind": "period", "grain": period[0], "value": period[1], "last": period[2]}
        merged.append(item)
    return merged


def _period(item: dict) -> tuple[str, str, str] | None:
    """
    (grain, period, last day) when a range of dates covers exactly one year or one month:
    ("year", "2023", "2023-12-31"), ("month", "2023-03", "2023-03-31"), the upper bound excluded or not.
    """
    try:
        low, high = date.fromisoformat(item["low"]), date.fromisoformat(item["high"])
    except ValueError:
        return None
    if item["low_op"] != ">=" or low.day != 1:
        return None
    last = high if item["high_op"] == "<=" else high - timedelta(days=1)
    if low.month == 1 and last == date(low.year, 12, 31):
        return "year", str(low.year), last.isoformat()
    next_month = date(low.year + low.month // 12, low.month % 12 + 1, 1)
    if last == next_month - timedelta(days=1):
        return "month", f"{low.year}-{low.month:02d}", last.isoformat()
    return None


def _base(sources: list[dict], outputs: list[dict]) -> str | None:
    """Alias of the source whose rows a step aggregates: the one its aggregations read most (the first on a tie), else its first source."""
    reads = [read["source"] for output in outputs if output["aggregated"] for read in output["reads"]]
    if reads:
        return max(dict.fromkeys(reads), key=reads.count)
    return sources[0]["alias"] if sources else None


def _keys(condition: exp.Expression) -> list[list[str]]:
    """[left, right] column names of each column equality of a join condition."""
    return [[eq.this.name, eq.expression.name] for eq in condition.find_all(exp.EQ)
            if isinstance(eq.this, exp.Column) and isinstance(eq.expression, exp.Column)]


def _conjuncts(node: exp.Expression | None) -> list[exp.Expression]:
    """The conditions of a WHERE or HAVING, split on AND (parentheses removed)."""
    if node is None:
        return []
    condition = node.this if isinstance(node, (exp.Where, exp.Having)) else node
    return [c.unnest() for c in condition.flatten()] if isinstance(condition, exp.And) else [condition.unnest()]


def _counts_rows(expression: exp.Expression, scope: Scope) -> bool:
    """Whether the expression holds a COUNT(*) of the scope."""
    return any(isinstance(count.this, exp.Star) for count in expression.find_all(exp.Count)
               if _owned_by(count, scope.expression))


def _owned_by(node: exp.Expression, select: exp.Expression) -> bool:
    """Whether the node belongs to the select itself, not to a subquery nested in it."""
    return node.find_ancestor(exp.Query) is select


def _find(mapping: dict, alias: str) -> str | None:
    """Key of the mapping naming the alias, compared case-insensitively."""
    return next((key for key in mapping if key.lower() == alias.lower()), None)


def _unique(items: list) -> list:
    """Items in their first order, without repeats (dicts compared by value)."""
    seen, unique = set(), []
    for item in items:
        key = tuple(sorted(item.items())) if isinstance(item, dict) else item
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique
