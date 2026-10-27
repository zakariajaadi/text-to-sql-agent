# offline/studio/metrics.py

"""
The Metrics screen of the studio (pure functions, unit-tested): the metrics of metrics.yaml as the
page shows them, and what a test of a metric measures on the data.

  - without the database: the startup validation of the agent (lib/metrics.py) against the columns
    it sees (a hidden column stops it), the joins of the metric through the relations (the query a
    test runs), the rules of the generator (skip_reason) applied to the metrics already written
  - test: one query of counts, never a value of the data: the rows of a summed table the joins count
    twice or lose, the rows the measure leaves empty, the rows without the date, the sign of the result
  - the proposals a run of the generator skipped and the metrics it added, read from its lines
"""

import re

import sqlglot
from sqlglot import exp

from lib.metrics import validate_metric
from lib.sql_tools import sqlglot_dialect
from offline.semantic_gen.utils import (
    COUNT_ONLY,
    DISTINCT_SUM,
    KNOWN_NAME,
    METRICS_KEY_ORDER,
    SAME_EXPRESSION,
    metric_from_clause,
    skip_reason,
)
from offline.studio.state import column_kind

FILLED    = 0.9    # a measure or a date filled under it is shown in amber (the 90 % of the relations)
ROWS_KEPT = 0.99   # the joins keeping fewer rows of a summed table: amber

# The reasons of skip_reason, as codes the page says in French
REASONS = {KNOWN_NAME: "known", SAME_EXPRESSION: "same", DISTINCT_SUM: "distinct", COUNT_ONLY: "count"}
# The fields of a metric that can't be empty, as a message names them
FIELDS = {"description": "La description", "expr": "La formule"}

# Lines of a run of the generator (offline/semantic_gen/metrics.py)
SKIPPED  = re.compile(r"\] Skipped metric '(?P<name>[^']+)': (?P<reason>.*)$")
REJECTED = re.compile(r"\] Rejected metric '(?P<name>[^']+)': (?P<reason>.*)$")
ADDED    = re.compile(r"\] \d+ new metric\(s\) added to \S+: (?P<names>.*) — review them\.$")
ANSI     = re.compile(r"\x1b\[[0-9;]*m")   # the underline of sqlglot in a parse error


def metric_row(name: str, entry: dict | None, entries: dict[str, dict], ignored: list[str],
               columns: dict[str, list[str]], hidden: dict[str, set[str]], fk_relations: list[tuple[str, str]],
               column_types: dict[str, str], formats: dict[str, dict], dialect: str) -> dict:
    """
    A metric as the page shows it, checked without the database: `error` when the agent would refuse
    it at startup or its tables can't be joined ({kind, detail}), `rule` when the generator would not
    propose it today (`rule_with`: the metric of the same expression), `sql` the query a test computes
    it on, `dates` the columns it can take as its date, `key` what a test result holds for.
    columns: {table: [column]} of the database ({}: syntax only), hidden: {table: {column}} the agent doesn't see.
    """
    entry = entry or {}
    row = {
        "name": name, "description": entry.get("description") or "", "synonyms": list(entry.get("synonyms") or []),
        "expr": entry.get("expr") or "", "time_dimension": entry.get("time_dimension") or "",
        "origin": entry.get("origin"), "edited": list(entry.get("edited") or []),
        "tables": [], "joins": 0, "sql": None, "from": None, "dates": [], "error": None, "rule": None, "rule_with": None,
    }
    others = {n: e for n, e in entries.items() if n != name}
    if reason := skip_reason(name, row["expr"], others, ignored, dialect):
        row["rule"] = REASONS[reason]
        if reason == SAME_EXPRESSION:
            row["rule_with"] = next(n for n, e in others.items()
                                    if skip_reason(name, row["expr"], {n: e}, [], dialect) == SAME_EXPRESSION)
    try:
        metric = validate_metric(name, entry, dialect, columns)
    except ValueError as e:
        row["error"] = {"kind": "invalid", "detail": ANSI.sub("", str(e))}
        return row
    row["tables"] = metric.tables
    row["dates"]  = date_columns(metric.tables, fk_relations, column_types, formats, row["time_dimension"])
    if seen := [f"{t}.{c}" for t, c in metric.columns if c in hidden.get(t, ())]:
        row["error"] = {"kind": "hidden", "detail": ", ".join(seen)}
    try:
        row["from"] = metric_from_clause(metric.tables, fk_relations, column_types, dialect)
    except ValueError as e:
        row["error"] = row["error"] or {"kind": "joins", "detail": str(e)}
        return row
    row["joins"] = len(metric.tables) - 1
    row["sql"]   = f"SELECT {metric.expr}\n{row['from']}"
    return row


def measure_key(row: dict) -> str:
    """What a test result holds for: the query and the date of the metric (an edit makes it stale)."""
    return f"{row['sql']}\n{row['time_dimension']}"


def date_columns(tables: list[str], fk_relations: list[tuple[str, str]], column_types: dict[str, str],
                 formats: dict[str, dict], current: str = "") -> list[str]:
    """
    The columns a metric can take as its date: DATE or TIMESTAMP, or text read as a date (`formats`
    of tables.yaml), in its tables then in the tables they join to. The current one always in.
    """
    near = list(tables)
    for source, target in fk_relations:
        for a, b in ((source, target), (target, source)):
            if a.split(".")[0] in tables and b.split(".")[0] not in near:
                near.append(b.split(".")[0])
    dates = []
    for ref, column_type in column_types.items():
        table, _, column = ref.partition(".")
        read_as = ((formats.get(table) or {}).get(column) or "")
        if table in near and (column_kind(column_type) == "date" or read_as.startswith("date")):
            dates.append(ref)
    dates.sort(key=lambda ref: near.index(ref.split(".")[0]))
    return dates if not current or current in dates else [current, *dates]


def measure_queries(row: dict, column_types: dict[str, str], fk_relations: list[tuple[str, str]], dialect: str) -> dict:
    """
    The queries of a test, never a value of the data. `main`: one row of counts on the joins of the
    metric (its rows, the rows where each measure is filled, the rows with the date, the sign of the
    result). `tables`: the rows of each summed table alone, when the metric joins, to tell the rows the
    joins count twice or lose; a table another one of the same argument points to is a lookup (a price,
    a duration), its rows are not the ones summed. `measures`: the argument of each SUM and AVG,
    `conditional` when a CASE empties it on purpose. COUNT(DISTINCT ...) counts once whatever the joins: not checked.
    """
    read = sqlglot_dialect(dialect)
    tree = sqlglot.parse_one(row["expr"], read=read)
    measures, summed = [], []
    for aggregate in tree.find_all(exp.Sum, exp.Avg, exp.Count):
        argument = aggregate.this
        if isinstance(argument, (exp.Distinct, exp.Star)):
            continue
        tables = list(dict.fromkeys(c.table for c in argument.find_all(exp.Column)))
        looked_up = {target.split(".")[0] for source, target in fk_relations
                     if source.split(".")[0] in tables and source.split(".")[0] != target.split(".")[0]}
        summed += [table for table in tables if table not in looked_up]
        sql = argument.sql(dialect=read)
        if not isinstance(aggregate, exp.Count) and sql not in [m["sql"] for m in measures]:
            measures.append({"sql": sql, "conditional": argument.find(exp.Case, exp.If) is not None})

    date, counts = row["time_dimension"], ["COUNT(*)", *(f"COUNT({m['sql']})" for m in measures)]
    if date:
        counts.append(f"COUNT(CASE WHEN TRIM({date}) <> '' THEN 1 END)" if column_kind(column_types.get(date)) == "text"
                      else f"COUNT({date})")
    value = f"({row['expr']})"
    counts.append(f"CASE WHEN {value} IS NULL THEN 'empty' WHEN {value} = 0 THEN 'zero' "
                  f"WHEN {value} > 0 THEN 'positive' ELSE 'negative' END")
    return {
        "main":     f"SELECT {', '.join(counts)}\n{row['from']}",
        "tables":   {table: f"SELECT COUNT(*) FROM {table}" for table in dict.fromkeys(summed)} if row["joins"] else {},
        "measures": measures,
        "date":     date,
    }


def read_measures(plan: dict, main: tuple, table_rows: dict[str, int], seconds: float) -> dict:
    """
    What the counts of a test say, each check with `warn` when it is worth a look: a summed table
    whose rows the joins count twice or lose, a measure or a date filled under FILLED, a result
    empty or zero on the whole data (a condition that finds no row).
    """
    rows, n = int(main[0] or 0), len(plan["measures"])
    share = lambda count: count / rows if rows else None  # noqa: E731
    tables = [{"table": t, "rows": int(r or 0), "counted": rows,
               "warn": rows > int(r or 0) or rows < int(r or 0) * ROWS_KEPT} for t, r in table_rows.items()]
    measures = [{**m, "filled": int(f or 0), "share": share(int(f or 0)),
                 "warn": bool(rows) and not m["conditional"] and int(f or 0) < rows * FILLED}
                for m, f in zip(plan["measures"], main[1:1 + n])]
    date = None
    if plan["date"]:
        dated = int(main[1 + n] or 0)
        date  = {"column": plan["date"], "filled": dated, "share": share(dated), "warn": bool(rows) and dated < rows * FILLED}
    sign = main[-1]
    return {
        "rows": rows, "tables": tables, "measures": measures, "date": date, "sign": sign, "seconds": round(seconds, 1),
        "warn": sign in ("empty", "zero") or any(c["warn"] for c in [*tables, *measures, *([date] if date else [])]),
    }


def metric_state(row: dict, test: dict | None) -> str:
    """error (the agent refuses it, or the database), check (a rule or a test to look at), ok (tested), untested."""
    if row["error"] or (test and test.get("error")):
        return "error"
    if row["rule"] or (test and test["warn"]):
        return "check"
    return "ok" if test else "untested"


def metrics_view(entries: dict[str, dict], ignored: list[str], columns: dict[str, list[str]],
                 hidden: dict[str, set[str]], formats: dict[str, dict], fk_relations: list[tuple[str, str]],
                 column_types: dict[str, str], dialect: str, tests: dict[str, dict], run: dict) -> dict:
    """
    Everything the Metrics screen shows: each metric with its checks and its last test (`tests`:
    {name: {key, result}}, kept while the metric is unchanged), the ignored names, what the last run
    of the generator skipped and added.
    """
    rows = []
    for name, entry in entries.items():
        row  = metric_row(name, entry, entries, ignored, columns, hidden, fk_relations, column_types, formats, dialect)
        test = tests.get(name)
        row["test"]  = test["result"] if test and row["sql"] and test["key"] == measure_key(row) else None
        row["state"] = metric_state(row, row["test"])
        rows.append(row)
    return {
        "metrics": rows, "ignored": ignored, "skipped": run.get("skipped") or [], "added": run.get("added") or [],
        "counts": {state: sum(r["state"] == state for r in rows) for state in ("ok", "check", "error", "untested")},
    }


def last_run(lines: list[dict]) -> dict:
    """The proposals a run of the generator skipped or rejected ({name, reason, detail}) and the metrics it added."""
    skipped, added = [], []
    for line in lines:
        message = line["message"]
        if found := SKIPPED.search(message):
            skipped.append({"name": found["name"], "reason": REASONS.get(found["reason"], "other"), "detail": found["reason"]})
        elif found := REJECTED.search(message):
            skipped.append({"name": found["name"], "reason": "rejected", "detail": found["reason"][:300]})
        elif found := ADDED.search(message):
            added = [] if found["names"] == "none" else found["names"].split(", ")
    return {"skipped": skipped, "added": added}


def edit_metric(entries: dict[str, dict], name: str, field: str, value) -> dict[str, dict]:
    """
    metrics.yaml with one field of a metric corrected in the studio (synonyms: a comma-separated list), the field
    added to `edited` when its value changed (the studio marks it corrected by you).
    """
    if name not in entries:
        raise ValueError(f"Métrique inconnue : {name}")
    entry = dict(entries[name])
    before = entry.get(field)
    text  = str(value or "").strip()
    if field == "synonyms":
        entry["synonyms"] = [s.strip() for s in text.split(",") if s.strip()]
    elif field == "time_dimension":
        entry["time_dimension"] = text
        if not text:
            del entry["time_dimension"]
    elif not text:
        raise ValueError(f"{FIELDS.get(field, field)} d'une métrique ne peut pas être vide")
    else:
        entry[field] = text
    if entry.get(field) != before:
        edited = {*(entry.get("edited") or []), field}
        entry["edited"] = [key for key in METRICS_KEY_ORDER if key in edited]
    return {**entries, name: entry}


def ignore_metric(entries: dict[str, dict], ignored: list[str], name: str,
                  ignore: bool = True) -> tuple[dict[str, dict], list[str]]:
    """
    metrics.yaml with a metric rejected: deleted, its name in `ignore` so the generator never
    proposes it again (the header of the file says so); or its name taken out of `ignore` (ignore False).
    """
    if ignore:
        if name not in entries:
            raise ValueError(f"Métrique inconnue : {name}")
        return {n: e for n, e in entries.items() if n != name}, [*ignored, name] if name not in ignored else ignored
    if name not in ignored:
        raise ValueError(f"{name} n'est pas parmi les métriques retirées")
    return entries, [n for n in ignored if n != name]
