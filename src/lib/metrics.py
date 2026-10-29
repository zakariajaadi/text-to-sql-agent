# src/lib/metrics.py

"""
Business metrics — loading and validating semantic_layer/<db>/metrics.yaml, rendering them for the prompts.
Used by: agent/nodes.py (rerank_tables, planner, sql_coder), offline/semantic_gen/metrics.py (generate_metrics)

A metric is an aggregate SQL expression defined once, with a description and synonyms:

    revenue:
      description: Amount sold, discounts applied.
      synonyms: [sales, turnover]
      expr: SUM(order_details.unitprice * order_details.quantity * (1 - order_details.discount))
      time_dimension: orders.orderdate     # optional: default date for "per month", "in 2022"...

The LLMs pick metrics by name and copy their expression: they never redefine them.
Columns are written table.column, so the tables of a metric are read from its expression.
The reserved key `ignore` lists rejected metric names, so the generator never proposes them again.
"""

from dataclasses import dataclass, field
from pathlib import Path

import sqlglot
import yaml
from sqlglot import exp
from sqlglot.errors import ParseError

from config import SEMANTIC_LAYER_DIR
from lib.column_pruning import get_column_names
from lib.sql_tools import sqlglot_dialect


@dataclass(frozen=True)
class Metric:
    name:           str
    description:    str
    expr:           str
    synonyms:       list[str] = field(default_factory=list)
    time_dimension: str = ""            # table.column, empty when the metric has no default date
    tables:         list[str] = field(default_factory=list)   # read from expr and time_dimension
    columns:        list[tuple[str, str]] = field(default_factory=list)   # (table, column) of expr and time_dimension


def metrics_path(db_name: str) -> Path:
    """semantic_layer/<db>/metrics.yaml: business metrics, generated (make metrics) or written, then reviewed."""
    return SEMANTIC_LAYER_DIR / db_name / "metrics.yaml"


def read_metrics_file(db_name: str) -> tuple[dict[str, dict], list[str]]:
    """Raw content of metrics.yaml: ({name: entry}, ignored names). The `ignore` key is reserved."""
    path = metrics_path(db_name)
    raw  = (yaml.safe_load(path.read_text()) if path.exists() else None) or {}
    return {name: entry for name, entry in raw.items() if name != "ignore"}, list(raw.get("ignore") or [])


def load_metrics(db_name: str, dialect: str, schemas: dict[str, str]) -> dict[str, Metric]:
    """
    Load and validate the metrics of a database, or {} if it has no metrics.yaml.

    Fails loudly (ValueError, see validate_metric) on a metric that can't be trusted.
    schemas ({table: DDL}, the indexed documents) are the reference for tables and columns;
    when empty (index not built yet), only the syntax is checked.
    """
    entries, _ = read_metrics_file(db_name)
    columns    = {table: get_column_names(ddl) for table, ddl in schemas.items()}
    where      = metrics_path(db_name).relative_to(SEMANTIC_LAYER_DIR.parent)

    metrics = {}
    for name, entry in entries.items():
        try:
            metrics[name] = validate_metric(name, entry, dialect, columns)
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from None
    return metrics


def validate_metric(name: str, entry: dict, dialect: str, columns: dict[str, list[str]]) -> Metric:
    """
    Build a Metric from its YAML entry, or raise ValueError naming the metric and the problem:
    missing field, expression that doesn't parse or doesn't aggregate, unqualified column,
    unknown table or column. columns ({table: [column, ...]}) may be empty: syntax checks only.
    """
    def fail(reason: str):
        raise ValueError(f"metric '{name}' — {reason}")

    if not entry or not entry.get("description") or not entry.get("expr"):
        fail("'description' and 'expr' are required")

    try:
        tree = sqlglot.parse_one(entry["expr"], read=sqlglot_dialect(dialect))
    except ParseError as e:
        fail(f"expr does not parse: {e}")
    if not tree.find(exp.AggFunc):
        fail("expr must aggregate (SUM, COUNT, AVG...)")

    # Columns as table.column: the tables of the metric come from its expression
    used = [(column.table, column.name) for column in tree.find_all(exp.Column)]
    if any(not table for table, _ in used):
        fail("write every column of expr as table.column")

    time_dimension = entry.get("time_dimension") or ""
    if time_dimension:
        table, _, column = time_dimension.partition(".")
        if not column:
            fail("write time_dimension as table.column")
        used.append((table, column))

    if columns:
        for table, column in used:
            if table not in columns:
                fail(f"unknown table '{table}'")
            if column not in columns[table]:
                fail(f"unknown column '{table}.{column}'")

    return Metric(
        name           = name,
        description    = entry["description"],
        expr           = entry["expr"],
        synonyms       = list(entry.get("synonyms") or []),
        time_dimension = time_dimension,
        tables         = list(dict.fromkeys(table for table, _ in used)),
        columns        = list(dict.fromkeys(used)),
    )


def render_metric_catalog(metrics: dict[str, Metric]) -> str:
    """Names, descriptions and synonyms of all the metrics, for rerank_tables to pick from."""
    return "\n".join(
        f"- {m.name}: {m.description}" + (f" Synonyms: {', '.join(m.synonyms)}." if m.synonyms else "")
        for m in metrics.values()
    ) or "None"


def render_metric_criteria(metrics: list[Metric]) -> str:
    """
    The criteria a clarification may offer (CLARIFICATION_CRITERIA), one "- label: description"
    line each. The label is the first synonym — short business words the model copies into the
    options, matched back by the synonyms on the follow-up turn — never the code name (the
    model copies whatever it sees) and never the expression: options name quantities, not SQL.
    """
    return "\n".join(
        f"- {m.synonyms[0] if m.synonyms else m.name.replace('_', ' ')}: {m.description}"
        for m in metrics
    )


def render_metric_definitions(metrics: list[Metric]) -> str:
    """One line per selected metric, name = expression (default date), for METRICS_SECTION (agent/prompts.py)."""
    return "\n".join(
        f"- {m.name} = {m.expr}" + (f"  (default date: {m.time_dimension})" if m.time_dimension else "")
        for m in metrics
    )
