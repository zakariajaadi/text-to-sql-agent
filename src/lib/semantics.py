# src/lib/semantics.py

"""
Semantics helpers — loading semantic_layer/<db>/tables.yaml and relationships.yaml, rendering table documents.
Used by: resources.py (FK graphs), offline/ (indexer, generators), agent/nodes.py (match_values)
"""

from pathlib import Path

import yaml
from omegaconf import OmegaConf

from config import SEMANTIC_LAYER_DIR
from lib.column_pruning import get_column_names, prune_ddl
from lib.sql_tools import join_condition


def tables_path(db_name: str) -> Path:
    """semantic_layer/<db>/tables.yaml: descriptions, categorical columns and read modes of the tables."""
    return SEMANTIC_LAYER_DIR / db_name / "tables.yaml"


def profile_path(db_name: str) -> Path:
    """semantic_layer/<db>/profile.yaml: facts measured on the data by `make profile` (no LLM)."""
    return SEMANTIC_LAYER_DIR / db_name / "profile.yaml"


def load_profile(db_name: str) -> dict:
    """Loads semantic_layer/<db>/profile.yaml, or an empty dict if the database was never profiled."""
    path = profile_path(db_name)
    return (yaml.safe_load(path.read_text()) if path.exists() else None) or {}


def load_semantics(db_name: str) -> dict:
    """Loads semantic_layer/<db>/tables.yaml, or an empty dict if missing."""
    path = tables_path(db_name)
    return (yaml.safe_load(path.read_text()) if path.exists() else None) or {}


def load_fk_relations(db_name: str) -> list[tuple[str, str]]:
    """Return (from_column, to_column) pairs, as table.column, declared in semantic_layer/<db>/relationships.yaml."""
    path = SEMANTIC_LAYER_DIR / db_name / "relationships.yaml"
    if not path.exists():
        return []
    relations = OmegaConf.load(path).get("relations") or []
    return [(r["from"], r["to"]) for r in relations]


# Keys of a table block written by the generator (make semantics); the other ones are manual
GENERATED_KEYS = ("description", "categorical_columns", "columns")


def manual_semantics(table_semantics: dict) -> dict:
    """A table block without its generated keys: formats, exclude_columns, data_as_of, read mode..."""
    return {key: value for key, value in table_semantics.items() if key not in GENERATED_KEYS}


def excluded_columns(table_semantics: dict | None) -> set[str]:
    """Columns of a table the agent must never see (`exclude_columns` in tables.yaml)."""
    return set((table_semantics or {}).get("exclude_columns") or [])


def render_table(table_name: str, table_ddl: str, table_semantics: dict,
                 db_fk_relations: list[tuple[str, str]],
                 categorical_values: dict[str, list[str]] | None = None,
                 column_types: dict[str, str] | None = None, dialect: str = "") -> str:
    """
    Render the document indexed in Chroma for one table, from reviewed semantics — no LLM.

    Prepends a comment header to the DDL and appends each documented column's
    description, the note of a column stored as text (`formats` in tables.yaml, converted
    following the dialect rules of agent/prompts.py), and its categorical values if any
    (read from the database by offline/indexer/categorical_values.py), as an inline comment.
    The columns listed in `exclude_columns` (empty, technical or personal data) are removed,
    sample rows included: the agent never sees them. `data_as_of`, the date the data stops,
    is the reference date of relative periods (agent/prompts.py):

        -- Table: invoice
        -- Description: One row per customer purchase.
        -- Data as of: 2024-07-10
        -- Joins: invoice.customerid = customer.customerid
        -- Read mode: LAST (partition: dt)

        CREATE TABLE `invoice` (
            `invoiceid` INT,  -- Unique identifier of the invoice.
            `invoicedate` STRING,  -- Date of the purchase. [stored as text: date dd/MM/yyyy]
            `billingcountry` STRING,  -- Billing country. Values: 'Brazil', 'Canada', ...
            ...
        )

    column_types: {"table.column": type} of the tables, from their DDL: a join key stored as text
    on one side and as a number on the other is cast in the "-- Joins" line, written for `dialect`
    (lib/sql_tools.py join_condition).

    Format contract: the "-- Joins" and "-- Read mode" lines are parsed back by
    lib/column_pruning.py — keep both in sync when changing the format.
    """
    columns  = table_semantics.get("columns") or {}
    formats  = table_semantics.get("formats") or {}   # {column: "date <pattern>" | "decimal"}
    excluded = excluded_columns(table_semantics)
    values   = categorical_values or {}   # {column: [value, ...]}
    lines    = [f"-- Table: {table_name}", f"-- Description: {table_semantics['description']}"]
    if table_semantics.get("data_as_of"):
        lines.append(f"-- Data as of: {table_semantics['data_as_of']}")

    # Excluded columns out of the CREATE TABLE body and of the sample rows
    if excluded:
        table_ddl = prune_ddl(table_ddl, set(get_column_names(table_ddl)) - excluded)

    # Join conditions involving this table, from relationships.yaml
    joins = [
        join_condition(source, target, column_types or {}, dialect)
        for source, target in db_fk_relations
        if table_name in (source.split(".")[0], target.split(".")[0])
    ]
    if joins:
        lines.append(f"-- Joins: {', '.join(joins)}")

    # How to read partitioned tables (manual, from tables.yaml)
    lines.extend(_render_read_mode(table_name, table_semantics))
    lines.append("")

    # Append each column description, format note and categorical values as an inline SQL comment
    in_create = False
    for line in table_ddl.splitlines():
        if line.startswith("CREATE TABLE"):
            in_create = True
        elif line.startswith(")"):
            in_create = False
        elif in_create and line.strip():
            name    = line.strip().split()[0].strip('`"[]')
            comment = columns.get(name, "")
            if name in formats:
                comment = f"{comment} {_format_note(table_name, name, formats[name])}".strip()
            if name in values:
                comment = f"{comment} Values: {', '.join(repr(str(v)) for v in values[name])}".strip()
            if comment:
                line = f"{line.rstrip()}  -- {comment}"
        lines.append(line)

    return "\n".join(lines)


def add_matching_values(table_ddl: str, values: dict[str, list[str]],
                        more: dict[str, int] | None = None) -> str:
    """
    Append to the inline comment of each given column the values of the value index that
    match the question (agent/nodes.py match_values):

        `city` STRING,  -- City of the company. Matching values: 'Sao Paulo'
        `name` STRING,  -- Agency name. Matching values: 'CASA ANFA', ..., 'CASA OASIS' (+12 more)

    values: {column: [value, ...]}; the other columns are left as they are.
    more:   {column: n}, the close values left out of the list.
    """
    more = more or {}
    if not values:
        return table_ddl

    lines     = []
    in_create = False
    for line in table_ddl.splitlines():
        if line.startswith("CREATE TABLE"):
            in_create = True
        elif line.startswith(")"):
            in_create = False
        elif in_create and line.strip():
            name = line.strip().split()[0].strip('`"[]')
            if name in values:
                matching = f"Matching values: {', '.join(repr(str(v)) for v in values[name])}"
                if more.get(name):
                    matching += f" (+{more[name]} more)"
                line     = f"{line.rstrip()} {matching}" if "--" in line else f"{line.rstrip()}  -- {matching}"
        lines.append(line)

    return "\n".join(lines)


def _format_note(table_name: str, column: str, column_format: str) -> str:
    """
    Note of a column whose values are stored as text (`formats` in tables.yaml): "date <pattern>"
    (Java pattern, e.g. dd/MM/yyyy) or "decimal". The dialect rules (agent/prompts.py) say how
    to convert the columns carrying it — keep both wordings in sync.
    """
    kind, _, pattern = column_format.strip().partition(" ")
    if kind == "date" and pattern.strip():
        return f"[stored as text: date {pattern.strip()}]"
    if kind == "decimal" and not pattern:
        return "[stored as text: decimal number]"
    raise ValueError(f"Unknown format '{column_format}' for {table_name}.{column} (expected: 'date <pattern>', 'decimal')")


def _render_read_mode(table_name: str, table_semantics: dict) -> list[str]:
    """Header line declaring how a partitioned or CDC table must be read (rules live in the prompts)."""
    mode = table_semantics.get("read_mode")
    if mode is None:
        return []

    if mode in ("last", "all"):
        return [f"-- Read mode: {mode.upper()} (partition: {table_semantics['partition_column']})"]
    if mode == "live":
        keys = ", ".join(table_semantics["key_columns"])
        return [f"-- Read mode: LIVE (key: {keys}, version: {table_semantics['version_column']})"]

    raise ValueError(f"Unknown read_mode '{mode}' for table {table_name} (expected: last, all, live)")
