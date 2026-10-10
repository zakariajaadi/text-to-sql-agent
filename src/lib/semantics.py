# src/lib/semantics.py

"""
Semantics helpers — loading semantic_layer/<db>/tables.yaml and relationships.yaml, rendering table documents.
Used by: resources.py (FK graphs), offline/ (indexer, generators)
"""

from pathlib import Path

import yaml
from omegaconf import OmegaConf

from config import SEMANTIC_LAYER_DIR


def tables_path(db_name: str) -> Path:
    """semantic_layer/<db>/tables.yaml: descriptions, categorical columns and read modes of the tables."""
    return SEMANTIC_LAYER_DIR / db_name / "tables.yaml"


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


def render_table(table_name: str, table_ddl: str, table_semantics: dict,
                 db_fk_relations: list[tuple[str, str]],
                 categorical_values: dict[str, list[str]] | None = None) -> str:
    """
    Render the document indexed in Chroma for one table, from reviewed semantics — no LLM.

    Prepends a comment header to the DDL and appends each documented column's
    description, and its categorical values if any (read from the database by
    offline/indexer/categorical_values.py), as an inline comment:

        -- Table: invoice
        -- Description: One row per customer purchase.
        -- Joins: invoice.customerid = customer.customerid
        -- Read mode: LAST (partition: dt)

        CREATE TABLE `invoice` (
            `invoiceid` INT,  -- Unique identifier of the invoice.
            `billingcountry` STRING,  -- Billing country. Values: 'Brazil', 'Canada', ...
            ...
        )

    Format contract: the "-- Joins" and "-- Read mode" lines are parsed back by
    lib/column_pruning.py — keep both in sync when changing the format.
    """
    columns = table_semantics.get("columns") or {}
    values  = categorical_values or {}   # {column: [value, ...]}
    lines   = [f"-- Table: {table_name}", f"-- Description: {table_semantics['description']}"]

    # Join conditions involving this table, from relationships.yaml
    joins = [
        f"{source} = {target}"
        for source, target in db_fk_relations
        if table_name in (source.split(".")[0], target.split(".")[0])
    ]
    if joins:
        lines.append(f"-- Joins: {', '.join(joins)}")

    # How to read partitioned tables (manual, from tables.yaml)
    lines.extend(_render_read_mode(table_name, table_semantics))
    lines.append("")

    # Append each column description and categorical values as an inline SQL comment
    in_create = False
    for line in table_ddl.splitlines():
        if line.startswith("CREATE TABLE"):
            in_create = True
        elif line.startswith(")"):
            in_create = False
        elif in_create and line.strip():
            name    = line.strip().split()[0].strip('`"[]')
            comment = columns.get(name, "")
            if name in values:
                comment = f"{comment} Values: {', '.join(repr(str(v)) for v in values[name])}".strip()
            if comment:
                line = f"{line.rstrip()}  -- {comment}"
        lines.append(line)

    return "\n".join(lines)


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
