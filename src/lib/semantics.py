# src/lib/semantics.py

"""
Semantics helpers — loading config/semantics/<db>.yaml and rendering table documents.
Used by: schema_retrieval/schema_indexer.py (build_documents), schema_retrieval/catalog_generators.py
"""

import yaml

from config import BASE_DIR

# Directory where semantics files are saved
SEMANTICS_DIR = BASE_DIR / "config" / "semantics"


def load_semantics(db_name: str) -> dict:
    """Loads config/semantics/<db>.yaml, or an empty dict if missing."""
    path = SEMANTICS_DIR / f"{db_name}.yaml"
    return (yaml.safe_load(path.read_text()) if path.exists() else None) or {}


def render_table(table_name: str, table_ddl: str, table_semantics: dict,
                 db_fk_relations: list[tuple[str, str]]) -> str:
    """
    Render the document indexed in Chroma for one table, from reviewed semantics — no LLM.

    Prepends a comment header to the DDL and appends each documented column's
    description as an inline comment:

        -- Table: invoice
        -- Description: One row per customer purchase.
        -- Joins: invoice.customerid = customer.customerid
        -- Read mode: LAST (partition: dt)

        CREATE TABLE `invoice` (
            `invoiceid` INT,  -- Unique identifier of the invoice.
            ...
        )

    Format contract: the "-- Joins" and "-- Read mode" lines are parsed back by
    lib/column_pruning.py — keep both in sync when changing the format.
    """
    columns = table_semantics.get("columns", {})
    lines   = [f"-- Table: {table_name}", f"-- Description: {table_semantics['description']}"]

    # Join conditions involving this table, from fk_relations.yaml
    joins = [
        f"{source} = {target}"
        for source, target in db_fk_relations
        if table_name in (source.split(".")[0], target.split(".")[0])
    ]
    if joins:
        lines.append(f"-- Joins: {', '.join(joins)}")

    # How to read partitioned tables (manual, from config/semantics)
    lines.extend(_render_read_mode(table_name, table_semantics))
    lines.append("")

    # Append each column description as an inline SQL comment
    in_create = False
    for line in table_ddl.splitlines():
        if line.startswith("CREATE TABLE"):
            in_create = True
        elif line.startswith(")"):
            in_create = False
        elif in_create:
            name = line.strip().split()[0].strip('`"[]')
            if name in columns:
                line = f"{line.rstrip()}  -- {columns[name]}"
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
