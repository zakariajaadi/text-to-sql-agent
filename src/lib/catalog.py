"""
The table catalog read by the table selector (agent/nodes.py rerank_tables): every table of the
database in a compact form — name, description, column names, joined tables — about 60 tokens
per table, so a database of 100 tables fits in one prompt. The selector sees the whole
database and names the tables holding the answer; their full DDL is loaded after, once.

Used by: agent/nodes.py (retrieve_schema, rerank_tables)
"""

import re

from lib.column_pruning import get_column_names


def render_catalog_entry(table: str, document: str, neighbors: list[str] = ()) -> str:
    """
    One table of the catalog, from its indexed document (lib/semantics.py render_table):

        ### orders
        One row per customer order, containing shipping details and logistics information.
        Columns: orderid, customerid, employeeid, orderdate, shippeddate, shipvia, freight
        Joins: customers, employees, order_details, shippers

    The excluded columns are already out of the document. neighbors: the tables joined to this
    one (resources.fk_graphs), the selector includes the ones on a join path.
    """
    lines = [f"### {table}"]
    description = _header_value(document, "Description")
    if description:
        lines.append(description)
    columns = get_column_names(document)
    if columns:
        lines.append("Columns: " + ", ".join(columns))
    if neighbors:
        lines.append("Joins: " + ", ".join(sorted(neighbors)))
    return "\n".join(lines)


def render_catalog(tables: list[str], documents: dict[str, str], fk_graph: dict[str, list[str]]) -> str:
    """The catalog of `tables`, in that order (the closest to the question first)."""
    return "\n\n".join(
        render_catalog_entry(table, documents[table], fk_graph.get(table, []))
        for table in tables if table in documents
    )


def _header_value(document: str, key: str) -> str:
    match = re.search(rf"^-- {key}: (.*)$", document, re.MULTILINE)
    return match.group(1).strip() if match else ""
