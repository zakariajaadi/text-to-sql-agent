"""
FK expnading helpers
Used by: nodes.py (rerank_tables)
"""

def expand_fk(selected: list[str], fk_graph: dict[str, list[str]]) -> list[str]:
    """
    Add bridge tables that connect pairs of already-selected tables via FK.

    LLMs systematically miss fact and bridge tables not explicitly named in the
    question. Two-pass expansion recovers them without flooding the context
    with all FK-linked tables:
        Pass 1 — direct neighbors of each selected table.
        Pass 2 — bridge tables shared between two selected tables.
    """
    expanded = set(selected)

    # Pass 1 — direct FK neighbors
    for table in selected:
        for neighbor in fk_graph.get(table, []):
            expanded.add(neighbor)

    # Pass 2 — bridge tables (common FK neighbors between pairs)
    for i, table_a in enumerate(selected):
        for table_b in selected[i + 1:]:
            neighbors_a = set(fk_graph.get(table_a, []))
            neighbors_b = set(fk_graph.get(table_b, []))
            expanded.update(neighbors_a & neighbors_b)

    return list(expanded)