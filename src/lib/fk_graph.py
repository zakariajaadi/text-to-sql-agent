"""
FK expnading helpers
Used by: agent/nodes.py (rerank_tables)
"""

from collections import deque

# Longest join path (in joins) that FK expansion bridges between two selected tables: 3 joins
# take two tables in between (client → client_master → personne → impaye)
MAX_BRIDGE_JOINS = 3


def expand_fk(selected: list[str], fk_graph: dict[str, list[str]],
              include_neighbors: bool = False, max_joins: int = MAX_BRIDGE_JOINS) -> list[str]:
    """
    Add bridge tables that connect pairs of already-selected tables via FK.

    LLMs systematically miss fact and bridge tables not explicitly named in the
    question. Two passes recover them:
        Pass 1 — direct neighbors of each selected table (opt-in: floods the context
                 on hub tables with many FKs).
        Pass 2 — every table on a shortest FK path between two selected tables, up to
                 max_joins joins (always on). Without them the model invents a direct join,
                 which can run and be wrong (two id families that overlap).
    Returns the selected tables first, then the added ones.
    """
    expanded = list(selected)

    # Pass 1 — direct FK neighbors (opt-in)
    if include_neighbors:
        for table in selected:
            expanded += sorted(fk_graph.get(table, []))

    # Pass 2 — bridge tables (on a shortest path between a pair)
    for i, table_a in enumerate(selected):
        for table_b in selected[i + 1:]:
            expanded += _shortest_path_tables(table_a, table_b, fk_graph, max_joins)

    return list(dict.fromkeys(expanded))


def _distances(start: str, fk_graph: dict[str, list[str]], max_joins: int) -> dict[str, int]:
    """Number of joins from start to every table reachable in at most max_joins joins."""
    dist = {start: 0}
    queue = deque([start])
    while queue:
        table = queue.popleft()
        if dist[table] == max_joins:
            continue
        for neighbor in fk_graph.get(table, []):
            if neighbor not in dist:
                dist[neighbor] = dist[table] + 1
                queue.append(neighbor)
    return dist


def _shortest_path_tables(a: str, b: str, fk_graph: dict[str, list[str]], max_joins: int) -> list[str]:
    """
    The tables between a and b on any shortest FK path (all of them when several paths tie),
    none when a and b are joined directly, unconnected, or further than max_joins joins apart.
    """
    from_a, from_b = _distances(a, fk_graph, max_joins), _distances(b, fk_graph, max_joins)
    joins = from_a.get(b)
    if joins is None or joins < 2:
        return []
    return sorted(t for t in from_a if t not in (a, b) and t in from_b and from_a[t] + from_b[t] == joins)
