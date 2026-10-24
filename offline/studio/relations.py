# offline/studio/relations.py

"""
The Relations screen of the studio (pure functions, unit-tested): the relations of relationships.yaml
with their evidence, the map of the tables they join, and the join path the agent takes between two.

  - evidence: the share of the values of `from` found in `to` (found, measured by the generator or
    when a relation is added in the studio), and the cardinality read from the keys of the profile;
    a relation added on purpose to a column that is not a key carries its fanout (rows of `to` each
    row of `from` joins, on average) and the cardinality measured then (links), and stays in amber
  - map: the tables laid out in layers (grandalf, Sugiyama), the referenced tables under the ones
    pointing to them; each group of joined tables side by side, the isolated tables on a last row
  - path: the tables the agent adds between two (lib/fk_graph.py, the same functions), and the
    number of joins, beyond MAX_BRIDGE_JOINS the agent does not bridge
"""

from grandalf.graphs import Edge, Graph, Vertex
from grandalf.layouts import SugiyamaLayout

from lib.fk_graph import MAX_BRIDGE_JOINS, _distances, _shortest_path_tables

KEY_OVERLAP = 0.9    # under it a relation is shown in amber, for a human to keep or ignore (relationships.py)
NODE_HEIGHT = 26
CHAR_WIDTH  = 7      # width of a character of the table names (monospace, 11px)
GAP         = 40     # between two groups of joined tables, and around the map
MAX_PATH    = 8      # joins the page looks for a path, beyond the agent's MAX_BRIDGE_JOINS


def side(ref: str, profile: dict) -> str:
    """'1' when the column is unique in its table (a key of the profile), 'n' when not, '?' when not measured."""
    table, _, column = ref.partition(".")
    facts  = profile.get(table) or {}
    counts = (facts.get("keys") or {}).get(column)
    if not counts:
        return "?"
    rows = facts.get("rows") or 0
    return "1" if rows and counts["distinct"] == counts["non_null"] == rows else "n"


def relation_row(relation: dict, profile: dict) -> dict:
    """
    A relation as the page shows it: its tables, evidence, cardinality and origin. Weak (amber) when few of
    its values are found, or when it joins several rows of `to` to a row of `from` (fanout).
    """
    found, fanout = relation.get("found"), relation.get("fanout")
    return {
        "from":   relation["from"],
        "to":     relation["to"],
        "source": relation["from"].split(".")[0],
        "target": relation["to"].split(".")[0],
        "found":  found,
        "fanout": fanout,
        "weak":   (found is not None and found < KEY_OVERLAP) or (fanout or 0) > 1,
        "links":  relation.get("links") or f"{side(relation['from'], profile)} → {side(relation['to'], profile)}",
        "origin": relation.get("origin"),
    }


def fk_graph(relations: list[dict], tables: list[str]) -> dict[str, list[str]]:
    """The undirected graph of the agent (resources.build_fk_graph): each table and the tables it joins."""
    graph = {table: set() for table in tables}
    for r in relations:
        source, target = r["from"].split(".")[0], r["to"].split(".")[0]
        if source in graph and target in graph and source != target:
            graph[source].add(target)
            graph[target].add(source)
    return {table: sorted(neighbors) for table, neighbors in graph.items()}


def join_path(a: str, b: str, graph: dict[str, list[str]]) -> dict:
    """
    The join path between two tables: `layers`, the tables at each step from a to b (several in one
    step when shortest paths tie: the agent adds them all), `joins`, and whether the agent bridges it.
    """
    joins = _distances(a, graph, MAX_PATH).get(b)
    if joins is None:
        return {"joins": None, "layers": [], "bridged": False}
    middle = _shortest_path_tables(a, b, graph, MAX_PATH)
    from_a = _distances(a, graph, joins)
    layers = [[a]] + [sorted(t for t in middle if from_a.get(t) == step) for step in range(1, joins)] + [[b]]
    return {"joins": joins, "layers": layers, "bridged": joins <= MAX_BRIDGE_JOINS}


class _View:
    """Size of a node for grandalf, and the position it sets."""

    def __init__(self, width: int):
        self.w, self.h, self.xy = width, NODE_HEIGHT, (0, 0)


def node_width(table: str) -> int:
    return CHAR_WIDTH * len(table) + 24


def layout(tables: list[str], relations: list[dict]) -> dict:
    """
    Positions of the tables on the map: {"nodes": {table: {x, y, w, h}} (top-left corner), "width", "height"}.
    The tables pointing to another one above it, each group of joined tables beside the previous one,
    the isolated tables in a row under them all.
    """
    vertices = {table: Vertex(table) for table in tables}
    for table, vertex in vertices.items():
        vertex.view = _View(node_width(table))
    edges = []
    for r in relations:
        source, target = r["from"].split(".")[0], r["to"].split(".")[0]
        if source in vertices and target in vertices and source != target:
            edges.append(Edge(vertices[source], vertices[target]))
    graph = Graph(list(vertices.values()), edges)

    nodes, x, height = {}, GAP, GAP
    groups = sorted((c for c in graph.C if len(c.sV) > 1), key=lambda c: -len(c.sV))
    for group in groups:
        sugiyama = SugiyamaLayout(group)
        sugiyama.init_all()
        sugiyama.draw()
        left = min(v.view.xy[0] - v.view.w / 2 for v in group.sV)
        top  = min(v.view.xy[1] - v.view.h / 2 for v in group.sV)
        right = 0
        for v in group.sV:
            nx = x + v.view.xy[0] - v.view.w / 2 - left
            ny = GAP + v.view.xy[1] - v.view.h / 2 - top
            nodes[v.data] = {"x": round(nx), "y": round(ny), "w": v.view.w, "h": NODE_HEIGHT}
            right  = max(right, nx + v.view.w)
            height = max(height, ny + NODE_HEIGHT)
        x = right + GAP

    # The isolated tables: a row under the groups, wrapped at the width of the map
    width = max(x, 600)
    isolated = [t for t in tables if t not in nodes]
    row_x, row_y = GAP, height + (GAP if nodes else 0)
    for table in isolated:
        w = node_width(table)
        if row_x + w > width - GAP and row_x > GAP:
            row_x, row_y = GAP, row_y + NODE_HEIGHT + 14
        nodes[table] = {"x": row_x, "y": row_y, "w": w, "h": NODE_HEIGHT}
        row_x += w + 14
    if isolated:
        height = row_y + NODE_HEIGHT
    return {"nodes": nodes, "width": round(width), "height": round(height + GAP)}


def relations_view(tables: list[str], entry: dict, profile: dict) -> dict:
    """Everything the Relations screen draws: the relations, the ignored ones, the count per table, the map."""
    relations = [relation_row(r, profile) for r in entry.get("relations") or []]
    ignored   = [relation_row(r, profile) for r in entry.get("ignore") or []]
    counts    = {table: 0 for table in tables}
    for r in relations:
        for table in {r["source"], r["target"]}:
            if table in counts:
                counts[table] += 1
    return {
        "relations": relations,
        "ignored":   ignored,
        "counts":    counts,
        "isolated":  [t for t, n in counts.items() if not n],
        "weak":      sum(r["weak"] for r in relations),
        "unchecked": sum(r["found"] is None for r in relations),
        "layout":    layout(tables, entry.get("relations") or []),
    }


def add_relation(entry: dict, source: str, target: str, found: float | None, spread: dict | None = None) -> dict:
    """
    relationships.yaml with a relation added in the studio (origin manual); taken out of `ignore` if it was
    there. When `to` is not a key (spread, measured by key_spread), the relation carries its fanout and the
    cardinality measured.
    """
    entry  = {**entry, "relations": list(entry.get("relations") or []), "ignore": list(entry.get("ignore") or [])}
    same   = lambda r: (r["from"], r["to"]) == (source, target)  # noqa: E731
    entry["ignore"]    = [r for r in entry["ignore"] if not same(r)]
    entry["relations"] = [r for r in entry["relations"] if not same(r)]
    relation = {"from": source, "to": target, "found": found}
    if fanout := relation_fanout(spread):
        relation["fanout"] = fanout
        relation["links"]  = f"{'n' if spread['source_values'] < spread['source_rows'] else '1'} → n"
    entry["relations"].append({**relation, "origin": "manual"})
    return entry


def relation_fanout(spread: dict | None) -> float | None:
    """Rows of `to` a row of `from` joins on average, when `to` is not a key (None when it is, or nothing joins)."""
    if not spread or spread["per_value"][1] <= 1 or not spread.get("matched"):
        return None
    return round(spread["joined"] / spread["matched"], 1)


def ignore_relation(entry: dict, source: str, target: str, ignore: bool = True) -> dict:
    """relationships.yaml with a relation moved to `ignore` (it never comes back), or back from it."""
    entry = {**entry, "relations": list(entry.get("relations") or []), "ignore": list(entry.get("ignore") or [])}
    take, put = ("relations", "ignore") if ignore else ("ignore", "relations")
    moved = [r for r in entry[take] if (r["from"], r["to"]) == (source, target)]
    if not moved:
        raise ValueError(f"No relation {source} -> {target} in {take}")
    entry[take] = [r for r in entry[take] if (r["from"], r["to"]) != (source, target)]
    entry[put].extend(moved)
    return entry
