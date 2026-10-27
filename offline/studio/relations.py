# offline/studio/relations.py

"""
The Relations screen of the studio (pure functions, unit-tested): the relations of relationships.yaml
with their evidence, the map of the tables they join, and the join path the agent takes between two.

  - evidence: the share of the values of `from` found in `to` (found, measured by the generator or
    when a relation is added in the studio), and the cardinality read from the keys of the profile;
    a relation added on purpose to a column that is not a key carries its fanout (rows of `to` each
    row of `from` joins, on average) and the cardinality measured then (links)
  - weak (amber, "à trancher"): a guessed relation (proposed by the LLM, or between two columns named
    differently) whose values are few found: maybe not a relation at all. Few values found by a relation
    of same-name keys, or one added by a human, is a partial coverage: the relation holds, `to` covers
    part of `from` only (the agent loses the other rows when it joins). A weak relation kept by a human
    (kept: true, "Garder") is no longer weak
  - map: the tables laid out in layers from left to right (grandalf, Sugiyama), the referenced tables on the
    right of the ones pointing to them; each group of joined tables under the previous one, the isolated
    tables on a last row
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
COLUMN_GAP  = 96     # between two columns of tables: room for the curves of the relations
ROW_GAP     = 14     # between two tables of a column
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
    A relation as the page shows it: its tables, evidence, cardinality and origin. Weak (amber) when it is
    guessed and few of its values are found, unless a human kept it; partial when few are found otherwise
    (or once kept). Empty when it was measured but `from` holds no value (found: null in relationships.yaml):
    nothing to compare, unlike a relation never measured (no found).
    """
    found, fanout, kept = relation.get("found"), relation.get("fanout"), bool(relation.get("kept"))
    few     = found is not None and found < KEY_OVERLAP
    guessed = relation.get("origin") == "llm" or (
        relation.get("origin") != "manual" and relation["from"].split(".")[1] != relation["to"].split(".")[1])
    return {
        "from":    relation["from"],
        "to":      relation["to"],
        "source":  relation["from"].split(".")[0],
        "target":  relation["to"].split(".")[0],
        "found":   found,
        "empty":   "found" in relation and found is None,
        "fanout":  fanout,
        "weak":    few and guessed and not kept,
        "partial": few and (kept or not guessed),
        "kept":    kept,
        "links":   relation.get("links") or f"{side(relation['from'], profile)} → {side(relation['to'], profile)}",
        "origin":  relation.get("origin"),
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
    """Size of a node for grandalf (w across a layer, h along the layers), and the position it sets."""

    def __init__(self, w: int, h: int):
        self.w, self.h, self.xy = w, h, (0, 0)


def node_width(table: str) -> int:
    return CHAR_WIDTH * len(table) + 24


def layout(tables: list[str], relations: list[dict]) -> dict:
    """
    Positions of the tables on the map: {"nodes": {table: {x, y, w, h}} (top-left corner), "width", "height"}.
    From left to right: a table pointing to another one on its left, the tables of a column aligned on their
    left side; each group of joined tables under the previous one, the isolated tables in a row under them all.
    grandalf lays its layers out from top to bottom: each table is given to it on its side (its width along
    the layers), and its positions are turned a quarter.
    """
    vertices = {table: Vertex(table) for table in tables}
    for table, vertex in vertices.items():
        vertex.view = _View(NODE_HEIGHT, node_width(table))
    edges = []
    for r in relations:
        source, target = r["from"].split(".")[0], r["to"].split(".")[0]
        if source in vertices and target in vertices and source != target:
            edges.append(Edge(vertices[source], vertices[target]))
    graph = Graph(list(vertices.values()), edges)

    nodes, y, width = {}, GAP, 600
    groups = sorted((c for c in graph.C if len(c.sV) > 1), key=lambda c: -len(c.sV))
    for group in groups:
        sugiyama = SugiyamaLayout(group)
        sugiyama.xspace, sugiyama.yspace = ROW_GAP, COLUMN_GAP
        sugiyama.init_all()
        sugiyama.draw()
        # A layer of grandalf is a column of the map, its tables aligned on the left side of the widest
        column = {}
        for v in group.sV:
            center = round(v.view.xy[1])
            column[center] = min(column.get(center, center), v.view.xy[1] - v.view.h / 2)
        left = min(column.values())
        top  = min(v.view.xy[0] - v.view.w / 2 for v in group.sV)
        bottom = y
        for v in group.sV:
            nx = GAP + column[round(v.view.xy[1])] - left
            ny = y + v.view.xy[0] - v.view.w / 2 - top
            nodes[v.data] = {"x": round(nx), "y": round(ny), "w": v.view.h, "h": NODE_HEIGHT}
            width  = max(width, nx + v.view.h + GAP)
            bottom = max(bottom, ny + NODE_HEIGHT)
        y = bottom + GAP

    # The isolated tables: a row under the groups, wrapped at the width of the map
    isolated = [t for t in tables if t not in nodes]
    row_x, row_y = GAP, y
    for table in isolated:
        w = node_width(table)
        if row_x + w > width - GAP and row_x > GAP:
            row_x, row_y = GAP, row_y + NODE_HEIGHT + ROW_GAP
        nodes[table] = {"x": row_x, "y": row_y, "w": w, "h": NODE_HEIGHT}
        row_x += w + ROW_GAP
    height = row_y + NODE_HEIGHT + GAP if isolated else y
    return {"nodes": nodes, "width": round(width), "height": round(height)}


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
        "unchecked": sum(r["found"] is None and not r["empty"] for r in relations),   # an empty one: nothing to measure
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


def keep_relation(entry: dict, source: str, target: str, keep: bool = True) -> dict:
    """relationships.yaml with a weak relation kept by a human (kept: true, no longer in amber), or no longer kept."""
    relations = [dict(r) for r in entry.get("relations") or []]
    found = [r for r in relations if (r["from"], r["to"]) == (source, target)]
    if not found:
        raise ValueError(f"Pas de jointure {source} → {target} dans les jointures")
    for relation in found:
        if keep:
            relation["kept"] = True
        else:
            relation.pop("kept", None)
    return {**entry, "relations": relations}


def ignore_relation(entry: dict, source: str, target: str, ignore: bool = True) -> dict:
    """relationships.yaml with a relation moved to `ignore` (it never comes back), or back from it."""
    entry = {**entry, "relations": list(entry.get("relations") or []), "ignore": list(entry.get("ignore") or [])}
    take, put = ("relations", "ignore") if ignore else ("ignore", "relations")
    moved = [r for r in entry[take] if (r["from"], r["to"]) == (source, target)]
    if not moved:
        where = "dans les jointures" if ignore else "parmi les jointures retirées"
        raise ValueError(f"Pas de jointure {source} → {target} {where}")
    entry[take] = [r for r in entry[take] if (r["from"], r["to"]) != (source, target)]
    entry[put].extend(moved)
    return entry
