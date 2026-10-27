# offline/studio/scope.py

"""
The Périmètre screen of the studio (pure functions, unit-tested): every table of the database against its scope,
the tables the agent and the pipeline see (semantic_layer/<db>/scope.yaml, lib/semantics.load_scope), and
against the last index.

The scope is every table of the database, the ones to come too, or a list: then a table enters it only when
chosen, and one the database gains stays out of it. A table taken out of the scope keeps its files (profile,
descriptions); its document stays in the index, read by the agent, until the next index. A table of the scope
the database no longer has is left out of the pipeline and of the agent.
"""

from offline.studio.index import UNKNOWN

# State of a table out of the scope, or of the database: the ones in both have the status of their index
# (offline/studio/index.py: fresh, changed, missing, unknown)
OUTSIDE, LEAVING, GONE = "outside", "leaving", "gone"


def change_scope(scope: list[str] | None, base: list[str], add: list[str], remove: list[str]) -> list[str]:
    """
    The scope once `add` and `remove` applied, sorted. None (no scope chosen yet) stands for every table of the
    database: removing one keeps the others. Raises ValueError for a table to add the database doesn't have,
    or a table both added and removed.
    """
    if both := sorted(set(add) & set(remove)):
        raise ValueError(f"Table à la fois ajoutée et retirée : {', '.join(both)}")
    if unknown := sorted(set(add) - set(base)):
        raise ValueError(f"{'Tables inconnues' if len(unknown) > 1 else 'Table inconnue'} dans la base : {', '.join(unknown)}")
    current = set(base if scope is None else scope)
    return sorted((current | set(add)) - set(remove))


def scope_rows(base: list[str] | None, scope: list[str] | None, statuses: dict[str, str], indexed: set[str],
               profiled: set[str], described: set[str], columns: dict[str, int]) -> list[dict]:
    """
    Every table of the database, of the scope or of the index, sorted, with its state:
    - in the scope and the database: the status of its index (statuses, from index_view; unknown without one)
    - out of the scope, in the database: OUTSIDE, or LEAVING while the index still holds it
    - not in the database (base known): GONE, whether the scope names it or the index holds it
    base: the tables of the database, None when it can't be read; scope: None, every table of the database.
    profiled, described: the tables of profile.yaml and the described ones of tables.yaml (kept out of the scope).
    columns: {table: number of columns} of the tables of the scope the database has.
    """
    rows = []
    for name in sorted({*(base or []), *(scope or []), *indexed}):
        in_base  = None if base is None else name in base
        in_scope = name in scope if scope is not None else in_base is not False
        if in_base is False:
            state = GONE
        elif not in_scope:
            state = LEAVING if name in indexed else OUTSIDE
        else:
            state = statuses.get(name, UNKNOWN)
        rows.append({"name": name, "state": state, "in_scope": in_scope, "in_base": in_base, "indexed": name in indexed,
                     "profiled": name in profiled, "described": name in described, "columns": columns.get(name)})
    return rows


def scope_view(rows: list[dict], base: list[str] | None, origin: str) -> dict:
    """
    The Périmètre screen: its rows (scope_rows), the counts of the database, of the scope and of the index, the
    count of each state, and where the scope comes from (lib/semantics.scope_origin: all, list, none).
    """
    states = {}
    for row in rows:
        states[row["state"]] = states.get(row["state"], 0) + 1
    return {
        "tables": rows,
        "origin": origin,
        "counts": {
            "base":    None if base is None else len(base),
            "scope":   sum(row["in_scope"] and row["in_base"] is not False for row in rows),
            "indexed": sum(row["indexed"] for row in rows),
            **states,
        },
    }
