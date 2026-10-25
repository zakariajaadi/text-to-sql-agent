# offline/studio/home.py

"""
Home screen of the studio: what the agent knows of every database, before the steps of one. The search
looks through the semantic layer of all of them: the names and descriptions of their tables and columns,
never a value of the data.
"""

import unicodedata

from lib.semantics import excluded_columns


def _plain(text) -> str:
    """Lower case without accents, for a search to find « Adresse » with « adresse »."""
    decomposed = unicodedata.normalize("NFD", str(text or ""))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def search_layer(query: str, layers: dict[str, tuple[dict, dict | None]], limit: int = 12) -> dict:
    """
    The tables and columns of every database whose name or description holds the query: the names first,
    then the descriptions. layers: {db: (semantics, trace)}; the columns of a table are the ones its document
    shows (the trace of the last index), the ones tables.yaml describes and the hidden ones.
    {"total", "bases", "results"}: how many results, in how many databases, and the first `limit` of them,
    each {db, table, column (None for a table), description, hidden, categorical}.
    """
    needle = _plain(query).strip()
    if not needle:
        return {"total": 0, "bases": 0, "results": []}
    found = []
    for db, (semantics, trace) in layers.items():
        indexed = (trace or {}).get("tables", {})
        for table in dict.fromkeys([*semantics, *indexed]):
            entry       = semantics.get(table) or {}
            hidden      = excluded_columns(entry)
            described   = entry.get("columns") or {}
            categorical = set(entry.get("categorical_columns") or [])
            in_name     = needle in _plain(table)
            if in_name or needle in _plain(entry.get("description")):
                found.append((0 if in_name else 2, {"db": db, "table": table, "column": None,
                                                    "description": entry.get("description"),
                                                    "hidden": False, "categorical": False}))
            shown = (indexed.get(table) or {}).get("columns", {})
            for column in dict.fromkeys([*shown, *described, *hidden]):
                in_name = needle in _plain(column)
                if in_name or needle in _plain(described.get(column)):
                    found.append((1 if in_name else 3, {"db": db, "table": table, "column": column,
                                                        "description": described.get(column),
                                                        "hidden": column in hidden,
                                                        "categorical": column in categorical and column not in hidden}))
    found.sort(key=lambda pair: pair[0])
    return {"total": len(found), "bases": len({result["db"] for _, result in found}),
            "results": [result for _, result in found[:limit]]}
