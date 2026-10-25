# offline/studio/table_types.py

"""
The type of each table, read from the data without an LLM: what the profile measured (rows, keys), the
columns (their types, the formats of tables.yaml) and the relations (which tables a table points to, which
point to it). A human corrects it in the studio (`table_type` in tables.yaml), which wins over the detection.

  events     one row per thing that happened (an order, an invoice line): dated, pointing to other tables
             and many rows for each of their rows; or measured and pointing to several without being pointed
             to. An entity has dates too (a birth, an opening) but about one row for each row it points to
  entity     one row per client, product, employee: a unique key, or pointed to, or the versions of one
             (read mode keeping the last version or the last snapshot, or partitions each holding a snapshot)
  reference  a short list of codes (category, country, status): pointed to, undated, at most 4 columns and
             1 000 rows, pointing only to other references
  bridge     links two entities, many to many: its columns almost all point to other tables
  (none)     unclassified: empty, or nothing tells its role

The rules run in that order: a bridge first (its structure says it even when empty), an empty table
unclassified, then the versions of an entity, the events, the references, the entities.
"""

from collections import defaultdict

from lib.semantics import excluded_columns
from offline.semantic_gen.profiling import KEY_NAME
from offline.studio.reading import suggest
from offline.studio.state import TABLE_TYPES, column_kind

REFERENCE_ROWS    = 1000   # a list of codes holds no more rows
REFERENCE_COLUMNS = 4      # nor more columns
BRIDGE_COLUMNS    = 4      # a bridge: its columns pointing elsewhere, and at most one more
EVENTS_PER_ROW    = 3      # dated events: more rows than this many times the largest table they point to
VERSIONS = ("live", "last")   # read modes of an entity's history


def _count(n: int, word: str, plural: str | None = None) -> str:
    return f"{n:,}".replace(",", " ") + f" {word if n <= 1 else plural or word + 's'}"


def _names(tables) -> str:
    tables = sorted(tables)
    shown, more = ", ".join(tables[:3]), len(tables) - 3
    return shown + (f" et {more} autre{'s' if more > 1 else ''}" if more > 0 else "")


def _facts(columns: dict[str, str | None], facts: dict, entry: dict, pointing: set[str]) -> dict:
    """
    What the rules read of a table: its columns (the ones the database lists, else the ones the profile knows),
    rows, the dates (columns typed as dates or formatted as dates in tables.yaml, but not the partitions nor
    the version of its read mode), the measures (numbers that are no identifier) and its unique keys.
    """
    keys  = facts.get("keys") or {}
    names = list(columns) or list(dict.fromkeys([*keys, *(facts.get("text_columns") or {})]))
    hidden, formats = excluded_columns(entry), entry.get("formats") or {}
    layout = {entry.get("partition_column"), entry.get("version_column"), *(facts.get("partitioned_by") or [])}
    rows = facts.get("rows")
    dates = [] if entry.get("read_mode") in VERSIONS else [
        c for c in names if c not in hidden and c not in layout
        and (column_kind(columns.get(c)) == "date" or str(formats.get(c, "")).startswith("date"))]
    identifier = lambda c: c in keys or c in pointing or KEY_NAME.search(c.lower())   # noqa: E731
    measures = [c for c in names if c not in hidden and not identifier(c)
                and (column_kind(columns.get(c)) == "number" or formats.get(c) == "decimal")]
    unique = [k for k, n in keys.items() if rows and n["distinct"] == n["non_null"] == rows]
    return {"columns": names, "rows": rows, "dates": dates, "measures": measures, "unique": unique}


def table_types(tables: list[str], columns: dict[str, dict[str, str | None]], profile: dict, semantics: dict,
                relations: list[tuple[str, str]]) -> dict[str, dict]:
    """
    {table: {"type", "detected", "reason", "edited"}}: the type shown (the one corrected by hand, else the one
    detected, None when unclassified), the type the rules give and why, in a few words. columns: {table:
    {column: type}} as the database lists them (the types may be unknown: None); relations: (from, to) as
    table.column.
    """
    out, into, pointing = defaultdict(set), defaultdict(set), defaultdict(set)
    for source, target in relations:
        (s_table, s_column), t_table = source.split(".", 1), target.split(".", 1)[0]
        pointing[s_table].add(s_column)   # pointing to its own table too (a manager): an identifier, no measure
        if s_table != t_table:
            out[s_table].add(t_table)
            into[t_table].add(s_table)
    facts = {t: _facts(columns.get(t) or {}, profile.get(t) or {}, semantics.get(t) or {}, pointing[t]) for t in tables}

    def reference(t: str, references: set[str] | None) -> bool:
        f = facts[t]
        return (bool(into[t]) and not f["dates"] and len(f["columns"]) <= REFERENCE_COLUMNS
                and (f["rows"] or 0) <= REFERENCE_ROWS
                and (not out[t] if references is None else out[t] <= references))

    # A reference may point to another one (a territory to its region): read once alone, then with them
    references = {t for t in tables if reference(t, None)}
    references |= {t for t in tables if reference(t, references)}

    def many(t: str) -> bool:
        """Many rows for each row of the tables it points to (unknown sizes: not against it)."""
        sizes = [facts[o]["rows"] for o in out[t] if o in facts and facts[o]["rows"]]
        return facts[t]["rows"] is None or not sizes or facts[t]["rows"] > EVENTS_PER_ROW * max(sizes)

    types = {}
    for t in tables:
        f, entry = facts[t], semantics.get(t) or {}
        rows, n = f["rows"], len(f["columns"])
        if len(out[t]) >= 2 and n <= BRIDGE_COLUMNS and len(pointing[t] & set(f["columns"])) >= n - 1:
            detected, reason = "bridge", (f"{_count(n, 'colonne')}, dont {len(pointing[t] & set(f['columns']))} "
                                          f"vers d'autres tables ({_names(out[t])})")
        elif rows == 0:
            detected, reason = None, "table vide : rien à mesurer"
        elif entry.get("read_mode") in VERSIONS:
            detected, reason = "entity", f"garde les versions de chaque ligne (mode {entry['read_mode'].upper()})"
        elif suggest(profile.get(t) or {})[0] == "last":
            by = profile[t]["partitioned_by"][0]
            detected, reason = "entity", f"une photo complète de ses lignes par partition de {by}"
        elif (f["dates"] and out[t] and many(t)) or (f["measures"] and len(out[t]) >= 2 and not into[t]):
            parts = [f["dates"] and _count(len(f["dates"]), "date"), f["measures"] and _count(len(f["measures"]), "mesure"),
                     f"vise {_count(len(out[t]), 'table')} ({_names(out[t])})", rows is not None and _count(rows, "ligne")]
            detected, reason = "events", ", ".join(p for p in parts if p)
        elif t in references:
            detected, reason = "reference", (f"visée par {_names(into[t])}, {_count(rows or 0, 'ligne')}, "
                                             f"{_count(n, 'colonne')}, pas de date")
        elif into[t] or f["unique"]:
            parts = [f["unique"] and f"une ligne par {f['unique'][0]}", into[t] and f"visée par {_names(into[t])}",
                     _count(n, "colonne")]
            detected, reason = "entity", ", ".join(p for p in parts if p)
        else:
            detected, reason = None, "ni clé unique ni relation : rien ne dit son rôle"
        chosen = entry.get("table_type") if entry.get("table_type") in TABLE_TYPES else None
        types[t] = {"type": chosen or detected, "detected": detected, "reason": reason, "edited": chosen is not None}
    return types
