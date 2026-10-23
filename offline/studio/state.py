# offline/studio/state.py

"""
What the studio shows of a database, read from its semantic layer files (pure functions, unit-tested):
the status of each table at each step, and what the profile detects in each column.

What the profile detects is data-driven, nobody reviews it: `make profile` writes the formats and the
empty columns into tables.yaml (offline/semantic_gen/profiling.apply_profile). The studio shows each
detection with its state, and a click on its pill makes an exception:
  - a format: applied in `formats`, refused in `keep_as_text` (or another format in `formats`),
    not applied in neither (a table whose formats were reviewed before the profile found it)
  - an exclusion: applied in `exclude_columns`, refused in `keep_columns` (kept visible on purpose),
    not applied in neither
Personal data is hidden in the whole database at once (hide_personal).
"""

import re

from offline.semantic_gen.profiling import DATE_SHAPES, NUMBER_SHAPE

# Status of a table at a step
NONE, DONE = "none", "done"
PERSONAL = "personal data"


def proposals(facts: dict | None, entry: dict | None, personal: bool) -> list[dict]:
    """
    The formats and exclusions the profile detects in a table, each with its state in tables.yaml
    (accepted: applied, rejected: an exception, pending: not applied), what tables.yaml holds for the
    column, and whether make profile applies it (`default`: personal data only when the database says so).
    """
    facts, entry = facts or {}, entry or {}
    formats   = entry.get("formats") or {}
    kept_text = set(entry.get("keep_as_text") or [])
    excluded  = set(entry.get("exclude_columns") or [])
    kept      = set(entry.get("keep_columns") or [])
    result    = []
    for column, value in (facts.get("formats") or {}).items():
        current = formats.get(column)
        state   = ("accepted" if current == value else "rejected" if current or column in kept_text else "pending")
        result.append({"kind": "format", "column": column, "value": value, "reason": None,
                       "state": state, "current": current, "default": True})
    for column, reason in (facts.get("exclude") or {}).items():
        state = "accepted" if column in excluded else "rejected" if column in kept else "pending"
        result.append({"kind": "exclude", "column": column, "value": None, "reason": reason,
                       "state": state, "current": None, "default": reason != PERSONAL or personal})
    return result


def set_format(entry: dict | None, column: str, value: str | None) -> dict:
    """The table block with `column` read in the format `value`, or kept as text on purpose (value None)."""
    entry   = dict(entry or {})
    formats = dict(entry.get("formats") or {})
    kept    = [c for c in entry.get("keep_as_text") or [] if c != column]
    if value:
        formats[column] = value
    else:
        formats.pop(column, None)
        kept.append(column)
    entry["formats"] = formats
    if kept:
        entry["keep_as_text"] = kept
    else:
        entry.pop("keep_as_text", None)
    return entry


def set_excluded(entry: dict | None, columns, hide: bool) -> dict:
    """
    The table block with `columns` hidden (exclude_columns) or kept visible on purpose (keep_columns).
    No column, no change: an empty exclude_columns key would stop make semantics adding its exclusions.
    """
    entry, columns = dict(entry or {}), list(columns)
    if not columns:
        return entry
    add, drop = ("exclude_columns", "keep_columns") if hide else ("keep_columns", "exclude_columns")
    current = list(entry.get(add) or [])
    entry[add] = current + [c for c in columns if c not in current]
    if drop in entry:
        entry[drop] = [c for c in entry[drop] or [] if c not in columns]
    if not entry.get("keep_columns"):
        entry.pop("keep_columns", None)
    return entry


def personal_columns(profile: dict, semantics: dict) -> dict[str, dict[str, str]]:
    """{table: {column: hidden | kept | visible}}: the personal data the profile found, and how tables.yaml reads it."""
    result = {}
    for table, facts in profile.items():
        entry    = semantics.get(table) or {}
        excluded = set(entry.get("exclude_columns") or [])
        kept     = set(entry.get("keep_columns") or [])
        columns  = {c: "hidden" if c in excluded else "kept" if c in kept else "visible"
                    for c, reason in ((facts or {}).get("exclude") or {}).items() if reason == PERSONAL}
        if columns:
            result[table] = columns
    return result


def hide_personal(profile: dict, semantics: dict, hide: bool) -> dict:
    """
    tables.yaml with the personal data of every profiled table hidden, the columns kept visible on purpose
    left so; or shown again (hide False), the exceptions kept. The tables without personal data are untouched.
    """
    semantics = dict(semantics)
    for table, columns in personal_columns(profile, semantics).items():
        entry = dict(semantics.get(table) or {})
        if hide:
            entry = set_excluded(entry, [c for c, state in columns.items() if state == "visible"], hide=True)
        elif excluded := [c for c, state in columns.items() if state == "hidden"]:
            entry["exclude_columns"] = [c for c in entry.get("exclude_columns") or [] if c not in excluded]
        semantics[table] = entry
    return semantics


def metric_tables(metrics: dict) -> set[str]:
    """Tables the metrics of metrics.yaml read (their expression and time dimension, written table.column)."""
    tables = set()
    for name, metric in (metrics or {}).items():
        if name == "ignore" or not isinstance(metric, dict):
            continue
        text = f"{metric.get('expr') or ''} {metric.get('time_dimension') or ''}"
        tables |= {m.group(1).lower() for m in re.finditer(r"\b([A-Za-z_]\w*)\.[A-Za-z_]\w*", text)}
    return tables


def format_share(text_column: dict | None, value: str) -> float | None:
    """
    Share of the filled values of a text column written in the format proposed ("date dd/MM/yyyy",
    "decimal"), from the shapes of the profile (a share of every value sampled, NULL included).
    The profile keeps the main shapes only: a lower bound. None when nothing was sampled.
    """
    if not text_column or not text_column.get("filled"):
        return None
    if value == "decimal":
        match = NUMBER_SHAPE.match
    else:
        regexes = [regex for regex, pattern in DATE_SHAPES if f"date {pattern}" == value]
        match = lambda shape: any(regex.match(shape) for regex in regexes)  # noqa: E731
    found = sum(share for shape, share in (text_column.get("shapes") or {}).items() if match(shape))
    return round(min(found / text_column["filled"], 1.0), 3)


def column_kind(type_name: str | None) -> str | None:
    """What a column type holds: text, date, number, boolean, other (None: type unknown)."""
    if not type_name:
        return None
    name = type_name.lower()
    for kind, words in (("text", ("string", "char", "text")), ("date", ("date", "time")),
                        ("number", ("int", "dec", "num", "float", "double", "real")), ("boolean", ("bool",))):
        if any(word in name for word in words):
            return kind
    return "other"


def table_rows(tables: list[str], columns: dict[str, int], profile: dict, semantics: dict,
               relations: list[tuple[str, str]], metrics: dict, personal: bool) -> list[dict]:
    """
    One row per table of the home screen: its size (rows measured by the profile, columns of the database)
    and its status at each step. Profile: done once profiled (what it detects needs no review).
    Semantics: done once described. Relations, metrics: done when a relation or a metric reads the table.
    """
    related  = {side.split(".")[0] for relation in relations for side in relation}
    measured = metric_tables(metrics)
    rows = []
    for table in tables:
        facts = profile.get(table)
        rows.append({
            "name":    table,
            "rows":    facts["rows"] if facts else None,
            "columns": columns.get(table),
            "status": {
                "profile":   DONE if facts else NONE,
                "semantics": DONE if (semantics.get(table) or {}).get("description") else NONE,
                "relations": DONE if table in related else NONE,
                "metrics":   DONE if table.lower() in measured else NONE,
            },
        })
    return rows


def table_detail(table: str, facts: dict | None, entry: dict | None, personal: bool,
                 db_columns: list[dict] | None = None) -> dict:
    """
    The profile of a table as the detail panel shows it, one line per column of the database
    (`db_columns`: name and type, in DDL order; without them, the columns the profile knows):
    its type, filled share, distinct values of a key, how tables.yaml reads it (format, excluded),
    what the profile detects (a format with the share of the values written in it, a date whose day
    and month order no value tells, an empty column) and, for personal data, whether it is hidden,
    kept visible on purpose or visible. Counts only, never a value.
    """
    if not facts:
        return {"name": table, "profiled": False, "detected": 0, "unique_key": None, "column_facts": []}
    rows     = facts.get("rows") or 0
    keys     = facts.get("keys") or {}
    texts    = facts.get("text_columns") or {}
    others   = facts.get("filled") or {}
    types    = {c["name"]: c["type"] for c in db_columns or []}
    entry    = entry or {}
    formats  = entry.get("formats") or {}
    excluded = set(entry.get("exclude_columns") or [])
    kept     = set(entry.get("keep_columns") or [])
    personal_data = {c for c, reason in (facts.get("exclude") or {}).items() if reason == PERSONAL}
    ambiguous     = set(facts.get("ambiguous_dates") or [])
    # A column may get a format and be personal data too (date_naissance stored as text)
    by_column = {}
    for proposal in (p for p in proposals(facts, entry, personal) if p["reason"] != PERSONAL):
        if proposal["kind"] == "format":
            proposal["share"] = format_share(texts.get(proposal["column"]), proposal["value"])
        by_column.setdefault(proposal["column"], []).append(proposal)

    column_facts = []
    for column in dict.fromkeys([*types, *keys, *texts, *others, *by_column, *personal_data]):
        counts = keys.get(column)
        if counts:
            filled = round(counts["non_null"] / rows, 3) if rows else None
        else:
            filled = (texts.get(column) or {}).get("filled", others.get(column))
        type_name = types.get(column)
        column_facts.append({
            "column":   column,
            "type":     type_name,
            "kind":     column_kind(type_name) or ("text" if column in texts else None),
            "filled":   filled,
            "distinct": counts["distinct"] if counts else None,
            "key":      None if not counts else "unique" if rows and counts["distinct"] == counts["non_null"] == rows else "key",
            "format":   formats.get(column),
            "ambiguous": column in ambiguous,
            "excluded": column in excluded,
            "personal": None if column not in personal_data
                        else "hidden" if column in excluded else "kept" if column in kept else "visible",
            "proposals": by_column.get(column, []),
        })
    return {
        "name":         table,
        "profiled":     True,
        "rows":         rows,
        "detected":     sum(p["kind"] == "format" for column in by_column.values() for p in column),
        "unique_key":   next((c["column"] for c in column_facts if c["key"] == "unique"), None),
        "column_facts": column_facts,
    }
