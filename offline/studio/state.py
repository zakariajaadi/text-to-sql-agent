# offline/studio/state.py

"""
What the studio shows of a database, read from its semantic layer files (pure functions, unit-tested):
the status of each table at each step, what the profile detects in each column, and the descriptions
the LLM writes (semantics_detail, edit_semantics).

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
from datetime import date

from offline.semantic_gen.profiling import DATE_SHAPES, NUMBER_SHAPE

# Status of a table at a step
NONE, PARTIAL, DONE = "none", "partial", "done"
TABLE_TYPES = ("events", "entity", "reference", "bridge")   # the types a human may choose (offline/studio/table_types.py)
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


def undescribed(entry: dict | None, columns: list[str]) -> list[str]:
    """The columns the agent sees (not excluded) that have no description yet."""
    entry = entry or {}
    excluded, described = set(entry.get("exclude_columns") or []), entry.get("columns") or {}
    return [c for c in columns if c not in excluded and not described.get(c)]


def marks(entry: dict | None) -> dict[str, int]:
    """
    How many descriptions of a table the agent sees were corrected by hand (edited, kept when the table is
    described again), rewritten by the LLM over another text (rewritten, to read again) and written by the
    LLM without being able to tell the meaning or the unit of their column (unclear, to state by hand).
    """
    entry = entry or {}
    excluded = set(entry.get("exclude_columns") or [])
    count = lambda kind: (bool(entry.get(f"{kind}_description"))  # noqa: E731
                          + sum(c not in excluded for c in entry.get(f"{kind}_columns") or []))
    return {"edited": count("edited"), "rewritten": count("rewritten"), "unclear": count("unclear")}


def table_rows(tables: list[str], columns: dict[str, list[str]], profile: dict, semantics: dict,
               relations: list[tuple[str, str]], metrics: dict, personal: bool) -> list[dict]:
    """
    One row per table of the home screen: its size (rows measured by the profile, columns of the database)
    and its status at each step. Profile: done once profiled (what it detects needs no review).
    Semantics: done once the table and its columns are described, partial while some columns are not
    (`undescribed`: added since the last description); with the descriptions corrected by hand, the
    ones the LLM rewrote, to read again, and the ones it could not tell, to state (marks). Relations,
    metrics: done when a relation or a metric reads the table.
    """
    related  = {side.split(".")[0] for relation in relations for side in relation}
    measured = metric_tables(metrics)
    rows = []
    for table in tables:
        facts = profile.get(table)
        entry = semantics.get(table) or {}
        missing = undescribed(entry, columns.get(table) or []) if entry.get("description") else []
        rows.append({
            "name":        table,
            "rows":        facts["rows"] if facts else None,
            "columns":     len(columns[table]) if columns.get(table) else None,
            "undescribed": len(missing),
            **marks(entry),
            "status": {
                "profile":   DONE if facts else NONE,
                "semantics": NONE if not entry.get("description") else PARTIAL if missing else DONE,
                "relations": DONE if table in related else NONE,
                "metrics":   DONE if table.lower() in measured else NONE,
            },
        })
    return rows


def summary(rows: list[dict], relations: list[tuple[str, str]], metrics: dict) -> dict:
    """
    The counts of each step over the rows of table_rows: the filters of the tables screen, the pipeline of
    a database on the home screen. rewritten: tables with a description the LLM rewrote, to read again;
    unclear: tables with a description the LLM could not tell, to state; isolated: tables no relation reads.
    """
    count = lambda step, status: sum(r["status"][step] == status for r in rows)  # noqa: E731
    return {
        "tables":    len(rows),
        "profiled":  len(rows) - count("profile", NONE),
        "described": len(rows) - count("semantics", NONE),
        "partial":   count("semantics", PARTIAL),
        "rewritten": sum(bool(r["rewritten"]) for r in rows),
        "unclear":   sum(bool(r["unclear"]) for r in rows),
        "isolated":  count("relations", NONE),
        "relations": len(relations),
        "metrics":   len([name for name in metrics if name != "ignore"]),
    }


def table_detail(table: str, facts: dict | None, entry: dict | None, personal: bool,
                 db_columns: list[dict] | None = None) -> dict:
    """
    The profile of a table as the detail panel shows it, one line per column of the database
    (`db_columns`: name and type, in DDL order; without them, the columns the profile knows):
    its type, filled share, distinct values (of the whole table for a key, of the sample for a text
    column: `sampled`), how tables.yaml reads it (format, excluded), what the profile detects (a format
    with the share of the values written in it, a date whose day and month order no value tells, an
    empty column) and, for personal data, whether it is hidden, kept visible on purpose or visible.
    `outdated`: what an older profiler did not measure (filled, distinct). Counts only, never a value.
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
    columns = list(dict.fromkeys([*types, *keys, *texts, *others, *by_column, *personal_data]))
    for column in columns:
        counts = keys.get(column)
        if counts:
            filled = round(counts["non_null"] / rows, 3) if rows else None
        else:
            filled = (texts.get(column) or {}).get("filled", others.get(column))
        type_name = types.get(column)
        on_sample = None if counts else (texts.get(column) or {}).get("distinct")
        column_facts.append({
            "column":   column,
            "type":     type_name,
            "kind":     column_kind(type_name) or ("text" if column in texts else None),
            "filled":   filled,
            "distinct": counts["distinct"] if counts else on_sample,
            "sampled":  on_sample is not None,
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
        # the formats the agent applies: written in tables.yaml, detected or by hand, of the columns it sees
        "formats":      sum(bool(c["format"]) and not c["excluded"] for c in column_facts),
        "unique_key":   next((c["column"] for c in column_facts if c["key"] == "unique"), None),
        "outdated":     outdated(facts, columns),
        "column_facts": column_facts,
    }


def outdated(facts: dict, columns: list[str]) -> list[str]:
    """
    What an older profiler did not measure in a table ([] when nothing): "distinct", the distinct values of
    its text columns; "filled", the filled share of the other columns (numbers, dates) the table has.
    The two came at different times: a profile may lack one only.
    """
    keys, texts = facts.get("keys") or {}, facts.get("text_columns") or {}
    missing = []
    if "filled" not in facts and any(c not in keys and c not in texts for c in columns):
        missing.append("filled")
    if any("distinct" not in (text or {}) for text in texts.values()):
        missing.append("distinct")
    return missing


# ------------------------------------------------------------
# Semantics: the descriptions the LLM writes, read and corrected in the studio
# ------------------------------------------------------------

DATE = re.compile(r"^\d{4}-\d\d-\d\d$")


def semantics_detail(table: str, entry: dict | None, facts: dict | None, db_columns: list[dict] | None) -> dict:
    """
    A table as the Semantics screen shows it: its description and the date its data stops (data_as_of),
    then one line per column of the database (`db_columns`, in DDL order; without them the described ones):
    its type, description, whether it is categorical, and its distinct values measured by the profile
    (of the whole table for a key, of the sample for a text column; None when not measured). Each
    description says whether it was corrected by hand (edited), rewritten by the LLM (rewritten) or
    written by it without being able to tell the meaning or the unit of the column (unclear).
    An excluded column is never described: the agent does not see it.
    """
    entry, facts = entry or {}, facts or {}
    described   = entry.get("columns") or {}
    categorical = set(entry.get("categorical_columns") or [])
    excluded    = set(entry.get("exclude_columns") or [])
    edited      = set(entry.get("edited_columns") or [])
    rewritten   = set(entry.get("rewritten_columns") or [])
    unclear     = set(entry.get("unclear_columns") or [])
    keys, texts = facts.get("keys") or {}, facts.get("text_columns") or {}
    rows        = facts.get("rows") or 0
    types       = {c["name"]: c["type"] for c in db_columns or []}
    columns = []
    for column in dict.fromkeys([*types, *described]):
        counts = keys.get(column)
        columns.append({
            "column":      column,
            "type":        types.get(column),
            "kind":        column_kind(types.get(column)),
            "key":         None if not counts else "unique" if rows and counts["distinct"] == counts["non_null"] == rows else "key",
            "description": described.get(column) or "",
            "categorical": column in categorical,
            "distinct":    counts["distinct"] if counts else (texts.get(column) or {}).get("distinct"),
            "sampled":     not counts and (texts.get(column) or {}).get("distinct") is not None,
            "excluded":    column in excluded,
            "edited":      column in edited,
            "rewritten":   column in rewritten,
            "unclear":     column in unclear,
        })
    visible = [c for c in columns if not c["excluded"]]
    return {
        "name":        table,
        "described":   bool(entry.get("description")),
        "description": entry.get("description") or "",
        "description_edited":    bool(entry.get("edited_description")),
        "description_rewritten": bool(entry.get("rewritten_description")),
        "data_as_of":  str(entry["data_as_of"]) if entry.get("data_as_of") else "",
        "counts": {
            "columns":     len(columns),
            "described":   sum(bool(c["description"]) for c in visible),
            "visible":     len(visible),
            "categorical": sum(c["categorical"] for c in visible),
            **marks(entry),
        },
        "columns": columns,
    }


def edit_semantics(entry: dict | None, field: str, value, column: str | None = None,
                   order: list[str] | None = None) -> dict:
    """
    The table block with one field corrected in the studio:
      - description, data_as_of (YYYY-MM-DD): the text, removed when empty
      - column: the description of `column`, removed when empty (make semantics then writes it again)
      - categorical: `column` in categorical_columns or not, kept in the DDL `order`
      - unclear: `column` in unclear_columns or not (false: its description is fine as it is), in the DDL `order`
      - read: the descriptions the LLM rewrote are read (no longer rewritten_*)
      - table_type: the type of the table chosen by hand (events, entity, reference, bridge), over the one
        detected (offline/studio/table_types.py); empty, the detection again
    A description corrected is marked edited (kept when the table is described again), read and no longer
    unclear; emptied, it is no longer edited. ValueError on a date that is not one, or a type that is none.
    """
    entry = dict(entry or {})
    if field == "table_type":
        if value and value not in TABLE_TYPES:
            raise ValueError(f"Type inconnu : {value}")
        if value:
            entry["table_type"] = value
        else:
            entry.pop("table_type", None)
        return entry
    if field == "description":
        _mark(entry, "description", None, bool(str(value or "").strip()))
    elif field == "column":
        _mark(entry, "columns", column, bool(str(value or "").strip()))
    elif field == "read":
        entry.pop("rewritten_description", None)
        entry.pop("rewritten_columns", None)
        return entry
    if field in ("description", "data_as_of"):
        text = str(value or "").strip()
        if field == "data_as_of" and text:
            if not DATE.match(text):
                raise ValueError(f"Ce n'est pas une date (AAAA-MM-JJ) : {text}")
            text = date.fromisoformat(text)   # a YAML date, unquoted as written by hand
        if text:
            entry[field] = text
        else:
            entry.pop(field, None)
    elif field == "column":
        described = dict(entry.get("columns") or {})
        if text := str(value or "").strip():
            described[column] = text
        else:
            described.pop(column, None)
        entry["columns"] = described
    elif field in ("categorical", "unclear"):
        key = f"{field}_columns"
        current = [c for c in entry.get(key) or [] if c != column]
        if value:
            current.append(column)
        rank = {name: i for i, name in enumerate(order or [])}
        entry[key] = sorted(current, key=lambda c: rank.get(c, len(rank)))
        if field == "unclear" and not current:
            entry.pop(key)
    else:
        raise ValueError(f"Champ inconnu : {field}")
    return entry


def _mark(entry: dict, key: str, column: str | None, edited: bool) -> None:
    """A description corrected by hand: edited (or no longer, when emptied), no longer rewritten nor unclear."""
    if column is None:
        entry.pop(f"rewritten_{key}", None)
        if edited:
            entry[f"edited_{key}"] = True
        else:
            entry.pop(f"edited_{key}", None)
        return
    for kind, keep in (("rewritten", False), ("unclear", False), ("edited", edited)):
        current = [c for c in entry.get(f"{kind}_{key}") or [] if c != column] + ([column] if keep else [])
        if current:
            entry[f"{kind}_{key}"] = current
        else:
            entry.pop(f"{kind}_{key}", None)
