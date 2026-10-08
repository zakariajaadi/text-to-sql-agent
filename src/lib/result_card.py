"""
Result card — props of the ResultCard custom element (public/elements/ResultCard.jsx)
Used by: lib/chainlit_ui.py (attached to the final answer message)

The visualization is picked by deterministic rules on the result shape, never by an LLM:
  - 1 row with measures          → KPI tiles
  - temporal column + measures   → line chart
  - label column(s) + measures   → horizontal bar chart
  - anything else                → table only
"""

import re

import sqlparse

MAX_KPIS     = 4
MAX_SERIES   = 4
MAX_LABELS   = 2

_ISO_DATE    = re.compile(r"^\d{4}-\d{2}(-\d{2})?([T ]\d{2}:\d{2}(:\d{2})?.*)?$")
_YEAR_NAME   = re.compile(r"(^|_)(year|annee|année|yr)($|_)", re.IGNORECASE)
_MONTH_NAME  = re.compile(r"(^|_)(month|mois)($|_)", re.IGNORECASE)
_PERIOD_NAME = re.compile(r"(^|_)(date|day|jour|period|periode|période|quarter|trimestre)($|_)", re.IGNORECASE)
_ID_NAME     = re.compile(r"(^|_)id$|[a-z]Id$|^id_", re.IGNORECASE)


def build_result_card_props(
    columns:   list[str],
    rows:      list[list],
    sql:       str,
    truncated: bool,
    tables:    list[str],
    retries:   int,
    db_name:   str,
) -> dict:
    """Assemble the JSON props of the ResultCard element."""
    columns = [_display_name(c) for c in columns]
    viz     = pick_visualization(columns, rows)

    return {
        "columns":   columns,
        "rows":      rows,
        "rowCount":  len(rows),
        "truncated": truncated,
        "tables":    tables,
        "sql":       sqlparse.format(sql, reindent=True, keyword_case="upper"),
        "retries":   retries,
        "dbName":    db_name,
        "chart":     viz["chart"],
        "kpis":      viz["kpis"],
    }


def pick_visualization(columns: list[str], rows: list[list]) -> dict:
    """
    Return {"chart": {...} | None, "kpis": [...] | None}.
    chart = {"type": "line" | "bar", "x": [col idx...], "y": [col idx...]}: x columns are
    joined into one label, each y column is a series the user can switch between.
    """
    none = {"chart": None, "kpis": None}
    if not rows or not columns:
        return none

    kinds    = [_column_kind(name, [row[i] for row in rows]) for i, name in enumerate(columns)]
    measures = [i for i, k in enumerate(kinds) if k == "measure"]
    temporal = [i for i, k in enumerate(kinds) if k == "temporal"]
    labels   = [i for i, k in enumerate(kinds) if k == "label"]
    labels   = labels or [i for i, k in enumerate(kinds) if k == "id"]  # e.g. "CustomerId, total"

    if not measures:
        return none

    # Single row → KPI tiles, labelled by the row's text values (e.g. "top customer")
    if len(rows) == 1:
        caption = " · ".join(str(rows[0][i]) for i in labels + temporal if rows[0][i] is not None)
        kpis = [
            {"label": columns[i], "value": rows[0][i], "caption": caption}
            for i in measures[:MAX_KPIS]
        ]
        return {"chart": None, "kpis": kpis}

    # Time series → line chart (year + month columns are joined into one x label)
    if temporal:
        return {"chart": {"type": "line", "x": temporal, "y": measures[:MAX_SERIES]}, "kpis": None}

    # Categories → bar chart (first name + last name columns are joined into one label)
    if labels:
        return {"chart": {"type": "bar", "x": labels[:MAX_LABELS], "y": measures[:MAX_SERIES]}, "kpis": None}

    return none


# ── Private helpers ───────────────────────────────────────────────────────────


def _display_name(column: str) -> str:
    """Hive prefixes result columns with their table ("t.col"): keep the column part."""
    return column.rsplit(".", 1)[-1]


def _column_kind(name: str, values: list) -> str:
    """Classify a column as "temporal", "measure", "id" or "label" from its name and values."""
    present = [v for v in values if v is not None]
    if not present:
        return "label"

    is_int     = all(isinstance(v, int) and not isinstance(v, bool) for v in present)
    is_numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present)

    if all(isinstance(v, str) and _ISO_DATE.match(v) for v in present):
        return "temporal"
    if is_int and _YEAR_NAME.search(name) and all(1900 <= v <= 2100 for v in present):
        return "temporal"
    if is_int and _MONTH_NAME.search(name) and all(1 <= v <= 12 for v in present):
        return "temporal"
    if _PERIOD_NAME.search(name) and (is_int or all(isinstance(v, str) for v in present)):
        return "temporal"
    if is_numeric:
        return "id" if is_int and _ID_NAME.search(name) else "measure"
    return "label"
