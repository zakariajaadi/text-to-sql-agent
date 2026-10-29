"""
Charts — what the result card draws for a query result: KPI tiles or a ready-to-draw chart
Used by: ui/result_card.py (props of the ResultCard element)

Picked by deterministic rules on the result shape and the question words, never by an LLM, in three steps:
  1. classify   each column: temporal (year, quarter, month, date, period), measure, id or label
  2. normalize  to one shape per meaning, so the SQL wording doesn't change the chart:
                - a month date next to a year column ("2022-01" beside 2022) becomes the month (1)
                - measures pivoted per year, as a CASE WHEN writes them (ca_2022, ca_2023),
                  are melted into a year column + one measure
  3. pick       the visualization:
                  - 1 row with measures                    → KPI tiles
                  - 2 crossed dimensions, one with 2..MAX_GROUPS values → one series per value
                    (lines on a temporal axis, grouped bars otherwise), e.g. year × month
                  - temporal column(s) + measures          → line chart
                  - label column(s) + measures             → bar chart, largest first (pie on a toggle)
                  - anything else                         → table only

The chart is sent ready to draw, so the element only renders it:
  {"type": "line" | "bar",
   "measures": [{"name": str, "labels": [x label...], "series": [{"name": str | None, "values": [...]}]}],
   "timeline": same list as "measures" | None}
One entry per measure the user can switch between. "timeline" is the chronological view of a
year comparison (2022 vs 2023 by month → 24 months in a row), offered as a toggle.

A bar chart of categories also reads as shares of a whole, drawn as a pie on a toggle. Each of
its measures then carries its slices, and the chart says which view opens first:
  measure["parts"]:     {"labels": [...], "values": [...], "others": {"count": int, "value": num} | None}
                        or None when the values make no whole (an average, a price, a value ≤ 0,
                        rows cut by the row limit). Past MAX_SLICES categories, the smallest are
                        folded into "others".
  chart["parts_first"]: the question asks for a distribution ("répartition", "part"...): the pie
                        opens first. Otherwise bars do: a ranking reads better than angles.
"""

import re
from dataclasses import dataclass

MAX_KPIS     = 4
MAX_MEASURES = 4    # measures the user can switch between
MAX_LABELS   = 2    # label columns joined into one bar label (first name + last name)
MAX_GROUPS   = 4    # most series a dimension may split the chart into
MAX_SLICES   = 6    # most slices of a pie, "others" included

MONTHS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]

_ISO_DATE     = re.compile(r"^\d{4}-\d{2}(-\d{2})?([T ]\d{2}:\d{2}(:\d{2})?.*)?$")
_MIDNIGHT     = re.compile(r"[T ]00:00:00(\.0+)?$")
_YEAR_NAME    = re.compile(r"(^|_)(year|annee|année|yr)($|_)", re.IGNORECASE)
_QUARTER_NAME = re.compile(r"(^|_)(quarter|trimestre|qtr)($|_)", re.IGNORECASE)
_MONTH_NAME   = re.compile(r"(^|_)(month|mois)($|_)", re.IGNORECASE)
_PERIOD_NAME  = re.compile(r"(^|_)(date|day|jour|period|periode|période|quarter|trimestre)($|_)", re.IGNORECASE)
_ID_NAME      = re.compile(r"(^|_)id$|[a-z]Id$|^id_|(^|_)(numero|num)($|_)", re.IGNORECASE)
_MONTH_DATE   = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})(-01([T ]00:00:00(\.0+)?)?)?$")
_YEAR_SUFFIX  = re.compile(r"^(?P<base>.*?)[_ ]?(?P<year>(19|20)\d{2})$|^(?P<year2>(19|20)\d{2})[_ ]?(?P<base2>.*)$")
_NAME_WORDS   = re.compile(r"[A-Z]?[a-zà-ÿ]+|[A-Z]+(?![a-z])|\d+")  # unit_price, UnitPrice → unit, price

# Words of a measure whose values don't add up across categories: a pie of them has no whole
_NOT_PARTS = {
    "avg", "average", "mean", "moy", "moyen", "moyenne", "median", "mediane", "médiane", "min", "max",
    "prix", "price", "taux", "rate", "ratio", "pct", "percent", "percentage", "pourcent", "pourcentage",
}
# Words of a question asking how a whole splits: the pie opens first
_PARTS_QUESTION = re.compile(
    r"\b(r[ée]partitions?|parts?|proportions?|pourcentages?|distributions?|ventilations?|shares?|breakdowns?)\b",
    re.IGNORECASE,
)

# Temporal units from coarsest to finest: the sort order of a time axis. "period" (free text,
# e.g. "2022-S1") has no known order and keeps the query's row order.
_UNITS = ["year", "quarter", "month", "date"]


@dataclass
class Field:
    """One column of the result, as the chart sees it."""
    name:   str
    kind:   str         # "temporal" | "measure" | "id" | "label"
    unit:   str | None  # temporal only: "year" | "quarter" | "month" | "date" | "period"
    values: list        # one per row


def pick_visualization(columns: list[str], rows: list[list], question: str = "", truncated: bool = False) -> dict:
    """
    Return {"chart": {...} | None, "kpis": [...] | None} for a query result.
    question: the user's question, its words pick the first view of a pie.
    truncated: the rows were cut by the row limit, so they make no whole to share.
    """
    none = {"chart": None, "kpis": None}
    if not rows or not columns:
        return none

    fields = [_classify(name, [row[i] for row in rows]) for i, name in enumerate(columns)]
    if not any(f.kind == "measure" for f in fields):
        return none

    if len(rows) == 1:
        return {"chart": None, "kpis": _kpis(fields)}
    chart = _chart(_melt_year_pivot(_month_of_dated_year(fields)), whole=not truncated,
                   asks_parts=bool(_PARTS_QUESTION.search(question)))
    return {"chart": chart, "kpis": None}


# ── 1. Classify ───────────────────────────────────────────────────────────────


def _classify(name: str, values: list) -> Field:
    """Classify a column as temporal (with its unit), measure, id or label from its name and values."""
    def field(kind: str, unit: str | None = None) -> Field:
        return Field(name, kind, unit, values)

    present = [v for v in values if v is not None]
    if not present:
        return field("label")

    is_int     = all(isinstance(v, int) and not isinstance(v, bool) for v in present)
    is_numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present)
    # Months and quarters come as ints or as zero-padded strings (DATE_FORMAT(d, 'MM') → "01")
    as_ints    = [_as_int(v) for v in present]

    if all(isinstance(v, str) and _ISO_DATE.match(v) for v in present):
        return field("temporal", "date")
    if is_int and _YEAR_NAME.search(name) and all(1900 <= v <= 2100 for v in present):
        return field("temporal", "year")
    if _QUARTER_NAME.search(name) and all(v is not None and 1 <= v <= 4 for v in as_ints):
        return field("temporal", "quarter")
    if _MONTH_NAME.search(name) and all(v is not None and 1 <= v <= 12 for v in as_ints):
        return field("temporal", "month")
    if _PERIOD_NAME.search(name) and (is_int or all(isinstance(v, str) for v in present)):
        return field("temporal", "period")
    if is_numeric:
        return field("id" if is_int and _ID_NAME.search(name) else "measure")
    return field("label")


# ── 2. Normalize ──────────────────────────────────────────────────────────────


def _month_of_dated_year(fields: list[Field]) -> list[Field]:
    """
    Turn a month date into its month when a year column already holds its year, as
    YEAR(d), DATE_FORMAT(d, 'yyyy-MM') writes them: ("2022-01", 2022) reads as (1, 2022).
    The explicit year column is the sign of years compared: a month date alone stays a timeline.
    """
    years = [f for f in fields if f.unit == "year"]
    if len(years) != 1:
        return fields

    def month_of(f: Field) -> Field:
        matches = [_MONTH_DATE.match(v) if isinstance(v, str) else None for v in f.values]
        same_year = all(m and int(m["year"]) == y for m, y in zip(matches, years[0].values))
        return Field(f.name, "temporal", "month", [int(m["month"]) for m in matches]) if same_year else f

    return [month_of(f) if f.unit == "date" else f for f in fields]



def _melt_year_pivot(fields: list[Field]) -> list[Field]:
    """
    Melt measures pivoted per year (ca_2022, ca_2023 or 2022_ca, 2023_ca) into a year column and
    one measure, so the chart sees the same shape as a GROUP BY year: one row per (x, year).
    Only for a single x dimension (else the melted result has no clear axis), with 2..MAX_GROUPS
    years; measures outside the pivot (e.g. a variation %) stay in the table only.
    """
    dims = [f for f in fields if f.kind in ("temporal", "label")]
    if len(dims) != 1:
        return fields

    families: dict[str, list[tuple[int, Field, str]]] = {}
    for f in fields:
        match = _YEAR_SUFFIX.match(f.name) if f.kind == "measure" else None
        if match:
            base = (match["base"] if match["year"] else match["base2"]) or ""
            families.setdefault(base.lower(), []).append((int(match["year"] or match["year2"]), f, base))
    family = next((sorted(m, key=lambda m: m[0]) for m in families.values() if 2 <= len(m) <= MAX_GROUPS), None)
    if not family:
        return fields

    [x], rows = dims, range(len(dims[0].values))
    return [
        Field(x.name, x.kind, x.unit, [x.values[r] for r in rows for _ in family]),
        Field("year", "temporal", "year", [year for _ in rows for year, _, _ in family]),
        Field(family[0][2], "measure", None, [f.values[r] for r in rows for _, f, _ in family]),
    ]


# ── 3. Pick ───────────────────────────────────────────────────────────────────


def _kpis(fields: list[Field]) -> list[dict]:
    """KPI tiles of a single row, captioned by its text and time values (e.g. "top customer")."""
    context = [f for f in fields if f.kind in ("label", "temporal")] or [f for f in fields if f.kind == "id"]
    caption = " · ".join(_format(f, f.values[0]) for f in context if f.values[0] is not None)
    return [
        {"label": f.name, "value": f.values[0], "caption": caption}
        for f in fields if f.kind == "measure"
    ][:MAX_KPIS]


def _chart(fields: list[Field], whole: bool, asks_parts: bool) -> dict | None:
    measures = [f for f in fields if f.kind == "measure"][:MAX_MEASURES]
    temporal = [f for f in fields if f.kind == "temporal"]
    labels   = [f for f in fields if f.kind == "label"]

    split = _crossed(temporal + labels)
    if split:
        x, series = split
        if x.kind != "temporal":
            return {"type": "bar", "measures": [_by_size(_split(x, series, m)) for m in measures], "timeline": None}
        # Years compared month by month can also be read as one timeline: offered as a toggle
        timeline = [_sequence([series, x], m) for m in measures] if series.kind == "temporal" else None
        return {"type": "line", "measures": [_split(x, series, m) for m in measures], "timeline": timeline}

    if temporal:
        return {"type": "line", "measures": [_sequence(temporal, m) for m in measures], "timeline": None}

    # Categories, or ids when nothing names the rows (e.g. "CustomerId, total")
    labels = labels or [f for f in fields if f.kind == "id"]
    if labels:
        bars = [_by_size(_sequence(labels[:MAX_LABELS], m)) for m in measures]
        bars = [{**b, "parts": _parts(b) if whole else None} for b in bars]
        parts_first = asks_parts and any(b["parts"] for b in bars)
        return {"type": "bar", "measures": bars, "timeline": None, "parts_first": parts_first}
    return None


def _parts(data: dict) -> dict | None:
    """
    Pie slices of a single-series bar chart, or None when its values make no whole: a measure
    that doesn't add up (an average, a price, a rate) or a value ≤ 0 has no share.
    Bars come largest first: past MAX_SLICES, the tail is folded into one "others" slice.
    """
    [series] = data["series"]
    values   = series["values"]
    words    = {w.lower() for w in _NAME_WORDS.findall(data["name"])}
    if "%" in data["name"] or words & _NOT_PARTS or not all(v is not None and v > 0 for v in values):
        return None

    if len(values) <= MAX_SLICES:
        return {"labels": data["labels"], "values": values, "others": None}
    kept = MAX_SLICES - 1
    return {
        "labels": data["labels"][:kept],
        "values": values[:kept],
        "others": {"count": len(values) - kept, "value": sum(values[kept:])},
    }


def _crossed(dims: list[Field]) -> tuple[Field, Field] | None:
    """
    Return (x, series) when the result crosses exactly two dimensions and one of them has
    2..MAX_GROUPS values, else None. Crossed means each (x, series) pair appears once and most
    x values appear under several series: a hierarchy like (country, city) is not a comparison,
    nor is a person's (first name, last name).
    """
    if len(dims) != 2:
        return None

    # Fewest values become the series. On a tie, keep a temporal dimension on the x axis, and of
    # two temporal ones keep the finer there (quarter × year → one series per year)
    def series_rank(f: Field) -> tuple:
        return len(set(f.values)), f.kind == "temporal", _UNITS.index(f.unit) if f.unit in _UNITS else len(_UNITS)

    series = min(dims, key=series_rank)
    x      = dims[0] if series is dims[1] else dims[1]
    if not 2 <= len(set(series.values)) <= MAX_GROUPS or len(set(x.values)) < 2:
        return None
    # Time compares a coarse unit across a finer one (years by month), never the reverse:
    # 10 years × 2 months is one timeline, not a line per month
    if _sortable([x, series]) and _UNITS.index(series.unit) > _UNITS.index(x.unit):
        return None

    pairs = list(zip(x.values, series.values))
    if len(set(pairs)) != len(pairs):
        return None
    groups: dict = {}
    for xv, sv in pairs:
        groups.setdefault(xv, set()).add(sv)
    shared = sum(1 for g in groups.values() if len(g) > 1)
    return (x, series) if shared * 2 >= len(groups) else None


# ── Chart data ────────────────────────────────────────────────────────────────


def _sequence(x: list[Field], measure: Field) -> dict:
    """One point per row, labelled by the x fields joined (e.g. "janv. 2022"), in time order."""
    points = list(zip(zip(*(f.values for f in x)), measure.values))
    if _sortable(x):
        points.sort(key=lambda p: _time_key(x, p[0]))
    return {
        "name":   measure.name,
        "labels": [_label(x, key) for key, _ in points],
        "series": [{"name": None, "values": [_number(v) for _, v in points]}],
    }


def _split(x: Field, series: Field, measure: Field) -> dict:
    """One series per value of the series field, aligned on the x values (gaps stay None)."""
    xs     = _distinct(x)
    groups = _distinct(series)
    values = {(xv, sv): mv for xv, sv, mv in zip(x.values, series.values, measure.values)}
    return {
        "name":   measure.name,
        "labels": [_format(x, xv) for xv in xs],
        "series": [
            {"name": _format(series, sv), "values": [_number(values.get((xv, sv))) for xv in xs]}
            for sv in groups
        ],
    }


def _by_size(data: dict) -> dict:
    """Bars largest first (summed over series): a ranking reads better than the query order."""
    def total(i: int) -> float:
        return sum(s["values"][i] or 0 for s in data["series"])

    order = sorted(range(len(data["labels"])), key=total, reverse=True)
    return {
        **data,
        "labels": [data["labels"][i] for i in order],
        "series": [{**s, "values": [s["values"][i] for i in order]} for s in data["series"]],
    }


def _distinct(field: Field) -> list:
    """Distinct values of a field: in time order when sortable, else in first-seen order."""
    values = list(dict.fromkeys(field.values))
    return sorted(values, key=lambda v: _time_key([field], (v,))) if _sortable([field]) else values


def _sortable(fields: list[Field]) -> bool:
    return all(f.unit in _UNITS for f in fields)


def _time_key(fields: list[Field], key: tuple) -> tuple:
    """Sort key of a time point: coarsest unit first (year, then month), None last."""
    parts = sorted(zip(fields, key), key=lambda p: _UNITS.index(p[0].unit))
    return tuple((v is None, _as_int(v) if _as_int(v) is not None else str(v)) for _, v in parts)


def _label(fields: list[Field], key: tuple) -> str:
    """Join the x values of a point: time from finest to coarsest ("janv. 2022"), else in column order."""
    parts = list(zip(fields, key))
    if _sortable(fields):
        parts.sort(key=lambda p: -_UNITS.index(p[0].unit))
    return " ".join(_format(f, v) for f, v in parts if v is not None) or "—"


def _format(field: Field, value) -> str:
    """Display text of a dimension value: month and quarter names, dates without a midnight time."""
    if value is None:
        return "—"
    if field.unit == "month":
        return MONTHS[_as_int(value) - 1]
    if field.unit == "quarter":
        return f"T{_as_int(value)}"
    return _MIDNIGHT.sub("", str(value))


def _number(value) -> float | int | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _as_int(value) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
