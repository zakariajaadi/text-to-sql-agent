"""
Result card — props of the ResultCard custom element (public/elements/ResultCard.jsx)
Used by: ui/streaming.py (attached to the final answer message)

The KPI tiles or chart come from ui/charts.py, picked by rules on the result shape and the question.
The lineage of the Calcul tab comes from lib/query_lineage.py, read by ui/streaming.py.
"""

import sqlparse

from ui.charts import pick_visualization


def build_result_card_props(
    columns:   list[str],
    rows:      list[list],
    sql:       str,
    truncated: bool,
    tables:    list[str],
    lineage:   dict | None,
    retries:   int,
    db_name:   str,
    question:  str,
) -> dict:
    """Assemble the JSON props of the ResultCard element."""
    columns = [_display_name(c) for c in columns]
    viz     = pick_visualization(columns, rows, question=question, truncated=truncated)

    return {
        "columns":   columns,
        "rows":      rows,
        "rowCount":  len(rows),
        "truncated": truncated,
        "tables":    tables,
        "lineage":   lineage,
        "sql":       sqlparse.format(sql, reindent=True, keyword_case="upper"),
        "retries":   retries,
        "dbName":    db_name,
        "chart":     viz["chart"],
        "kpis":      viz["kpis"],
    }


# ── Private helpers ───────────────────────────────────────────────────────────


def _display_name(column: str) -> str:
    """Hive prefixes result columns with their table ("t.col"): keep the column part."""
    return column.rsplit(".", 1)[-1]
