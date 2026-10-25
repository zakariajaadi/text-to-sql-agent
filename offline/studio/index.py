# offline/studio/index.py

"""
The Index screen of the studio (pure functions, unit-tested): what the agent reads of each table, its
document as the last index wrote it, against the semantic layer files today.

The index (make index, offline/indexer/schema_indexer.py) turns the files of steps 1 to 4 into one document
per table and reads the values of the categorical columns from the database. The agent reads the documents,
not the files: a correction reaches it once the database is indexed again. The trace of the last index
(offline/indexer/trace.py) holds what went into each document; compared with the files today and the columns
of the database, it tells which tables changed since and how. The metrics are checked against the documents
as the agent does at startup (lib/metrics.load_metrics). Counts and shapes, never a value.
"""

from datetime import datetime

from lib.metrics import validate_metric
from lib.semantics import excluded_columns, render_table

# Status of a table in the index
FRESH, CHANGED, MISSING, UNKNOWN = "fresh", "changed", "missing", "unknown"

# Keys of a table block that write its "-- Read mode" line (lib/semantics._render_read_mode)
READ_MODE_KEYS = ("read_mode", "partition_column", "key_columns", "version_column")

# What the text of a document shows in place of the data: ‹...› marks it, the page draws it apart
MASK = "‹{}›"


def table_relations(table: str, relations) -> list[list[str]]:
    """The relations involving a table, as [source, target], sorted."""
    return sorted([source, target] for source, target in relations
                  if table in (source.split(".")[0], target.split(".")[0]))


def document_inputs(entry: dict | None, relations: list[list[str]]) -> dict:
    """
    What the document of a table is made of in the semantic layer (lib/semantics.render_table): its block of
    tables.yaml, None for a raw DDL, and its relations. Only what changes the document: the exceptions to the
    profile (keep_as_text, keep_columns) don't, their effect is in formats and exclude_columns.
    """
    if entry is None:
        return {"described": False}
    hidden = excluded_columns(entry)
    return {
        "described":   True,
        "description": entry.get("description") or "",
        "data_as_of":  str(entry["data_as_of"]) if entry.get("data_as_of") else "",
        "columns":     {c: d for c, d in (entry.get("columns") or {}).items() if c not in hidden},
        "formats":     {c: f for c, f in (entry.get("formats") or {}).items() if c not in hidden},
        "hidden":      sorted(hidden),
        "categorical": sorted(c for c in entry.get("categorical_columns") or [] if c not in hidden),
        "read_mode":   {key: entry[key] for key in READ_MODE_KEYS if entry.get(key) is not None},
        "relations":   sorted(list(relation) for relation in relations),
    }


def _added_removed(before, after) -> tuple[list, list]:
    return [x for x in after if x not in before], [x for x in before if x not in after]


def changes(indexed: dict, entry: dict | None, relations: list[list[str]],
            db_columns: list[str] | None) -> list[dict]:
    """
    What changed in a document since the index: its trace (`indexed`) against the files today (`entry`,
    `relations`) and the columns of the database (None when it can't be read). [] when nothing did.
    Each change: {"kind", ...}, the columns or relations concerned.
    """
    before = document_inputs(indexed["semantics"], indexed["relations"])
    after  = document_inputs(entry, relations)
    found  = []
    if before["described"] != after["described"]:
        found.append({"kind": "described" if after["described"] else "undescribed"})
    elif after["described"]:
        for key in ("description", "data_as_of", "read_mode"):
            if before[key] != after[key]:
                found.append({"kind": key})
        hidden, shown = _added_removed(before["hidden"], after["hidden"])
        moved = {*hidden, *shown}
        for key in ("columns", "formats"):
            edited = sorted(c for c in before[key].keys() | after[key].keys()
                            if c not in moved and before[key].get(c) != after[key].get(c))
            if edited:
                found.append({"kind": key, "columns": edited})
        if hidden:
            found.append({"kind": "hidden", "columns": hidden})
        if shown:
            found.append({"kind": "shown", "columns": shown})
        added, removed = _added_removed(before["categorical"], after["categorical"])
        if added or removed:
            found.append({"kind": "categorical", "added": added, "removed": removed})
        added, removed = _added_removed(before["relations"], after["relations"])
        if added or removed:
            found.append({"kind": "relations", "added": added, "removed": removed})

    # The database itself: a column the document lists is gone, or a new one the agent doesn't see
    if db_columns is not None:
        hidden  = set(before.get("hidden", [])) | set(after.get("hidden", []))
        added   = [c for c in db_columns if c not in indexed["columns"] and c not in hidden]
        removed = [c for c in indexed["columns"] if c not in db_columns]
        if added or removed:
            found.append({"kind": "db_columns", "added": added, "removed": removed})
    return found


def dropped_columns(indexed: dict) -> list[str]:
    """Categorical columns of a document without any value indexed: empty, or more than value_search.max_values."""
    inputs = document_inputs(indexed["semantics"], indexed["relations"])
    return [c for c in inputs.get("categorical", []) if c not in indexed["values"]]


def table_status(table: str, trace: dict | None, entry: dict | None, relations,
                 db_columns: list[str] | None) -> tuple[str, list[dict]]:
    """The status of a table in the index, and what changed in its document since."""
    if trace is None:
        return UNKNOWN, []
    if table not in trace["tables"]:
        return MISSING, []
    found = changes(trace["tables"][table], entry, table_relations(table, relations), db_columns)
    return (CHANGED if found else FRESH), found


def document_text(table: str, trace: dict, dialect: str) -> str | None:
    """
    The text of the document of a table as the last index wrote it, rendered by the function of the agent
    (lib/semantics.render_table) from the trace, the data masked: the values of a categorical column listed in
    the document become ‹24 valeurs›, its sample rows ‹3 lignes d'exemple›. The CREATE TABLE is rebuilt from the
    columns and types of the trace (the database is not read): its layout may differ from the DDL of the
    database by a space. None when the table is not in the trace.
    """
    indexed = (trace or {}).get("tables", {}).get(table)
    if indexed is None:
        return None
    body = ",\n".join(f"\t`{column}` {column_type}" for column, column_type in indexed["columns"].items())
    ddl  = f"CREATE TABLE `{table}` (\n{body}\n)"
    if rows := trace.get("sample_rows"):
        ddl += f"\n\n/*\n{MASK.format(f'{rows} ligne{"s" if rows > 1 else ""} d\'exemple')}\n*/"
    entry = indexed["semantics"]
    if not entry or not entry.get("description"):
        return ddl
    inline = {column: [MASK.format(f"{counts['count']} valeur{'s' if counts['count'] > 1 else ''}")]
              for column, counts in indexed["values"].items() if counts.get("inline")}
    types  = {f"{t}.{column}": column_type for t, other in trace["tables"].items() for column, column_type in other["columns"].items()}
    text   = render_table(table, ddl, entry, [tuple(r) for r in indexed["relations"]], inline, types, dialect)
    return text.replace(f"'{MASK[0]}", MASK[0]).replace(f"{MASK[-1]}'", MASK[-1])   # a mask is no quoted value


def metric_errors(metrics: dict[str, dict], trace: dict, dialect: str) -> list[dict]:
    """
    The metrics the agent would refuse at startup: it checks metrics.yaml against the indexed documents
    (lib/metrics.load_metrics), the columns they show. [{"name", "detail"}].
    """
    columns = {table: list(indexed["columns"]) for table, indexed in trace["tables"].items()}
    errors  = []
    for name, entry in metrics.items():
        try:
            validate_metric(name, entry, dialect, columns)
        except ValueError as e:
            errors.append({"name": name, "detail": str(e).split(" — ", 1)[-1]})
    return errors


def eval_run(results: dict | None, modified: float | None) -> dict | None:
    """
    The last eval of the database (offline/eval/results/<db>.json): when, its questions, how many
    returned the expected rows among the ones compared on their rows. None when it never ran.
    """
    if results is None:
        return None
    run_at = (results.get("meta") or {}).get("run_at")
    scores = [r.get("exec_exact") for r in results.get("results") or []]
    scored = [s for s in scores if s is not None]
    return {
        "at":        datetime.fromisoformat(run_at).timestamp() if run_at else modified,
        "questions": len(scores),
        "scored":    len(scored),
        "exact":     sum(1 for s in scored if s),
    }


def index_view(tables: list[str], db_columns: dict[str, list[str]], semantics: dict, relations,
               trace: dict | None, metrics: dict[str, dict], dialect: str, evaluation: dict | None) -> dict:
    """
    The Index screen: each table of the database with its status, the tables the index holds that the
    database no longer has (dropped at the next index), what the agent receives (documents and the joins
    between them, the columns they show or hide, values, sample rows, metrics) and the last eval. Without
    a trace, the statuses are unknown.
    """
    rows = []
    for table in tables:
        status, found = table_status(table, trace, semantics.get(table), relations, db_columns.get(table))
        rows.append({"name": table, "status": status, "changes": found})
    view = {"tables": rows, "trace": trace is not None, "eval": evaluation,
            "counts": {status: sum(r["status"] == status for r in rows) for status in (FRESH, CHANGED, MISSING, UNKNOWN)}}
    if trace is None:
        return view

    indexed = trace["tables"]
    values  = [(t, c, v) for t, entry in indexed.items() for c, v in entry["values"].items()]
    view.update({
        "built_at":    trace["built_at"],
        "sample_rows": trace["sample_rows"],
        "gone":        sorted(set(indexed) - set(tables)),
        "documents":   len(indexed),
        "joins":       len({tuple(relation) for entry in indexed.values() for relation in entry["relations"]}),
        "shown":       sum(len(entry["columns"]) for entry in indexed.values()),
        "hidden":      sorted(f"{t}.{c}" for t, entry in indexed.items() for c in excluded_columns(entry["semantics"])),
        "raw":         sorted(t for t, entry in indexed.items() if entry["semantics"] is None),
        "inline":      {"columns": sum(v["inline"] for _, _, v in values),
                        "values":  sum(v["count"] for _, _, v in values if v["inline"])},
        "searched":    {"columns": sum(not v["inline"] for _, _, v in values),
                        "values":  sum(v["count"] for _, _, v in values if not v["inline"])},
        "dropped":     [f"{t}.{c}" for t, entry in indexed.items() for c in dropped_columns(entry)],
        "metrics":     {"total": len(metrics), "errors": metric_errors(metrics, trace, dialect)},
    })
    return view


def document_view(table: str, trace: dict | None, entry: dict | None, relations,
                  db_columns: list[dict] | None) -> dict:
    """
    The document of a table as the agent reads it: its size, its header (description, date of the data,
    joins), each column with its type (of the database, as the other screens show it), its format, its values
    (listed in the document, searched, none) or hidden, and what changed since the index: a column hidden
    since leaves the document at the next index, one shown since enters it. db_columns: [{name, type}] of the
    database, None when unreachable.
    """
    names  = [c["name"] for c in db_columns] if db_columns is not None else None
    status, found = table_status(table, trace, entry, relations, names)
    view = {"name": table, "status": status, "changes": found}
    if status in (UNKNOWN, MISSING):
        return view

    indexed = trace["tables"][table]
    inputs  = document_inputs(indexed["semantics"], indexed["relations"])
    dropped = set(dropped_columns(indexed))
    hidden  = set(inputs.get("hidden", []))
    now     = excluded_columns(entry)   # hidden today
    types   = {c["name"]: c["type"] for c in db_columns or []}
    order   = list(dict.fromkeys([*(names or []), *indexed["columns"]]))

    columns = []
    for column in order:
        line = {"column": column, "type": types.get(column) or indexed["columns"].get(column)}
        if column in indexed["columns"]:
            line["format"]  = inputs.get("formats", {}).get(column)
            line["values"]  = indexed["values"].get(column) or ({"dropped": True} if column in dropped else None)
            line["gone"]    = names is not None and column not in names
            line["leaving"] = column in now
        else:
            # In the database, not in the document: hidden then (and maybe shown since), or added since the index
            line["hidden"]   = column in hidden
            line["entering"] = column in hidden and column not in now
            line["added"]    = column not in hidden
        columns.append(line)

    view.update({
        "chars":       indexed["chars"],
        "described":   inputs["described"],
        "data_as_of":  inputs.get("data_as_of") or None,
        "joins":       len(indexed["relations"]) if inputs["described"] else 0,
        "sample_rows": trace["sample_rows"],
        "built_at":    trace["built_at"],
        "columns":     columns,
    })
    return view
