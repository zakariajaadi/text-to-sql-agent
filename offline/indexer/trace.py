# offline/indexer/trace.py

"""
Trace of the last index of a database: what went into the document of each table. Written by
offline/indexer/schema_indexer.py once the index is written, read by the studio (offline/studio/index.py)
to tell which tables changed in the semantic layer since, without loading Chroma.

One JSON file per database, next to its value index: <value_search.directory>/<db>.index.json
    built_at     when the index was written (seconds since the epoch)
    builds       when the indexes before it were written, oldest first (kept from the previous trace)
    sample_rows  rows under each DDL of the documents (databases[].sample_rows)
    tables       {table: {semantics, relations, columns, chars, values}}
        semantics  its block of tables.yaml when the run started, None for a raw DDL
        relations  [[source, target], ...]: the relations of relationships.yaml involving the table
        columns    {column: type}: the columns its document shows (the hidden ones are not in it)
        chars      length of the document
        values     {column: {count, inline}}: the values indexed of each categorical column,
                   listed in the document (inline) or searched at query time only

The semantic layer and counts only, never a value of the data.
"""

import json
from pathlib import Path

from config import cfg


def trace_path(db_name: str) -> Path:
    """<value_search.directory>/<db>.index.json: the trace of the last index of a database."""
    return Path(cfg.value_search.directory) / f"{db_name}.index.json"


# The dates of the indexes before the last one kept in its trace: the studio marks them on the score of the evals
BUILDS_KEPT = 200


def write_trace(db_name: str, trace: dict) -> None:
    """
    Write the trace, replacing the previous one: a temporary file renamed, never read half-written. The dates of
    the previous indexes carry over (`builds`).
    """
    path = trace_path(db_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        previous = json.loads(path.read_text()) if path.exists() else None
    except json.JSONDecodeError:
        previous = None
    if previous and previous.get("built_at"):
        trace = {**trace, "builds": [*previous.get("builds", []), previous["built_at"]][-BUILDS_KEPT:]}
    temporary = path.with_name(f"{path.name}.tmp")
    # default=str: a data_as_of read by YAML as a date
    temporary.write_text(json.dumps(trace, ensure_ascii=False, indent=1, default=str))
    temporary.replace(path)


def read_trace(db_name: str) -> dict | None:
    """The trace of the last index, or None when the database was never indexed since traces exist."""
    path = trace_path(db_name)
    return json.loads(path.read_text()) if path.exists() else None
