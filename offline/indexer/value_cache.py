# offline/indexer/value_cache.py

"""
The categorical values the last index read, with the version of the data of their table (data_version.py):
the next index keeps them for a table whose data has not changed since, instead of one GROUP BY per column
(make index; --full reads them all again).

One JSON file per database, next to its value index: <value_search.directory>/<db>.values.json
    max_values  value_search.max_values of the run: another one reads every value again
    tables      {table: {version, columns, values}}, only the tables whose version could be read
        version  "iceberg:<snapshot>" or "files:<hash>"
        columns  the categorical columns read, those without a value list included (none, or too many)
        values   {column: {value: row count}}, as the run read them

It holds values of the data, like the value index beside it: it stays on the machine of the index.
"""

import json
from pathlib import Path

from config import cfg


def value_cache_path(db_name: str) -> Path:
    """<value_search.directory>/<db>.values.json: the values the last index read, with their data version."""
    return Path(cfg.value_search.directory) / f"{db_name}.values.json"


def load_value_cache(db_name: str) -> dict[str, dict]:
    """{table: {version, columns, values}} of the last index; {} when there is none, or another max_values."""
    path = value_cache_path(db_name)
    if not path.exists():
        return {}
    try:
        cache = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return cache.get("tables", {}) if cache.get("max_values") == cfg.value_search.max_values else {}


def write_value_cache(db_name: str, tables: dict[str, dict]) -> None:
    """Write the cache of this index, replacing the previous one: a temporary file renamed, never read half-written."""
    path = value_cache_path(db_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps({"max_values": cfg.value_search.max_values, "tables": tables}, ensure_ascii=False))
    temporary.replace(path)


def kept_columns(cached: dict | None, version: str | None, columns: list[str]) -> list[str]:
    """The columns whose values the last index read on the same data: none without a version or a cache."""
    if not version or not cached or cached.get("version") != version:
        return []
    return [c for c in columns if c in cached.get("columns", [])]
