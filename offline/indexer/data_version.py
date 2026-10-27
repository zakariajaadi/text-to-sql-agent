# offline/indexer/data_version.py

"""
The version of the data of a table, read from the metastore without a scan of the data: the index keeps the
categorical values it read last when the data has not changed since (offline/indexer/value_cache.py), the photo
of the database it gives staying the same.

    - Impala, an Iceberg table: its current snapshot (current-snapshot-id), which every write changes
    - Impala, another table on files (Parquet, ORC, text): its files and their sizes (SHOW FILES). A file on
      HDFS is never changed in place: a load writes new files, under new names
    - a view (no data of its own), a table of a storage handler other than Iceberg (Kudu and HBase change rows
      in place, JDBC reads elsewhere), Hive (its statistics are not kept up to date by a job writing files),
      SQLite and the others: None, the values are read again, as expected (no warning)
    - the least doubt (Impala did not answer): None too, with a warning

Read from what Impala itself serves: a load its catalog does not see yet is not seen by the agent either.
Used by: offline/indexer/schema_indexer.py (fetch_values)
"""

import hashlib

from langchain_community.utilities import SQLDatabase
from loguru import logger
from sqlalchemy import text

IMPALA = ("impala", "impala4")   # impala4: the dialect of impyla for Impala >= 4 (config.py)


def data_version(db: SQLDatabase, table: str) -> str | None:
    """
    "iceberg:<snapshot>" or "files:<hash of the files and their sizes>" of an Impala table; None when the
    engine or the table can't tell (the values are then read again), with a warning when Impala did not answer.
    """
    try:
        if db._engine.dialect.name not in IMPALA:
            return None
        quote = db._engine.dialect.identifier_preparer.quote
        with db._engine.connect() as conn:
            described  = conn.execute(text(f"DESCRIBE FORMATTED {quote(table)}")).fetchall()
            parameters = table_parameters(described)
            if parameters.get("table_type", "").upper() == "ICEBERG":
                snapshot = parameters.get("current-snapshot-id")
                return f"iceberg:{snapshot}" if snapshot else None
            # A view, or rows kept by a storage handler (Kudu, HBase, JDBC...): no files of its own to list
            if table_type(described) == "VIRTUAL_VIEW" or parameters.get("storage_handler"):
                return None
            return files_version(conn.execute(text(f"SHOW FILES IN {quote(table)}")).fetchall())
    except Exception as e:
        logger.warning(f"Data version of {table} not read, its values are read again — {str(e).splitlines()[0][:200]}")
        return None


def table_parameters(rows) -> dict[str, str]:
    """
    The table parameters of DESCRIBE FORMATTED: the rows under "Table Parameters:", the key in the second
    column, the value in the third, up to the next section ("# Storage Information").
    """
    parameters, inside = {}, False
    for row in rows:
        name = str(row[0] or "").strip()
        if name.startswith("Table Parameters"):
            inside = True
        elif inside and name:
            break
        elif inside and (key := str(row[1] or "").strip()):
            parameters[key] = str(row[2] or "").strip()
    return parameters


def table_type(rows) -> str:
    """The "Table Type:" of DESCRIBE FORMATTED (MANAGED_TABLE, EXTERNAL_TABLE, VIRTUAL_VIEW), "" when not said."""
    return next((str(row[1] or "").strip().upper() for row in rows if str(row[0] or "").strip() == "Table Type:"), "")


def files_version(rows) -> str:
    """'files:' and a hash of the rows of SHOW FILES (path, size, partition...), in any order."""
    listed = sorted("\t".join(str(field) for field in row) for row in rows)
    return "files:" + hashlib.sha256("\n".join(listed).encode()).hexdigest()[:32]
