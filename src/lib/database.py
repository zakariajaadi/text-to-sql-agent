# src/lib/database.py

"""
Database connections: the langchain SQLAlchemy wrapper of a configured database.
Used by: resources.py (the registry of the app and the generators), offline/studio (built on demand,
without the models, the Chroma index or the cache resources.py builds at import).
"""

from langchain_community.utilities import SQLDatabase

from config import DatabaseConfig
from lib.db_errors import is_unreachable


def build_database(db_cfg: DatabaseConfig) -> SQLDatabase:
    """
    Build a langchain SQLAlchemy wrapper from the URI.
    With include_tables, the agent, the indexer and the generators only see those tables
    (an unknown name stops the startup). sample_rows: rows under each DDL of get_table_info.
    """
    try:
        db = SQLDatabase.from_uri(
            db_cfg.uri,
            engine_args=db_cfg.engine_args() or None,
            include_tables=list(db_cfg.include_tables) or None,
            sample_rows_in_table_info=db_cfg.sample_rows,
            lazy_table_reflection=True,  # don't reflect the whole metastore at startup
        )
    except Exception as e:
        if is_unreachable(e):
            raise RuntimeError(
                f"The {db_cfg.dialect} database {db_cfg.name} is unreachable at startup ({type(e).__name__}: "
                f"{str(e)[:200]}). Check the network or VPN (in Docker: restart Docker Desktop), then start again."
            ) from e
        raise
    if db_cfg.dialect == "impala":
        quote_every_identifier(db._engine.dialect)
    return db


def quote_every_identifier(dialect) -> None:
    """
    Backtick every identifier SQLAlchemy writes for this dialect (sample rows, categorical values).
    impyla's reserved word list stops at Impala 2: a column named like an Impala 4 reserved word
    (position, ...) breaks the query unquoted. The agent's own queries are quoted by normalize_sql.
    """
    preparer = dialect.identifier_preparer
    preparer._requires_quotes = lambda value: True
    preparer._strings.clear()   # identifiers already formatted, cached unquoted
