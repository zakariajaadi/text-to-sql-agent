# src/lib/database.py

"""
Database connections: the langchain SQLAlchemy wrapper of a configured database.
Used by: resources.py (the registry of the app and the generators), offline/studio (built on demand,
without the models, the Chroma index or the cache resources.py builds at import).
"""

from langchain_community.utilities import SQLDatabase
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy import inspect as sa_inspect

from config import DatabaseConfig
from lib.db_errors import is_unreachable
from lib.semantics import load_scope


def build_database(db_cfg: DatabaseConfig) -> SQLDatabase:
    """
    Build a langchain SQLAlchemy wrapper from the URI.
    The agent, the indexer and the generators only see the tables of the scope (lib/semantics.load_scope):
    a table of the scope the database no longer has is left out, with a warning. sample_rows: rows
    under each DDL of get_table_info.
    """
    scope = load_scope(db_cfg.name)
    try:
        engine = create_engine(db_cfg.uri, **db_cfg.engine_args())
        tables = {}
        if scope is not None:
            base = set(sa_inspect(engine).get_table_names())
            if missing := sorted(set(scope) - base):
                logger.warning(f"[{db_cfg.name}] Tables of the scope not in the database, left out: {', '.join(missing)}")
            kept = [table for table in scope if table in base]
            # An empty include_tables means every table: a scope with none of them ignores them all
            tables = {"include_tables": kept} if kept else {"ignore_tables": sorted(base)}
        db = SQLDatabase(
            engine,
            **tables,
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
