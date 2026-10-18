# offline/semantic_gen/utils.py

"""
Helpers shared by the generators (tables.py, relationships.py, metrics.py):
reading the database structure, rendering schemas for the LLM, dumping reviewed YAML files.
"""

import yaml
from langchain_community.utilities import SQLDatabase
from sqlalchemy import inspect as sa_inspect

from config import cfg
from lib.column_pruning import get_column_types
from lib.semantics import load_fk_relations, load_semantics, render_table


def table_columns(db: SQLDatabase) -> dict[str, list[str]]:
    """Current columns of each table, in DDL order."""
    inspector = sa_inspect(db._engine)
    return {t: [c["name"] for c in inspector.get_columns(t)] for t in db.get_usable_table_names()}


def render_schemas(db: SQLDatabase, db_name: str, tables: list[str] | None = None) -> str:
    """
    Tables (default: all) as one string: rendered with semantics when documented, raw DDL otherwise.
    Join keys stored as text are cast in the "-- Joins" lines when both tables are rendered.
    """
    semantics    = load_semantics(db_name)
    fk_relations = load_fk_relations(db_name)
    dialect      = cfg.get_db(db_name).dialect

    ddls = {
        table: db.get_table_info(table_names=[table]).strip()
        for table in (tables if tables is not None else db.get_usable_table_names())
    }
    column_types = {
        f"{table}.{column}": column_type
        for table, ddl in ddls.items()
        for column, column_type in get_column_types(ddl).items()
    }
    return "\n\n".join(
        render_table(table, ddl, semantics[table], fk_relations, column_types=column_types, dialect=dialect)
        if table in semantics else ddl
        for table, ddl in ddls.items()
    )


class FlowListDumper(yaml.SafeDumper):
    """Lists on one line ([a, b, c]): column lists, synonyms and ignore lists stay compact for review."""


FlowListDumper.add_representer(
    list, lambda dumper, data: dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
)
