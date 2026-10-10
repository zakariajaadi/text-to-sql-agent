# offline/semantic_gen/utils.py

"""
Helpers shared by the generators (tables.py, relationships.py, metrics.py):
reading the database structure, rendering schemas for the LLM, dumping reviewed YAML files.
"""

import yaml
from langchain_community.utilities import SQLDatabase
from sqlalchemy import inspect as sa_inspect

from lib.semantics import load_fk_relations, load_semantics, render_table


def table_columns(db: SQLDatabase) -> dict[str, list[str]]:
    """Current columns of each table, in DDL order."""
    inspector = sa_inspect(db._engine)
    return {t: [c["name"] for c in inspector.get_columns(t)] for t in db.get_usable_table_names()}


def render_schemas(db: SQLDatabase, db_name: str, tables: list[str] | None = None) -> str:
    """Tables (default: all) as one string: rendered with semantics when documented, raw DDL otherwise."""
    semantics    = load_semantics(db_name)
    fk_relations = load_fk_relations(db_name)

    schemas = []
    for table in tables if tables is not None else db.get_usable_table_names():
        ddl = db.get_table_info(table_names=[table]).strip()
        schemas.append(render_table(table, ddl, semantics[table], fk_relations) if table in semantics else ddl)
    return "\n\n".join(schemas)


class FlowListDumper(yaml.SafeDumper):
    """Lists on one line ([a, b, c]): column lists, synonyms and ignore lists stay compact for review."""


FlowListDumper.add_representer(
    list, lambda dumper, data: dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
)
