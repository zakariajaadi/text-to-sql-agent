# offline/semantic_gen/utils.py

"""
Helpers shared by the generators (tables.py, relationships.py, metrics.py) and offline/studio:
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


# ------------------------------------------------------------
# tables.yaml (written by tables.py and offline/studio)
# ------------------------------------------------------------

SEMANTICS_HEADER = """\
# Semantic layer, generated (make semantics) and reviewed manually.
# The generator only ADDS missing content, it never overwrites reviewed content.
#   description, columns : business meaning of the table and of each column
#   categorical_columns  : columns holding a closed set of labels (country, status...),
#                          whose values are read from the database at index time.
#                          Remove a column from the list to reject it: it won't come back
#   read_mode, key_columns, partition_column, version_column : Hive table layout
#   formats              : (manual) columns whose values are stored as text,
#                          {column: "date <pattern>" (e.g. date dd/MM/yyyy) or "decimal"}
#   exclude_columns      : (manual) columns the agent never sees (empty, technical, personal data):
#                          out of the documents, the sample rows and the LLM prompts
#   keep_columns         : (manual) columns the profile proposed to exclude, kept visible on purpose
#   keep_as_text         : (manual) columns the profile reads as dates or numbers, kept as text on purpose
#   data_as_of          : (manual) date the data stops (YYYY-MM-DD), reference of relative periods

"""

# Key order of a table block in the YAML (unknown keys last)
SEMANTICS_KEY_ORDER = [
    "description", "data_as_of", "read_mode", "key_columns", "partition_column", "version_column",
    "categorical_columns", "formats", "keep_as_text", "exclude_columns", "keep_columns", "columns",
]


def dump_table_semantics(semantics: dict) -> str:
    """Dump one YAML block per table, separated by a blank line for readability."""
    def ordered(entry: dict) -> dict:
        rank = {key: i for i, key in enumerate(SEMANTICS_KEY_ORDER)}
        return dict(sorted(entry.items(), key=lambda item: rank.get(item[0], len(rank))))

    return SEMANTICS_HEADER + "\n".join(
        yaml.dump({table: ordered(entry)}, Dumper=FlowListDumper, sort_keys=False, allow_unicode=True, width=1000)
        for table, entry in semantics.items()
    )
