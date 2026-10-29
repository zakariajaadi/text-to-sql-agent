# offline/semantic_gen/utils.py

"""
Helpers shared by the generators (tables.py, relationships.py, metrics.py) and offline/studio:
reading the database structure, rendering schemas for the LLM, dumping reviewed YAML files,
the rules a metric is checked against.
"""

import re
from pathlib import Path

import yaml
from langchain_community.utilities import SQLDatabase
from sqlalchemy import inspect as sa_inspect

from config import SEMANTIC_LAYER_DIR, cfg
from lib.column_pruning import get_column_types
from lib.metrics import metrics_path
from lib.semantics import load_fk_relations, load_semantics, render_table, scope_path
from lib.sql_tools import contains_expression, join_condition


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
# scope.yaml (written by offline/studio)
# ------------------------------------------------------------

SCOPE_HEADER = """\
# The tables of this database the agent and the pipeline see, chosen in the studio (Périmètre):
#   all: true   every table of the database, the ones it will gain too
#   tables      only these: a table of the database not listed is never read (not profiled, not
#               described, not indexed, a query of the agent reading it is blocked), and a table
#               the database gains stays out until it is added
# Read at startup by the agent and by each run of the pipeline. Without this file: every table of the
# database.

"""


def write_scope(db_name: str, tables: list[str] | None) -> None:
    """
    semantic_layer/<db>/scope.yaml: `all: true` when tables is None (every table, the ones to come too), else the
    tables sorted. Written at once (a temporary file renamed).
    """
    path = scope_path(db_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"all": True} if tables is None else {"tables": sorted(set(tables))}
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(SCOPE_HEADER + yaml.safe_dump(entry, sort_keys=False, allow_unicode=True))
    temporary.replace(path)


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
#   table_type           : (studio) the type of the table chosen by hand (events, entity, reference, bridge),
#                          over the one the studio detects from the profile and the relations
#   edited_description, edited_columns       : (studio) descriptions corrected by hand, kept when the
#                          table is described again (--fresh), unless --replace-edited
#   rewritten_description, rewritten_columns : descriptions the LLM wrote again over another text
#                          (--fresh), to read again; the studio drops them once read
#   unclear_columns      : columns whose meaning or unit the LLM could not tell, to state by hand
#                          (« à préciser » in the studio); a correction drops them

"""

# Key order of a table block in the YAML (unknown keys last)
SEMANTICS_KEY_ORDER = [
    "description", "data_as_of", "table_type", "read_mode", "key_columns", "partition_column", "version_column",
    "categorical_columns", "formats", "keep_as_text", "exclude_columns", "keep_columns", "columns",
    "edited_description", "edited_columns", "rewritten_description", "rewritten_columns", "unclear_columns",
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


# ------------------------------------------------------------
# relationships.yaml (written by relationships.py and offline/studio)
# ------------------------------------------------------------

FK_RELATIONS_HEADER = """\
# Declarative FK relations of this database, for engines that expose no FK metadata (Hive, Impala).
# Generated by LLM (make relations), then reviewed manually:
#   - the generator only ADDS relations, never modifies or removes existing ones
#   - to reject a generated relation, move it to `ignore` (otherwise it comes back)
#   - found: share of the values of `from` found in `to`; origin: profile, llm or manual
#   - fanout, links: a relation added on purpose to a column of `to` that is not a key (studio): the rows
#     of `to` each row of `from` joins on average, and the cardinality measured then
# Table and column names must match the engine (lowercase for Hive).

"""


def relations_path(db_name: str) -> Path:
    return SEMANTIC_LAYER_DIR / db_name / "relationships.yaml"


def load_relations_file(db_name: str) -> dict:
    """The whole file: `relations` (with their evidence) and `ignore`, both lists, empty when missing."""
    path  = relations_path(db_name)
    entry = (yaml.safe_load(path.read_text()) if path.exists() else None) or {}
    entry["relations"] = entry.get("relations") or []
    entry["ignore"]    = entry.get("ignore") or []
    return entry


def write_relations_file(db_name: str, entry: dict) -> None:
    """
    Written at once (a temporary file renamed): offline/studio reads it while a run goes on.
    The comments on top of the file are kept (what a reviewer wrote there), the standard header otherwise.
    """
    path = relations_path(db_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = FK_RELATIONS_HEADER
    if path.exists():
        lines = path.read_text().splitlines()
        end   = next((i for i, line in enumerate(lines) if line.strip() and not line.startswith("#")), len(lines))
        top   = lines[:end]
        if any(line.startswith("#") for line in top):
            header = "\n".join(top).rstrip() + "\n\n"
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(header + yaml.safe_dump(entry, sort_keys=False, default_flow_style=None,
                                                 allow_unicode=True, width=1000))
    temporary.replace(path)


# ------------------------------------------------------------
# metrics.yaml (written by metrics.py and offline/studio)
# ------------------------------------------------------------

METRICS_HEADER = """\
# Business metrics, generated (make metrics) and reviewed manually (lib/metrics.py).
# The generator only ADDS metrics, it never modifies or removes existing ones.
#   description    : what the metric measures, read by the LLM to pick it
#   synonyms       : other words for it in the questions, any language
#   expr           : aggregate SQL expression, every column written table.column
#   time_dimension : default date for periods and trends (optional)
#   origin         : (generator) "ai" on a metric it proposed; none on one written before the studio tracked it
#   edited         : (studio) the fields corrected by hand since, out of description, synonyms, expr, time_dimension
# To reject a generated metric, delete it and add its name to `ignore` (otherwise it comes back).
# Checked at startup: an unknown table or column stops the app.

"""

# Key order of a metric block in the YAML: what the agent reads, then where it comes from
METRICS_KEY_ORDER = ["description", "synonyms", "expr", "time_dimension", "dimensions", "origin", "edited"]

# Never proposed, whatever the name: two rows with the same amount are two amounts
DISTINCT_AMOUNT = re.compile(r"\b(SUM|AVG)\s*\(\s*DISTINCT\b", re.IGNORECASE)

# Never proposed either: a plain count of one column (COUNT(DISTINCT t.id), COUNT(*)) holds no business
# rule, the agent counts entities by itself, and a metric forces its table into every question it matches
PLAIN_COUNT = re.compile(r"^\s*COUNT\s*\(\s*(DISTINCT\s+)?(\*|[\w.`\"]+)\s*\)\s*$", re.IGNORECASE)

# The reasons of skip_reason, written in the log of a run and read back by offline/studio
KNOWN_NAME      = "already defined or rejected"
SAME_EXPRESSION = "same expression as an existing metric"
DISTINCT_SUM    = "SUM(DISTINCT ...) or AVG(DISTINCT ...) drops the equal amounts of different rows"
COUNT_ONLY      = "a plain count of one column holds no business rule: the agent counts entities by itself"


def skip_reason(name: str, expr: str, known: dict[str, dict], ignored: list[str], dialect: str) -> str | None:
    """
    Why a proposed metric is not added (None: it goes on to validation): its name is defined or
    rejected, its expression is the one of a known metric, it sums distinct values, or it only counts
    one column. The last three are checked on the expression, as the LLM proposes a rejected metric
    again under another name.
    """
    if name in known or name in ignored:
        return KNOWN_NAME
    # Same expression whatever the aliases: each one contains the other (a ratio only contains its parts)
    if any(contains_expression(expr, e.get("expr", ""), dialect) and contains_expression(e.get("expr", ""), expr, dialect)
           for e in known.values()):
        return SAME_EXPRESSION
    if DISTINCT_AMOUNT.search(expr):
        return DISTINCT_SUM
    if PLAIN_COUNT.match(expr):
        return COUNT_ONLY
    return None


def metric_from_clause(tables: list[str], fk_relations: list[tuple[str, str]], column_types: dict[str, str],
                       dialect: str) -> str:
    """
    The FROM and JOIN lines a metric is computed on: its first table, the others joined through the
    declared relationships, a join key stored as text cast as in the "-- Joins" lines (join_condition;
    column_types: {"table.column": type}). Raises ValueError when they can't all be joined.
    """
    def link(table: str, joined: list[str]) -> tuple[str, str] | None:
        """A relationship between the table and one already joined."""
        for source, target in fk_relations:
            ends = (source.split(".")[0], target.split(".")[0])
            if table in ends and (set(ends) - {table}) & set(joined):
                return source, target
        return None

    joined, lines, pending = [tables[0]], [f"FROM {tables[0]}"], list(tables[1:])
    while pending:
        table, relation = next(((t, link(t, joined)) for t in pending if link(t, joined)), (None, None))
        if table is None:
            raise ValueError(f"tables {', '.join(pending)} not joined to {', '.join(joined)} by a relationship")
        lines.append(f"JOIN {table} ON {join_condition(*relation, column_types, dialect)}")
        joined.append(table)
        pending.remove(table)
    return "\n".join(lines)


def dump_metrics(metrics: dict[str, dict], ignored: list[str]) -> str:
    """Dump one YAML block per metric, separated by a blank line, then the ignored names."""
    def ordered(entry: dict) -> dict:
        rank = {key: i for i, key in enumerate(METRICS_KEY_ORDER)}
        return dict(sorted(entry.items(), key=lambda item: rank.get(item[0], len(rank))))

    blocks = [
        yaml.dump({name: ordered(entry)}, Dumper=FlowListDumper, sort_keys=False, allow_unicode=True, width=1000)
        for name, entry in metrics.items()
    ]
    blocks.append(yaml.dump({"ignore": ignored}, Dumper=FlowListDumper, sort_keys=False, allow_unicode=True))
    return METRICS_HEADER + "\n".join(blocks)


def write_metrics_file(db_name: str, metrics: dict[str, dict], ignored: list[str]) -> None:
    """Written at once (a temporary file renamed): offline/studio reads it while a run goes on."""
    path = metrics_path(db_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(dump_metrics(metrics, ignored))
    temporary.replace(path)
