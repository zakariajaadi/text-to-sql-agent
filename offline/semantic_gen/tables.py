# offline/semantic_gen/tables.py

"""
Offline LLM generator of the table semantics: semantic_layer/<db>/tables.yaml
(table and column descriptions, categorical columns).

Additive: it only fills what is missing and never overwrites reviewed content. It reads
the database directly (DDL + existing semantics), never Chroma, so it can run before the
first index. The values of the categorical columns are not generated here: they are data,
read from the database at index time (offline/indexer/categorical_values.py).

Pipeline order: semantics → relations → metrics → (review) → index   (see offline/cli.py)
"""

from concurrent.futures import ThreadPoolExecutor

import yaml
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from lib.semantics import load_semantics, tables_path
from offline.semantic_gen.prompts import TABLE_SEMANTICS_PROMPT
from offline.semantic_gen.schemas import TableSemantics
from offline.semantic_gen.utils import FlowListDumper, table_columns
from resources import db_registry, fast_model

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

SEMANTICS_HEADER = """\
# Semantic layer, generated (make semantics) and reviewed manually.
# The generator only ADDS missing content, it never overwrites reviewed content.
#   description, columns : business meaning of the table and of each column
#   categorical_columns  : columns holding a closed set of labels (country, status...),
#                          whose values are read from the database at index time.
#                          Remove a column from the list to reject it: it won't come back
#   read_mode, key_columns, partition_column, version_column : Hive table layout

"""

# Key order of a table block in the YAML (unknown keys last)
SEMANTICS_KEY_ORDER = [
    "description", "read_mode", "key_columns", "partition_column", "version_column",
    "categorical_columns", "columns",
]

# ------------------------------------------------------------
# Chain
# ------------------------------------------------------------

# One table DDL → its table and column descriptions
semantics_chain = (
    ChatPromptTemplate.from_messages([("human", TABLE_SEMANTICS_PROMPT)])
    | fast_model.with_structured_output(TableSemantics)
)

# ------------------------------------------------------------
# Generator
# ------------------------------------------------------------

def generate_table_semantics(db_name: str):
    """
    Generates a semantic layer via LLM into semantic_layer/<db>/tables.yaml: missing table
    and column descriptions, and which columns are categorical.

    Compares current DB tables and columns against `semantic_layer/<db>/tables.yaml`.
    Only missing tables, tables with undocumented columns, or tables without
    `categorical_columns` are routed to the LLM.

    Additive only:
    existing descriptions are never overwritten, and a reviewed `categorical_columns`
    list only gains columns that are new to the table.
    """
    db         = db_registry[db_name]
    existing   = load_semantics(db_name)
    columns    = table_columns(db)
    all_tables = ", ".join(columns)

    # 1. Only tables with something missing go to the LLM
    todo = [
        t for t in columns
        if t not in existing
        or "categorical_columns" not in existing[t]
        or set(columns[t]) - set(existing[t].get("columns") or {})
    ]

    def generate(table: str) -> tuple[str, TableSemantics]:
        ddl = db.get_table_info(table_names=[table]).strip()
        logger.info(f"[{db_name}] Describing: {table}")
        return table, semantics_chain.invoke({
            "db_name":    db_name,
            "all_tables": all_tables,
            "ddl":        ddl,
        })

    # 2. Generate descriptions in parallel
    # (max_workers stays under the SQLAlchemy pool size: 5 + 10 overflow)
    with ThreadPoolExecutor(max_workers=8) as executor:
        generated = dict(executor.map(generate, todo))

    # 3. Merge — fill only what is missing
    added = 0
    for table, result in generated.items():
        entry = existing.setdefault(table, {})
        entry.setdefault("description", result.description)
        entry["columns"] = entry.get("columns") or {}

        # Categorical columns, in DDL order (columns the LLM invented are dropped).
        # A reviewed list only gains the columns new to the table: rejections stay rejected
        flagged     = {c.name for c in result.columns if c.categorical}
        new_columns = [c for c in columns[table] if c not in entry["columns"]]
        if "categorical_columns" not in entry:
            entry["categorical_columns"] = [c for c in columns[table] if c in flagged]
        else:
            entry["categorical_columns"] += [c for c in new_columns if c in flagged]

        described = {c.name: c.description for c in result.columns}
        for column in new_columns:
            if column in described:
                entry["columns"][column] = described[column]
                added += 1

    # Flag descriptions of columns that no longer exist
    for table, entry in existing.items():
        for column in entry.get("columns") or {}:
            if column not in columns.get(table, []):
                logger.warning(f"[{db_name}] Stale description: {table}.{column} no longer exists")

    path = tables_path(db_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump_table_semantics(existing))

    logger.info(f"[{db_name}] {len(todo)} table(s) described, {added} column description(s) added — review them.")


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def _dump_table_semantics(semantics: dict) -> str:
    """Dump one YAML block per table, separated by a blank line for readability."""
    def ordered(entry: dict) -> dict:
        rank = {key: i for i, key in enumerate(SEMANTICS_KEY_ORDER)}
        return dict(sorted(entry.items(), key=lambda item: rank.get(item[0], len(rank))))

    return SEMANTICS_HEADER + "\n".join(
        yaml.dump({table: ordered(entry)}, Dumper=FlowListDumper, sort_keys=False, allow_unicode=True, width=1000)
        for table, entry in semantics.items()
    )
