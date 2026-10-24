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

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from lib.column_pruning import get_column_names, prune_ddl
from config import cfg
from lib.semantics import excluded_columns, load_profile, load_semantics, manual_semantics, tables_path
from offline.semantic_gen.profiling import apply_profile, row_facts
from offline.semantic_gen.prompts import TABLE_SEMANTICS_PROMPT
from offline.semantic_gen.schemas import TableSemantics
from offline.semantic_gen.utils import dump_table_semantics, table_columns
from resources import db_registry, fast_model

# Former name, imported by the patch scripts of semantic_layer/<db>/
_dump_table_semantics = dump_table_semantics

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

REWRITTEN_KEYS = ("rewritten_description", "rewritten_columns")
EDITED_KEYS    = ("edited_description", "edited_columns")


def kept_on_fresh(entry: dict, replace_edited: bool = False) -> dict:
    """
    What a table block keeps when it is described again (--fresh): its manual keys (formats,
    exclude_columns, data_as_of, read mode...) and the descriptions corrected by hand in the studio
    (edited_description, edited_columns), unless they are replaced too.
    """
    kept = {key: value for key, value in manual_semantics(entry).items() if key not in REWRITTEN_KEYS}
    if replace_edited:
        return {key: value for key, value in kept.items() if key not in EDITED_KEYS}
    if entry.get("edited_description") and entry.get("description"):
        kept["description"] = entry["description"]
    edited = set(entry.get("edited_columns") or [])
    if columns := {c: text for c, text in (entry.get("columns") or {}).items() if c in edited and text}:
        kept["columns"] = columns
    return kept


def mark_rewritten(before: dict, entry: dict, order: list[str]) -> int:
    """
    The descriptions the LLM wrote again over another text, marked in the block (rewritten_*) for a
    human to read again; the columns put back in the order of the table. Returns how many.
    """
    entry["columns"] = {c: entry["columns"][c] for c in [*order, *entry.get("columns", {})] if c in entry.get("columns", {})}
    old = before.get("columns") or {}
    columns = [c for c, text in entry["columns"].items() if old.get(c) and old[c] != text]
    description = bool(before.get("description")) and before["description"] != entry.get("description")
    for key, value in (("rewritten_description", description), ("rewritten_columns", columns)):
        if value:
            entry[key] = value
    return len(columns) + description


def generate_table_semantics(db_name: str, fresh: bool = False, tables: list[str] | None = None,
                             replace_edited: bool = False):
    """
    Generates a semantic layer via LLM into semantic_layer/<db>/tables.yaml: missing table
    and column descriptions, and which columns are categorical, written in the database's
    `semantic_language` (the language users ask in).

    Compares current DB tables and columns against `semantic_layer/<db>/tables.yaml`.
    Only missing tables, tables with undocumented columns, or tables without
    `categorical_columns` are routed to the LLM.

    Additive only:
    existing descriptions are never overwritten, and a reviewed `categorical_columns`
    list only gains columns that are new to the table.
    fresh: drop the generated content first (descriptions, categorical columns) and generate
    it again, keeping the manual keys (formats, exclude_columns, data_as_of, read mode...) and the
    descriptions corrected by hand in the studio (all replaced with replace_edited). The texts the
    LLM changes are marked rewritten, for a human to read again.
    tables: only these tables (the others left as they are); ValueError on an unknown one.
    """
    db         = db_registry[db_name]
    existing   = load_semantics(db_name)
    db_columns = table_columns(db)
    if tables and (unknown := sorted(set(tables) - set(db_columns))):
        raise ValueError(f"Unknown or excluded table(s): {', '.join(unknown)}")
    before = {}
    if fresh:
        before   = {table: entry for table, entry in existing.items() if not tables or table in tables}
        existing = {table: kept_on_fresh(entry, replace_edited) if table in before else entry
                    for table, entry in existing.items()}
    language   = cfg.get_db(db_name).semantic_language
    profile    = load_profile(db_name)
    personal   = cfg.get_db(db_name).exclude_personal_data

    # Formats and exclusions measured by `make profile`, for a table that has none yet (`make profile`
    # writes them already; again here for a profile made before it did): excluded columns are then
    # neither sent nor described
    for change in apply_profile(existing, profile, personal, [t for t in profile if t in db_columns]):
        logger.info(f"[{db_name}] Applied from the profile — {change}")
    # Excluded columns (exclude_columns, set beforehand) are neither sent to the LLM nor described
    columns    = {
        table: [c for c in table_cols if c not in excluded_columns(existing.get(table))]
        for table, table_cols in db_columns.items()
    }
    all_tables = ", ".join(columns)

    # 1. Only tables with something missing go to the LLM
    todo = [
        t for t in columns
        if (not tables or t in tables) and (
            t not in existing
            or "categorical_columns" not in existing[t]
            or set(columns[t]) - set(existing[t].get("columns") or {})
        )
    ]

    def generate(table: str) -> tuple[str, TableSemantics]:
        ddl = db.get_table_info(table_names=[table]).strip()
        excluded = excluded_columns(existing.get(table))
        if excluded:
            ddl = prune_ddl(ddl, set(get_column_names(ddl)) - excluded)
        logger.info(f"[{db_name}] Describing: {table}")
        return table, semantics_chain.invoke({
            "db_name":    db_name,
            "all_tables": all_tables,
            "ddl":        ddl,
            "language":   language,
            "facts":      row_facts(profile.get(table)),
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

    # Described again: what the LLM changed is to read again, what was corrected by hand is kept
    for table in generated.keys() & before.keys():
        rewritten = mark_rewritten(before[table], existing[table], db_columns.get(table, []))
        kept = 0 if replace_edited else (bool(before[table].get("edited_description"))
                                         + len(before[table].get("edited_columns") or []))
        logger.info(f"[{db_name}] {table}: {rewritten} description(s) rewritten, to read again"
                    + (f"; {kept} corrected by hand kept" if kept else ""))

    # Flag descriptions of columns that no longer exist
    for table, entry in existing.items():
        for column in entry.get("columns") or {}:
            if column not in db_columns.get(table, []):
                logger.warning(f"[{db_name}] Stale description: {table}.{column} no longer exists")

    # Written at once (a temporary file renamed): offline/studio reads it while a run goes on
    path = tables_path(db_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(dump_table_semantics(existing))
    temporary.replace(path)

    logger.info(f"[{db_name}] {len(todo)} table(s) described, {added} column description(s) added — review them.")
