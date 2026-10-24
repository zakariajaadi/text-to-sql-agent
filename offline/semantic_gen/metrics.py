# offline/semantic_gen/metrics.py

"""
Offline LLM generator of the business metrics: semantic_layer/<db>/metrics.yaml.

Like the other generators, it is additive: reviewed metrics are never
modified, rejected ones (listed in `ignore`) never come back. It reads the database directly
(DDL, column types, reviewed tables.yaml and relationships.yaml), never Chroma.

Pipeline order: semantics → relations → metrics → (review) → index   (see offline/cli.py)
"""

from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from sqlalchemy import inspect as sa_inspect

from config import cfg
from lib.metrics import metrics_path, read_metrics_file, validate_metric
from lib.semantics import excluded_columns, load_fk_relations, load_semantics
from offline.semantic_gen.prompts import METRICS_PROMPT
from offline.semantic_gen.schemas import MetricProposals
from offline.semantic_gen.utils import metric_from_clause, render_schemas, skip_reason, write_metrics_file
from resources import db_registry, model

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

# Languages of the metric synonyms, besides English: the languages users ask in
SYNONYM_LANGUAGES = "French"

# Column types, from the type name reported by SQLAlchemy (Hive types don't subclass the generic ones)
NUMERIC_TYPES  = ("int", "decimal", "numeric", "float", "double", "real")
TEMPORAL_TYPES = ("date", "time")

# ------------------------------------------------------------
# Chain
# ------------------------------------------------------------

# Tables holding measures → business metrics
metrics_chain = (
    ChatPromptTemplate.from_messages([("human", METRICS_PROMPT)])
    | model.with_structured_output(MetricProposals)
)

# ------------------------------------------------------------
# Generator
# ------------------------------------------------------------

def generate_metrics(db_name: str):
    """
    Proposes business metrics via LLM and appends them to semantic_layer/<db>/metrics.yaml, for review.

    1. Frame (no LLM): the tables holding measures (see _column_roles), the fact tables among
       them (many side of a relationship), and the tables they join to for dates and labels.
    2. Propose: the LLM writes up to 8 metrics from that context, avoiding the metrics
       already defined or listed in `ignore`.
    3. Check: a proposal named or written like a known metric, or summing distinct values, is
       skipped (skip_reason); the others go through the startup validation (lib/metrics.py), then the
       database compiles it on an empty result (WHERE 1 = 0); a failing one is dropped with a warning.

    Additive only: reviewed metrics are never modified, rejected ones never come back.
    """
    db           = db_registry[db_name]
    dialect      = cfg.get_db(db_name).dialect
    fk_relations = load_fk_relations(db_name)
    semantics    = load_semantics(db_name)
    roles        = _column_roles(db, fk_relations, semantics)
     
    # Measure tables: have at least one column of role measure
    measure_tables = [table for table, role in roles.items() if role["measures"]]
    if not measure_tables:
        logger.info(f"[{db_name}] No numeric measure found — no metric to propose.")
        return

    # Fact tables: have measures AND are on the "many" side of a FK relationship
    fact_tables = {source.split(".")[0] for source, _ in fk_relations} & set(measure_tables)
    
    # --- 1. Context: measure tables + tables they join to (dimensions, date tables ...)
    related     = set(measure_tables)
    for source, target in fk_relations:
        tables = {source.split(".")[0], target.split(".")[0]}
        if tables & set(measure_tables):
            related |= tables

    # Load already defined metrics and voluntarily ignored ones
    entries, ignored = read_metrics_file(db_name)

    # ---  2. LLM proposals
    proposals = metrics_chain.invoke({
        "db_name":   db_name,
        "dialect":   dialect,
        "languages": SYNONYM_LANGUAGES,
        "language":  cfg.get_db(db_name).semantic_language,
        "roles":     _render_roles(roles, measure_tables, fact_tables),
        "existing":  "\n".join(f"- {name}: {entry.get('expr')}" for name, entry in entries.items()) or "None",
        "rejected":  ", ".join(ignored) or "None",
        "schemas":   render_schemas(db, db_name, tables=[t for t in roles if t in related]),
    }).metrics

    # 3. Validate, compile, merge — only metrics neither defined nor rejected
    columns = {table: list(role["all"]) for table, role in roles.items()}
    added   = {}
    for proposal in proposals:
        name  = proposal.name.strip()
        entry = {
            "description":    proposal.description,
            "synonyms":       proposal.synonyms,
            "expr":           proposal.expr,
            "time_dimension": proposal.time_dimension.strip(),
        }
        if not entry["time_dimension"]:
            del entry["time_dimension"]

        if reason := skip_reason(name, proposal.expr, {**entries, **added}, ignored, dialect):
            logger.info(f"[{db_name}] Skipped metric '{name}': {reason}")
            continue

        try:
            metric = validate_metric(name, entry, dialect, columns)
            _compile_metric(db, metric.expr, metric.tables, fk_relations, dialect)
        except Exception as e:
            logger.warning(f"[{db_name}] Rejected metric '{name}': {e}")
            continue
        added[name] = entry

    write_metrics_file(db_name, {**entries, **added}, ignored)
    logger.info(f"[{db_name}] {len(added)} new metric(s) added to {metrics_path(db_name).name}: {', '.join(added) or 'none'} — review them.")


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def _column_roles(db: SQLDatabase, fk_relations: list[tuple[str, str]], semantics: dict) -> dict[str, dict]:
    """
    Sort the columns of each table by role, from their type and the relationships — no LLM.
    Returns {table: {"all", "identifiers", "dates", "measures", "dimensions": [column, ...]}}.

    First matching role wins:
        identifiers — a relationship end, or a numeric column named *id
        dates       — a DATE / TIMESTAMP column
        measures    — any other numeric column, unless declared categorical
        dimensions  — the rest (labels, codes, categorical columns)

    Types are matched by name (NUMERIC_TYPES, TEMPORAL_TYPES): Hive types don't subclass SQLAlchemy's.
    """
    inspector = sa_inspect(db._engine)
    keys      = {ref for relation in fk_relations for ref in relation}

    roles = {}
    for table in db.get_usable_table_names():
        categorical = set((semantics.get(table) or {}).get("categorical_columns") or [])
        excluded    = excluded_columns(semantics.get(table))   # never proposed in a metric
        role = {"all": [], "identifiers": [], "dates": [], "measures": [], "dimensions": []}
        for column in inspector.get_columns(table):
            name      = column["name"]
            if name in excluded:
                continue
            type_name = type(column["type"]).__name__.lower()
            numeric   = any(t in type_name for t in NUMERIC_TYPES)
            role["all"].append(name)
            if f"{table}.{name}" in keys or (numeric and name.lower().endswith("id")):
                role["identifiers"].append(name)
            elif any(t in type_name for t in TEMPORAL_TYPES):
                role["dates"].append(name)
            elif numeric and name not in categorical:
                role["measures"].append(name)
            else:
                role["dimensions"].append(name)
        roles[table] = role
    return roles


def _render_roles(roles: dict[str, dict], measure_tables: list[str], fact_tables: set[str]) -> str:
    """Format the roles of measure tables into a readable string for the LLM.
       One line per table, fact tables first

    Output example:
        - orders (fact table): measures amount, quantity; identifiers order_id, customer_id; dates order_date
        - customers: measures none; identifiers customer_id; dates registration_date
    """
    ordered = sorted(measure_tables, key=lambda t: t not in fact_tables)
    return "\n".join(
        f"- {table}{' (fact table)' if table in fact_tables else ''}: "
        f"measures {', '.join(roles[table]['measures'])}; "
        f"identifiers {', '.join(roles[table]['identifiers']) or 'none'}; "
        f"dates {', '.join(roles[table]['dates']) or 'none'}"
        for table in ordered
    )


def _compile_metric(db: SQLDatabase, expr: str, tables: list[str], fk_relations: list[tuple[str, str]],
                    dialect: str):
    """
    Have the database compile the metric on an empty result (WHERE 1 = 0): catches what static
    checks can't (types, functions). Its tables are joined through the declared relationships
    (metric_from_clause); raises ValueError when they can't all be joined.
    """
    # Type names by their SQLAlchemy class (STRING, INT, DECIMAL...): enough to tell text from numbers
    inspector    = sa_inspect(db._engine)
    column_types = {
        f"{table}.{column['name']}": type(column["type"]).__name__
        for table in tables
        for column in inspector.get_columns(table)
    }

    db.run(f"SELECT {expr} {metric_from_clause(tables, fk_relations, column_types, dialect)} WHERE 1 = 0")
