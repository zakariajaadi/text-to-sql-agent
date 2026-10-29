# src/agent/nodes.py

import time
from functools import cache
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from loguru import logger
from sqlalchemy.exc import SQLAlchemyError

from agent.guardrails import is_safe_query
from agent.prompts import (
    CONDENSE_QUESTION_PROMPT,
    DIALECT_RULES,
    FORMAT_ANSWER_PROMPT,
    METRICS_SECTION,
    PLANNER_PROMPT,
    PRUNE_COLUMNS_PROMPT,
    RERANK_TABLES_PROMPT,
    SQL_CODER_PROMPT,
    SQL_VALIDATOR_FEEDBACK,
    VALUES_NOT_FOUND_SECTION,
)
from agent.schemas import ColumnSelection, RankedTablesAndScope
from agent.state import AgentState
from agent.tools import get_run_query_tool, run_capped_query
from config import cfg
from lib import (
    cached_node,
    expand_fk,
    extract_plan_tables,
    extract_sql,
    extract_text_content,
    get_last_cycle,
    get_question,
    normalize_sql,
    not_available_reason,
)
from lib.column_pruning import get_column_names, get_structural_columns, prune_ddl
from lib.db_errors import is_unreachable
from lib.metrics import (
    Metric,
    render_metric_catalog,
    render_metric_definitions,
)
from lib.semantics import add_matching_values, excluded_columns, load_semantics
from lib.sql_tools import (
    contains_expression,
    find_excluded_columns,
    find_unavailable_tables,
    get_query_tables,
    get_value_filters,
    is_refusal,
)
from lib.value_index import find_matching_values, find_unknown_values
from resources import (
    db_registry,
    fast_model,
    fk_graphs,
    get_schema_retriever,
    metrics_store,
    model,
    schema_store,
)

# What format_answer is told when no query was run, by unavailable_reason: the information exists
# but is withheld (excluded columns, confidential), or is in no data
NO_QUERY = {
    "restricted": "No query executed — the information asked for is confidential: withheld from the assistant, for data protection.",
    "absent":     "No query executed — the information asked for is not in the available data.",
}


def not_available(reason: str) -> dict:
    """
    State of a question whose information is in no column: no query, the out-of-scope answer
    (format_answer). reason: "restricted" or "absent", from the "Not available (...)" line of the
    planner or sql_coder, "restricted" when the excluded columns guardrail blocked the query, "absent"
    when the scope guardrail did.
    """
    return {"generated_query": "", "is_in_scope": False, "unavailable_reason": reason, "validation_feedback": ""}

# An unreachable database is retried once, on fresh connections, after this delay
RECONNECT_DELAY_S = 3

# Answer of report_unreachable, in the language of the UI
UNREACHABLE_ANSWER = (
    "**La base {db} est injoignable pour le moment** : {engine} ne répond pas "
    "(réseau, VPN ou maintenance du cluster). La question n'a pas pu être exécutée, "
    "réessayez dans quelques minutes."
)

# Feedback of sql_validator for a query reading excluded columns (hidden from the documents)
EXCLUDED_COLUMNS_ERROR = (
    "BLOCKED: the query reads columns that are not available: {columns}. They are confidential, not in "
    "the schema and never queried. Rewrite it with the columns of the schema only, never SELECT *. If "
    "the question cannot be answered without them, write no query: output only the line "
    "-- Not available (restricted): [the information, in business words]"
)


# Feedback of sql_validator for a query reading tables out of the scope of the agent (scope.yaml)
UNAVAILABLE_TABLES_ERROR = (
    "BLOCKED: the query reads tables that are not available: {tables}. They are not in the schema and never "
    "queried. Rewrite it with the tables of the schema only. If the question cannot be answered without "
    "them, write no query: output only the line -- Not available (absent): [the information, in business words]"
)


@cache
def get_scope_tables(db_name: str) -> tuple[set[str], set[str]]:
    """The tables the agent may read (its scope, as the database had it at startup), and the other tables of the database."""
    db      = db_registry[db_name]
    allowed = set(db.get_usable_table_names())
    return allowed, set(db._all_tables) - allowed


@cache
def get_excluded_columns(db_name: str) -> dict[str, set[str]]:
    """{table: columns} of the `exclude_columns` of tables.yaml, read once (like the indexed documents)."""
    return {table: columns for table, entry in load_semantics(db_name).items()
            if (columns := excluded_columns(entry))}


def get_selected_metrics(state: AgentState) -> list[Metric]:
    """Metrics rerank_tables picked for this question."""
    db_metrics = metrics_store[state["selected_db"]]
    return [db_metrics[name] for name in state.get("selected_metrics") or [] if name in db_metrics]


def render_metrics_section(metrics: list[Metric]) -> str:
    """BUSINESS METRICS section of the planner and sql_coder prompts, empty when no metric is selected."""
    return METRICS_SECTION.format(definitions=render_metric_definitions(metrics)) if metrics else ""


def render_values_section(state: AgentState) -> str:
    """VALUES NOT FOUND section of the planner and sql_coder prompts, empty unless match_values found none."""
    if not state.get("values_not_found"):
        return ""
    terms = ", ".join(f"'{term}'" for term in state.get("filter_values") or [])
    return VALUES_NOT_FOUND_SECTION.format(terms=terms)

# ── Nodes & Conditional Edges ─────────────────────────────────────────────────

def condense_question(state: AgentState) -> dict:
    """
    LLM node — rewrites the current question using conversation history
    to resolve pronouns and implicit references.
    No-op when history has no prior turns (skips the LLM call).
    """
    # Fetch all human messages
    human_msgs = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
    
    # if first question, no condensation
    if len(human_msgs) <= 1:
        return {"standalone_question": human_msgs[-1] if human_msgs else ""}
    
    # Last question 
    question = human_msgs[-1]

    # Previous questions
    prior_questions = human_msgs[:-1][-5:]  # last 5 prior questions
    
    # Condesation
    response = fast_model.invoke([
        SystemMessage(content=CONDENSE_QUESTION_PROMPT),
        HumanMessage(content=(
            "Prior questions:\n"
            + "\n".join(f"- {q}" for q in prior_questions)
            + f"\n\nFollow-up question: {question}"
        )),
    ])
    condensed = extract_text_content(response)

    logger.debug(f"condense_question: '{question}' → '{condensed}'")
    return {"standalone_question": condensed}

def resolve_database(state: AgentState) -> dict:
    """
    Deterministic node — resolves the target database and resets per-turn fields.

    The database is chosen by the caller, once per conversation:
        - Chainlit: the chat profile picked by the user (app.py)
        - eval:     the benchmark database (offline/eval/run.py)

    Without a caller choice, the only configured database is used.
    With several databases, it fails loudly rather than guess.
    """
    # Per-turn fields, reset at every new question
    resets = {
        "generated_query":     "",
        "query_plan":          "",
        "validation_feedback": "",
        "validation_error":    "",
        "validation_check":    "",
        "matched_values":      {},
        "values_not_found":    False,
        "result_row_count":    0,
        "result_truncated":    False,
        "result_columns":      [],
        "result_rows":         [],
    }

    db_name = state.get("selected_db")

    if not db_name:
        if len(cfg.databases) > 1:
            raise ValueError("No database selected: pass selected_db in the graph input.")
        db_name = cfg.databases[0].name

    # Fail early on an unknown name rather than in retrieve_schema
    cfg.get_db(db_name)

    return {"selected_db": db_name, **resets}


def retrieve_schema(state: AgentState) -> dict:
    """
    Deterministic node — broad vector search for candidate table schemas.
    First stage of the two-step schema selection (retrieve → rerank).

    Scoped to selected_db via Chroma metadata filter — only tables
    belonging to the routed database are returned as candidates.
    """
    # Get question
    question = get_question(state)

    # Retriever scoped to the database selected by route_database
    db_name          = state["selected_db"]
    schema_retriever = get_schema_retriever(db_name=db_name, k=cfg.retrieval.top_k)

    # Vector similarity search against the schema index
    docs = schema_retriever.invoke(question)

    # Build a dict {table_name: DDL} for downstream LLM re-ranking
    candidates = {doc.metadata["table_name"]: doc.page_content for doc in docs}

    return {"candidate_schemas": candidates}


def rerank_tables(state: AgentState) -> dict:
    """
    LLM-driven node — narrows candidates to only the tables needed,
    picks the business metrics the question asks for, and checks if the question is in scope.

    Three-pass table resolution after LLM selection:
        Pass 1 — load LLM-selected tables and the tables of the selected metrics
                 missing from vector retrieval
        Pass 2 — FK expansion to recover bridge/fact tables
        Pass 3 — load FK-expanded tables missing from candidates (schema store)

    Outputs selected_schemas ({table: full DDL}); columns are pruned downstream by prune_columns.
    """
    # Fetch user question
    question = get_question(state)

    # Work on a copy — candidates may be extended with store-loaded schemas
    candidates = state["candidate_schemas"].copy()
    db_name    = state["selected_db"]
    db_cfg     = cfg.get_db(db_name)
    db_metrics = metrics_store[db_name]

    # ── 1. LLM reranking: table and metric selection + scope check ────────────
    schemas_text = "\n\n".join(
        f"### {name}\n{ddl}" for name, ddl in candidates.items()
    )

    structured_llm = fast_model.with_structured_output(RankedTablesAndScope)
    response = structured_llm.invoke([
        SystemMessage(content=RERANK_TABLES_PROMPT.format(
            dialect=db_cfg.dialect,
            metrics=render_metric_catalog(db_metrics),
            schemas=schemas_text,
        )),
        HumanMessage(content=question),
    ])

    # Metric names are exact keys of metrics.yaml: anything else is ignored
    selected_metrics = [m for m in dict.fromkeys(response.metrics) if m in db_metrics]
    metric_tables    = [t for m in selected_metrics for t in db_metrics[m].tables]
    filter_values    = list(dict.fromkeys(v.strip() for v in response.filter_values if v.strip()))

    logger.debug(f"rerank_tables: {response.tables} | metrics={selected_metrics} | "
                 f"filter_values={filter_values} | in_scope={response.is_in_scope} — {response.reason}")

    # ── 2. Load LLM-selected and metric tables missing from candidates ────────
    # LLM may select tables known via schema descriptions but not returned by Chroma
    loaded = []
    for table in [*response.tables, *metric_tables]:
        if table not in candidates and table in schema_store[db_name]:
            candidates[table] = schema_store[db_name][table]
            loaded.append(table)
            logger.debug(f"[rerank_tables] Loaded selected table from store: {table}")

    # Override scope when the LLM judged out of scope for want of tables it named, now loaded:
    # not when it saw them all (the information asked for is in none of their columns)
    if (not response.is_in_scope and any(t in loaded for t in response.tables)
            and all(t in candidates for t in response.tables)):
        response.is_in_scope = True
        logger.debug("[rerank_tables] Overrode is_in_scope → True (all tables resolved from store)")

    # Why no query will answer it: withheld (confidential) or in no data, the default
    reason = "" if response.is_in_scope else (
        "restricted" if response.unavailable.strip().lower() == "restricted" else "absent")

    # Metric tables come right after the LLM's choice, before FK expansion bridges them
    selected_names = [t for t in dict.fromkeys([*response.tables, *metric_tables]) if t in candidates]

    if not selected_names:
        logger.warning("LLM returned no valid tables — fallback to all candidates.")
        return {
            "selected_schemas":    candidates,
            "selected_metrics":    selected_metrics,
            "filter_values":       filter_values,
            "is_in_scope":         response.is_in_scope,
            "unavailable_reason":  reason,
            "llm_selected_tables": list(response.tables),
        }

    # ── 3. FK expansion — recover bridge/fact tables ──────────────────────────
    expanded_names = expand_fk(selected_names, fk_graphs[db_name],
                               include_neighbors=cfg.retrieval.fk_neighbors)
    logger.debug(f"rerank_tables: FK expansion {selected_names} → {expanded_names}")

    # ── 4. Load FK-expanded tables missing from candidates ────────────────────
    for table in expanded_names:
        if table not in candidates and table in schema_store[db_name]:
            candidates[table] = schema_store[db_name][table]
            logger.debug(f"[rerank_tables] FK-expanded from store: {table}")

    # ── 5. Build final schemas — preserve LLM order, deduplicate ─────────────
    final_tables     = list(dict.fromkeys(selected_names + expanded_names))
    selected_schemas = {t: candidates[t] for t in final_tables if t in candidates}

    if not selected_schemas:
        logger.warning("No valid tables after FK expansion — fallback to all candidates.")
        selected_schemas = candidates

    return {
        "selected_schemas":    selected_schemas,
        "selected_metrics":    selected_metrics,
        "filter_values":       filter_values,
        "is_in_scope":         response.is_in_scope,
        "unavailable_reason":  reason,
        "llm_selected_tables": list(response.tables),
    }


def route_by_scope(state: AgentState) -> Literal["in_scope", "out_of_scope"]:
    if state["is_in_scope"]:
        return "in_scope"
    return "out_of_scope"

def match_values(state: AgentState) -> dict:
    """
    Deterministic node — finds the values the question filters on (filter_values, from
    rerank_tables) in the value index (lib/value_index.py), and writes them in the DDL of
    their column, for planner and sql_coder to filter on the exact value:

        `city` STRING,  -- City of the company. Matching values: 'Sao Paulo'

    Only the columns of the selected tables are searched, and only the ones whose values are
    not listed in the schema get matching values: a listed column already shows every value.
    A term best matched by a listed column adds nothing ('Mexico' → country, listed).

    When no term is close to any value, planner and sql_coder are told (values_not_found):
    they must not swap the term for another value of the data ('santa' → state = 'SP').
    """
    db_name          = state["selected_db"]
    selected_schemas = state["selected_schemas"]
    terms            = state.get("filter_values") or []

    found = find_matching_values(db_name, terms, list(selected_schemas))

    if not found.values:
        return {"matched_values": {}, "values_not_found": found.not_found}

    logger.debug(f"match_values: {terms} → {found.values} (left out: {found.more})")
    return {
        "matched_values":   found.values,
        "values_not_found": False,
        # Retries of sql_coder read selected_schemas: they keep the matching values too
        "selected_schemas": {
            table: add_matching_values(ddl, found.values.get(table, {}), found.more.get(table))
            for table, ddl in selected_schemas.items()
        },
    }


def prune_columns(state: AgentState) -> dict:
    """
    LLM node — removes the columns irrelevant to the question from wide tables
    , then builds retrieved_schema for planner and sql_coder.

    Only tables with more than column_pruning.min_columns columns are sent to the LLM:
    narrow tables have few tokens to save and recall to lose.

    Recall guarantees — pruning must never remove a column the query needs:
        - structural columns (join keys, read mode) are always kept
        - columns of the selected metrics (expression and default date) are always kept
        - columns where match_values found a value of the question are always kept
        - column names invented by the LLM are ignored
        - a table the LLM skipped, or a failed LLM call, keeps its full DDL
    """
    selected_schemas = state["selected_schemas"]
    pruning_cfg      = cfg.column_pruning

    # ── 1. Keep only the wide tables as pruning targets ───────────────────────
    wide_schemas = {
        table: ddl
        for table, ddl in selected_schemas.items()
        if pruning_cfg.enabled and len(get_column_names(ddl)) > pruning_cfg.min_columns
    }

    if pruning_cfg.enabled and not wide_schemas:
        logger.debug(f"prune_columns: no table above {pruning_cfg.min_columns} columns — schemas kept whole")

    # ── 2. LLM selects the semantic columns of the wide tables ────────────────
    selected_columns: dict[str, set[str]] = {}

    if wide_schemas:
        schemas_text = "\n\n".join(f"### {table}\n{ddl}" for table, ddl in wide_schemas.items())

        try:
            selection = fast_model.with_structured_output(ColumnSelection).invoke([
                SystemMessage(content=PRUNE_COLUMNS_PROMPT.format(schemas=schemas_text)),
                HumanMessage(content=get_question(state)),
            ])
            # Hive names are lowercase, the LLM does not always keep the case
            selected_columns = {
                entry.table.lower(): {column.lower() for column in entry.columns}
                for entry in selection.tables
            }
        except Exception as e:
            # Pruning is an optimization: its failure must never block the answer
            logger.warning(f"prune_columns: LLM call failed, tables kept whole — {e}")

    # ── 3. Prune the wide tables, structural, metric and matched columns always kept
    pruned_schemas = dict(selected_schemas)   # narrow tables pass through unchanged
    metrics        = get_selected_metrics(state)
    matched_values = state.get("matched_values") or {}

    for table, ddl in wide_schemas.items():
        column_names     = set(get_column_names(ddl))
        semantic_columns = selected_columns.get(table, set()) & column_names   # drops invented names

        # Nothing valid selected (table skipped or LLM failure) → keep the table whole rather than guess
        if not semantic_columns:
            logger.warning(f"prune_columns: no valid column selected for {table} — kept whole")
            continue

        metric_columns        = {column for m in metrics for t, column in m.columns if t == table}
        matched_columns       = set(matched_values.get(table, {}))
        columns_to_keep       = (semantic_columns | metric_columns | matched_columns
                                 | get_structural_columns(table, ddl, query_tables=set(selected_schemas)))
        pruned_schemas[table] = prune_ddl(ddl, columns_to_keep)
        logger.debug(f"prune_columns: {table} {len(column_names)} → {len(columns_to_keep)} columns")

    return {"retrieved_schema": "\n\n".join(pruned_schemas.values())}

def planner(state: AgentState) -> dict:
    """
    LLM node — produces a structured execution plan for sql_coder.
    Runs on fast_model. Skipped when validation_feedback is set (retry loop).

    The "-- Joins:" hints of a DDL name tables rerank_tables may have left out
    (e.g. order_details for a revenue question). When the plan uses such a table,
    its full DDL is loaded so sql_coder writes its columns from the schema, not from memory.
    """

    schema = state["retrieved_schema"]
    db_name = state["selected_db"]
    db_cfg = cfg.get_db(db_name)
    question = get_question(state)

    # Free-text plan: a structured output would constrain the planner's reasoning
    response = fast_model.invoke([
        SystemMessage(content=PLANNER_PROMPT.format(
            dialect=db_cfg.dialect,
            metrics_section=render_metrics_section(get_selected_metrics(state)),
            values_section=render_values_section(state),
            schema=schema,
        )),
        HumanMessage(content=question),
    ])

    plan = extract_text_content(response)
    logger.debug(f"planner: {plan[:200]}")

    # ── Load the plan's tables missing from the schema ────────────────────────
    selected_schemas = state["selected_schemas"]
    missing = [
        table for table in extract_plan_tables(plan, schema_store[db_name])
        if table not in selected_schemas
    ]

    if not missing:
        return {"query_plan": plan}

    logger.debug(f"planner: tables of the plan loaded from store: {missing}")
    added = {table: schema_store[db_name][table] for table in missing}
    return {
        "query_plan":       plan,
        "selected_schemas": {**selected_schemas, **added},   # retries use the full schema too
        "retrieved_schema": "\n\n".join([schema, *added.values()]),
    }

@cached_node(
    key_fn=lambda state: f"{state['selected_db']}:{get_question(state)}",
    return_key="generated_query"
)
def sql_coder(state: AgentState) -> dict:
    """
    LLM node — generates SQL without executing.
    Receives schema and question, optionally validation feedback on retry.
    Outputs raw SQL into state['generated_query'].

    First attempt uses the pruned schema (retrieved_schema); retries use the full
    schema (selected_schemas), so a column wrongly pruned cannot fail every attempt.
    """
    # Fetch db config, user question, plan and retry feedback
    db_cfg   = cfg.get_db(state["selected_db"])
    question = get_question(state)
    plan     = state.get("query_plan", "")
    feedback = state.get("validation_feedback", "")

    # The planner found the information asked for in no column: no query to write
    if not feedback and (reason := not_available_reason(plan.strip().split("\n", 1)[0])):
        logger.info(f"sql_coder: no query, the plan says the information is not available ({reason}) — {plan[:200]}")
        return not_available(reason)

    # Pruned schema first; full schema on retry — the error may come from a pruned column
    if feedback:
        schema = "\n\n".join(state["selected_schemas"].values())
    else:
        schema = state["retrieved_schema"]

    # Format prompt with schema, dialect and the selected metrics — all db-specific
    metrics = get_selected_metrics(state)
    system_prompt = SQL_CODER_PROMPT.format(
        dialect=db_cfg.dialect,
        dialect_rules=DIALECT_RULES.get(db_cfg.dialect, ""),
        metrics_section=render_metrics_section(metrics),
        values_section=render_values_section(state),
        schema=schema,
    )

    # Build user message: plan + question, plus the validator's feedback on retry. The plan comes
    # back on retry too — the strategy must survive the fix — but the error wins over it: the plan
    # is sometimes what the failed query followed
    if feedback:
        plan_section = (f"## EXECUTION PLAN (still applies, except where the error below says otherwise)\n"
                        f"{plan}\n\n") if plan else ""
        user_content = f"{plan_section}{feedback}\n\nOriginal question: {question}"
    elif plan:
        user_content = f"## EXECUTION PLAN\n{plan}\n\n## QUESTION\n{question}"
    else:
        user_content = question

    # Generate SQL
    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ])

    # Extract raw SQL from the response
    query = extract_sql(extract_text_content(response))

    # No query to run ("-- Not available (...): ...", or a text message dressed as a query): out of
    # scope. After the excluded columns guardrail, the information is withheld whatever the line says
    if is_refusal(query, db_cfg.dialect):
        reason = not_available_reason(query) or "absent"
        if feedback and state.get("validation_check") == "excluded":
            reason = "restricted"
        logger.info(f"sql_coder: no query, the information is not available ({reason}) — {query[:200]}")
        return not_available(reason)

    # Quote query identifiers (reserved words on Hive/Impala)
    # Deterministic fixes (Hive/Impala): backticks, NULLs last, assumption comments kept
    query = normalize_sql(query, db_cfg.dialect)

    # Light check: a selected metric must be computed with its own expression (logged only)
    for metric in metrics:
        if not contains_expression(query, metric.expr, db_cfg.dialect):
            logger.warning(f"sql_coder: metric '{metric.name}' not computed with its expression {metric.expr}")

    return {"generated_query": query}


def sql_validator(state: AgentState) -> dict:
    """
    Deterministic node — validates and executes the generated SQL.
    Checks: guardrail → tables out of the scope → excluded columns → filtered values → execute → result handling.
    On failure: writes structured feedback for sql_coder.
    On success: injects the query result into messages for format_answer.

    The query runs against selected_db and at most `max_rows` rows are read:
    the safety cap is applied here, never as a LIMIT in the SQL.
    An unreachable database (network, cluster down, Kerberos) is not the query's fault: after
    one retry on fresh connections, validation_check "unreachable" sends the turn to
    report_unreachable, without feedback (sql_coder never rewrites a correct query for it).
    """
    # Fetch generated query and target database from state
    query   = state["generated_query"]
    db_name = state["selected_db"]
    db_cfg  = cfg.get_db(db_name)

    # 1. Guardrail — only SELECT allowed
    if not is_safe_query(query):
        error = "BLOCKED: Only SELECT queries are allowed. DML/DDL detected."
        return {
            "validation_feedback": SQL_VALIDATOR_FEEDBACK.format(query=query, error=error, extra_context=""),
            "validation_error":    error,
            "validation_check":    "guardrail",
        }

    # 2. Tables out of the scope (semantic_layer/<db>/scope.yaml), or of another database: never
    #    read, even under a guessed name. One rewrite: when the next query reads them too, the
    #    question needs them and gets the out-of-scope answer
    allowed, outside = get_scope_tables(db_name)
    if blocked := find_unavailable_tables(query, db_cfg.dialect, allowed, outside, db_name):
        logger.warning(f"sql_validator: query reading tables out of the scope blocked: {', '.join(blocked)}")
        if state.get("validation_feedback") and state.get("validation_check") == "scope":
            return {**not_available("absent"), "validation_error": "", "validation_check": "scope"}
        return {
            "validation_feedback": SQL_VALIDATOR_FEEDBACK.format(
                query=query, error=UNAVAILABLE_TABLES_ERROR.format(tables=", ".join(blocked)), extra_context="",
            ),
            "validation_error":    f"Tables hors du périmètre de l'agent : {', '.join(blocked)}",
            "validation_check":    "scope",
        }

    # 3. Excluded columns (exclude_columns of tables.yaml): hidden from the documents, never read
    #    either, even under a guessed name. One rewrite: when the next query reads them too, the
    #    question needs them and gets the out-of-scope answer
    if blocked := find_excluded_columns(query, db_cfg.dialect, get_excluded_columns(db_name)):
        logger.warning(f"sql_validator: query reading excluded columns blocked: {', '.join(blocked)}")
        if state.get("validation_feedback") and state.get("validation_check") == "excluded":
            return {**not_available("restricted"), "validation_error": "", "validation_check": "excluded"}
        return {
            "validation_feedback": SQL_VALIDATOR_FEEDBACK.format(
                query=query, error=EXCLUDED_COLUMNS_ERROR.format(columns=", ".join(blocked)), extra_context="",
            ),
            "validation_error":    f"Colonnes non disponibles pour l'agent : {', '.join(blocked)}",
            "validation_check":    "excluded",
        }

    # 4. Filtered values — a value its column doesn't hold silently returns no row
    #    (city = 'Sao Paolo'). First attempt only: at most one correction, a retry runs as written
    #    (the value index may be older than the data)
    if not state.get("validation_feedback"):
        unknown = find_unknown_values(db_name, get_value_filters(query, db_cfg.dialect),
                                      get_query_tables(query, db_cfg.dialect))
        if unknown:
            return {
                "validation_feedback": SQL_VALIDATOR_FEEDBACK.format(
                    query=query, error=_unknown_values_error(unknown), extra_context="",
                ),
                "validation_error":    "\n".join(
                    f"{u.table}.{u.column} = '{u.value}' : "
                    + (f"proche de {', '.join(repr(v) for v in u.closest)}" if u.closest else "aucune valeur proche")
                    for u in unknown
                ),
                "validation_check":    "values",
            }

    # 5. Execute SQL query, reading at most max_rows rows
    try:
        result = _run_with_reconnect(db_name, query, db_cfg.max_rows)
    except Exception as e:
        if is_unreachable(e):
            logger.warning(f"sql_validator: {db_name} unreachable — {type(e).__name__}: {str(e)[:200]}")
            return {
                "validation_feedback": "",   # nothing for sql_coder to fix
                "validation_error":    str(e),
                "validation_check":    "unreachable",
            }
        if not isinstance(e, SQLAlchemyError):
            raise
        # On failure return feedback to sql_coder, and the bare error to the Chainlit UI
        return {
            "validation_feedback": SQL_VALIDATOR_FEEDBACK.format(
                query=query,
                error=f"Execution error: Error: {e}",
                extra_context="",
            ),
            "validation_error":    str(e),
            "validation_check":    "database",
        }

    # On success, clear feedback and inject result into messages as a tool call + result,
    # in the format of the sql_db_query tool (read by format_answer and the Chainlit UI)
    tool_name = get_run_query_tool(db_name).name
    ai_msg    = AIMessage(content="", tool_calls=[{
        "name": tool_name, "args": {"query": query}, "id": "sql_validator_call", "type": "tool_call",
    }])
    tool_msg  = ToolMessage(content=result.content, name=tool_name, tool_call_id="sql_validator_call")

    return {
        "messages":            [ai_msg, tool_msg],
        "validation_feedback": "",  # clear feedback on success
        "validation_error":    "",
        "validation_check":    "",
        "result_row_count":    len(result.rows),
        "result_truncated":    result.truncated,
        "result_columns":      result.columns,
        "result_rows":         result.rows,
    }


def _run_with_reconnect(db_name: str, query: str, max_rows: int):
    """
    run_capped_query, run again once on fresh connections when the database was unreachable:
    a connection of the pool may have died (cluster restart, network change) while a new one works.
    """
    try:
        return run_capped_query(db_name, query, max_rows)
    except Exception as e:
        if not is_unreachable(e):
            raise
        logger.info(f"sql_validator: {db_name} unreachable, retrying on fresh connections in {RECONNECT_DELAY_S}s")
        db_registry[db_name]._engine.dispose()   # drop the pooled connections
        time.sleep(RECONNECT_DELAY_S)
        return run_capped_query(db_name, query, max_rows)


def report_unreachable(state: AgentState) -> dict:
    """
    Deterministic node — the answer when the database could not be reached (sql_validator):
    a fixed message rather than format_answer, so no LLM call and nothing cached.
    """
    db_name = state["selected_db"]
    engine  = cfg.get_db(db_name).dialect.capitalize()
    return {"messages": [AIMessage(content=UNREACHABLE_ANSWER.format(db=db_name, engine=engine))]}


def _unknown_values_error(unknown: list) -> str:
    """Error of sql_validator for the values a query filters on that their column doesn't hold."""
    lines = [
        f"- {u.table}.{u.column} = '{u.value}': "
        + (f"closest values held by the column: {', '.join(repr(v) for v in u.closest)}" if u.closest
           else "no close value in this column: check the column, or use LOWER(column) LIKE LOWER('%value%')")
        for u in unknown
    ]
    return ("Unknown values: the query filters on values these columns don't hold, "
            "so it would return no row.\n" + "\n".join(lines))


def route_after_coder(state: AgentState) -> Literal["query", "not_available"]:
    """To sql_validator with a query, to format_answer (out-of-scope answer) without one."""
    return "query" if state.get("generated_query") else "not_available"


def route_after_validation(state: AgentState) -> Literal["error", "success", "unreachable"]:
    """Route back to coder on failure, to report_unreachable without a database, to format_answer on success."""
    if state.get("validation_check") == "unreachable":
        return "unreachable"
    feedback = state.get("validation_feedback", "")
 
    if feedback:
        if state["remaining_steps"] <= 3:
            logger.warning("Not enough steps remaining, aborting.")
            return "success"
        return "error"
 
    return "success"


@cached_node(
    key_fn=lambda state: f"{state['selected_db']}:{get_question(state)}",
    return_key="messages",
    cache_only_first_attempt=False
)
def format_answer(state: AgentState) -> dict:
    """
    LLM-driven node — single exit point for all user-facing responses.
    Handles success (question + SQL + results), errors (failed query),
    and out-of-scope (no query executed). Scoped to the current turn
    via get_last_cycle to avoid leaking previous turns' data.
    """
    db_cfg   = cfg.get_db(state["selected_db"])
    messages = state["messages"]
    
    # Get last question 
    question = get_question(state)
    
    # Fetch query (empty if out of scope)
    query = state.get("generated_query") or ""
    
    # Result of the query run this turn (get_last_cycle): a previous turn's ToolMessage must never
    # stand in for a query that did not run (retries exhausted, route_after_validation)
    executed = next(
        (m.content for m in reversed(get_last_cycle(messages)) if isinstance(m, ToolMessage)),
        None,
    ) if query else None

    # Give the formatter the row count, and tell it when sql_validator left rows unread (safety cap)
    results_label = "Results"
    if query and executed is None:
        result = f"Error: the query could not be executed. {state.get('validation_error', '')}".strip()
    elif executed is None:
        result = "No results."
    else:
        result = executed
        if state.get("result_truncated"):
            results_label = f"Results (truncated: first {state['result_row_count']} rows only, the query returned more)"
        elif "result_row_count" in state:
            results_label = f"Results ({state['result_row_count']} rows)"

    # Format answer
    response = fast_model.invoke([
        SystemMessage(content=FORMAT_ANSWER_PROMPT.format(dialect=db_cfg.dialect)),
        HumanMessage(content=(
            f"Question: {question}\n\n"
            f"Executed SQL:\n{query or NO_QUERY.get(state.get('unavailable_reason'), NO_QUERY['absent'])}\n\n"
            f"{results_label}: {result}"
        )),
    ])

    return {"messages": [response]}