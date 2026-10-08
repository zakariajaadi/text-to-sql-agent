# src/agent/nodes.py

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from loguru import logger
from sqlalchemy.exc import SQLAlchemyError

from agent.guardrails import is_safe_query
from agent.prompts import (
    CONDENSE_QUESTION_PROMPT,
    DIALECT_RULES,
    FORMAT_ANSWER_PROMPT,
    PLANNER_PROMPT,
    PRUNE_COLUMNS_PROMPT,
    RERANK_TABLES_PROMPT,
    SQL_CODER_PROMPT,
    SQL_VALIDATOR_FEEDBACK,
)
from agent.schemas import ColumnSelection, RankedTablesAndScope
from agent.state import AgentState
from agent.tools import get_run_query_tool, run_capped_query
from config import (
    cfg,
    fast_model,
    fk_graphs,
    model,
    schema_store,
)
from lib import (
    cached_node,
    expand_fk,
    extract_plan_tables,
    extract_sql,
    extract_text_content,
    get_last_cycle,
    get_question,
    normalize_sql,
)
from lib.column_pruning import get_column_names, get_structural_columns, prune_ddl
from schema_retrieval.retriever import get_schema_retriever

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
        - eval:     the benchmark database (eval/run.py)

    Without a caller choice, the only configured database is used.
    With several databases, it fails loudly rather than guess.
    """
    # Per-turn fields, reset at every new question
    resets = {
        "generated_query":     "",
        "validation_feedback": "",
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
    LLM-driven node — narrows candidates to only the tables needed
    and checks if the question is in scope.

    Three-pass table resolution after LLM selection:
        Pass 1 — load LLM-selected tables missing from vector retrieval (via DDL semantic reasoning)
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

    # ── 1. LLM reranking: table selection + scope check ───────────────────────
    schemas_text = "\n\n".join(
        f"### {name}\n{ddl}" for name, ddl in candidates.items()
    )

    structured_llm = fast_model.with_structured_output(RankedTablesAndScope)
    response = structured_llm.invoke([
        SystemMessage(content=RERANK_TABLES_PROMPT.format(
            dialect=db_cfg.dialect,
            schemas=schemas_text,
        )),
        HumanMessage(content=question),
    ])

    logger.debug(f"rerank_tables: {response.tables} | in_scope={response.is_in_scope} — {response.reason}")

    # ── 2. Load LLM-selected tables missing from candidates ───────────────────
    # LLM may select tables known via schema descriptions but not returned by Chroma
    for table in list(response.tables):
        if table not in candidates and table in schema_store[db_name]:
            candidates[table] = schema_store[db_name][table]
            logger.debug(f"[rerank_tables] Loaded LLM-selected table from store: {table}")

    # Override scope if all LLM-selected tables are now available
    # (an empty selection means out of scope — all() is True on an empty list)
    if response.tables and not response.is_in_scope and all(t in candidates for t in response.tables):
        response.is_in_scope = True
        logger.debug("[rerank_tables] Overrode is_in_scope → True (all tables resolved from store)")

    selected_names = [t for t in response.tables if t in candidates]

    if not selected_names:
        logger.warning("LLM returned no valid tables — fallback to all candidates.")
        return {
            "selected_schemas":    candidates,
            "is_in_scope":         response.is_in_scope,
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
        "is_in_scope":         response.is_in_scope,
        "llm_selected_tables": list(response.tables),
    }


def route_by_scope(state: AgentState) -> Literal["in_scope", "out_of_scope"]:
    if state["is_in_scope"]:
        return "in_scope"
    return "out_of_scope"

def prune_columns(state: AgentState) -> dict:
    """
    LLM node — removes the columns irrelevant to the question from wide tables
    , then builds retrieved_schema for planner and sql_coder.

    Only tables with more than column_pruning.min_columns columns are sent to the LLM:
    narrow tables have few tokens to save and recall to lose.

    Recall guarantees — pruning must never remove a column the query needs:
        - structural columns (join keys, read mode) are always kept
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

    # ── 3. Prune the wide tables, structural columns always kept ──────────────
    pruned_schemas = dict(selected_schemas)   # narrow tables pass through unchanged

    for table, ddl in wide_schemas.items():
        column_names     = set(get_column_names(ddl))
        semantic_columns = selected_columns.get(table, set()) & column_names   # drops invented names

        # Nothing valid selected (table skipped or LLM failure) → keep the table whole rather than guess
        if not semantic_columns:
            logger.warning(f"prune_columns: no valid column selected for {table} — kept whole")
            continue

        columns_to_keep       = semantic_columns | get_structural_columns(table, ddl, query_tables=set(selected_schemas))
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

    # Pruned schema first; full schema on retry — the error may come from a pruned column
    if feedback:
        schema = "\n\n".join(state["selected_schemas"].values())
    else:
        schema = state["retrieved_schema"]

    # Format prompt with schema and dialect — both db-specific
    system_prompt = SQL_CODER_PROMPT.format(
        dialect=db_cfg.dialect,
        dialect_rules=DIALECT_RULES.get(db_cfg.dialect, ""),
        schema=schema,
    )

    # Build user message: plan + question + feedback if retry
    if feedback:
        user_content = f"{feedback}\n\nOriginal question: {question}"
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

    # Quote query identifiers (reserved words on Hive/Impala)
    # Deterministic fixes (Hive/Impala): backticks, NULLs last, assumption comments kept
    query = normalize_sql(query, db_cfg.dialect)

    return {"generated_query": query}


def sql_validator(state: AgentState) -> dict:
    """
    Deterministic node — validates and executes the generated SQL.
    Three checks: guardrail → execute → result handling.
    On failure: writes structured feedback for sql_coder.
    On success: injects the query result into messages for format_answer.

    The query runs against selected_db and at most `max_rows` rows are read:
    the safety cap is applied here, never as a LIMIT in the SQL.
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
        }

    # 2. Execute SQL query, reading at most max_rows rows
    try:
        result = run_capped_query(db_name, query, db_cfg.max_rows)
    except SQLAlchemyError as e:
        # On failure return feedback to sql_coder, and the bare error to the Chainlit UI
        return {
            "validation_feedback": SQL_VALIDATOR_FEEDBACK.format(
                query=query,
                error=f"Execution error: Error: {e}",
                extra_context="",
            ),
            "validation_error":    str(e),
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
        "result_row_count":    len(result.rows),
        "result_truncated":    result.truncated,
        "result_columns":      result.columns,
        "result_rows":         result.rows,
    }


def route_after_validation(state: AgentState) -> Literal["error", "success"]:
    """Route back to coder on failure, to format_answer on success."""
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
    
    # Fetch query result ("No results" if no query was executed)
    result = next(
        (m.content for m in reversed(messages) if isinstance(m, ToolMessage)),
        "No results."
    ) if query else "No results."

    # Give the formatter the row count, and tell it when sql_validator left rows unread (safety cap)
    results_label = "Results"
    if query and state.get("result_truncated"):
        results_label = f"Results (truncated: first {state['result_row_count']} rows only, the query returned more)"
    elif query and "result_row_count" in state:
        results_label = f"Results ({state['result_row_count']} rows)"

    # Format answer
    response = fast_model.invoke([
        SystemMessage(content=FORMAT_ANSWER_PROMPT.format(dialect=db_cfg.dialect)),
        HumanMessage(content=(
            f"Question: {question}\n\n"
            f"Executed SQL:\n{query or 'No query executed — question is out of scope.'}\n\n"
            f"{results_label}: {result}"
        )),
    ])

    return {"messages": [response]}