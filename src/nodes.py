# src/nodes.py

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from loguru import logger

from agent_schemas import RankedTablesAndScope
from config import (
    cfg,
    db_embeddings,
    domain_rules,
    fast_model,
    fk_graphs,
    model,
    schema_store,
)
from guardrails import is_safe_query
from lib import (
    cached_node,
    expand_fk,
    extract_sql,
    extract_text_content,
    get_last_cycle,
    get_question,
    select_database,
)
from prompts import (
    CONDENSE_QUESTION_PROMPT,
    FORMAT_ANSWER_PROMPT,
    PLANNER_PROMPT,
    RERANK_TABLES_PROMPT,
    SQL_CODER_PROMPT,
    SQL_VALIDATOR_FEEDBACK,
)
from schema_retrieval.retriever import get_schema_retriever
from state import AgentState
from tools import get_run_query_tool

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

def route_database(state: AgentState) -> dict:
    """
    Deterministic node — selects the target database once per conversation.
    Routing logic is delegated to select_database() in lib/routing.py.
    Skipped when a database is already set: previous turn (checkpointer) or eval mode.
    """

    # Resets always apply, even when routing is skipped
    resets = {
        "generated_query":     "",   # reset for the new turn
        "validation_feedback": "",   # reset for the new turn
    }

    # One database per conversation — keep the one already selected
    if state.get("selected_db"):
        return {"selected_db": state["selected_db"], **resets}

    # First question: route it
    selected = select_database(
        question=get_question(state),
        domain_rules=domain_rules,
        db_embeddings=db_embeddings,
    )

    return {"selected_db": selected, **resets}


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
    if not response.is_in_scope and all(t in candidates for t in response.tables):
        response.is_in_scope = True
        logger.debug("[rerank_tables] Overrode is_in_scope → True (all tables resolved from store)")

    selected_names = [t for t in response.tables if t in candidates]

    if not selected_names:
        logger.warning("LLM returned no valid tables — fallback to all candidates.")
        return {
            "retrieved_schema": "\n\n".join(candidates.values()),
            "is_in_scope":      response.is_in_scope,
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

    # ── 5. Build final schema — preserve LLM order, deduplicate ──────────────
    final_tables = list(dict.fromkeys(selected_names + expanded_names))
    selected     = [candidates[t] for t in final_tables if t in candidates]

    if not selected:
        logger.warning("No valid tables after FK expansion — fallback to all candidates.")
        selected = list(candidates.values())

    return {
        "retrieved_schema": "\n\n".join(selected),
        "is_in_scope":      response.is_in_scope,
        "llm_selected_tables": list(response.tables),
    }


def route_by_scope(state: AgentState) -> Literal["in_scope", "out_of_scope"]:
    if state["is_in_scope"]:
        return "in_scope"
    return "out_of_scope"

def planner(state: AgentState) -> dict:
    """
    LLM node — produces a structured execution plan for sql_coder.
    Runs on fast_model. Skipped when validation_feedback is set (retry loop).
    """
    
    schema = state["retrieved_schema"]
    db_cfg = cfg.get_db(state["selected_db"])
    question = get_question(state)

    response = fast_model.invoke([
        SystemMessage(content=PLANNER_PROMPT.format(
            dialect=db_cfg.dialect,
            schema=schema,
        )),
        HumanMessage(content=question),
    ])

    plan = extract_text_content(response)
    logger.debug(f"planner: {plan[:200]}")

    return {"query_plan": plan}

@cached_node(
    key_fn=lambda state: f"{state['selected_db']}:{get_question(state)}",
    return_key="generated_query"
)
def sql_coder(state: AgentState) -> dict:
    """
    LLM node — generates SQL without executing.
    Receives schema and question, optionally validation feedback on retry.
    Outputs raw SQL into state['generated_query'].
    """
    # Fetch tables schema and db config
    schema = state["retrieved_schema"]
    db_cfg = cfg.get_db(state["selected_db"])

    # Fetch user question
    question = get_question(state)

    # Format prompt with schema, dialect and limit — all db-specific
    system_prompt = SQL_CODER_PROMPT.format(
        dialect=db_cfg.dialect,
        limit=db_cfg.limit,
        schema=schema,
    )

    # Build user message: plan + question + feedback if retry
    feedback = state.get("validation_feedback", "")
    plan = state.get("query_plan", "")

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

    return {"generated_query": query}


def sql_validator(state: AgentState) -> dict:
    """
    Deterministic node — validates and executes the generated SQL.
    Three checks: guardrail → execute → result handling.
    On failure: writes structured feedback for sql_coder.
    On success: injects the query result into messages for format_answer.

    The run_query_tool is resolved from the tools registry using selected_db
    to ensure the query is always executed against the correct database.
    """
    # Fetch generated query and target database from state
    query   = state["generated_query"]
    db_name = state["selected_db"]

    # 1. Guardrail — only SELECT allowed
    if not is_safe_query(query):
        return {
            "validation_feedback": SQL_VALIDATOR_FEEDBACK.format(
                query=query,
                error="BLOCKED: Only SELECT queries are allowed. DML/DDL detected.",
                extra_context="",
            )
        }

    # 2. Execute SQL query via the db-scoped tool
    run_query = get_run_query_tool(db_name)

    tool_call = {
        "name": run_query.name,
        "args": {"query": query},
        "id":   "sql_validator_call",
        "type": "tool_call",
    }

    ai_msg   = AIMessage(content="", tool_calls=[tool_call])
    tool_msg = run_query.invoke(tool_call)

    # On failure return feedback to sql_coder (SQLAlchemy error msgs always start with "Error")
    if tool_msg.status == "error" or tool_msg.content.strip().lower().startswith("error"):
        return {
            "validation_feedback": SQL_VALIDATOR_FEEDBACK.format(
                query=query,
                error=f"Execution error: {tool_msg.content}",
                extra_context="",
            )
        }

    # On success, clear feedback and inject result into messages (AI msg + tool msg)
    return {
        "messages":            [ai_msg, tool_msg],
        "validation_feedback": "",  # clear feedback on success
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
    
    # Format answer
    response = fast_model.invoke([
        SystemMessage(content=FORMAT_ANSWER_PROMPT.format(dialect=db_cfg.dialect)),
        HumanMessage(content=(
            f"Question: {question}\n\n"
            f"Executed SQL:\n{query or 'No query executed — question is out of scope.'}\n\n"
            f"Results: {result}"
        )),
    ])

    return {"messages": [response]}