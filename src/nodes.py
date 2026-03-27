# src/nodes.py

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from loguru import logger

from guardrails import is_safe_query
from prompts import (
    FORMAT_ANSWER_PROMPT,
    RERANK_TABLES_PROMPT,
    SQL_CODER_PROMPT,
    SQL_VALIDATOR_FEEDBACK,
)
from schema_retrieval.retriever import get_schema_retriever
from schemas import RankedTablesAndScope
from state import AgentState
from tools import get_run_query_tool
from utils import extract_sql, get_last_cycle, select_database
from config import cfg, db_embeddings, domain_rules, fast_model, model


# ── Nodes & Conditional Edges ─────────────────────────────────────────────────

def route_database(state: AgentState) -> dict:
    """
    Deterministic node — selects the target database for the incoming question.
    Routing logic is delegated to select_database() in utils.py.
    """
    question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
    )

    selected = select_database(
        question=question,
        domain_rules=domain_rules,
        db_embeddings=db_embeddings,
    )

    return {"selected_db": selected}


def retrieve_schema(state: AgentState) -> dict:
    """
    Deterministic node — broad vector search for candidate table schemas.
    First stage of the two-step schema selection (retrieve → rerank).

    Scoped to selected_db via Chroma metadata filter — only tables
    belonging to the routed database are returned as candidates.
    """
    question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )

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
    and checks if the question is in scope. Single LLM call replacing
    the former rerank_tables + check_scope two-node sequence.
    """
    question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    candidates = state["candidate_schemas"]
    db_cfg     = cfg.get_db(state["selected_db"])
 
    # Format all candidate schemas for the LLM prompt
    schemas_text = "\n\n".join(
        f"### {name}\n{ddl}" for name, ddl in candidates.items()
    )
 
    # Single LLM call: table selection + scope check
    structured_llm = fast_model.with_structured_output(RankedTablesAndScope)
    response = structured_llm.invoke([
        SystemMessage(content=RERANK_TABLES_PROMPT.format(
            dialect=db_cfg.dialect,
            schemas=schemas_text,
        )),
        HumanMessage(content=question),
    ])
 
    logger.debug(f"rerank_tables: {response.tables} | in_scope={response.is_in_scope} — {response.reason}")
 
    # Keep only the DDLs for tables the LLM selected
    selected = [candidates[t] for t in response.tables if t in candidates]
 
    # Fallback: if LLM returned no valid tables, pass all candidates through
    if not selected:
        logger.warning("LLM re-ranking returned no valid tables, using all candidates.")
        selected = list(candidates.values())
 
    return {
        "retrieved_schema": "\n\n".join(selected),
        "is_in_scope": response.is_in_scope,
    }


def route_by_scope(state: AgentState) -> Literal["in_scope", "out_of_scope"]:
    if state["is_in_scope"]:
        return "in_scope"
    return "out_of_scope"


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
    question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )

    # Format prompt with schema, dialect and limit — all db-specific
    system_prompt = SQL_CODER_PROMPT.format(
        dialect=db_cfg.dialect,
        limit=db_cfg.limit,
        schema=schema,
    )

    # Build user message: question + feedback if retry
    feedback = state.get("validation_feedback", "")
    if feedback:
        user_content = f"{feedback}\n\nOriginal question: {question}"
    else:
        user_content = question

    # Generate SQL
    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ])

    # Extract raw SQL from the response
    query = extract_sql(response.content)

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

    # On failure return feedback to sql_coder
    if tool_msg.status == "error":
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

def format_answer(state: AgentState) -> dict:
    """
    LLM-driven node — single exit point for all user-facing responses.
    Handles success (question + SQL + results), errors (failed query),
    and out-of-scope (no query executed). Scoped to the current turn
    via get_last_cycle to avoid leaking previous turns' data.

    Minimal context: last question, executed SQL (if any), query results (if any).
    The run_query_tool is resolved via selected_db to match tool message names
    correctly across databases.
    """
    messages = state["messages"]
    turn     = get_last_cycle(messages)  # messages after the last HumanMessage
    db_cfg   = cfg.get_db(state["selected_db"])

    # Get last question (get_last_cycle excludes HumanMessage)
    question = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )

    # Resolve tool name for the selected database
    run_query = get_run_query_tool(state["selected_db"])

    # Last SQL tool_call in this turn (absent for out-of-scope questions)
    query = next(
        (m.tool_calls[0]["args"]["query"] for m in reversed(turn)
         if isinstance(m, AIMessage) and m.tool_calls
         and m.tool_calls[0]["name"] == run_query.name),
        "No query executed — question is out of scope for this database."
    )

    # Last SQL result in this turn (absent for out-of-scope or if query was never run)
    result = next(
        (m.content for m in reversed(turn)
         if isinstance(m, ToolMessage) and m.name == run_query.name),
        "No results."
    )

    # Dialect injected so the LLM formats the answer appropriately per database
    response = fast_model.invoke([
        SystemMessage(content=FORMAT_ANSWER_PROMPT.format(dialect=db_cfg.dialect)),
        HumanMessage(content=(
            f"Question: {question}\n\n"
            f"Executed SQL:\n{query}\n\n"
            f"Results: {result}"
        )),
    ])

    return {"messages": [response]}