from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from loguru import logger

from config import cfg, model
from prompts import (
    FORMAT_ANSWER_PROMPT,
    RERANK_TABLES_PROMPT,
    SCOPE_CHECK_PROMPT,
    SQL_CODER_PROMPT,
    SQL_VALIDATOR_FEEDBACK
)
from schema_retrieval.retriever import get_schema_retriever
from schemas import QuestionScope, SelectedTables
from state import AgentState
from tools import get_schema_tool, run_query_tool
from utils import extract_sql,get_last_cycle
from guardrails import is_safe_query

# ── Nodes & Conditional Edges ─────────────────────────────────────────────────

def retrieve_schema(state: AgentState) -> dict:
    """
    Deterministic node — broad vector search for candidate table schemas.
    First stage of the two-step schema selection (retrieve → rerank).
    """
    question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    
    schema_retriever = get_schema_retriever(k=cfg.retrieval.top_k)

    # Vector similarity search against the schema index
    docs = schema_retriever.invoke(question)

    # Build a dict {table_name: DDL} for downstream LLM re-ranking
    candidates = {doc.metadata["table_name"]: doc.page_content for doc in docs}

    return {"candidate_schemas": candidates}


def rerank_tables(state: AgentState) -> dict:
    """
    LLM-driven node — narrows candidates to only the tables needed.
    Second stage: reasons about FK chains that embeddings can't capture
    (e.g., Genre → Track → InvoiceLine for a revenue question).
    """
    question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    candidates = state["candidate_schemas"]

    # Format all candidate schemas for the LLM prompt
    schemas_text = "\n\n".join(
        f"### {name}\n{ddl}" for name, ddl in candidates.items()
    )

    # LLM structured output: returns table names + reasoning
    structured_llm = model.with_structured_output(SelectedTables)
    response = structured_llm.invoke([
        SystemMessage(content=RERANK_TABLES_PROMPT.format(schemas=schemas_text)),
        HumanMessage(content=question),
    ])

    logger.debug(f"LLM re-ranking: {response.tables} — {response.reason}")

    # Keep only the DDLs for tables the LLM selected
    selected = [candidates[t] for t in response.tables if t in candidates]

    # Fallback: if LLM returned no valid tables (hallucinated names, etc.),
    # pass all candidates through rather than failing silently
    if not selected:
        logger.warning("LLM re-ranking returned no valid tables, using all candidates.")
        selected = list(candidates.values())

    return {"retrieved_schema": "\n\n".join(selected)}

def check_scope(state: AgentState) -> dict: 
    """Deterministic gate — filters out questions unrelated to the database."""
    
    # Fetch schema from state 
    schema = state["retrieved_schema"]

    # LLM with structured output
    structured_llm= model.with_structured_output(QuestionScope)
    
    # Check question scope
    messages=[
        SystemMessage(content=SCOPE_CHECK_PROMPT.format(schema=schema)),
        *state["messages"]
    ]
    response= structured_llm.invoke(messages)
    logger.debug(f"Scope check: in_scope={response.is_in_scope} — {response.reason}")

    # return scope 
    return {"is_in_scope": response.is_in_scope}

def route_by_scope(state: AgentState) -> Literal["sql_coder","format_answer"]:
    if state["is_in_scope"]:
        return "sql_coder"
    return "format_answer"


# ── SQL Coder ─────────────────────────────────────────────────────────────────

def sql_coder(state: AgentState) -> dict:
    """
    LLM node — generates SQL without executing.
    Receives schema and question, optionally validation feedback on retry.
    Outputs raw SQL into state['generated_query'].
    """
    # Fetch tables schema
    schema = state["retrieved_schema"]

    # Fetch user question
    question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    
    # Format prompt with tables schemas
    system_prompt = SQL_CODER_PROMPT.format(schema=schema)

    # Build user message: question + feedback if retry
    feedback = state.get("validation_feedback", "")
    if feedback:
        user_content = f"{feedback}\n\nOriginal question: {question}"
    else:
        user_content = question
    
    # Generate SQL
    response = model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_content)])

    # Extract raw SQL from the response
    query = extract_sql(response.content)
   
    return {"generated_query": query}
  

def sql_validator(state: AgentState) -> dict:
    """
    Deterministic node — validates and executes the generated SQL.
    Three checks: guardrail → EXPLAIN → execute.
    On failure: writes structured feedback for sql_coder.
    On success: injects the query result into messages for format_answer.
    """
    # Fetch generated query from state
    query = state["generated_query"]

    # 1. Guardrail — only SELECT allowed
    if not is_safe_query(query):
        return {
            "validation_feedback": SQL_VALIDATOR_FEEDBACK.format(
                query=query,
                error="BLOCKED: Only SELECT queries are allowed. DML/DDL detected.",
                extra_context="",
            )
        }

    # 2. Execute sql query via tool
    tool_call = {
        "name": run_query_tool.name,
        "args": {"query": query},
        "id": "sql_validator_call",
        "type": "tool_call",
    }

    ai_msg = AIMessage(content="", tool_calls=[tool_call])
    tool_msg = run_query_tool.invoke(tool_call)

    # on failure return feedback to sql coder 
    if tool_msg.status == "error":
        return {
            "validation_feedback": SQL_VALIDATOR_FEEDBACK.format(
                query=query,
                error=f"Execution error: {tool_msg.content}",
                extra_context="",
            )
        }
    
    # on success, clear feedback and inject result into messages (AI msg + tool msg)
    return {
        "messages": [ai_msg, tool_msg],
        "validation_feedback": "",  # clear feedback on success
    }


def route_after_validation(state: AgentState) -> Literal["sql_coder", "format_answer"]:
    """Route back to coder on failure, to format_answer on success."""
    feedback = state.get("validation_feedback", "")
    
    if feedback:
        if state["remaining_steps"] <= 3:
            logger.warning("Not enough steps remaining, aborting.")
            return "format_answer"
        return "sql_coder"
    
    return "format_answer"

def format_answer(state: AgentState) -> dict:
    """
    LLM-driven node — single exit point for all user-facing responses.
    Handles success (question + SQL + results), errors (failed query),
    and out-of-scope (no query executed). Scoped to the current turn
    via get_last_cycle to avoid leaking previous turns' data.

    Minimal context: last question, executed SQL (if any), query results (if any).
    """

    messages = state["messages"]
    turn = get_last_cycle(messages)  # messages after the last HumanMessage

    # Get last question (get_last_cycle excludes HumanMessage)
    question = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
        ""
    )

    # Last SQL tool_call in this turn (absent for out-of-scope questions)
    query = next(
        (m.tool_calls[0]["args"]["query"] for m in reversed(turn)
         if isinstance(m, AIMessage) and m.tool_calls
         and m.tool_calls[0]["name"] == run_query_tool.name),
        "No query executed — question is out of scope for this database."
    )

    # Last SQL result in this turn (absent for out-of-scope or if query was never run)
    result = next(
        (m.content for m in reversed(turn)
         if isinstance(m, ToolMessage) and m.name == run_query_tool.name),
        "No results."
    )

    response = model.invoke([
        SystemMessage(content=FORMAT_ANSWER_PROMPT),
        HumanMessage(content=f"Question: {question}\n\n"
                             f"Executed SQL:\n{query}\n\n"
                             f"Results: {result}")
    ])

    return {"messages": [response]}

