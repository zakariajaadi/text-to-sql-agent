from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from loguru import logger

from config import model
from guardrails import is_safe_query
from prompts import (
    CLASSIFY_QUESTION_PROMPT,
    FORMAT_ANSWER_PROMPT,
    GENERATE_QUERY_PROMPT,
    PLAN_SQL_PROMPT,
)
from schemas import QuestionComplexity
from state import AgentState
from tools import get_schema_tool, list_tables_tool, run_query_tool
from utils import get_last_cycle

# ── Tool Nodes ────────────────────────────────────────────────────────────────

get_schema_node = ToolNode([get_schema_tool],
                           name="get_schema",
                           handle_tool_errors=True
)
run_query_node  = ToolNode([run_query_tool],
                            name="run_query",
                            handle_tool_errors=True)

# ── Nodes & Conditional Edges ─────────────────────────────────────────────────

def list_tables(state: AgentState) -> dict:
    """
    Deterministic node: retrieves all available tables without invoking the LLM.
    
    Manually constructs and executes the `sql_db_list_tables` tool call, bypassing
    the LLM entirely. 
    """

    tool_call = {
        "name": list_tables_tool.name,
        "args": {},
        "id": "list_tables_call",
        "type": "tool_call",
    }

    # Simulate the AIMessage a LLM would have sent to trigger the tool call (AI Message + tool call)
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])

    # Actually execute the tool, returns a ToolMessage with tool_call_id set and response
    tool_message = list_tables_tool.invoke(tool_call)
    
    return {"messages": [tool_call_message, tool_message]}


def relevant_tables(state: AgentState) -> dict:
    """
    LLM-driven node : selects the tables relevant to the user's question.
    
    Forces the LLM to make a tool call `sql_db_schema` on the tables it deems relevant.
    """

    llm_with_tools = model.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}


def question_complexity(state: AgentState) -> dict: 

    # LLM with structured output
    structured_llm= model.with_structured_output(QuestionComplexity)
    
    # Assess complexity
    messages=[SystemMessage(content=CLASSIFY_QUESTION_PROMPT),*state["messages"]]
    response= structured_llm.invoke(messages)
    logger.debug(f"query complexity = {response.question_complexity}")
    
    # return complexity
    return {"question_complexity":response.question_complexity}

def route_question_by_complexity(state: AgentState) -> Literal["generate_query","plan_query_generation","format_answer"]:
    question_complexity=state["question_complexity"]

    if question_complexity == "out_of_scope":
        return "format_answer"
    
    elif question_complexity == "complex":
        return "plan_query_generation"
    else:
        return "generate_query"


def plan_query_generation(state: AgentState) -> dict: 

    messages=[SystemMessage(content=PLAN_SQL_PROMPT),*state["messages"]]
    response= model.invoke(messages)
    return {"query_plan": response.content}


def generate_query(state: AgentState) -> dict:
    """
    LLM-driven node — generates a SQL query

    """
    llm_with_tools = model.bind_tools([run_query_tool],tool_choice="any")

    # If plan
    system_prompt = GENERATE_QUERY_PROMPT
    if state.get("query_plan"):
        system_prompt += f"\n\n## Execution Plan\nFollow this plan:\n{state['query_plan']}"
    
    # No Plan
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def safe_run_query(state: AgentState) -> dict:
    """Wrapper Node for run_query_node ToolNode.
    
    Guardrailed query execution: safety check → syntexe check → execute."""
    
    last_ai = state["messages"][-1]
    tool_call = last_ai.tool_calls[0]
    query = tool_call["args"]["query"]

    # 1. Guardrail
    if not is_safe_query(query):
        raise ValueError(f"Unsafe query blocked: {query}")

    # 2. EXPLAIN — validate structure without executing
    from config import db
    try:
        db.run(f"EXPLAIN {query}")
    except Exception as e:
        # Return error as ToolMessage so retry loop can handle it
        return {"messages": [ToolMessage(
            content=f"Query validation failed (EXPLAIN): {e}",
            name=run_query_tool.name,
            tool_call_id=tool_call["id"],
            status="error",
        )]}

    # 3. Execute
    return run_query_node.invoke(state)

def route_after_run_query(state: AgentState) -> Literal["generate_query", "format_answer"]:
    """Route to generate_query on SQL error, format_answer on success."""
    
    last_message = state["messages"][-1]
    
    if isinstance(last_message, ToolMessage) and last_message.status == "error":
        if state["remaining_steps"] <= 3: # avoid infinite loop
            logger.warning("Not enough steps remaining, aborting.")
            return "format_answer"
        return "generate_query"
    
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

