"""
LangChain messages helpers
Used by: nodes.py (check_query, generate_query)

"""

from langchain_core.messages import HumanMessage, ToolMessage

from tools import get_run_query_tool


def get_last_cycle(messages: list) -> list:
    """Return technical messages from the last cycle (after the last user question)."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i + 1:]
    return []


def extract_sql(text: str) -> str:
    """Strip markdown code fences from LLM-generated SQL."""
    query = text.strip()
    if query.startswith("```"):
        query = query.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return query


def extract_schema_message(messages: list, db_name: str) -> ToolMessage | None:
    """
    Scan state messages and return the ToolMessage produced by get_schema for a given db.
    Returns the last one found (in case of multi-turn conversations).
    """
    schema_tool = get_run_query_tool(db_name)
    result = None
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == schema_tool.name:
            result = msg
    return result


def get_question(state: dict) -> str:
    """Return the standalone question (post-condensation) or fall back to last HumanMessage."""
    return state.get("standalone_question") or next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
    )