"""
LangChain messages helpers
Used by: agent/nodes.py, ui/streaming.py, ui/follow_ups.py

"""

import re

from langchain_core.messages import HumanMessage


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
    # Hive/Impala reject a trailing ';' — harmless to drop for every dialect
    return query.rstrip().rstrip(";").rstrip()


# "Not available (restricted): ..." line of the planner or sql_coder (agent/prompts.py), "--" or bold allowed
_NOT_AVAILABLE = re.compile(r"^[\W_]*not available\b\W*(restricted|absent)?", re.IGNORECASE)


def not_available_reason(text: str) -> str | None:
    """
    Reason of the first "Not available" line of a plan or of sql_coder's answer: "restricted" (the
    information exists but is withheld, confidential) or "absent" (in no data, also when the line
    gives no reason); None when no line says "Not available".
    """
    for line in text.splitlines():
        if match := _NOT_AVAILABLE.match(line.strip()):
            return (match.group(1) or "absent").lower()
    return None


def extract_plan_tables(plan: str, tables) -> list[str]:
    """
    Tables of the database named on the "Tables:" line of the planner's free-text plan.
    Only the line is located; table names are then matched against the known tables,
    so the line's formatting (bold, brackets, quotes, backticks) does not matter.
    """
    line = next((line for line in plan.splitlines() if "tables:" in line.lower()), "")
    return [
        table for table in tables
        if re.search(rf"(?<![\w.]){re.escape(table)}(?!\w)", line, re.IGNORECASE)
    ]


def get_question(state: dict) -> str:
    """Return the standalone question (post-condensation) or fall back to last HumanMessage."""
    return state.get("standalone_question") or next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
    )