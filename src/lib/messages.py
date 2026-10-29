"""
LangChain messages helpers
Used by: agent/nodes.py, ui/streaming.py, ui/follow_ups.py

"""

import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage


def extract_text_content(val: Any) -> str:
    """
    Recursively extracts plain text from various LangChain/LLM output formats.
    Handles:
      - Lists of messages (takes the last one).
      - LangChain Message objects (AIMessage, ToolMessage).
      - Gemini-style content blocks (list of dicts with 'text').
      - Standard dictionaries or raw strings.
    """
    if not val:
        return ""

    # 1. If it's a list (like state['messages']), recurse on the last element
    if isinstance(val, list):
        return extract_text_content(val[-1])

    # 2. Extract content from Message objects (if present)
    content = getattr(val, "content", val)

    # 3. Handle Gemini-style content blocks: [{'text': '...'}, {'text': '...'}]
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content
            if isinstance(block, dict)
        ).strip()

    # 4. Handle dictionary returns (e.g., {'text': '...'})
    if isinstance(content, dict):
        return content.get("text", str(content))

    # 5. Fallback to string representation
    return str(content)


def get_qa_pairs(messages: list) -> list[tuple[str, str]]:
    """
    (question, answer) of each turn: the content of every HumanMessage, paired with the last final
    AIMessage of its turn — an answer written for the user, never the tool-call messages of
    sql_validator. "" for a turn without one: the current question, or a stopped turn.
    Used by condense_question to resolve references to the answers.
    """
    pairs: list[list[str]] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            pairs.append([extract_text_content(message), ""])
        elif pairs and isinstance(message, AIMessage) and not message.tool_calls:
            if answer := extract_text_content(message):
                pairs[-1][1] = answer
    return [(question, answer) for question, answer in pairs]


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


# "Clarify: question | option | option" line of the planner (agent/prompts.py), "--" or bold allowed
_CLARIFY = re.compile(r"^[\W_]*clarify\s*:\s*(.+)$", re.IGNORECASE)


def clarify_question(text: str) -> tuple[str, list[str]] | None:
    """
    The question the planner asks the user instead of planning, with its options:
    the first "Clarify: [question] | [option] | [option]" line of the plan.
    None when no line asks one (the usual case: the plan stands).
    """
    for line in text.splitlines():
        if match := _CLARIFY.match(line.strip()):
            parts    = [part.strip(" *\t") for part in match.group(1).split("|")]
            question = parts[0]
            if question:
                return question, [part for part in parts[1:] if part]
    return None


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