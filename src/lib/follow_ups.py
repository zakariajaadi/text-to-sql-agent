"""
Follow-up questions — suggested under each answer, one click asks them
Used by: lib/chainlit_ui.py (started when sql_validator succeeds, so the LLM call runs
while format_answer writes the answer). UI only: the agent graph and the eval never call it.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from agent.prompts import FOLLOW_UPS_PROMPT
from agent.schemas import FollowUpQuestions
from resources import fast_model

MAX_FOLLOW_UPS = 3
SAMPLE_ROWS    = 25  # rows of the result shown to the LLM: enough to see what it already answers


async def suggest_follow_ups(
    question: str,
    schema:   str,
    sql:      str,
    columns:  list[str],
    rows:     list[list],
) -> list[str]:
    """Return up to MAX_FOLLOW_UPS follow-up questions, or [] when the LLM call fails."""
    sample = "\n".join(" | ".join(str(v) for v in row) for row in rows[:SAMPLE_ROWS])
    try:
        result = await fast_model.with_structured_output(FollowUpQuestions).ainvoke([
            SystemMessage(content=FOLLOW_UPS_PROMPT),
            HumanMessage(content=(
                f"Question: {question}\n\n"
                f"Schema:\n{schema}\n\n"
                f"Executed SQL:\n{sql}\n\n"
                f"Results ({len(rows)} rows, first {min(len(rows), SAMPLE_ROWS)} shown):\n"
                f"{' | '.join(columns)}\n{sample}"
            )),
        ])
    except Exception as e:
        # Suggestions are a bonus: never let them break the answer
        logger.warning(f"suggest_follow_ups failed: {e!r}")
        return []

    questions = [q.strip() for q in (result.questions if result else []) if q.strip()]
    return questions[:MAX_FOLLOW_UPS]
