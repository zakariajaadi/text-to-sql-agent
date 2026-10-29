"""
A retry of sql_coder keeps the execution plan (agent/nodes.py): the feedback fixes one error, the
strategy of the planner (aggregation levels, read modes, period filters) must survive the rewrite,
but the error wins over the plan — the plan is sometimes what the failed query followed.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent import nodes


@pytest.fixture
def coder(monkeypatch):
    """sql_coder on a fake Impala database, its LLM answering a fixed query."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="SELECT COUNT(*) AS n FROM person")
    monkeypatch.setattr(nodes, "model", llm)
    monkeypatch.setattr(nodes, "cfg", SimpleNamespace(get_db=lambda name: SimpleNamespace(dialect="impala")))
    return llm


STATE = {"selected_db": "bank", "messages": [HumanMessage(content="Average income?")],
         "retrieved_schema": "CREATE TABLE person (person_id INT)",
         "selected_schemas": {"person": "CREATE TABLE person (person_id INT)"}}

FEEDBACK = "The query you generated failed validation.\n\n## ERROR\nAnalysisException: could not resolve column"


def sent_content(llm) -> str:
    return llm.invoke.call_args.args[0][-1].content


def test_a_retry_keeps_the_plan_under_the_error(coder):
    nodes.sql_coder({**STATE, "query_plan": "- Tables: person\n- Aggregation strategy: two levels",
                     "validation_feedback": FEEDBACK})
    content = sent_content(coder)
    # The plan is back, under a title saying the error wins, then the feedback, then the question
    assert "## EXECUTION PLAN (still applies, except where the error below says otherwise)" in content
    assert "- Aggregation strategy: two levels" in content
    assert content.index("EXECUTION PLAN") < content.index("AnalysisException")
    assert content.rstrip().endswith("Original question: Average income?")


def test_a_retry_without_a_plan_sends_the_feedback_alone(coder):
    nodes.sql_coder({**STATE, "validation_feedback": FEEDBACK})
    content = sent_content(coder)
    assert "EXECUTION PLAN" not in content and content.startswith("The query you generated")


def test_a_first_attempt_sends_the_plan_as_before(coder):
    nodes.sql_coder({**STATE, "query_plan": "- Tables: person"})
    content = sent_content(coder)
    assert content.startswith("## EXECUTION PLAN\n") and "says otherwise" not in content
    assert content.rstrip().endswith("## QUESTION\nAverage income?")
