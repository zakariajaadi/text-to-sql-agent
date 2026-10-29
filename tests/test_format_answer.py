"""
format_answer reads only the current turn (agent/nodes.py): the result of the query run this turn,
an error when the retries were exhausted, never the ToolMessage or the row counts of a previous
turn. resolve_database resets every per-turn field at each new question.
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import nodes

# A finished turn, kept in messages by the checkpointer: its result must never be read again
PREVIOUS_TURN = [
    HumanMessage("CA 2024 ?"),
    ToolMessage(content="[(1250,)]", name="sql_db_query", tool_call_id="sql_validator_call"),
    AIMessage("Le CA 2024 est 1250."),
]


@pytest.fixture
def prompts(monkeypatch):
    """format_answer on a fake database, the prompts sent to the model collected."""
    sent = []

    def invoke(messages):
        sent.append(messages[-1].content)
        return AIMessage(content="answer")

    monkeypatch.setattr(nodes, "cfg", SimpleNamespace(get_db=lambda name: SimpleNamespace(dialect="impala")))
    monkeypatch.setattr(nodes, "fast_model", SimpleNamespace(invoke=invoke))
    return sent


def test_exhausted_retries_answer_with_an_error_not_the_previous_results(prompts):
    # route_after_validation aborts the retry loop: a query, feedback still set, nothing run this turn
    nodes.format_answer({
        "selected_db":         "shop",
        "standalone_question": "CA 2025 ?",
        "messages":            [*PREVIOUS_TURN, HumanMessage("CA 2025 ?")],
        "generated_query":     "SELECT SUM(total) FROM invoice",
        "validation_feedback": "The query you generated failed validation.",
        "validation_error":    "AnalysisException: could not resolve column",
        "result_row_count":    0,
        "result_truncated":    False,
    })
    prompt = prompts[0]
    assert "Error: the query could not be executed. AnalysisException: could not resolve column" in prompt
    assert "[(1250,)]" not in prompt and "rows)" not in prompt


def test_the_result_read_is_the_one_of_this_turn(prompts):
    nodes.format_answer({
        "selected_db":         "shop",
        "standalone_question": "CA 2025 ?",
        "messages": [
            *PREVIOUS_TURN,
            HumanMessage("CA 2025 ?"),
            ToolMessage(content="[(999,)]", name="sql_db_query", tool_call_id="sql_validator_call"),
        ],
        "generated_query":     "SELECT SUM(total) FROM invoice",
        "validation_feedback": "",
        "result_row_count":    1,
        "result_truncated":    False,
    })
    prompt = prompts[0]
    assert "[(999,)]" in prompt and "[(1250,)]" not in prompt
    assert "Results (1 rows)" in prompt


def test_a_question_without_a_query_reads_no_result(prompts):
    nodes.format_answer({
        "selected_db":         "shop",
        "standalone_question": "Salaire du directeur ?",
        "messages":            [*PREVIOUS_TURN, HumanMessage("Salaire du directeur ?")],
        "generated_query":     "",
        "unavailable_reason":  "restricted",
    })
    prompt = prompts[0]
    assert "confidential" in prompt and "[(1250,)]" not in prompt


def test_resolve_database_resets_the_fields_of_the_turn(monkeypatch):
    monkeypatch.setattr(nodes, "cfg", SimpleNamespace(
        databases=[SimpleNamespace(name="shop")], get_db=lambda name: SimpleNamespace(dialect="impala"),
    ))
    output = nodes.resolve_database({"selected_db": "shop", "result_rows": [[1250]], "result_row_count": 1})
    assert output["result_rows"] == [] and output["result_columns"] == []
    assert output["result_row_count"] == 0 and output["result_truncated"] is False
    assert output["generated_query"] == "" and output["query_plan"] == ""
    assert output["validation_feedback"] == "" and output["validation_error"] == "" and output["validation_check"] == ""
