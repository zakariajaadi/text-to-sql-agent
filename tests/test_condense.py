"""
condense_question resolves references to the prior answers and to the rows of the last result
(agent/nodes.py): the condenser reads question→answer pairs (lib/messages.get_qa_pairs) and the
first rows of the previous turn's result, still in the state at this point of the graph.
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import nodes
from lib.messages import get_qa_pairs

# A finished turn as the checkpointer keeps it: the tool-call messages of sql_validator
# between the question and the answer written for the user
TURN = [
    HumanMessage("Top 5 des compagnies ?"),
    AIMessage("", tool_calls=[{"name": "sql_db_query", "args": {"query": "SELECT 1"},
                               "id": "sql_validator_call", "type": "tool_call"}]),
    ToolMessage(content="[(1250,)]", name="sql_db_query", tool_call_id="sql_validator_call"),
    AIMessage("Nile Air mène avec 1250 vols."),
]


def test_each_question_pairs_with_the_answer_written_for_the_user():
    assert get_qa_pairs([*TURN, HumanMessage("Et la deuxième ?")]) == [
        ("Top 5 des compagnies ?", "Nile Air mène avec 1250 vols."),
        ("Et la deuxième ?", ""),
    ]
    # A stopped turn has no answer; an answer before any question belongs to no turn
    assert get_qa_pairs([HumanMessage("CA ?"), AIMessage(""), HumanMessage("CA 2024 ?")]) == [
        ("CA ?", ""), ("CA 2024 ?", ""),
    ]
    assert get_qa_pairs([AIMessage("Bienvenue")]) == []


@pytest.fixture
def prompts(monkeypatch):
    """condense_question with its LLM answering a fixed rewrite, the prompts sent collected."""
    sent = []

    def invoke(messages):
        sent.append(messages[-1].content)
        return AIMessage(content="condensed")

    monkeypatch.setattr(nodes, "fast_model", SimpleNamespace(invoke=invoke))
    return sent


def test_the_first_question_is_not_condensed(prompts):
    output = nodes.condense_question({"messages": [HumanMessage("Top 5 des compagnies ?")]})
    assert output == {"standalone_question": "Top 5 des compagnies ?"} and prompts == []


def test_the_condenser_reads_the_answers_and_the_last_rows(prompts):
    output = nodes.condense_question({
        "messages":       [*TURN, HumanMessage("Et la deuxième ?")],
        "result_columns": ["compagnie", "vols"],
        "result_rows":    [["Nile Air", 1250], ["Air Europa", 980]],
    })
    assert output == {"standalone_question": "condensed"}
    prompt = prompts[0]
    assert "Q: Top 5 des compagnies ?" in prompt
    assert "A: Nile Air mène avec 1250 vols." in prompt
    assert "compagnie | vols" in prompt and "Air Europa | 980" in prompt
    assert prompt.rstrip().endswith("Follow-up question: Et la deuxième ?")


def test_the_rows_sent_are_capped_and_their_values_cut(prompts):
    rows = [[f"compagnie {n}", "x" * 200] for n in range(30)]
    nodes.condense_question({
        "messages":       [*TURN, HumanMessage("Et la deuxième ?")],
        "result_columns": ["compagnie", "description"],
        "result_rows":    rows,
    })
    prompt = prompts[0]
    assert "compagnie 19" in prompt and "compagnie 20" not in prompt and "(+10 more rows)" in prompt
    assert "x" * 81 not in prompt and "x" * 80 + "…" in prompt


def test_without_a_result_no_rows_block_is_sent(prompts):
    nodes.condense_question({"messages": [*TURN, HumanMessage("Et la deuxième ?")], "result_rows": []})
    assert "Rows of the last results" not in prompts[0]
