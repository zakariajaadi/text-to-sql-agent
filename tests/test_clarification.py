"""
Active clarification (agent/nodes.py, lib/messages.py): when two readings of the question
diverge, the planner writes a "Clarify:" line instead of a plan; the turn ends on that one
question with its options, and the user's answer is the next turn (condense_question merges
the exchange). The eval never asks: the section only enters the planner prompt when the
caller sets `clarify` (app.py, from the database config).
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent import nodes
from agent.prompts import CLARIFICATION_SECTION
from lib.messages import clarify_question


@pytest.mark.parametrize("text, expected", [
    ("Clarify: Actif au sens ? | contrat ouvert | a opéré sur 12 mois",
     ("Actif au sens ?", ["contrat ouvert", "a opéré sur 12 mois"])),
    # "--" or bold, and the question alone
    ("-- Clarify: Quel encours ? | sain | total", ("Quel encours ?", ["sain", "total"])),
    ("**Clarify:** Quelle date ? | **création** | **validation**", ("Quelle date ?", ["création", "validation"])),
    ("Clarify: Quelle période ?", ("Quelle période ?", [])),
    # A plan, a refusal, an empty text: no question
    ("- Tables: invoice\n- Filters: none", None),
    ("Not available (absent): the weather", None),
    ("", None),
])
def test_the_clarify_line_of_a_plan(text, expected):
    assert clarify_question(text) == expected


def test_the_route_after_the_planner():
    assert nodes.route_after_planner({"query_plan": "Clarify: Actif ? | ouvert | a opéré"}) == "clarify"
    assert nodes.route_after_planner({"query_plan": "- Tables: invoice"}) == "plan"
    assert nodes.route_after_planner({}) == "plan"


def test_ask_clarification_ends_the_turn_on_the_question():
    output = nodes.ask_clarification({"query_plan": "Clarify: Actif au sens ? | contrat ouvert | a opéré"})
    assert output["messages"][0].content == "Actif au sens ?"
    assert output["clarification_question"] == "Actif au sens ?"
    assert output["clarification_options"] == ["contrat ouvert", "a opéré"]
    assert output["generated_query"] == ""


@pytest.fixture
def prompts(monkeypatch):
    """planner on a fake database, the system prompts sent to the model collected."""
    sent = []

    def invoke(messages):
        sent.append(messages[0].content)
        return AIMessage(content="- Tables: person")

    monkeypatch.setattr(nodes, "cfg", SimpleNamespace(get_db=lambda name: SimpleNamespace(dialect="impala")))
    monkeypatch.setattr(nodes, "fast_model", SimpleNamespace(invoke=invoke))
    monkeypatch.setattr(nodes, "schema_store", {"bank": {}})
    return sent


STATE = {"selected_db": "bank", "messages": [HumanMessage("Encours des clients actifs ?")],
         "retrieved_schema": "CREATE TABLE person (person_id INT)", "selected_schemas": {}}


def test_the_clarification_section_enters_the_planner_prompt_only_when_asked(prompts):
    nodes.planner({**STATE, "clarify": True})
    nodes.planner(STATE)   # the eval never sets clarify: the section is absent
    assert "## CLARIFICATION" in prompts[0] and CLARIFICATION_SECTION.strip() in prompts[0]
    assert "## CLARIFICATION" not in prompts[1] and "Clarify:" not in prompts[1]


def test_resolve_database_resets_the_clarification(monkeypatch):
    monkeypatch.setattr(nodes, "cfg", SimpleNamespace(
        databases=[SimpleNamespace(name="bank")], get_db=lambda name: SimpleNamespace(dialect="impala"),
    ))
    output = nodes.resolve_database({"selected_db": "bank", "clarification_question": "Actif ?",
                                     "clarification_options": ["ouvert"]})
    assert output["clarification_question"] == "" and output["clarification_options"] == []
