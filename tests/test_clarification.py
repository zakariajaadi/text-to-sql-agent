"""
Active clarification (agent/nodes.py, lib/messages.py): when two readings of the question
diverge, the planner writes a "Clarify:" line instead of a plan; the turn ends on that one
question with its options, and the user's answer is the next turn (condense_question merges
the exchange). The eval never asks: the section only enters the planner prompt when the
caller sets `clarify` (app.py, from the database config).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent import nodes
from agent.schemas import RankedTablesAndScope
from lib.messages import clarify_question
from lib.metrics import Metric


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


def test_ask_clarification_caps_the_options():
    # The prompt asks for at most 3 options; a model that writes 6 is capped to the first 3
    output = nodes.ask_clarification({"query_plan": "Clarify: Par quoi ? | a | b | c | d | e | f"})
    assert output["clarification_options"] == ["a", "b", "c"]


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
    monkeypatch.setattr(nodes, "metrics_store", {"bank": {}})
    return sent


SAVINGS = Metric(name="total_savings", description="Total savings held.", expr="SUM(savings.amount)",
                 synonyms=["épargne totale"], time_dimension="", tables=["savings"], columns=[("savings", "amount")])


STATE = {"selected_db": "bank", "messages": [HumanMessage("Encours des clients actifs ?")],
         "retrieved_schema": "CREATE TABLE person (person_id INT)", "selected_schemas": {}}


def test_the_clarification_section_enters_the_planner_prompt_only_when_asked(prompts):
    nodes.planner({**STATE, "clarify": True})
    nodes.planner(STATE)   # the eval never sets clarify: the section is absent
    assert "## CLARIFICATION" in prompts[0] and "Clarify:" in prompts[0]
    # no metric in the store: no criteria catalog either
    assert "### Business criteria" not in prompts[0]
    assert "## CLARIFICATION" not in prompts[1] and "Clarify:" not in prompts[1]


def test_the_chosen_criteria_enter_the_clarification(prompts, monkeypatch):
    monkeypatch.setattr(nodes, "metrics_store", {"bank": {"total_savings": SAVINGS}})
    nodes.planner({**STATE, "clarify": True, "clarify_criteria": ["total_savings"]})
    nodes.planner({**STATE, "clarify": True})                        # the rerank chose none
    nodes.planner({**STATE, "clarify_criteria": ["total_savings"]})  # the eval never sets clarify
    assert "### Business criteria" in prompts[0]
    assert "- épargne totale: Total savings held." in prompts[0]   # first synonym, business words
    assert "total_savings" not in prompts[0]         # the code name would be copied into the options
    assert "SUM(savings.amount)" not in prompts[0]   # labels and descriptions only, never expressions
    assert "### Business criteria" not in prompts[1]
    assert "épargne totale" not in prompts[2]


def test_the_rerank_chooses_the_criteria_from_the_catalog(monkeypatch):
    """Asked only when clarify is on; validated against the catalog keys, tables not loaded."""
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = RankedTablesAndScope(
        tables=["person"], is_in_scope=True, reason="",
        criteria=["total_savings", "invented", "total_savings"])
    monkeypatch.setattr(nodes, "fast_model", llm)
    monkeypatch.setattr(nodes, "cfg", SimpleNamespace(get_db=lambda name: SimpleNamespace(dialect="impala"),
                                                      retrieval=SimpleNamespace(fk_neighbors=False)))
    monkeypatch.setattr(nodes, "schema_store", {"bank": {"person": "ddl", "savings": "ddl"}})
    monkeypatch.setattr(nodes, "metrics_store", {"bank": {"total_savings": SAVINGS}})
    state = {"selected_db": "bank", "messages": [HumanMessage("meilleurs ?")],
             "candidate_schemas": {"person": "ddl"}}

    with_clarify = nodes.rerank_tables({**state, "clarify": True})
    without      = nodes.rerank_tables(state)
    assert with_clarify["clarify_criteria"] == ["total_savings"]       # catalog keys only, deduplicated
    assert "savings" not in with_clarify["selected_schemas"]           # a criterion loads no table
    assert without["clarify_criteria"] == []
    sent = [call.args[0][0].content for call in llm.with_structured_output.return_value.invoke.call_args_list]
    assert "9. In criteria" in sent[0]
    assert "9. In criteria" not in sent[1]   # the eval's rerank prompt is unchanged


def test_resolve_database_resets_the_clarification(monkeypatch):
    monkeypatch.setattr(nodes, "cfg", SimpleNamespace(
        databases=[SimpleNamespace(name="bank")], get_db=lambda name: SimpleNamespace(dialect="impala"),
    ))
    output = nodes.resolve_database({"selected_db": "bank", "clarification_question": "Actif ?",
                                     "clarification_options": ["ouvert"]})
    assert output["clarification_question"] == "" and output["clarification_options"] == []
