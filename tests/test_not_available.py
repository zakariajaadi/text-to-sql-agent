"""
A question whose information is in no column of the schema gets the out-of-scope answer, never a
query on another column nor a text message dressed as a query (agent/nodes.py, lib/sql_tools.py).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent import nodes
from agent.schemas import RankedTablesAndScope
from lib.messages import not_available_reason
from lib.sql_tools import is_refusal


@pytest.mark.parametrize("query, refusal", [
    ("-- Not available: the document numbers are in no column", True),
    ("-- Assumption: x\n-- Not available: y", True),
    ("SELECT 'The average income cannot be computed' AS message", True),
    ("SELECT 'a' AS m, 'b'", True),
    # Queries
    ("-- Assumption: interpreted 'clients' as person\nSELECT COUNT(*) FROM person", False),
    ("SELECT 1", False),
    ("SELECT 'a' AS label FROM person", False),
    ("SELECT CONCAT('a', 'b')", False),
])
def test_is_refusal(query, refusal):
    assert is_refusal(query, "impala") is refusal


# =============================================================================
# sql_coder: no query → format_answer
# =============================================================================

@pytest.fixture
def coder(monkeypatch):
    """sql_coder on a fake Impala database; its LLM answers what the test sets in coder.answer."""
    llm = MagicMock()
    monkeypatch.setattr(nodes, "model", llm)
    monkeypatch.setattr(nodes, "cfg", SimpleNamespace(get_db=lambda name: SimpleNamespace(dialect="impala")))
    return llm


STATE = {"selected_db": "bank", "messages": [HumanMessage(content="Average income of the persons?")],
         "retrieved_schema": "CREATE TABLE person (person_id INT)", "selected_schemas": {}}


@pytest.mark.parametrize("plan, reason", [
    ("Not available (restricted): the identity document number", "restricted"),
    ("**Not available (absent)**: the weather", "absent"),
    ("**Not available**: no column holds the income.", "absent"),   # no reason given
])
def test_a_plan_saying_not_available_writes_no_query(coder, plan, reason):
    output = nodes.sql_coder({**STATE, "query_plan": plan})
    assert output == {"generated_query": "", "is_in_scope": False, "unavailable_reason": reason, "validation_feedback": ""}
    coder.invoke.assert_not_called()
    assert nodes.route_after_coder({**STATE, **output}) == "not_available"


@pytest.mark.parametrize("answer, reason", [
    ("-- Not available (restricted): the income of the persons", "restricted"),
    ("-- Assumption: x\n-- Not available (absent): the income", "absent"),
    ("-- Not available: no column holds the income", "absent"),
    ("```sql\nSELECT 'The average income cannot be computed' AS message;\n```", "absent"),
])
def test_a_refusal_of_the_coder_writes_no_query(coder, answer, reason):
    coder.invoke.return_value = AIMessage(content=answer)
    output = nodes.sql_coder({**STATE, "query_plan": "- Tables: person"})
    assert output["generated_query"] == "" and output["is_in_scope"] is False
    assert output["unavailable_reason"] == reason
    assert nodes.route_after_coder({**STATE, **output}) == "not_available"


def test_a_refusal_after_the_excluded_columns_guardrail_is_restricted(coder):
    # The guardrail blocked the previous query: the information exists, withheld, whatever the line says
    coder.invoke.return_value = AIMessage(content="-- Not available (absent): the document number")
    output = nodes.sql_coder({**STATE, "validation_feedback": "BLOCKED: ...", "validation_check": "excluded"})
    assert output["unavailable_reason"] == "restricted"


def test_a_query_goes_to_the_validator(coder):
    coder.invoke.return_value = AIMessage(content="SELECT COUNT(*) AS n FROM person")
    output = nodes.sql_coder({**STATE, "query_plan": "- Tables: person"})
    assert "FROM `person`" in output["generated_query"]
    assert nodes.route_after_coder({**STATE, **output}) == "query"


# =============================================================================
# rerank_tables: an out-of-scope verdict on tables it saw is kept
# =============================================================================

@pytest.fixture
def rerank(monkeypatch):
    """rerank_tables on the catalog person, account; returns a function setting the LLM's answer."""
    llm = MagicMock()
    monkeypatch.setattr(nodes, "fast_model", llm)
    monkeypatch.setattr(nodes, "cfg", SimpleNamespace(get_db=lambda name: SimpleNamespace(dialect="impala"),
                                                      retrieval=SimpleNamespace(fk_neighbors=False, top_k=8)))
    monkeypatch.setattr(nodes, "schema_store", {"bank": {"person": "ddl", "account": "ddl", "loan": "ddl"}})

    def run(tables: list[str], in_scope: bool, **unavailable) -> dict:
        llm.with_structured_output.return_value.invoke.return_value = RankedTablesAndScope(
            tables=tables, is_in_scope=in_scope, reason="", **unavailable)
        return nodes.rerank_tables({"selected_db": "bank", "messages": [HumanMessage(content="q")],
                                    "candidate_tables": ["person", "account"]})
    return run


def test_out_of_scope_on_tables_it_saw_is_kept(rerank):
    assert rerank(["person"], in_scope=False)["is_in_scope"] is False


def test_in_scope_stays_in_scope(rerank):
    output = rerank(["person"], in_scope=True)
    assert output["is_in_scope"] is True and output["unavailable_reason"] == ""


@pytest.mark.parametrize("unavailable, reason", [
    ({"unavailable": "restricted"}, "restricted"),
    ({"unavailable": "absent"}, "absent"),
    ({"unavailable": " Restricted "}, "restricted"),
    ({"unavailable": "unknown"}, "absent"),
    ({}, "absent"),                       # the LLM left it out: in no data
])
def test_out_of_scope_says_why(rerank, unavailable, reason):
    assert rerank([], in_scope=False, **unavailable)["unavailable_reason"] == reason


# =============================================================================
# The reason of a "Not available" line, the answer and the steps it gives
# =============================================================================

@pytest.mark.parametrize("text, reason", [
    ("Not available (restricted): the document number", "restricted"),
    ("-- Not available (absent): the weather", "absent"),
    ("**Not available (RESTRICTED)**: x", "restricted"),
    ("- Not available: x", "absent"),
    ("-- Assumption: interpreted 'x' as y\n-- Not available (restricted): z", "restricted"),
    ("- Tables: person\n- Filters: None", None),
    ("SELECT COUNT(*) FROM person", None),
])
def test_not_available_reason(text, reason):
    assert not_available_reason(text) == reason


@pytest.mark.parametrize("reason, said", [
    ("restricted", "confidential"),
    ("absent", "not in the available data"),
])
def test_format_answer_is_told_why_there_is_no_query(monkeypatch, reason, said):
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="answer")
    monkeypatch.setattr(nodes, "fast_model", llm)
    monkeypatch.setattr(nodes, "cfg", SimpleNamespace(get_db=lambda name: SimpleNamespace(dialect="impala")))
    state = {**STATE, **nodes.not_available(reason)}
    nodes.format_answer(state)
    question = llm.invoke.call_args.args[0][1].content
    assert said in question and "No query executed" in question
    # Still a refusal for the eval: no SQL (offline/eval/run.py reads generated_query or None)
    assert (state.get("generated_query") or None) is None


@pytest.mark.parametrize("reason, title", [
    ("restricted", "information confidentielle"),
    ("absent", "information absente des données disponibles"),
])
def test_the_steps_say_why_in_french_without_the_plan_words(reason, title):
    from ui.streaming import TurnRenderer
    plan = f"Not available ({reason}): numero_piece_identite is excluded from piece_identite"
    plan_title, plan_detail = asyncio.run(TurnRenderer._end_planner(None, {}, {"query_plan": plan}))
    coder_title, _ = asyncio.run(TurnRenderer._end_sql_coder(None, {}, nodes.not_available(reason)))
    assert plan_title == f"Plan : {title}" and coder_title == f"Aucune requête : {title}"
    assert "numero_piece_identite" not in plan_detail and "Not available" not in plan_title + plan_detail
