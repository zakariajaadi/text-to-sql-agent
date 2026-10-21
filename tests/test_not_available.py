"""
A question whose information is in no column of the schema gets the out-of-scope answer, never a
query on another column nor a text message dressed as a query (agent/nodes.py, lib/sql_tools.py).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent import nodes
from agent.schemas import RankedTablesAndScope
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


def test_a_plan_saying_not_available_writes_no_query(coder):
    output = nodes.sql_coder({**STATE, "query_plan": "**Not available**: no column holds the income."})
    assert output == {"generated_query": "", "is_in_scope": False, "validation_feedback": ""}
    coder.invoke.assert_not_called()
    assert nodes.route_after_coder({**STATE, **output}) == "not_available"


@pytest.mark.parametrize("answer", [
    "-- Not available: no column holds the income",
    "```sql\nSELECT 'The average income cannot be computed' AS message;\n```",
])
def test_a_refusal_of_the_coder_writes_no_query(coder, answer):
    coder.invoke.return_value = AIMessage(content=answer)
    output = nodes.sql_coder({**STATE, "query_plan": "- Tables: person"})
    assert output["generated_query"] == "" and output["is_in_scope"] is False
    assert nodes.route_after_coder({**STATE, **output}) == "not_available"


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
    """rerank_tables with the candidates person and account; returns a function setting the LLM's answer."""
    llm = MagicMock()
    monkeypatch.setattr(nodes, "fast_model", llm)
    monkeypatch.setattr(nodes, "cfg", SimpleNamespace(get_db=lambda name: SimpleNamespace(dialect="impala"),
                                                      retrieval=SimpleNamespace(fk_neighbors=False)))
    monkeypatch.setattr(nodes, "schema_store", {"bank": {"person": "ddl", "account": "ddl", "loan": "ddl"}})

    def run(tables: list[str], in_scope: bool) -> dict:
        llm.with_structured_output.return_value.invoke.return_value = RankedTablesAndScope(
            tables=tables, is_in_scope=in_scope, reason="")
        return nodes.rerank_tables({"selected_db": "bank", "messages": [HumanMessage(content="q")],
                                    "candidate_schemas": {"person": "ddl", "account": "ddl"}})
    return run


def test_out_of_scope_on_tables_it_saw_is_kept(rerank):
    assert rerank(["person"], in_scope=False)["is_in_scope"] is False


def test_out_of_scope_for_want_of_a_table_loaded_since_is_overridden(rerank):
    assert rerank(["person", "loan"], in_scope=False)["is_in_scope"] is True


def test_in_scope_stays_in_scope(rerank):
    assert rerank(["person"], in_scope=True)["is_in_scope"] is True
