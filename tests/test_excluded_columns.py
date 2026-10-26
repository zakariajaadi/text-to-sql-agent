"""
Excluded columns (exclude_columns of tables.yaml): hidden from the documents, and never read by a
query either (lib/sql_tools.py find_excluded_columns, agent/nodes.py sql_validator).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent import nodes
from lib.semantics import excluded_columns, load_semantics
from lib.sql_tools import find_excluded_columns

EXCLUDED = {"identity_document": {"document_number"}, "person": {"monthly_income", "last_name"}}


# =============================================================================
# find_excluded_columns
# =============================================================================

@pytest.mark.parametrize("query, found", [
    # Qualified, unqualified, through an alias, quoted, of another case, the database named
    ("SELECT identity_document.document_number FROM identity_document", ["identity_document.document_number"]),
    ("SELECT document_number FROM identity_document", ["identity_document.document_number"]),
    ("SELECT d.document_number FROM bank.identity_document d", ["identity_document.document_number"]),
    ("SELECT `d`.`Document_Number` FROM `identity_document` AS `d`", ["identity_document.document_number"]),
    # In a WHERE, a JOIN condition, an aggregate
    ("SELECT COUNT(*) FROM person WHERE monthly_income > 1000", ["person.monthly_income"]),
    ("SELECT COUNT(*) FROM person p JOIN account a ON a.holder = p.last_name", ["person.last_name"]),
    # In a subquery, a derived table, a CTE, a correlated reference to the outer query
    ("SELECT x.n FROM (SELECT AVG(p.monthly_income) AS n FROM person p) x", ["person.monthly_income"]),
    ("SELECT person_id FROM identity_document WHERE person_id IN "
     "(SELECT person_id FROM person WHERE last_name = 'X')", ["person.last_name"]),
    ("WITH c AS (SELECT person_id, monthly_income FROM person) SELECT AVG(c.monthly_income) FROM c",
     ["person.monthly_income"]),
    ("SELECT d.person_id FROM identity_document d WHERE EXISTS (SELECT 1 FROM person p "
     "WHERE p.person_id = d.person_id AND d.document_number IS NOT NULL)", ["identity_document.document_number"]),
    # SELECT * and t.* over a table having excluded columns, in a CTE too
    ("SELECT * FROM person", ["person.*"]),
    ("SELECT p.* FROM person p JOIN account a ON a.person_id = p.person_id", ["person.*"]),
    ("WITH c AS (SELECT * FROM person) SELECT COUNT(*) FROM c", ["person.*"]),
    # Unparseable: every word naming an excluded column of a table the query names
    ("SELECT last_name FROM person WHERE city = 'unclosed", ["person.last_name"]),
])
def test_excluded_columns_are_found(query, found):
    assert find_excluded_columns(query, "impala", EXCLUDED) == found


@pytest.mark.parametrize("query", [
    # The allowed columns of tables having excluded ones, COUNT(*)
    "SELECT p.person_id, COUNT(*) AS n FROM person p JOIN identity_document d ON p.person_id = d.person_id "
    "GROUP BY p.person_id ORDER BY n DESC",
    # A column of the same name in a table that doesn't exclude it
    "SELECT a.last_name FROM account a",
    # SELECT * over a table without excluded columns, or over a subquery of allowed columns
    "SELECT * FROM account",
    "SELECT * FROM (SELECT person_id FROM person) s",
])
def test_allowed_queries_pass(query):
    assert find_excluded_columns(query, "impala", EXCLUDED) == []


@pytest.mark.parametrize("db_name", ["Chinook", "Northwind"])
def test_the_eval_queries_of_chinook_and_northwind_pass(db_name):
    """No exclusion there today; with one, the accepted queries must not read it."""
    excluded = {table: columns for table, entry in load_semantics(db_name).items()
                if (columns := excluded_columns(entry))}
    dataset  = yaml.safe_load(Path(f"offline/eval/datasets/{db_name.lower()}.yaml").read_text())
    queries  = [query for q in dataset for query in q.get("accepted_queries") or []]
    assert queries
    # A column excluded in every table, read by no query: each query resolved, none blocked
    tables = {table.lower() for table in load_semantics(db_name)}
    hidden = {table: {"hidden_column"} for table in tables}
    for dialect in ("sqlite", "hive"):
        assert all(find_excluded_columns(query, dialect, excluded) == [] for query in queries)
        assert all(find_excluded_columns(query, dialect, hidden) == [] for query in queries
                   if "*" not in query.replace("(*)", ""))


# =============================================================================
# sql_validator
# =============================================================================

@pytest.fixture
def validator(monkeypatch):
    """sql_validator on a fake Impala database excluding EXCLUDED, its query runs counted."""
    runs = []
    monkeypatch.setattr(nodes, "cfg", SimpleNamespace(get_db=lambda name: SimpleNamespace(dialect="impala", max_rows=100)))
    monkeypatch.setattr(nodes, "get_excluded_columns", lambda db_name: EXCLUDED)
    monkeypatch.setattr(nodes, "get_scope_tables", lambda db_name: ({"identity_document", "person"}, set()))
    monkeypatch.setattr(nodes, "find_unknown_values", lambda *args: [])
    monkeypatch.setattr(nodes, "get_run_query_tool", lambda db_name: SimpleNamespace(name="sql_db_query"))
    monkeypatch.setattr(nodes, "_run_with_reconnect", lambda db_name, query, max_rows: runs.append(query) or
                        SimpleNamespace(content="[(1,)]", rows=[[1]], truncated=False, columns=["n"]))
    return runs


def test_a_query_reading_an_excluded_column_is_never_run(validator):
    state  = {"generated_query": "SELECT document_number FROM identity_document", "selected_db": "bank"}
    output = nodes.sql_validator(state)
    assert validator == []
    assert output["validation_check"] == "excluded"
    assert "identity_document.document_number" in output["validation_feedback"]
    assert "-- Not available (restricted)" in output["validation_feedback"]
    assert nodes.route_after_validation({**state, **output, "remaining_steps": 10}) == "error"


def test_a_second_query_reading_them_gets_the_out_of_scope_answer(validator):
    state  = {"generated_query": "SELECT * FROM person", "selected_db": "bank",
              "validation_feedback": "BLOCKED: ...", "validation_check": "excluded"}
    output = nodes.sql_validator(state)
    assert validator == []
    assert output["generated_query"] == "" and output["is_in_scope"] is False
    assert output["unavailable_reason"] == "restricted"
    assert nodes.route_after_validation({**state, **output, "remaining_steps": 10}) == "success"


def test_an_allowed_query_runs(validator):
    state  = {"generated_query": "SELECT COUNT(DISTINCT p.person_id) AS n FROM person p", "selected_db": "bank"}
    output = nodes.sql_validator(state)
    assert validator == [state["generated_query"]]
    assert output["validation_check"] == "" and output["result_rows"] == [[1]]


def test_the_excluded_columns_come_from_tables_yaml(monkeypatch):
    semantics = {"person": {"exclude_columns": ["monthly_income"]}, "account": {"description": "x"}}
    monkeypatch.setattr(nodes, "load_semantics", lambda db_name: semantics)
    nodes.get_excluded_columns.cache_clear()
    try:
        assert nodes.get_excluded_columns("bank") == {"person": {"monthly_income"}}
    finally:
        nodes.get_excluded_columns.cache_clear()
