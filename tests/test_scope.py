"""
The scope of a database (semantic_layer/<db>/scope.yaml): the tables the agent and the pipeline see
(lib/semantics.load_scope, lib/database.build_database), the queries reading another table blocked
(lib/sql_tools.find_unavailable_tables, agent/nodes.py sql_validator), and the Périmètre screen of the
studio (offline/studio/scope.py, its routes).
"""

from types import SimpleNamespace

import pytest
import sqlalchemy
from fastapi.testclient import TestClient

import lib.semantics
from agent import nodes
from lib.database import build_database
from lib.semantics import load_scope, scope_origin, scope_path
from lib.sql_tools import find_unavailable_tables
from offline.semantic_gen.utils import write_scope
from offline.studio import app as studio_app
from offline.studio.index import CHANGED, FRESH, MISSING, UNKNOWN
from offline.studio.scope import GONE, LEAVING, OUTSIDE, change_scope, scope_rows, scope_view

TABLES = ["customer", "invoice", "person", "track"]


@pytest.fixture
def layer(tmp_path, monkeypatch):
    """semantic_layer/ in tmp_path, the config of "shop" listing `config_tables` (none at first)."""
    config = SimpleNamespace(include_tables=[])
    monkeypatch.setattr(lib.semantics, "SEMANTIC_LAYER_DIR", tmp_path)
    monkeypatch.setattr(lib.semantics, "cfg", SimpleNamespace(get_db=lambda name: config))
    return config


@pytest.fixture
def sqlite_shop(tmp_path):
    """A SQLite database with TABLES, and the config of a database reading it."""
    engine = sqlalchemy.create_engine(f"sqlite:///{tmp_path / 'shop.db'}")
    with engine.begin() as conn:
        for table in TABLES:
            conn.execute(sqlalchemy.text(f"CREATE TABLE {table} (id INTEGER, name TEXT)"))
    return SimpleNamespace(name="shop", dialect="sqlite", uri=f"sqlite:///{tmp_path / 'shop.db'}", include_tables=[],
                           sample_rows=0, exclude_personal_data=False, engine_args=lambda: {})


# =============================================================================
# The scope: scope.yaml, else the include_tables of the config, else every table
# =============================================================================

def test_the_scope_comes_from_its_file_then_the_config(layer):
    assert load_scope("shop") is None and scope_origin("shop") == "none"
    layer.include_tables = ["invoice", "customer"]
    assert load_scope("shop") == ["invoice", "customer"] and scope_origin("shop") == "config"
    # Saved by the studio: the file wins, sorted, its header says what it is
    write_scope("shop", ["track", "invoice", "track"])
    assert load_scope("shop") == ["invoice", "track"] and scope_origin("shop") == "list"
    assert scope_path("shop").read_text().startswith("# The tables of this database the agent and the pipeline see")
    # A scope without a table: the agent sees none, not all of them
    write_scope("shop", [])
    assert load_scope("shop") == []
    # Every table, the ones to come too: chosen, not a list
    write_scope("shop", None)
    assert load_scope("shop") is None and scope_origin("shop") == "all"
    assert "all: true" in scope_path("shop").read_text()


def test_the_database_wrapper_sees_the_tables_of_the_scope(layer, sqlite_shop):
    assert build_database(sqlite_shop).get_usable_table_names() == TABLES
    # A table of the scope the database no longer has is left out: the agent still starts
    write_scope("shop", ["invoice", "customer", "legacy"])
    db = build_database(sqlite_shop)
    assert db.get_usable_table_names() == ["customer", "invoice"]
    assert db._all_tables == set(TABLES)
    write_scope("shop", [])
    assert build_database(sqlite_shop).get_usable_table_names() == []
    write_scope("shop", None)
    assert build_database(sqlite_shop).get_usable_table_names() == TABLES


# =============================================================================
# The queries: a table out of the scope is never read
# =============================================================================

ALLOWED, OUTSIDE_TABLES = {"invoice", "customer"}, {"person", "track"}


@pytest.mark.parametrize("query, found", [
    ("SELECT COUNT(*) FROM invoice i JOIN customer c ON i.id = c.id", []),
    # A table out of the scope, in a join, a subquery, a CTE body; of another case; the database named
    ("SELECT COUNT(*) FROM invoice i JOIN person p ON i.id = p.id", ["person"]),
    ("SELECT * FROM invoice WHERE id IN (SELECT id FROM `Track`)", ["track"]),
    ("WITH t AS (SELECT id FROM person) SELECT COUNT(*) FROM t", ["person"]),
    ("SELECT COUNT(*) FROM shop.invoice", []),
    # A name the agent guessed, a table of another database even when the scope has its name
    ("SELECT COUNT(*) FROM invoices", ["invoices"]),
    ("SELECT COUNT(*) FROM hr.invoice", ["hr.invoice"]),
    # A CTE named like a table of the scope is no table
    ("WITH person AS (SELECT id FROM invoice) SELECT COUNT(*) FROM person", []),
])
def test_the_tables_out_of_the_scope_a_query_reads(query, found):
    assert find_unavailable_tables(query, "impala", ALLOWED, OUTSIDE_TABLES, "Shop") == found


def test_an_unparseable_query_naming_a_table_out_of_the_scope():
    assert find_unavailable_tables("SELECT FROM person WHERE (", "impala", ALLOWED, OUTSIDE_TABLES, "shop") == ["person"]
    assert find_unavailable_tables("SELECT FROM invoice WHERE (", "impala", ALLOWED, OUTSIDE_TABLES, "shop") == []


@pytest.fixture
def validator(monkeypatch):
    """sql_validator on a fake Impala database whose scope is ALLOWED, its query runs counted."""
    runs = []
    monkeypatch.setattr(nodes, "cfg", SimpleNamespace(get_db=lambda name: SimpleNamespace(dialect="impala", max_rows=100)))
    monkeypatch.setattr(nodes, "get_scope_tables", lambda db_name: (ALLOWED, OUTSIDE_TABLES))
    monkeypatch.setattr(nodes, "get_excluded_columns", lambda db_name: {})
    monkeypatch.setattr(nodes, "find_unknown_values", lambda *args: [])
    monkeypatch.setattr(nodes, "get_run_query_tool", lambda db_name: SimpleNamespace(name="sql_db_query"))
    monkeypatch.setattr(nodes, "_run_with_reconnect", lambda db_name, query, max_rows: runs.append(query) or
                        SimpleNamespace(content="[(1,)]", rows=[[1]], truncated=False, columns=["n"]))
    return runs


def test_a_query_reading_a_table_out_of_the_scope_is_never_run(validator):
    state  = {"generated_query": "SELECT COUNT(*) FROM person", "selected_db": "shop"}
    output = nodes.sql_validator(state)
    assert validator == []
    assert output["validation_check"] == "scope" and output["validation_error"] == "Tables hors du périmètre de l'agent : person"
    assert "person" in output["validation_feedback"] and "-- Not available (absent)" in output["validation_feedback"]
    assert nodes.route_after_validation({**state, **output, "remaining_steps": 10}) == "error"

    # Read again after the rewrite: the question gets the out-of-scope answer
    output = nodes.sql_validator({**state, **output})
    assert validator == []
    assert output["generated_query"] == "" and output["unavailable_reason"] == "absent"

    output = nodes.sql_validator({"generated_query": "SELECT COUNT(*) FROM invoice", "selected_db": "shop"})
    assert validator == ["SELECT COUNT(*) FROM invoice"] and output["validation_check"] == ""


# =============================================================================
# The Périmètre screen
# =============================================================================

def test_a_change_of_the_scope():
    # No scope chosen yet: every table of the database, one taken out keeps the others
    assert change_scope(None, TABLES, [], ["person"]) == ["customer", "invoice", "track"]
    assert change_scope(["invoice"], TABLES, ["track", "customer"], []) == ["customer", "invoice", "track"]
    # A table the database no longer has can still be taken out
    assert change_scope(["invoice", "legacy"], TABLES, [], ["legacy"]) == ["invoice"]
    with pytest.raises(ValueError, match="inconnue"):
        change_scope(["invoice"], TABLES, ["legacy"], [])
    with pytest.raises(ValueError, match="à la fois"):
        change_scope(["invoice"], TABLES, ["track"], ["track"])


def test_every_table_of_the_database_against_the_scope_and_the_index():
    rows = scope_rows(TABLES, ["invoice", "customer", "legacy"], {"invoice": CHANGED, "customer": FRESH},
                      indexed={"invoice", "customer", "person", "old"}, profiled={"person"}, described={"invoice"},
                      columns={"invoice": 2, "customer": 2})
    states = {row["name"]: row["state"] for row in rows}
    assert states == {"customer": FRESH, "invoice": CHANGED, "legacy": GONE, "old": GONE, "person": LEAVING, "track": OUTSIDE}
    person = next(row for row in rows if row["name"] == "person")
    assert person == {"name": "person", "state": LEAVING, "in_scope": False, "in_base": True, "indexed": True,
                      "profiled": True, "described": False, "columns": None}
    view = scope_view(rows, TABLES, "list", 0)
    assert view["counts"] == {"base": 4, "scope": 2, "indexed": 4, FRESH: 1, CHANGED: 1, GONE: 2, LEAVING: 1, OUTSIDE: 1}

    # No scope: every table of the database is in it; the database not read: the tables of the scope and the index
    assert {row["name"]: row["state"] for row in scope_rows(TABLES, None, {"invoice": MISSING}, set(), set(), set(), {})} \
        == {"customer": UNKNOWN, "invoice": MISSING, "person": UNKNOWN, "track": UNKNOWN}
    rows = scope_rows(None, ["invoice"], {"invoice": FRESH}, {"invoice", "person"}, set(), set(), {})
    assert [(row["name"], row["state"], row["in_base"]) for row in rows] == [("invoice", FRESH, None), ("person", LEAVING, None)]
    assert scope_view(rows, None, "list", 0)["counts"]["base"] is None


@pytest.fixture
def client(layer, sqlite_shop, monkeypatch):
    """The studio on the SQLite database, the config of "shop" one object for the studio and lib/semantics."""
    monkeypatch.setattr(lib.semantics, "cfg", SimpleNamespace(get_db=lambda name: sqlite_shop))
    monkeypatch.setattr(studio_app, "_db_config", lambda db: sqlite_shop)
    monkeypatch.setattr(studio_app, "_listings", {})
    monkeypatch.setattr(studio_app, "_databases", {})
    monkeypatch.setattr(studio_app, "read_trace", lambda db: None)
    return TestClient(studio_app.app)


def test_the_scope_chosen_in_the_studio(client, sqlite_shop):
    sqlite_shop.include_tables = ["invoice", "customer"]
    view = client.get("/api/db/shop/scope").json()
    assert view["origin"] == "config" and view["config_tables"] == 2 and view["version"] == "0"
    assert client.get("/api/db/shop/index").json()["scope"] == {"tables": 2, "base": 4, "origin": "config", "version": "0"}
    assert view["counts"]["base"] == 4 and view["counts"]["scope"] == 2 and view["counts"][OUTSIDE] == 2
    assert client.get("/api/db/shop/tables").json()["summary"]["tables"] == 2

    # Saved as it is (the list of the config written to scope.yaml), then a table added: the other tabs see it at once
    view = client.post("/api/db/shop/scope", json={"version": "0"}).json()
    assert view["origin"] == "list" and load_scope("shop") == ["customer", "invoice"]
    view = client.post("/api/db/shop/scope", json={"add": ["track"], "version": view["version"]}).json()
    assert load_scope("shop") == ["customer", "invoice", "track"] and view["counts"]["scope"] == 3
    assert [t["name"] for t in client.get("/api/db/shop/tables").json()["tables"]] == ["customer", "invoice", "track"]
    assert client.get("/api/db/shop/tables/track").json()["columns"] == 2

    # A page opened before the change, a table the database doesn't have: nothing written
    assert client.post("/api/db/shop/scope", json={"remove": ["track"], "version": "0"}).status_code == 409
    response = client.post("/api/db/shop/scope", json={"add": ["legacy"], "version": view["version"]})
    assert response.status_code == 400 and "legacy" in response.json()["detail"]
    assert load_scope("shop") == ["customer", "invoice", "track"]

    # Every table, the ones to come too; then one taken out: a list again, of the others
    view = client.post("/api/db/shop/scope", json={"all": True, "version": view["version"]}).json()
    assert view["origin"] == "all" and load_scope("shop") is None and view["counts"]["scope"] == 4
    assert client.get("/api/db/shop/tables").json()["summary"]["tables"] == 4
    assert client.post("/api/db/shop/scope", json={"all": True, "add": ["track"], "version": view["version"]}).status_code == 400
    view = client.post("/api/db/shop/scope", json={"remove": ["person"], "version": view["version"]}).json()
    assert view["origin"] == "list" and load_scope("shop") == ["customer", "invoice", "track"]
