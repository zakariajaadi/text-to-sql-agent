"""Metrics screen of the studio (offline/studio/metrics.py): checks without the database, tests on it, edits."""

from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

import lib.metrics
import lib.semantics
from offline.semantic_gen import utils
from offline.studio import app as studio_app
from offline.studio.metrics import (
    date_columns,
    edit_metric,
    ignore_metric,
    last_run,
    metric_row,
    metric_state,
    measure_queries,
    read_measures,
)

COLUMNS = {
    "invoice":     ["invoiceid", "customerid", "invoicedate", "total", "status"],
    "invoiceline": ["invoicelineid", "invoiceid", "unitprice", "quantity"],
    "customer":    ["customerid", "email", "since"],
}
TYPES = {
    "invoice.invoiceid": "INTEGER", "invoice.customerid": "INTEGER", "invoice.invoicedate": "DATE",
    "invoice.total": "NUMERIC", "invoice.status": "VARCHAR",
    "invoiceline.invoicelineid": "INTEGER", "invoiceline.invoiceid": "INTEGER", "invoiceline.unitprice": "NUMERIC",
    "invoiceline.quantity": "INTEGER",
    "customer.customerid": "INTEGER", "customer.email": "VARCHAR", "customer.since": "VARCHAR",
}
RELATIONS = [("invoiceline.invoiceid", "invoice.invoiceid"), ("invoice.customerid", "customer.customerid")]
FORMATS   = {"customer": {"since": "date dd/MM/yyyy"}}
METRICS = {
    "revenue":       {"description": "Invoiced", "expr": "SUM(invoiceline.unitprice * invoiceline.quantity)",
                      "time_dimension": "invoice.invoicedate"},
    "invoice_count": {"description": "Invoices", "expr": "COUNT(DISTINCT invoice.invoiceid)"},
    "billed":        {"description": "Billed, counted on the lines", "expr": "SUM(invoice.total) / COUNT(DISTINCT invoiceline.trackid)"},
}


def row(name: str, entry: dict, entries: dict | None = None, hidden: dict | None = None, ignored=()) -> dict:
    return metric_row(name, entry, entries or {name: entry}, list(ignored), COLUMNS, hidden or {}, RELATIONS, TYPES,
                      FORMATS, "sqlite")


def test_a_metric_checked_without_the_database():
    revenue = row("revenue", METRICS["revenue"])
    assert revenue["error"] is None and revenue["rule"] is None and revenue["joins"] == 1
    assert revenue["sql"] == ("SELECT SUM(invoiceline.unitprice * invoiceline.quantity)\n"
                              "FROM invoiceline\nJOIN invoice ON invoiceline.invoiceid = invoice.invoiceid")
    assert revenue["dates"] == ["invoice.invoicedate", "customer.since"]   # a text read as a date, one join away

    # The rules of the generator, on a metric already written
    assert row("invoice_count", METRICS["invoice_count"])["rule"] == "count"
    twin = row("sales", {"description": "Sales", "expr": "SUM(InvoiceLine.unitprice * `invoiceline`.quantity)"},
               {"revenue": METRICS["revenue"], "sales": {}})
    assert (twin["rule"], twin["rule_with"]) == ("same", "revenue")
    # the same columns of another table: another formula
    other = row("track_value", {"description": "Value", "expr": "SUM(track.unitprice * track.milliseconds)"},
                {"revenue": METRICS["revenue"], "track_value": {}})
    assert (other["rule"], other["rule_with"]) == (None, None)
    assert row("revenue", METRICS["revenue"], ignored=["revenue"])["rule"] == "known"


def test_a_metric_the_agent_would_refuse():
    # A hidden column stops the agent at startup: it is out of the documents the metrics are checked against
    hidden = row("revenue", METRICS["revenue"], hidden={"invoiceline": {"unitprice"}})
    assert hidden["error"] == {"kind": "hidden", "detail": "invoiceline.unitprice"}
    assert row("x", {"description": "x", "expr": "SUM(invoice.amount)"})["error"]["kind"] == "invalid"
    lonely = metric_row("x", {"description": "x", "expr": "SUM(invoice.total) + SUM(invoiceline.quantity)"},
                        {}, [], COLUMNS, {}, [], TYPES, {}, "sqlite")
    assert lonely["error"]["kind"] == "joins" and lonely["sql"] is None
    assert metric_state(lonely, None) == "error" and metric_state(row("r", METRICS["revenue"]), None) == "untested"


@pytest.fixture
def engine():
    """Five invoices, one with no date, whose lines (one orphan) the joins count."""
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE invoice (invoiceid INTEGER, customerid INTEGER, invoicedate DATE, total NUMERIC, status VARCHAR)"))
        conn.execute(text("CREATE TABLE invoiceline (invoicelineid INTEGER, invoiceid INTEGER, unitprice NUMERIC, quantity INTEGER, trackid INTEGER)"))
        conn.execute(text("CREATE TABLE customer (customerid INTEGER, email VARCHAR, since VARCHAR)"))
        conn.execute(text("INSERT INTO invoice VALUES (1, 1, '2024-01-02', 10, 'paid'), (2, 1, '2024-02-03', 20, 'paid'), "
                          "(3, 2, NULL, 5, 'paid'), (4, 2, '2024-03-01', 7, 'paid'), (5, 3, '2024-03-09', 4, 'open')"))
        conn.execute(text("INSERT INTO invoiceline VALUES (1, 1, 1, 2, 1), (2, 1, 2, 1, 2), (3, 2, 5, 1, 3), (4, 3, NULL, 1, 4), "
                          "(5, 4, 1, 1, 5), (6, 5, 1, 4, 6), (7, 9, 3, 1, 7)"))
    return engine


def run(engine, metric: dict, name: str = "m") -> dict:
    """The test of the studio, on SQLite."""
    columns = {**COLUMNS, "invoiceline": [*COLUMNS["invoiceline"], "trackid"]}
    checked = metric_row(name, metric, {name: metric}, [], columns, {}, RELATIONS, TYPES, FORMATS, "sqlite")
    plan = measure_queries(checked, TYPES, RELATIONS, "sqlite")
    with engine.connect() as conn:
        main = tuple(conn.execute(text(plan["main"])).fetchone())
        rows = {table: conn.execute(text(sql)).scalar() for table, sql in plan["tables"].items()}
    return read_measures(plan, main, rows, 0.1)


def test_a_test_counts_rows_never_values(engine):
    revenue = run(engine, METRICS["revenue"])
    # The orphan line is lost by the join, the line with no price leaves the measure empty, one invoice has no date
    assert revenue["rows"] == 6 and revenue["tables"] == [{"table": "invoiceline", "rows": 7, "counted": 6, "warn": True}]
    assert revenue["measures"][0]["filled"] == 5 and revenue["measures"][0]["warn"]     # 5/6 under 90 %
    assert revenue["date"]["filled"] == 5 and revenue["sign"] == "positive" and revenue["warn"]

    # A total of the invoices joined to their lines: each invoice counted once per line
    billed = run(engine, METRICS["billed"])
    assert billed["tables"] == [{"table": "invoice", "rows": 5, "counted": 6, "warn": True}]

    # The amount of each line, priced from its invoice: the invoices are a lookup, only the lines are summed
    priced = run(engine, {"description": "Priced", "expr": "SUM(invoiceline.quantity * invoice.total)"})
    assert [t["table"] for t in priced["tables"]] == ["invoiceline"]

    # A count by itself is not checked for joins; a condition that finds no row gives a zero result
    assert run(engine, METRICS["invoice_count"])["tables"] == []
    late = run(engine, {"description": "Late", "expr": "SUM(CASE WHEN invoice.status = 'late' THEN invoice.total ELSE 0 END)"})
    assert late["sign"] == "zero" and late["warn"]
    assert not run(engine, {"description": "Open", "expr": "SUM(CASE WHEN invoice.status = 'open' THEN invoice.total END)"})["warn"]


def test_date_columns_take_the_current_one_first():
    assert date_columns(["invoice"], RELATIONS, TYPES, FORMATS, "invoice.custom") == ["invoice.custom", "invoice.invoicedate", "customer.since"]


def test_the_last_run_of_the_generator():
    lines = [{"message": m} for m in [
        "[Chinook] Skipped metric 'invoices': a plain count of one column holds no business rule: the agent counts entities by itself",
        "[Chinook] Skipped metric 'sales': same expression as an existing metric",
        "[Chinook] Rejected metric 'margin': metric 'margin' — unknown column 'invoiceline.cost'",
        "[Chinook] 2 new metric(s) added to metrics.yaml: revenue, tracks_sold — review them.",
    ]]
    run = last_run(lines)
    assert [(s["name"], s["reason"]) for s in run["skipped"]] == [("invoices", "count"), ("sales", "same"), ("margin", "rejected")]
    assert run["added"] == ["revenue", "tracks_sold"]
    assert last_run([{"message": "[Chinook] 0 new metric(s) added to metrics.yaml: none — review them."}])["added"] == []


def test_edit_and_ignore_a_metric():
    edited = edit_metric(METRICS, "revenue", "synonyms", "sales, chiffre d'affaires, ")
    assert edited["revenue"]["synonyms"] == ["sales", "chiffre d'affaires"] and list(edited) == list(METRICS)
    assert "time_dimension" not in edit_metric(METRICS, "revenue", "time_dimension", "")["revenue"]
    with pytest.raises(ValueError):
        edit_metric(METRICS, "revenue", "description", "  ")

    # a field corrected is noted in `edited`, in the order of the block; a value left as it was is not
    assert edited["revenue"]["edited"] == ["synonyms"]
    again = edit_metric(edited, "revenue", "description", "Amount invoiced, every invoice line.")
    assert again["revenue"]["edited"] == ["description", "synonyms"]
    same = edit_metric(METRICS, "revenue", "expr", METRICS["revenue"]["expr"])
    assert "edited" not in same["revenue"]

    entries, ignored = ignore_metric(METRICS, [], "invoice_count")
    assert "invoice_count" not in entries and ignored == ["invoice_count"]
    assert ignore_metric(entries, ignored, "invoice_count", ignore=False) == (entries, [])
    with pytest.raises(ValueError):
        ignore_metric(entries, ignored, "unknown")


# ------------------------------------------------------------
# The routes: the screen, a test, an edit checked by the database, ignore
# ------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch, engine):
    for module in (lib.semantics, lib.metrics, studio_app):
        monkeypatch.setattr(module, "SEMANTIC_LAYER_DIR", tmp_path)
    monkeypatch.setattr(studio_app, "_db_config", lambda db: SimpleNamespace(exclude_personal_data=False, dialect="sqlite"))
    columns = {table: [{"name": c, "type": TYPES.get(f"{table}.{c}", "INTEGER")} for c in names]
               for table, names in {**COLUMNS, "invoiceline": [*COLUMNS["invoiceline"], "trackid"]}.items()}
    monkeypatch.setattr(studio_app, "_listing", lambda db, refresh=False: {
        "tables": list(COLUMNS), "columns": columns, "source": "database", "error": None, "read_at": 0})
    monkeypatch.setattr(studio_app, "_databases", {"shop": SimpleNamespace(_engine=engine)})
    monkeypatch.setattr(studio_app, "_tests", {})
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "relationships.yaml").write_text(yaml.safe_dump(
        {"relations": [{"from": s, "to": t} for s, t in RELATIONS]}))
    utils.write_metrics_file("shop", {k: METRICS[k] for k in ("revenue", "invoice_count")}, [])
    return TestClient(studio_app.app)


def test_a_metric_tested_edited_and_ignored_from_the_page(client, tmp_path):
    view = client.get("/api/db/shop/metrics").json()
    assert [(m["name"], m["state"]) for m in view["metrics"]] == [("revenue", "untested"), ("invoice_count", "check")]

    view = client.post("/api/db/shop/metrics/revenue/test").json()
    assert view["metrics"][0]["state"] == "check" and view["metrics"][0]["test"]["rows"] == 6

    # A new expression is checked by the database before it is written; the test of the old one is dropped
    post = lambda changes, version=view["version"]: client.post(  # noqa: E731
        "/api/db/shop/metrics/revenue", json={"changes": changes, "version": version})
    assert post({"expr": "SUM(invoiceline.price)"}).status_code == 400
    # a draft is written whole or not at all: its description stays out while its expression is refused
    assert post({"description": "Sold", "expr": "SUM(invoiceline.price)"}).status_code == 400
    assert yaml.safe_load((tmp_path / "shop" / "metrics.yaml").read_text())["revenue"]["description"] != "Sold"
    assert post({"expr": "SUM(invoiceline.unitprice * invoiceline.quantity)"}, "1").status_code == 409
    view = post({"description": "Units sold", "expr": "SUM(invoiceline.quantity)"}).json()
    saved = view["metrics"][0]
    assert saved["expr"] == "SUM(invoiceline.quantity)" and saved["description"] == "Units sold" and saved["test"] is None
    assert saved["edited"] == ["description", "expr"]

    view = client.post("/api/db/shop/metrics/invoice_count/ignore", json={"version": view["version"]}).json()
    assert [m["name"] for m in view["metrics"]] == ["revenue"] and view["ignored"] == ["invoice_count"]
    saved = yaml.safe_load((tmp_path / "shop" / "metrics.yaml").read_text())
    assert saved["revenue"]["expr"] == "SUM(invoiceline.quantity)" and saved["ignore"] == ["invoice_count"]
