"""Index screen of the studio (offline/studio/index.py): the trace of the index against the files today, the metrics
checked against the documents, the progress of an index and of an eval, the routes; and the home, which shows
every database the same way and searches through them (offline/studio/home.py)."""

import json
from datetime import date, datetime
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient

import lib.metrics
import lib.semantics
from offline.eval.history import history_path, read_history, record
from offline.indexer import trace as trace_module
from offline.semantic_gen import utils
from offline.studio import app as studio_app
from offline.studio.home import search_layer
from offline.studio.index import (
    CHANGED,
    FRESH,
    MISSING,
    UNKNOWN,
    changes,
    document_text,
    document_view,
    eval_run,
    index_view,
    metric_errors,
    reading_order,
)
from offline.studio.jobs import Job

INVOICE = {
    "description": "One row per purchase.",
    "data_as_of": date(2024, 7, 10),
    "columns": {"invoiceid": "Invoice id.", "total": "Amount.", "city": "Billing city.", "email": "Email."},
    "formats": {"total": "decimal"},
    "exclude_columns": ["email"],
    "categorical_columns": ["city", "country"],
}
RELATIONS = [("invoice.customerid", "customer.customerid"), ("album.artistid", "artist.artistid")]
DB_COLUMNS = {
    "invoice":  ["invoiceid", "customerid", "total", "city", "country", "email"],
    "customer": ["customerid", "name"],
}


def make_trace(semantics: dict | None = None) -> dict:
    """The trace an index writes, read back from its JSON (a date becomes a string)."""
    trace = {
        "built_at": 1000.0, "sample_rows": 3,
        "tables": {
            "invoice": {
                "semantics": semantics or INVOICE,
                "relations": [["invoice.customerid", "customer.customerid"]],
                "columns": {"invoiceid": "INT", "customerid": "INT", "total": "STRING", "city": "STRING", "country": "STRING"},
                "chars": 900,
                "values": {"city": {"count": 53, "inline": False}},   # country: no value read
            },
            "customer": {"semantics": None, "relations": [], "columns": {"customerid": "INT", "name": "STRING"},
                         "chars": 120, "values": {}},
            "legacy": {"semantics": None, "relations": [], "columns": {"id": "INT"}, "chars": 40, "values": {}},
        },
    }
    return json.loads(json.dumps(trace, default=str))


# ------------------------------------------------------------
# The trace of an index
# ------------------------------------------------------------

def test_the_trace_is_written_and_read_back(tmp_path, monkeypatch):
    monkeypatch.setattr(trace_module, "trace_path", lambda db: tmp_path / f"{db}.index.json")
    assert trace_module.read_trace("shop") is None
    trace_module.write_trace("shop", {"built_at": 1.0, "tables": {"invoice": {"semantics": INVOICE}}})
    assert trace_module.read_trace("shop")["tables"]["invoice"]["semantics"]["data_as_of"] == "2024-07-10"
    assert [p.name for p in tmp_path.iterdir()] == ["shop.index.json"]   # no temporary file left
    # the next index keeps the date of this one, then of both
    trace_module.write_trace("shop", {"built_at": 2.0, "tables": {}})
    trace_module.write_trace("shop", {"built_at": 3.0, "tables": {}})
    assert trace_module.read_trace("shop")["builds"] == [1.0, 2.0]


# ------------------------------------------------------------
# What changed in a document since the index
# ------------------------------------------------------------

def test_nothing_changed_when_the_files_are_the_ones_indexed():
    # A date read by YAML against its string in the trace; the relations in another order
    assert changes(make_trace()["tables"]["invoice"], INVOICE, [["invoice.customerid", "customer.customerid"]],
                   DB_COLUMNS["invoice"]) == []


def test_each_change_of_the_semantic_layer_is_named():
    indexed = make_trace()["tables"]["invoice"]
    entry = {**INVOICE, "description": "Purchases.", "columns": {**INVOICE["columns"], "total": "Amount with taxes."},
             "exclude_columns": ["email", "city"], "categorical_columns": ["city", "country", "total"],
             "keep_as_text": ["invoiceid"]}   # an exception to the profile: not in the document
    relations = [["invoice.customerid", "customer.customerid"], ["invoiceline.invoiceid", "invoice.invoiceid"]]
    assert changes(indexed, entry, relations, DB_COLUMNS["invoice"]) == [
        {"kind": "description"},
        {"kind": "columns", "columns": ["total"]},   # city, hidden now, is not counted twice
        {"kind": "hidden", "columns": ["city"]},
        {"kind": "categorical", "added": ["total"], "removed": ["city"]},
        {"kind": "relations", "added": [["invoiceline.invoiceid", "invoice.invoiceid"]], "removed": []},
    ]


def test_the_columns_of_the_database_against_the_document():
    indexed = make_trace()["tables"]["invoice"]
    # A column added to the table (email stays hidden), one the document lists is gone
    db_columns = ["invoiceid", "customerid", "total", "city", "email", "channel"]
    assert changes(indexed, INVOICE, [["invoice.customerid", "customer.customerid"]], db_columns) == [
        {"kind": "db_columns", "added": ["channel"], "removed": ["country"]}]
    # A raw DDL described since: its relations didn't count, now they do
    raw = make_trace()["tables"]["customer"]
    assert changes(raw, {"description": "Customers."}, [], ["customerid", "name"]) == [{"kind": "described"}]


# ------------------------------------------------------------
# The screen
# ------------------------------------------------------------

def test_the_index_of_a_database_against_its_files():
    semantics = {"invoice": {**INVOICE, "data_as_of": "2024-08-01"}}
    metrics = {"revenue": {"description": "Invoiced", "expr": "SUM(invoice.total)"},
               "mails": {"description": "Mails", "expr": "COUNT(DISTINCT invoice.email)"}}
    view = index_view(["invoice", "customer", "track"], DB_COLUMNS, semantics, RELATIONS, make_trace(), metrics,
                      "hive", None, base=["customer", "invoice", "track"])
    assert [(r["name"], r["status"]) for r in view["tables"]] == [("invoice", CHANGED), ("customer", FRESH), ("track", MISSING)]
    assert view["tables"][0]["changes"] == [{"kind": "data_as_of"}]
    assert view["gone"] == ["legacy"] and view["unscoped"] == [] and view["raw"] == ["customer", "legacy"] and view["documents"] == 3
    # One join (written in the trace of each table it links), email hidden from the 9 columns
    assert view["joins"] == 1 and view["shown"] == 8 and view["hidden"] == ["invoice.email"]
    assert view["searched"] == {"columns": 1, "values": 53} and view["inline"] == {"columns": 0, "values": 0}
    assert view["dropped"] == ["invoice.country"]
    # email is hidden: not in the document, the agent would refuse the metric at startup
    assert view["metrics"] == {"total": 2, "errors": [{"name": "mails", "detail": "unknown column 'invoice.email'"}]}

    # legacy still in the database, out of the scope; or the database not read: its document leaves the index all the same
    for base in (["customer", "invoice", "legacy", "track"], None):
        view = index_view(["invoice", "customer", "track"], DB_COLUMNS, semantics, RELATIONS, make_trace(), metrics,
                          "hive", None, base=base)
        assert view["gone"] == [] and view["unscoped"] == ["legacy"]

    # Never indexed since the traces exist: nothing is known of the tables
    view = index_view(["invoice"], DB_COLUMNS, semantics, RELATIONS, None, metrics, "hive", None)
    assert view["trace"] is False and view["tables"][0]["status"] == UNKNOWN and "metrics" not in view


def test_the_metrics_are_checked_against_the_documents_as_the_agent_does():
    trace = make_trace()
    assert metric_errors({"ok": {"description": "d", "expr": "SUM(invoice.total)", "time_dimension": "invoice.total"}},
                         trace, "hive") == []
    assert metric_errors({"x": {"description": "d", "expr": "SUM(track.milliseconds)"}}, trace, "hive") == [
        {"name": "x", "detail": "unknown table 'track'"}]


def test_the_document_of_a_table_column_by_column():
    db_columns = [{"name": c, "type": "VARCHAR"} for c in ["invoiceid", "customerid", "total", "city", "email", "channel"]]
    view = document_view("invoice", make_trace(), INVOICE, RELATIONS, db_columns)
    assert view["status"] == CHANGED and view["joins"] == 1 and view["data_as_of"] == "2024-07-10"
    lines = {c["column"]: c for c in view["columns"]}
    # the type of the database, as the other screens show it
    assert lines["total"]["format"] == "decimal" and lines["total"]["type"] == "VARCHAR"
    assert lines["city"]["values"] == {"count": 53, "inline": False}
    assert lines["country"]["values"] == {"dropped": True} and lines["country"]["gone"] is True
    assert lines["email"]["hidden"] is True and lines["channel"]["added"] is True
    assert document_view("track", make_trace(), None, RELATIONS, None)["status"] == MISSING


def test_the_columns_leaving_and_entering_the_document_at_the_next_index():
    db_columns = [{"name": c, "type": "STRING"} for c in ["invoiceid", "customerid", "total", "city", "country", "email"]]
    entry = {**INVOICE, "exclude_columns": ["city"]}   # city hidden since the index, email shown since
    lines = {c["column"]: c for c in document_view("invoice", make_trace(), entry, RELATIONS, db_columns)["columns"]}
    assert lines["city"]["leaving"] is True and lines["total"]["leaving"] is False
    assert (lines["email"]["hidden"], lines["email"]["entering"], lines["email"]["added"]) == (True, True, False)
    # the files as indexed: nothing leaves, nothing enters
    lines = document_view("invoice", make_trace(), INVOICE, RELATIONS, db_columns)["columns"]
    assert not any(c.get("leaving") or c.get("entering") for c in lines)


def test_the_last_eval_of_a_database():
    results = {"meta": {"run_at": "2026-10-22T20:00"},
               "results": [{"exec_exact": 1.0}, {"exec_exact": 0.0}, {"exec_exact": None}, {"exec_exact": 1.0}]}
    run = eval_run(results, 5.0)
    assert (run["questions"], run["scored"], run["exact"]) == (4, 3, 2) and run["at"] > 5.0
    assert eval_run({"results": []}, 5.0)["at"] == 5.0 and eval_run(None, None) is None
    assert run["results"][1] == {"q": "", "d": None, "x": 0, "latency": None, "tables": []} and run["results"][2]["x"] is None
    assert run["history"] == [{"at": run["at"], "scored": 3, "exact": 2}]   # no history yet: the run alone

    # The history: its last line is this run (dated like the panel says it), else this run follows it
    history = [{"at": "2026-10-01T10:00", "scored": 3, "exact": 1}, {"at": "2026-10-22T19:43", "scored": 3, "exact": 2}]
    points = eval_run(results, 5.0, history)["history"]
    assert [(p["scored"], p["exact"]) for p in points] == [(3, 1), (3, 2)] and points[-1]["at"] == run["at"]
    points = eval_run(results, 5.0, history[:1])["history"]
    assert [(p["scored"], p["exact"]) for p in points] == [(3, 1), (3, 2)]

    # Questions asked again, or the queries scored again, after the run: the panel and the last point say the day of
    # that last measure, while whether the eval measures the index stays the run's date
    assert run["measured_at"] == run["at"]
    meta = {**results["meta"], "rescored_at": "2026-10-25T09:00", "partial_reruns": [{"at": "2026-10-28T10:00", "questions": ["q"]}]}
    history = [{"at": "2026-10-22T20:00", "scored": 3, "exact": 2}, {"at": "2026-10-28T10:00", "scored": 3, "exact": 2}]
    later = eval_run({**results, "meta": meta}, 5.0, history)
    assert later["at"] == run["at"] and later["measured_at"] == datetime(2026, 10, 28, 10).timestamp()
    assert [p["at"] for p in later["history"]] == [run["at"], later["measured_at"]]


def test_the_history_of_the_evals(tmp_path):
    results = tmp_path / "shop.json"
    assert read_history(results) == [] and history_path(results).name == "shop.history.jsonl"
    record(results, [{"exec_exact": 1.0}, {"exec_exact": None}], "run", at="2026-10-22T20:00", commit="abc")
    record(results, [{"exec_exact": 1.0}, {"exec_exact": 0.0}], "partial", at="2026-10-21T09:00")
    history_path(results).open("a").write("not json\n")   # a broken line is skipped
    assert read_history(results) == [
        {"at": "2026-10-21T09:00", "kind": "partial", "commit": None, "questions": 2, "scored": 2, "exact": 1},
        {"at": "2026-10-22T20:00", "kind": "run", "commit": "abc", "questions": 2, "scored": 1, "exact": 1}]


def test_the_tables_in_one_reading_order():
    # the most joined table in the middle of its group, the tables it reaches by turns on its right and its left, the
    # largest group first, the ones without a join last
    pairs = [("invoice", "customer"), ("invoiceline", "invoice"), ("invoiceline", "track"), ("album", "artist"), ("employee", "employee")]
    tables = ["album", "artist", "customer", "employee", "genre", "invoice", "invoiceline", "track"]
    assert reading_order(tables, pairs) == ["customer", "invoice", "invoiceline", "track", "album", "artist", "employee", "genre"]


def test_what_the_cards_of_the_overview_draw():
    metrics = {"revenue": {"description": "d", "expr": "SUM(invoice.total)"},
               "mails": {"description": "d", "expr": "COUNT(DISTINCT invoice.email)"}}
    view = index_view(["invoice", "customer"], DB_COLUMNS, {"invoice": INVOICE}, RELATIONS, make_trace(), metrics, "hive", None,
                      sizes={"invoice": 412, "customer": 59})
    cards = view["cards"]
    assert [t["name"] for t in cards["tables"]] == ["customer", "invoice", "legacy"] and cards["joins"] == [["invoice", "customer"]]
    invoice = cards["tables"][1]   # customer first (two tables joined once: the first by name), invoice on its right
    # its rows, a kind per column shown, email hidden, city searched and country without values
    assert (invoice["rows"], invoice["kinds"], invoice["hidden"]) == (412, "nnsss", 1)
    assert invoice["values"] == [["city", 53, "searched"], ["country", 0, "dropped"]]
    assert cards["tables"][2]["rows"] is None   # not profiled
    assert cards["metrics"] == [{"name": "revenue", "tables": ["invoice"], "refused": False},
                                {"name": "mails", "tables": [], "refused": True}]


# ------------------------------------------------------------
# The progress of an index and of an eval, read from their lines
# ------------------------------------------------------------

def line(message: str) -> str:
    return f"2026-10-24 11:02:03 | INFO     | offline.indexer.schema_indexer:run:12 - {message}"


def test_the_progress_of_an_index():
    job = Job(id=1, db="Chinook", step="index", tables=None, command=["python"])
    job.add_line(line("[Chinook] Fetching values: album (0 column(s))"))
    job.add_line(line("[Chinook] Fetching values: customer (3 column(s))"))
    assert job.current == "customer" and job.done == ["album"]
    job.add_line(line("[Chinook] Built: album"))
    assert (job.current, job.phase, job.done) == (None, "documents", ["album", "customer"])
    job.add_line(line("[Chinook] Indexed 14 schemas to Chroma."))
    assert job.phase == "values"


def test_the_progress_of_an_eval():
    job = Job(id=1, db="Chinook", step="eval", tables=None, command=["python"])
    job.add_line(line("[3/23] How many customers per country?"))
    assert job.progress == [3, 23] and job.to_dict()["progress"] == [3, 23]


# ------------------------------------------------------------
# The routes
# ------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    for module in (lib.semantics, lib.metrics, studio_app):
        monkeypatch.setattr(module, "SEMANTIC_LAYER_DIR", tmp_path)
    monkeypatch.setattr(studio_app, "_db_config", lambda db: SimpleNamespace(
        exclude_personal_data=False, dialect="hive", sample_rows=0))
    columns = {table: [{"name": c, "type": "STRING"} for c in names] for table, names in DB_COLUMNS.items()}
    monkeypatch.setattr(studio_app, "_listing", lambda db, refresh=False: {
        "tables": list(DB_COLUMNS), "base": list(DB_COLUMNS), "columns": columns, "source": "database", "error": None,
        "read_at": 0})
    monkeypatch.setattr(studio_app, "read_trace", lambda db: make_trace())
    monkeypatch.setattr(studio_app, "value_index_path", lambda db: tmp_path / "none.sqlite")
    monkeypatch.setattr(studio_app, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(studio_app, "DATASETS_DIR", tmp_path / "datasets")
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "tables.yaml").write_text(yaml.safe_dump({"invoice": INVOICE}))
    utils.write_metrics_file("shop", {"revenue": {"description": "Invoiced", "expr": "SUM(invoice.total)"}}, [])
    return TestClient(studio_app.app)


def test_the_index_screen_and_a_document(client, tmp_path):
    view = client.get("/api/db/shop/index").json()
    assert [(r["name"], r["status"]) for r in view["tables"]] == [("invoice", CHANGED), ("customer", FRESH)]
    assert view["sample_rows"] == 3 and view["sample_rows_now"] == 0 and view["dataset"] is False
    assert view["eval"] is None and view["values_at"] is None and "tables.yaml" in view["files"]
    # The tile of the Périmètre: no scope chosen, every table of the database
    assert view["scope"] == {"tables": 2, "base": 2, "origin": "none", "version": "0"}

    document = client.get("/api/db/shop/index/legacy").json()
    assert document["gone"] is True and document["unscoped"] is False and document["chars"] == 40
    assert client.get("/api/db/shop/index/nothing").status_code == 404

    # No questions: no eval to run, no dashboard to open
    assert view["questions"] is None
    assert client.post("/api/db/shop/eval/run").status_code == 400
    assert client.get("/eval/shop").status_code == 404


def test_the_questions_of_the_next_eval(client, tmp_path):
    (tmp_path / "datasets").mkdir()
    dataset = tmp_path / "datasets" / "shop.yaml"
    dataset.write_text(yaml.safe_dump([{"question": "a"}, {"question": "b"}]))
    assert client.get("/api/db/shop/index").json()["questions"] == 2
    dataset.write_text(yaml.safe_dump({"questions": [{"question": "a"}]}))
    assert client.get("/api/db/shop/index").json()["questions"] == 1
    # a file the eval can't read either: no count
    dataset.write_text("questions: [a")
    view = client.get("/api/db/shop/index").json()
    assert view["questions"] is None and view["dataset"] is True


# ------------------------------------------------------------
# The home
# ------------------------------------------------------------

def test_the_search_finds_the_names_first_whatever_the_accents_and_case():
    semantics = {"invoice":  {**INVOICE, "columns": {**INVOICE["columns"], "total": "Montant de la facture."}},
                 "factures": {"description": "Une ligne par facture."}}
    found = search_layer("FACTURÉ", {"shop": (semantics, make_trace())})
    # The table named so first (its description says it too: one result), then the column whose description says it
    assert [(r["table"], r["column"]) for r in found["results"]] == [("factures", None), ("invoice", "total")]
    assert found["total"] == 2 and found["bases"] == 1
    assert search_layer("factur", {"shop": (semantics, make_trace())}, limit=1)["results"] == found["results"][:1]
    assert search_layer("  ", {"shop": (semantics, make_trace())}) == {"total": 0, "bases": 0, "results": []}


def test_the_home_asks_no_database(client, monkeypatch):
    monkeypatch.setattr(studio_app, "cfg", SimpleNamespace(databases=[
        SimpleNamespace(name="shop", dialect="hive", exclude_personal_data=False)]))
    monkeypatch.setattr(studio_app, "_listing", lambda db, refresh=False: pytest.fail("the home reads the files only"))
    [base] = client.get("/api/overview").json()
    assert base["name"] == "shop" and base["dialect"] == "hive" and base["source"] == "files"
    # The tables the files and the last index know: none is gone, the database was not read
    assert base["summary"]["tables"] == 3 and base["summary"]["described"] == 1 and base["summary"]["isolated"] == 3
    assert base["index"]["gone"] == [] and base["index"]["joins"] == 1 and base["index"]["hidden"] == ["invoice.email"]
    assert base["index"]["scope"] == {"tables": 3, "base": None, "origin": "none", "version": "0"}

    # The search goes through the semantic layer of every database: email is hidden from the agent
    found = client.get("/api/search", params={"q": "EMAIL"}).json()
    assert found == {"total": 1, "bases": 1, "results": [{"db": "shop", "table": "invoice", "column": "email",
                                                          "description": "Email.", "hidden": True, "categorical": False}]}


def test_the_text_of_a_document_masks_the_data():
    trace = make_trace()
    trace["tables"]["invoice"]["values"]["country"] = {"count": 12, "inline": True}
    text = document_text("invoice", trace, "hive")
    # what the agent reads, rendered by its own function
    assert text.splitlines()[:4] == ["-- Table: invoice", "-- Description: One row per purchase.", "-- Data as of: 2024-07-10",
                                     "-- Joins: invoice.customerid = customer.customerid"]
    assert "\t`total` STRING,  -- Amount. [stored as text: decimal number]" in text
    # the values listed in the document and the sample rows masked; the values searched only are not in it
    assert "\t`country` STRING  -- Values: ‹12 valeurs›" in text and "Values" not in text.split("`city`")[1].split("\n")[0]
    assert text.endswith("/*\n‹3 lignes d'exemple›\n*/")
    # a table without description: its raw DDL; a table the index does not hold: nothing
    assert document_text("customer", trace, "hive").startswith("CREATE TABLE `customer` (\n\t`customerid` INT,\n\t`name` STRING\n)")
    assert document_text("unknown", trace, "hive") is None
