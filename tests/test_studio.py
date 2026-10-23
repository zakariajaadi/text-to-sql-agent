"""Studio of the semantic layer (offline/studio): what the profile detects, table statuses, jobs, routes."""

from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient

import lib.semantics
from offline.studio import app as studio_app
from offline.studio.jobs import Job, profile_command
from offline.studio.state import (
    hide_personal,
    metric_tables,
    personal_columns,
    proposals,
    set_excluded,
    set_format,
    table_detail,
    table_rows,
)

FACTS = {
    "rows": 100,
    "keys": {"id_dossier": {"non_null": 100, "distinct": 100}},
    "text_columns": {"date_deblocage": {"filled": 0.99, "shapes": {"99/99/9999": 0.99}}},
    "formats": {"date_deblocage": "date dd/MM/yyyy", "encours": "decimal"},
    "exclude": {"commentaire": "empty", "nom_client": "personal data"},
}


def states(result: list[dict]) -> dict:
    return {(p["kind"], p["column"]): p["state"] for p in result}


# =============================================================================
# What the profile detects: applied, an exception, or not applied
# =============================================================================

def test_detections_not_applied_yet():
    result = proposals(FACTS, {"description": "Dossiers"}, personal=False)
    assert set(states(result).values()) == {"pending"}
    defaults = {p["column"]: p["default"] for p in result}
    # make profile applies the empty columns always, personal data only when the database asks for it
    assert defaults == {"date_deblocage": True, "encours": True, "commentaire": True, "nom_client": False}
    assert {p["column"]: p["default"] for p in proposals(FACTS, None, personal=True)}["nom_client"] is True


def test_detections_applied_or_made_exceptions():
    entry = {"formats": {"date_deblocage": "date dd/MM/yyyy", "encours": "date yyyy-MM-dd"},
             "exclude_columns": ["nom_client"]}
    result = proposals(FACTS, entry, personal=False)
    assert states(result) == {
        ("format", "date_deblocage"): "accepted",
        ("format", "encours"): "rejected",          # reviewed by hand to another format
        ("exclude", "commentaire"): "pending",      # neither excluded nor kept
        ("exclude", "nom_client"): "accepted",
    }
    assert next(p for p in result if p["column"] == "encours")["current"] == "date yyyy-MM-dd"
    exceptions = {"keep_as_text": ["encours"], "keep_columns": ["commentaire"]}
    assert states(proposals(FACTS, exceptions, False))[("format", "encours")] == "rejected"
    assert states(proposals(FACTS, exceptions, False))[("exclude", "commentaire")] == "rejected"


def test_a_format_kept_as_text_then_applied_again():
    entry = set_format({"formats": {"encours": "decimal", "montant": "decimal"}}, "encours", None)
    assert entry == {"formats": {"montant": "decimal"}, "keep_as_text": ["encours"]}
    assert set_format(entry, "encours", "decimal") == {"formats": {"montant": "decimal", "encours": "decimal"}}


# =============================================================================
# Personal data: hidden in the whole database, a column kept visible on purpose
# =============================================================================

PROFILE = {
    "client":  {"rows": 5, "exclude": {"nom": "personal data", "email": "personal data", "fax": "empty"}},
    "agence":  {"rows": 3, "exclude": {"nom_agence": "personal data"}},
    "produit": {"rows": 9},
}


def test_hide_personal_in_every_table_but_the_exceptions():
    semantics = {"client": {"description": "Clients", "exclude_columns": ["email"]},
                 "agence": {"keep_columns": ["nom_agence"]}}
    assert personal_columns(PROFILE, semantics) == {"client": {"nom": "visible", "email": "hidden"},
                                                    "agence": {"nom_agence": "kept"}}
    hidden = hide_personal(PROFILE, semantics, hide=True)
    assert hidden["client"] == {"description": "Clients", "exclude_columns": ["email", "nom"]}   # fax: a correction
    assert hidden["agence"] == {"keep_columns": ["nom_agence"]}
    assert "produit" not in hidden

    shown = hide_personal(PROFILE, hidden, hide=False)
    assert shown["client"]["exclude_columns"] == [] and shown["agence"] == {"keep_columns": ["nom_agence"]}


def test_a_column_kept_visible_then_hidden_again():
    entry = set_excluded({"exclude_columns": ["nom", "fax"]}, ["nom"], hide=False)
    assert entry == {"exclude_columns": ["fax"], "keep_columns": ["nom"]}
    assert set_excluded(entry, ["nom"], hide=True) == {"exclude_columns": ["fax", "nom"]}
    assert set_excluded({"description": "x"}, [], hide=True) == {"description": "x"}


# =============================================================================
# Status of the tables at each step
# =============================================================================

def test_metric_tables_reads_expressions_and_time_dimensions():
    metrics = {
        "revenue": {"expr": "SUM(Order_Details.unitprice * order_details.quantity)", "time_dimension": "orders.orderdate"},
        "ratio":   {"expr": "SUM(t.x) / 1.5"},
        "ignore":  ["old_metric"],
    }
    assert metric_tables(metrics) == {"order_details", "orders", "t"}


def test_table_rows():
    profile   = {"dossier": FACTS, "client": {"rows": 5}}
    semantics = {"client": {"description": "Clients"}}
    rows = table_rows(["dossier", "client", "agence"], {"dossier": 12}, profile, semantics,
                      relations=[("dossier.id_client", "client.id_client")],
                      metrics={"n": {"expr": "COUNT(DISTINCT client.id_client)"}}, personal=False)
    assert [(r["name"], r["rows"], r["columns"]) for r in rows] == [("dossier", 100, 12), ("client", 5, None),
                                                                     ("agence", None, None)]
    # profiled is done: what the profile detects needs no review
    assert rows[0]["status"] == {"profile": "done", "semantics": "none", "relations": "done", "metrics": "none"}
    assert rows[1]["status"] == {"profile": "done", "semantics": "done", "relations": "done", "metrics": "done"}
    assert rows[2]["status"]["profile"] == "none"


DB_COLUMNS = [{"name": "id_dossier", "type": "BIGINT"}, {"name": "date_deblocage", "type": "STRING"},
              {"name": "montant", "type": "DECIMAL(18, 2)"}, {"name": "date_cloture", "type": "TIMESTAMP"},
              {"name": "id_client", "type": "INT"}, {"name": "date_naissance", "type": "STRING"},
              {"name": "actif", "type": "BOOLEAN"}]


def test_table_detail_one_line_per_column_of_the_database():
    facts = {**FACTS, "keys": {"id_dossier": {"non_null": 100, "distinct": 100},
                               "id_client": {"non_null": 80, "distinct": 40}},
             "filled": {"montant": 1.0, "date_cloture": 0.25},
             "formats": {"date_naissance": "date dd/MM/yyyy"}, "exclude": {"date_naissance": "personal data"},
             "ambiguous_dates": ["date_naissance"]}
    entry = {"exclude_columns": ["commentaire", "id_client"]}
    detail = table_detail("dossier", facts, entry, personal=False, db_columns=DB_COLUMNS)
    lines = {c["column"]: c for c in detail["column_facts"]}

    # every column of the database, in DDL order, measured or not (actif: no filled share)
    assert list(lines) == ["id_dossier", "date_deblocage", "montant", "date_cloture", "id_client", "date_naissance",
                           "actif"]
    assert [lines[c]["kind"] for c in ("montant", "date_cloture", "actif", "date_deblocage")] == [
        "number", "date", "boolean", "text"]
    assert (lines["montant"]["filled"], lines["date_cloture"]["filled"], lines["actif"]["filled"]) == (1.0, 0.25, None)
    assert (lines["id_dossier"]["key"], lines["id_client"]["key"]) == ("unique", "key")
    assert (lines["id_client"]["filled"], lines["id_client"]["distinct"]) == (0.8, 40)
    assert lines["id_client"]["excluded"] and not lines["id_dossier"]["excluded"]
    assert lines["date_deblocage"] == {"column": "date_deblocage", "type": "STRING", "kind": "text", "filled": 0.99,
                                       "distinct": None, "key": None, "format": None, "ambiguous": False,
                                       "excluded": False, "personal": None, "proposals": []}
    # without the database (unreachable): the columns the profile knows, a text column still said so
    lines = {c["column"]: c for c in table_detail("dossier", facts, entry, personal=False)["column_facts"]}
    assert list(lines)[:4] == ["id_dossier", "id_client", "date_deblocage", "montant"]
    assert (lines["date_deblocage"]["kind"], lines["montant"]["kind"]) == ("text", None)
    # a date of birth stored as text: a format whose day and month order no value tells, and personal data
    assert [(p["kind"], p["state"]) for p in lines["date_naissance"]["proposals"]] == [("format", "pending")]
    assert (lines["date_naissance"]["personal"], lines["date_naissance"]["ambiguous"]) == ("visible", True)
    assert detail["detected"] == 1
    assert table_detail("agence", None, None, personal=False) == {"name": "agence", "profiled": False, "detected": 0,
                                                                  "unique_key": None, "column_facts": []}


def test_format_share_and_unique_key():
    facts = {**FACTS, "text_columns": {
        "date_deblocage": {"filled": 0.8, "shapes": {"99/99/9999": 0.76, "a": 0.04}},
        "encours": {"filled": 1.0, "shapes": {"9.9": 0.5, "99": 0.46, "a a": 0.04}},
    }}
    detail = table_detail("dossier", facts, None, personal=False)
    shares = {p["column"]: p["share"] for c in detail["column_facts"] for p in c["proposals"] if p["kind"] == "format"}
    # a share of the filled values (0.76 of 0.8), not of every row
    assert shares == {"date_deblocage": 0.95, "encours": 0.96}
    assert detail["unique_key"] == "id_dossier"


# =============================================================================
# Jobs: the CLI command and the progress read from its output
# =============================================================================

def test_profile_command():
    assert profile_command("awb", ["a", "b"], False)[1:] == ["-m", "offline.cli", "profile", "--db", "awb", "--tables", "a,b"]
    assert profile_command("awb", None, True)[-1] == "--missing"


def test_job_follows_the_progress_of_the_cli():
    job = Job(id=1, db="awb", step="profile", tables=["a", "b", "c"], command=[])
    for line in [
        "\x1b[32m2026-10-23 10:00:01\x1b[0m | INFO     | offline.semantic_gen.profiling:profile_database:110 - [awb] Profiling: a",
        "2026-10-23 10:00:05 | INFO     | offline.semantic_gen.profiling:profile_database:110 - [awb] Profiling: b",
        "2026-10-23 10:00:06 | WARNING  | offline.semantic_gen.profiling:profile_database:114 - [awb] Not profiled: b — boom",
        "2026-10-23 10:00:07 | INFO     | offline.semantic_gen.profiling:profile_database:110 - [awb] Profiling: c",
        "a raw traceback line",
    ]:
        job.add_line(line)
    assert (job.current, job.done, job.failed) == ("c", ["a"], ["b"])
    assert job.current_since is not None
    assert job.lines[0] == {"time": "10:00:01", "level": "INFO", "message": "[awb] Profiling: a"}
    assert job.lines[-1]["message"] == "a raw traceback line"
    job.finish(0)
    assert (job.current, job.current_since, job.done, job.running) == (None, None, ["a", "c"], False)
    assert [line["message"] for line in job.to_dict(since=4)["lines"]] == ["a raw traceback line"]


def test_job_tells_whether_a_stop_cancelled_the_query():
    job = Job(id=1, db="awb", step="profile", tables=None, command=[])
    job.add_line("2026-10-23 10:00:01 | WARNING  | offline.query_cancel:_cancel:120 - "
                 "Stopped: the running query is cancelled on the cluster")
    assert job.cancel == "cancelled"
    job.add_line("2026-10-23 10:00:01 | WARNING  | offline.query_cancel:_cancel:122 - "
                 "Stopped, but the running query could not be cancelled, it may keep running on the cluster — x")
    assert job.cancel == "failed"


# =============================================================================
# Routes: an exception to a detection, personal data everywhere, a stale page refused
# =============================================================================

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(lib.semantics, "SEMANTIC_LAYER_DIR", tmp_path)
    monkeypatch.setattr(studio_app, "SEMANTIC_LAYER_DIR", tmp_path)
    monkeypatch.setattr(studio_app, "_db_config", lambda db: SimpleNamespace(exclude_personal_data=False))
    monkeypatch.setattr(studio_app, "_listing", lambda db, refresh=False: {
        "tables": ["dossier"], "columns": {"dossier": DB_COLUMNS}, "source": "database", "error": None, "read_at": 0})
    (tmp_path / "awb").mkdir()
    (tmp_path / "awb" / "profile.yaml").write_text(yaml.safe_dump({"dossier": FACTS}))
    (tmp_path / "awb" / "tables.yaml").write_text(yaml.safe_dump({"dossier": {"description": "Dossiers"}}))
    return TestClient(studio_app.app)


def change(client, column: str, action: str, version: str):
    return client.post("/api/db/awb/tables/dossier/columns", json={"column": column, "action": action, "version": version})


def test_a_click_on_a_pill_writes_the_exception(client, tmp_path):
    detail = client.get("/api/db/awb/tables/dossier").json()
    assert detail["columns"] == 7 and detail["detected"] == 2

    detail = change(client, "date_deblocage", "format", detail["version"]).json()
    detail = change(client, "encours", "text", detail["version"]).json()
    detail = change(client, "commentaire", "hide", detail["version"]).json()
    written = yaml.safe_load((tmp_path / "awb" / "tables.yaml").read_text())
    assert written["dossier"] == {"description": "Dossiers", "formats": {"date_deblocage": "date dd/MM/yyyy"},
                                  "exclude_columns": ["commentaire"], "keep_as_text": ["encours"]}
    # a detection the profile does not make is refused
    assert change(client, "montant", "text", detail["version"]).status_code == 400
    assert change(client, "montant", "hide", detail["version"]).status_code == 400


def test_a_change_over_a_newer_tables_yaml_is_refused(client):
    assert change(client, "encours", "text", "1").status_code == 409


def test_personal_data_hidden_everywhere_then_one_kept_visible(client, tmp_path):
    home = client.get("/api/db/awb/tables").json()
    assert home["personal"] == {"hidden": 0, "kept": 0, "visible": 1}

    home = client.post("/api/db/awb/personal", json={"hide": True, "version": home["version"]}).json()
    assert home["personal"] == {"hidden": 1, "kept": 0, "visible": 0}
    written = yaml.safe_load((tmp_path / "awb" / "tables.yaml").read_text())
    assert written["dossier"]["exclude_columns"] == ["nom_client"]

    detail = change(client, "nom_client", "show", home["version"]).json()
    assert next(c for c in detail["column_facts"] if c["column"] == "nom_client")["personal"] == "kept"
