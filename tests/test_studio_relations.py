"""Relations screen of the studio (offline/studio/relations.py): evidence, map, path, add and ignore."""

from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient

import lib.semantics
from offline.semantic_gen import utils
from offline.studio import app as studio_app
from offline.studio.relations import (
    add_relation,
    fk_graph,
    ignore_relation,
    join_path,
    keep_relation,
    layout,
    relation_fanout,
    relation_row,
    relations_view,
)

PROFILE = {
    "client":        {"rows": 10, "keys": {"id_client": {"non_null": 10, "distinct": 10}}},
    "client_master": {"rows": 10, "keys": {"id_client": {"non_null": 10, "distinct": 10},
                                           "id_personne": {"non_null": 10, "distinct": 9}}},
    "carte":         {"rows": 30, "keys": {"id_client": {"non_null": 30, "distinct": 8}}},
}
TABLES = ["client", "client_master", "personne", "carte", "impaye", "bilan"]
RELATIONS = [
    {"from": "client_master.id_client", "to": "client.id_client", "found": 1.0, "origin": "profile"},
    {"from": "client_master.id_personne", "to": "personne.id_personne", "found": 0.62, "origin": "llm"},
    {"from": "carte.id_client", "to": "client.id_client"},
    {"from": "impaye.id_personne", "to": "personne.id_personne", "found": 0.97, "origin": "manual"},
]


def test_relation_row_says_its_evidence():
    rows = [relation_row(r, PROFILE) for r in RELATIONS]
    assert (rows[0]["links"], rows[0]["weak"], rows[0]["origin"]) == ("1 → 1", False, "profile")
    assert (rows[1]["weak"], rows[1]["links"]) == (True, "n → ?")    # under 90 %: for a human to keep or ignore
    assert (rows[2]["found"], rows[2]["empty"], rows[2]["weak"], rows[2]["links"]) == (None, False, False, "n → 1")   # not checked yet


# customer.country is no key: 24 values for 59 rows, up to 13 rows per value; invoice joins 5.7 rows on average
SPREAD = {"target_rows": 59, "target_values": 24, "per_value": [1, 13],
          "source_rows": 412, "source_values": 24, "matched": 412, "joined": 2343}


def test_a_relation_to_a_column_that_is_no_key():
    assert relation_fanout(SPREAD) == 5.7
    assert relation_fanout({"target_rows": 59, "target_values": 59, "per_value": [1, 1]}) is None
    entry = add_relation({"relations": []}, "invoice.billingcountry", "customer.country", 1.0, SPREAD)
    assert entry["relations"] == [{"from": "invoice.billingcountry", "to": "customer.country", "found": 1.0,
                                   "fanout": 5.7, "links": "n → n", "origin": "manual"}]
    row = relation_row(entry["relations"][0], PROFILE)
    # all its values found, still in amber: a join on it counts each row several times
    assert (row["weak"], row["links"], row["fanout"]) == (True, "n → n", 5.7)


def test_relations_view_counts_and_isolated_tables():
    view = relations_view(TABLES, {"relations": RELATIONS, "ignore": []}, PROFILE)
    assert view["counts"] == {"client": 2, "client_master": 2, "personne": 2, "carte": 1, "impaye": 1, "bilan": 0}
    assert (view["isolated"], view["weak"], view["unchecked"]) == (["bilan"], 1, 1)


def test_a_relation_from_an_empty_table_is_measured_with_nothing_to_compare():
    empty = {"from": "bilan.id_client", "to": "client.id_client", "found": None}   # written by fk-relations --check
    row = relation_row(empty, PROFILE)
    assert (row["found"], row["empty"], row["weak"]) == (None, True, False)
    # never counted as a relation to measure: checking again would not change it
    view = relations_view(TABLES, {"relations": [*RELATIONS, empty], "ignore": []}, PROFILE)
    assert view["unchecked"] == 1


def test_layout_places_every_table_without_overlap():
    result = layout(TABLES, RELATIONS)
    nodes = result["nodes"]
    assert set(nodes) == set(TABLES)
    boxes = list(nodes.values())
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert a["x"] + a["w"] <= b["x"] or b["x"] + b["w"] <= a["x"] or a["y"] + a["h"] <= b["y"] or b["y"] + b["h"] <= a["y"]
    # a table pointing to another one is on its left, the tables of a column aligned on their left side;
    # the isolated table on a last row
    assert nodes["carte"]["x"] + nodes["carte"]["w"] < nodes["client"]["x"]
    assert nodes["carte"]["x"] == nodes["client_master"]["x"] and nodes["client"]["x"] == nodes["personne"]["x"]
    assert nodes["bilan"]["y"] > max(n["y"] for t, n in nodes.items() if t != "bilan")
    assert result["width"] >= max(n["x"] + n["w"] for n in boxes)


def test_join_path_is_the_one_of_the_agent():
    graph = fk_graph(RELATIONS, TABLES)
    assert join_path("carte", "impaye", graph) == {
        "joins": 4, "layers": [["carte"], ["client"], ["client_master"], ["personne"], ["impaye"]], "bridged": False}
    assert join_path("carte", "personne", graph)["bridged"]           # 3 joins: the agent bridges it
    assert join_path("carte", "bilan", graph) == {"joins": None, "layers": [], "bridged": False}


def test_add_and_ignore_a_relation():
    entry = {"relations": list(RELATIONS), "ignore": [{"from": "bilan.id", "to": "client.id_client"}]}
    entry = add_relation(entry, "bilan.id", "client.id_client", 0.4)
    assert entry["ignore"] == [] and entry["relations"][-1] == {"from": "bilan.id", "to": "client.id_client",
                                                                "found": 0.4, "origin": "manual"}
    entry = ignore_relation(entry, "carte.id_client", "client.id_client")
    assert entry["ignore"] == [{"from": "carte.id_client", "to": "client.id_client"}]
    assert ignore_relation(entry, "carte.id_client", "client.id_client", ignore=False)["ignore"] == []
    with pytest.raises(ValueError):
        ignore_relation(entry, "x.a", "y.b")


def test_a_weak_relation_kept_by_a_human_is_no_longer_weak():
    entry = keep_relation({"relations": list(RELATIONS), "ignore": []}, "client_master.id_personne", "personne.id_personne")
    row = relation_row(entry["relations"][1], PROFILE)
    assert (row["found"], row["weak"], row["kept"]) == (0.62, False, True)       # its evidence kept, out of the amber
    assert relations_view(TABLES, entry, PROFILE)["weak"] == 0
    entry = keep_relation(entry, "client_master.id_personne", "personne.id_personne", keep=False)
    assert "kept" not in entry["relations"][1] and relation_row(entry["relations"][1], PROFILE)["weak"]
    assert "kept" not in RELATIONS[1]                                               # the entry given is not changed
    with pytest.raises(ValueError):
        keep_relation(entry, "x.a", "y.b")


# =============================================================================
# Routes, and relationships.yaml written with its own comments kept
# =============================================================================

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(lib.semantics, "SEMANTIC_LAYER_DIR", tmp_path)
    monkeypatch.setattr(utils, "SEMANTIC_LAYER_DIR", tmp_path)
    monkeypatch.setattr(studio_app, "_db_config", lambda db: SimpleNamespace(exclude_personal_data=False))
    columns = {t: [{"name": "id_client", "type": "INT"}, {"name": "id_personne", "type": "INT"}] for t in TABLES}
    monkeypatch.setattr(studio_app, "_listing", lambda db, refresh=False: {
        "tables": TABLES, "columns": columns, "source": "database", "error": None, "read_at": 0})
    monkeypatch.setattr(studio_app, "_databases", {"awb": object()})
    monkeypatch.setattr(studio_app, "key_overlap", lambda db, source, target: 0.95)
    monkeypatch.setattr(studio_app, "key_spread", lambda db, source, target: (
        SPREAD if target == "personne.id_personne" else {"target_rows": 10, "target_values": 10, "per_value": [1, 1]}))
    (tmp_path / "awb").mkdir()
    (tmp_path / "awb" / "profile.yaml").write_text(yaml.safe_dump(PROFILE))
    (tmp_path / "awb" / "relationships.yaml").write_text(
        "# Two id families that overlap: never join client and personne directly.\n\n"
        + yaml.safe_dump({"relations": RELATIONS[:1], "ignore": []}))
    return TestClient(studio_app.app)


def test_a_relation_added_then_ignored_from_the_page(client, tmp_path):
    view = client.get("/api/db/awb/relations").json()
    assert view["isolated"] == ["personne", "carte", "impaye", "bilan"] and "id_client" in view["columns"]["carte"]

    view = client.post("/api/db/awb/relations", json={"source": "carte.id_client", "target": "client.id_client",
                                                      "version": view["version"]}).json()
    assert view["relations"][-1]["found"] == 0.95 and view["relations"][-1]["origin"] == "manual"
    view = client.post("/api/db/awb/relations/ignore", json={"source": "carte.id_client", "target": "client.id_client",
                                                             "version": view["version"]}).json()
    assert [r["from"] for r in view["ignored"]] == ["carte.id_client"]

    text = (tmp_path / "awb" / "relationships.yaml").read_text()
    assert text.startswith("# Two id families that overlap")      # the reviewer's comment kept
    assert client.get("/api/db/awb/relations/path", params={"a": "carte", "b": "client"}).json()["joins"] is None


def test_a_weak_relation_kept_then_no_longer_kept_from_the_page(client, tmp_path):
    (tmp_path / "awb" / "relationships.yaml").write_text(yaml.safe_dump({"relations": RELATIONS[:2], "ignore": []}))
    view = client.get("/api/db/awb/relations").json()
    assert view["weak"] == 1
    body = {"source": "client_master.id_personne", "target": "personne.id_personne", "version": view["version"]}
    view = client.post("/api/db/awb/relations/keep", json=body).json()
    assert view["weak"] == 0 and view["relations"][1]["kept"] is True
    assert yaml.safe_load((tmp_path / "awb" / "relationships.yaml").read_text())["relations"][1]["kept"] is True
    view = client.post("/api/db/awb/relations/keep", params={"back": "true"}, json={**body, "version": view["version"]}).json()
    assert view["weak"] == 1 and view["relations"][1]["kept"] is False
    assert client.post("/api/db/awb/relations/keep", json={**body, "version": "1"}).status_code == 409


def test_a_relation_to_a_column_that_is_no_key_is_added_on_demand_only(client, tmp_path):
    version = client.get("/api/db/awb/relations").json()["version"]
    post = lambda force, v: client.post("/api/db/awb/relations", json={  # noqa: E731
        "source": "carte.id_personne", "target": "personne.id_personne", "version": v, "force": force}).json()
    view = post(False, version)
    assert view["added"] is False and view["check"]["per_value"] == [1, 13] and view["check"]["found"] == 0.95
    assert view["version"] == version and len(view["relations"]) == 1          # nothing written
    view = post(True, version)
    assert view["added"] is True and view["relations"][-1]["fanout"] == 5.7 and view["weak"] == 1
    written = yaml.safe_load((tmp_path / "awb" / "relationships.yaml").read_text())["relations"][-1]
    assert (written["fanout"], written["links"], written["origin"]) == (5.7, "n → n", "manual")


def test_bad_relations_are_refused(client):
    version = client.get("/api/db/awb/relations").json()["version"]
    post = lambda source, target, v=version: client.post(  # noqa: E731
        "/api/db/awb/relations", json={"source": source, "target": target, "version": v})
    assert post("carte.inconnue", "client.id_client").status_code == 400
    assert post("carte.id_client", "carte.id_personne").status_code == 400       # one table
    assert post("carte.id_client", "client.id_client", "1").status_code == 409    # page older than the file
