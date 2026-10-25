"""FK relations generator (offline/semantic_gen/relationships.py): every relation measured, its evidence kept."""

import sqlite3
from types import SimpleNamespace

import pytest
import yaml
from langchain_community.utilities import SQLDatabase
from loguru import logger

import lib.semantics
from offline.semantic_gen import relationships, utils
from offline.semantic_gen.schemas import FKRelation, FKRelations

# Share of the values found, per relation (key_overlap, faked)
OVERLAP = {("carte.id_client", "client.id_client"): 0.99, ("carte.code_agence", "agence.code"): 0.4,
           ("compte.id_client", "client.id_client"): 0.97}


class FakeLLM:
    def invoke(self, inputs: dict) -> FKRelations:
        return FKRelations(relations=[
            FKRelation(from_column="carte.code_agence", to_column="agence.code"),    # 40 %: kept, to check
            FKRelation(from_column="carte.inconnue", to_column="client.id_client"),  # hallucinated: dropped
        ])


@pytest.fixture
def bank(tmp_path, monkeypatch):
    conn = sqlite3.connect(tmp_path / "bank.sqlite")
    conn.executescript("""
        CREATE TABLE client (id_client INTEGER); CREATE TABLE carte (id_client INTEGER, code_agence TEXT);
        CREATE TABLE agence (code TEXT); CREATE TABLE compte (id_client INTEGER);
    """)
    conn.commit()
    for module in (lib.semantics, utils):
        monkeypatch.setattr(module, "SEMANTIC_LAYER_DIR", tmp_path)
    monkeypatch.setattr(utils, "cfg", SimpleNamespace(get_db=lambda name: SimpleNamespace(dialect="sqlite")))
    monkeypatch.setattr(relationships, "db_registry", {"bank": SQLDatabase.from_uri(f"sqlite:///{tmp_path / 'bank.sqlite'}")})
    monkeypatch.setattr(relationships, "fk_relations_chain", FakeLLM())
    monkeypatch.setattr(relationships, "key_overlap", lambda db, source, target: OVERLAP.get((source, target)))
    (tmp_path / "bank").mkdir()
    (tmp_path / "bank" / "profile.yaml").write_text(yaml.safe_dump({
        "client": {"rows": 10, "keys": {"id_client": {"non_null": 10, "distinct": 10}}},
        "carte":  {"rows": 30, "keys": {"id_client": {"non_null": 30, "distinct": 8}}},
    }))
    (tmp_path / "bank" / "relationships.yaml").write_text(
        "# Written by hand: compte only through client.\n\n"
        + yaml.safe_dump({"relations": [{"from": "compte.id_client", "to": "client.id_client"}], "ignore": []}))
    return tmp_path / "bank" / "relationships.yaml"


def test_every_relation_carries_its_evidence(bank):
    relationships.infer_fk_relations("bank")
    entry = yaml.safe_load(bank.read_text())
    assert entry["relations"] == [
        {"from": "compte.id_client", "to": "client.id_client", "found": 0.97},      # written before: measured now
        {"from": "carte.id_client", "to": "client.id_client", "found": 0.99, "origin": "profile"},
        {"from": "carte.code_agence", "to": "agence.code", "origin": "llm", "found": 0.4},   # kept, in amber
    ]
    assert bank.read_text().startswith("# Written by hand")


def test_check_measures_again_without_the_llm(bank, monkeypatch):
    monkeypatch.setattr(relationships, "fk_relations_chain", None)    # never called
    monkeypatch.setitem(OVERLAP, ("compte.id_client", "client.id_client"), 0.5)
    relationships.check_relations("bank")
    assert yaml.safe_load(bank.read_text())["relations"] == [{"from": "compte.id_client", "to": "client.id_client",
                                                              "found": 0.5}]


def test_the_progress_of_a_search_is_logged(bank):
    messages = []
    sink = logger.add(messages.append, format="{message}")
    try:
        relationships.infer_fk_relations("bank")
        relationships.check_relations("bank")
    finally:
        logger.remove(sink)
    # read by the studio (offline/studio/jobs.py): the LLM asked, the relations to measure, the ones added
    assert [m.strip() for m in messages if "LLM" in m or "relation(s)" in m] == [
        "[bank] Asking the LLM for the other relations",
        "[bank] 2 relation(s) to check",   # the one written before and the LLM's, the profile's is measured
        "[bank] 2 new relation(s) added to relationships.yaml — review them.",
        "[bank] 3 relation(s) to check",
        "[bank] 3 relation(s) checked on the data",
    ]
