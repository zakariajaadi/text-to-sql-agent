"""Semantics generator (offline/semantic_gen/tables.py): only some tables, described again with --fresh,
the descriptions corrected by hand kept and the ones the LLM changed marked to read again."""

import sqlite3
from types import SimpleNamespace

import pytest
from langchain_community.utilities import SQLDatabase

import lib.semantics
from offline.semantic_gen import tables
from offline.semantic_gen.schemas import ColumnSemantics, TableSemantics
from offline.semantic_gen.utils import dump_table_semantics


class FakeLLM:
    """The chain of the generator: a description per table, the tables it was asked for kept."""

    def __init__(self):
        self.asked = []

    def invoke(self, inputs: dict) -> TableSemantics:
        table = inputs["ddl"].split("`")[1] if "`" in inputs["ddl"] else inputs["ddl"].split()[2].strip('"')
        self.asked.append(table)
        return TableSemantics(description=f"New {table}.", columns=[
            ColumnSemantics(name="id", description="New id.", categorical=False),
            ColumnSemantics(name="label", description="New label.", categorical=True),
        ])


@pytest.fixture
def llm(tmp_path, monkeypatch):
    conn = sqlite3.connect(tmp_path / "shop.sqlite")
    conn.executescript("CREATE TABLE client (id INTEGER, label TEXT); CREATE TABLE produit (id INTEGER, label TEXT);")
    conn.commit()
    monkeypatch.setattr(lib.semantics, "SEMANTIC_LAYER_DIR", tmp_path)
    monkeypatch.setattr(tables, "db_registry", {"shop": SQLDatabase.from_uri(f"sqlite:///{tmp_path / 'shop.sqlite'}")})
    monkeypatch.setattr(tables, "cfg", SimpleNamespace(get_db=lambda name: SimpleNamespace(
        user_language="French", exclude_personal_data=False)))
    fake = FakeLLM()
    monkeypatch.setattr(tables, "semantics_chain", fake)
    return fake


def described(name: str = "shop") -> dict:
    return {table: entry.get("description") for table, entry in lib.semantics.load_semantics(name).items()}


def test_only_the_tables_asked_for(llm):
    tables.generate_table_semantics("shop", tables=["produit"])
    assert llm.asked == ["produit"] and described() == {"produit": "New produit."}


def test_fresh_describes_again_the_tables_asked_for_only(llm):
    tables.generate_table_semantics("shop")
    path = lib.semantics.tables_path("shop")
    path.write_text(path.read_text().replace("New client.", "Reviewed client.").replace("New produit.", "Reviewed produit."))

    llm.asked.clear()
    tables.generate_table_semantics("shop", fresh=True, tables=["produit"])
    assert llm.asked == ["produit"]
    # the reviewed description of the other table is kept
    assert described() == {"client": "Reviewed client.", "produit": "New produit."}


def test_fresh_keeps_the_corrections_made_by_hand(llm):
    tables.generate_table_semantics("shop")
    semantics = lib.semantics.load_semantics("shop")
    semantics["client"].update(description="Clients, corrected.", edited_description=True, edited_columns=["label"],
                               columns={"label": "Label, corrected.", "id": "Old id."})
    lib.semantics.tables_path("shop").write_text(dump_table_semantics(semantics))

    tables.generate_table_semantics("shop", fresh=True, tables=["client"])
    entry = lib.semantics.load_semantics("shop")["client"]
    assert entry["description"] == "Clients, corrected."
    assert entry["columns"] == {"id": "New id.", "label": "Label, corrected."} and list(entry["columns"]) == ["id", "label"]
    # what the LLM changed is to read again; the corrections stay corrections
    assert (entry["rewritten_columns"], entry["edited_columns"]) == (["id"], ["label"])
    assert "rewritten_description" not in entry

    # replaced too on demand: the corrections go, the texts changed are to read again (the same text is not)
    tables.generate_table_semantics("shop", fresh=True, tables=["client"], replace_edited=True)
    entry = lib.semantics.load_semantics("shop")["client"]
    assert (entry["description"], entry["columns"]["label"]) == ("New client.", "New label.")
    assert "edited_description" not in entry and "edited_columns" not in entry
    assert (entry["rewritten_description"], entry["rewritten_columns"]) == (True, ["label"])


def test_an_unknown_table_stops(llm):
    with pytest.raises(ValueError, match="clients"):
        tables.generate_table_semantics("shop", tables=["clients"])
