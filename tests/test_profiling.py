"""Data profile (offline/semantic_gen/profiling.py): formats, row facts, key relations, personal data."""

import re
import sqlite3

import pytest
import yaml
from langchain_community.utilities import SQLDatabase

import lib.semantics
from offline.eval.eval_lib.execution import _normalize_row
from offline.semantic_gen import profiling
from offline.semantic_gen.profiling import (
    KEY_NAME,
    PERSONAL_NAME,
    describe_shapes,
    key_overlap_sql,
    key_relations,
    personal_values,
    row_facts,
    text_format,
)
from offline.semantic_gen.prompts import TABLE_SEMANTICS_PROMPT


# =============================================================================
# Formats of the text columns, from the shapes of their values
# =============================================================================

@pytest.mark.parametrize("shapes, column, expected", [
    # Dates, NULL and empty values aside
    ({"99/99/9999": 900, None: 80, "": 20}, "entry_date", "date dd/MM/yyyy"),
    ({"9999-99-99 99:99:99": 950, "9999-99-99 99:99:99.999": 5}, "closed_at", "date yyyy-MM-dd HH:mm:ss"),
    ({"9999-99-99": 100}, "start_day", "date yyyy-MM-dd"),
    # DB2 timestamps, with or without their microseconds
    ({"9999-99-99-99.99.99": 100}, "date_maj", "date yyyy-MM-dd-HH.mm.ss"),
    ({"9999-99-99-99.99.99.999999": 98, "  ": 2}, "horodatage", "date yyyy-MM-dd-HH.mm.ss.SSSSSS"),
    # Numbers: dot or comma decimals, exponents, whole numbers in a measure column
    ({"999.99": 500, ".99": 30, "9.9a9": 20}, "amount", "decimal"),
    ({"9,99a+99": 80, "99,99": 20}, "balance", "decimal"),
    ({"9999,99": 45, "999": 29, "999.9": 26}, "outstanding", "decimal"),
    ({"99": 55, "9": 45}, "duration_days", "decimal"),
    ({"  ": 30, "999": 40, "9999": 30}, "days_late", "decimal"),
    # Letters among digits are codes, not exponents
    ({"a999": 100}, "status", None),
    ({"99a999999999999": 92, "99a9999999999999": 8}, "contract", None),
    ({"a999999": 95, "999999": 5}, "user_modif", None),
    ({"a9": 100}, "plan", None),
    # Named like a code or an identifier: text, even mangled into exponents
    ({"999999": 100}, "code_agence", None),
    ({"99999999": 100}, "numero_contrat", None),
    ({"9999": 100}, "customerid", None),
    ({"9999": 56, "999": 44}, "ch_cod_eds", None),
    ({"9,99a+99": 100}, "numero_compte", None),
    # Whole numbers that always have the same number of digits: codes
    ({"999999999999999": 100}, "support_account", None),
    ({"99": 99, "  ": 1}, "service_level", None),
    ({"999999": 999, "9999a9": 1}, "product_group", None),
    # Labels, or no format covering 90% of the filled values
    ({"a a": 70, "a": 30}, "city", None),
    ({"99/99/9999": 60, "a": 40}, "mixed", None),
    ({None: 100}, "empty", None),
])
def test_text_format(shapes, column, expected):
    assert text_format(shapes, column) == expected


def test_describe_shapes_counts_null_empty_and_blank_as_not_filled():
    facts = describe_shapes({"a": 6, None: 2, "": 1, "   ": 1})
    assert facts == {"filled": 0.6, "shapes": {"a": 0.6}}


# =============================================================================
# Row facts and key relations
# =============================================================================

PROFILE = {
    "customer": {"rows": 59, "keys": {"customerid": {"non_null": 59, "distinct": 59}}},
    "invoice": {"rows": 412, "keys": {"invoiceid": {"non_null": 412, "distinct": 412},
                                      "customerid": {"non_null": 412, "distinct": 59}}},
    "invoiceline": {"rows": 2240, "keys": {"invoiceid": {"non_null": 2240, "distinct": 412}}},
    # Unique too, but smaller: customer stays the hub of customerid
    "customer_extra": {"rows": 10, "keys": {"customerid": {"non_null": 10, "distinct": 10}}},
}


def test_row_facts_say_one_or_several_rows_per_key():
    facts = row_facts(PROFILE["invoice"])
    assert "- 412 rows" in facts
    assert "invoiceid: 412 distinct values, one row per invoiceid" in facts
    assert "customerid: 59 distinct values, 7.0 rows per customerid on average" in facts
    assert "make profile" in row_facts(None)


def test_key_relations_point_to_the_largest_table_where_the_key_is_unique():
    assert sorted(key_relations(PROFILE)) == [
        ("customer_extra.customerid", "customer.customerid"),
        ("invoice.customerid", "customer.customerid"),
        ("invoiceline.invoiceid", "invoice.invoiceid"),
    ]


def test_a_key_with_nulls_or_duplicates_is_no_hub():
    profile = {
        "client": {"rows": 100, "keys": {"client_id": {"non_null": 99, "distinct": 99}}},    # one NULL: hub
        "link":   {"rows": 50, "keys": {"client_id": {"non_null": 50, "distinct": 48},        # duplicates
                                        "link_id": {"non_null": 50, "distinct": 50}}},
        "other":  {"rows": 50, "keys": {"link_id": {"non_null": 40, "distinct": 40}}},       # 80% filled
    }
    assert key_relations(profile) == [("link.client_id", "client.client_id"), ("other.link_id", "link.link_id")]


def _run_on_sqlite(query: str, rows: dict[str, list[tuple]]) -> tuple:
    """A query written for Hive / Impala run on SQLite: RLIKE → REGEXP, STRING → TEXT."""
    conn = sqlite3.connect(":memory:")
    conn.create_function("regexp", 2, lambda pattern, value: value is not None and re.search(pattern, value) is not None)
    for table, values in rows.items():
        conn.execute(f'CREATE TABLE {table} ("k")')
        conn.executemany(f"INSERT INTO {table} VALUES (?)", values)
    return conn.execute(query.replace(" RLIKE ", " REGEXP ").replace(" AS STRING)", " AS TEXT)")).fetchone()


def test_key_overlap_compares_digits_as_numbers():
    rows = {
        # Text with leading zeros or blanks, a code, and a value of no client
        "loan":   [("01234567",), ("00000042",), (" 7 ",), ("A12",), ("99",), (None,)],
        "client": [(1234567,), (42,), (7,), (8,)],
        "agency": [("a12",), ("A12 ",)],
    }
    quote = '"{}"'.format
    total, found = _run_on_sqlite(key_overlap_sql("loan.k", "client.k", quote, 100), rows)
    assert (total, found) == (5, 3)
    # Text that is not only digits stays text, trimmed (and case-sensitive)
    assert _run_on_sqlite(key_overlap_sql("loan.k", "agency.k", quote, 100), rows) == (5, 1)


def test_the_semantics_prompt_takes_the_row_facts():
    assert "{facts}" in TABLE_SEMANTICS_PROMPT


# =============================================================================
# Profiling some tables, merged into the existing profile
# =============================================================================

@pytest.fixture
def bank(tmp_path, monkeypatch):
    """A SQLite database of numbers only (shapes need REGEXP_REPLACE), its profile in tmp_path."""
    monkeypatch.setattr(lib.semantics, "SEMANTIC_LAYER_DIR", tmp_path)
    conn = sqlite3.connect(tmp_path / "bank.sqlite")
    conn.executescript("""
        CREATE TABLE client (id_client INTEGER, age INTEGER);
        INSERT INTO client VALUES (1, 30), (2, 40);
        CREATE TABLE compte (id_compte INTEGER, id_client INTEGER);
        INSERT INTO compte VALUES (10, 1), (11, 1), (12, 2);
    """)
    conn.commit()
    return SQLDatabase.from_uri(f"sqlite:///{tmp_path / 'bank.sqlite'}")


def _write_profile(profile: dict):
    path = lib.semantics.profile_path("bank")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(profile))


def test_profile_some_tables_keeps_the_others(bank):
    _write_profile({"client": {"rows": 999}, "compte": {"rows": 999}, "dropped_table": {"rows": 1}})
    profiling.profile_database(bank, "bank", tables=["client"])
    profile = lib.semantics.load_profile("bank")
    assert list(profile) == ["client", "compte"]            # a table no longer usable is dropped
    assert profile["client"]["rows"] == 2
    assert profile["client"]["keys"]["id_client"] == {"non_null": 2, "distinct": 2}
    assert profile["compte"] == {"rows": 999}


def test_profile_missing_tables_only(bank):
    _write_profile({"client": {"rows": 999}})
    profiling.profile_database(bank, "bank", missing=True)
    profile = lib.semantics.load_profile("bank")
    assert profile["client"] == {"rows": 999}
    assert profile["compte"]["rows"] == 3


def test_profile_every_table_keeps_the_entry_of_a_failing_one(bank, monkeypatch):
    _write_profile({"client": {"rows": 999}})
    measure = profiling.profile_table
    monkeypatch.setattr(profiling, "profile_table",
                        lambda db, table, *args: measure(db, table, *args) if table == "compte" else 1 / 0)
    profiling.profile_database(bank, "bank")
    profile = lib.semantics.load_profile("bank")
    assert profile["client"] == {"rows": 999}
    assert profile["compte"]["rows"] == 3


def test_profile_an_unknown_table_stops(bank):
    with pytest.raises(ValueError, match="clients"):
        profiling.profile_database(bank, "bank", tables=["clients"])


# =============================================================================
# Column names
# =============================================================================

@pytest.mark.parametrize("name, is_key", [
    ("numero_personne_host", True), ("numero_compte", True), ("id_claim", True),
    ("customerid", True), ("n_compte", True), ("montant", False), ("statut", False),
])
def test_key_names(name, is_key):
    assert bool(KEY_NAME.search(name)) is is_key


@pytest.mark.parametrize("name, personal", [
    ("nom_gestionnaire", True), ("prenom", True), ("raison_sociale_1", True), ("email", True),
    ("gsm", True), ("commentaire_creation", True), ("cin_tuteur", True), ("firstname", True),
    ("libelle_agence", False), ("statut", False), ("montant_impaye", False), ("nombre_rachat", False),
    # Birth dates, misspelled too
    ("date_naissance", True), ("date_naisance", True), ("datenaissance", True), ("dt_nais_client", True),
    # Names glued to the next word, not the words starting like a name
    ("nomintervenants", True), ("prenomintervenants", True), ("nom_client", True), ("nomsclient", True),
    ("nombre", False), ("nb_nominal", False), ("nominal", False), ("code_nomenclature", False),
    ("nomade", False), ("denominateur", False),
    # Authors of a row
    ("cree_par", True), ("modifiee_par", True), ("modifie_par_agent", True), ("updated_by", True),
    ("date_creation", False), ("parametre", False),
])
def test_personal_data_names(name, personal):
    assert bool(PERSONAL_NAME.search(name)) is personal


@pytest.mark.parametrize("shapes, personal", [
    # Logins (first.last, first-name.last, a number to tell homonyms) and emails
    ({"a.a": 80, "a-a.a": 15, "a.a9": 5}, True),
    ({"a.a@a.a": 90, None: 10}, True),
    ({"a999999": 40, "a.a": 60}, True),                  # a user column mixing codes and logins
    # Codes, labels, decimals
    ({"a999999": 100}, False),
    ({"a a": 70, "a.": 30}, False),
    ({"99.99": 100}, False),
    ({None: 100}, False),
])
def test_personal_data_values(shapes, personal):
    assert personal_values(shapes) is personal


# =============================================================================
# Eval: numbers written as text
# =============================================================================

def test_numbers_written_as_text_compare_as_numbers():
    assert _normalize_row(("2021", 2021, "12.50", "Casablanca", None)) == _normalize_row((2021, "2021", 12.5, "Casablanca", None))
    assert _normalize_row(("2021",)) == (2021,)
