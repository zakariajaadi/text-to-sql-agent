"""Value index of the categorical columns (src/lib/value_index.py): build, then fuzzy search."""

import sqlite3

import pytest

from lib import value_index
from lib.value_index import (
    find_matching_values,
    find_unknown_values,
    normalize,
    search_values,
    write_value_index,
)

VALUES = {
    "agences": {
        "lib_agence": {
            "CASABLANCA ANFA": 120, "CASABLANCA MAARIF": 80, "RABAT AGDAL": 60,
            "MARRAKECH GUELIZ": 40, "TANGER VILLE": 30, "FÈS MÉDINA": 10,
        },
        "region": {"Casablanca-Settat": 200, "Rabat-Salé-Kénitra": 60},
    },
    "customers": {
        "city": {"México D.F.": 5, "Sao Paulo": 4, "London": 6, "Buenos Aires": 3},
        "country": {"Mexico": 5, "Brazil": 9, "UK": 7, "USA": 13},
    },
    "departements": {
        "code": {"IT": 12, "RH": 4, "FIN": 6},
    },
}


@pytest.fixture(autouse=True)
def index(tmp_path, monkeypatch):
    """Value index of a test database, written under tmp_path."""
    monkeypatch.setattr(value_index, "value_index_path", lambda db_name: tmp_path / f"{db_name}.sqlite")
    write_value_index("test", VALUES, inline={("customers", "country"), ("departements", "code")})
    return tmp_path / "test.sqlite"


def found(term, columns=None, **kwargs):
    return [m.value for m in search_values("test", term, columns, **kwargs)]


# ── Normalization ────────────────────────────────────────────────────────────

def test_normalize_removes_case_accents_and_extra_spaces():
    assert normalize("  São   PAULO ") == "sao paulo"
    assert normalize("Fès Médina") == "fes medina"


# ── Build ────────────────────────────────────────────────────────────────────

def test_index_lists_every_column_with_its_inline_flag(index):
    with sqlite3.connect(index) as conn:
        columns = {(t, c): (n, bool(i)) for t, c, n, i in conn.execute("SELECT * FROM value_column")}
    assert columns[("agences", "lib_agence")] == (6, False)
    assert columns[("customers", "country")] == (4, True)
    assert len(columns) == 5


def test_rebuild_replaces_the_previous_index(index):
    write_value_index("test", {"t": {"c": {"only": 1}}}, inline=set())
    assert found("only") == ["only"]
    assert found("casablanca") == []
    assert not index.with_name("test.sqlite.tmp").exists()


# ── Search ───────────────────────────────────────────────────────────────────

def test_partial_name_finds_every_variant_most_frequent_first():
    assert found("Casablanca", [("agences", "lib_agence")]) == ["CASABLANCA ANFA", "CASABLANCA MAARIF"]


def test_typo_is_tolerated():
    assert "CASABLANCA ANFA" in found("casablnca anfa")


def test_accents_and_case_are_ignored():
    assert found("fes medina") == ["FÈS MÉDINA"]
    assert found("sao paulo") == ["Sao Paulo"]


def test_search_is_restricted_to_the_given_columns():
    assert found("Casablanca", [("agences", "region")]) == ["Casablanca-Settat"]
    assert found("Casablanca", []) == []


def test_matches_name_their_column():
    matches = search_values("test", "mexico")
    assert {(m.table, m.column, m.value) for m in matches} >= {
        ("customers", "city", "México D.F."), ("customers", "country", "Mexico"),
    }


def test_short_terms_match_exactly():
    assert found("it") == ["IT"]
    assert found("ti") == []


def test_unrelated_term_finds_nothing():
    assert found("chiffre d'affaires") == []


def test_top_k_per_column():
    assert len(found("Casablanca", [("agences", "lib_agence")], top_k=1)) == 1


def test_missing_index_finds_nothing(index):
    index.unlink()
    assert found("Casablanca") == []
    assert matching("test", ["Casablanca"], ["agences"]) == {}


# ── Matching values of a question ────────────────────────────────────────────

def matching(db_name, terms, tables):
    return find_matching_values(db_name, terms, tables).values


def test_matching_values_are_grouped_by_table_and_column():
    assert matching("test", ["Casablanca"], ["agences"]) == {
        "agences": {
            "lib_agence": ["CASABLANCA ANFA", "CASABLANCA MAARIF"],
            "region":     ["Casablanca-Settat"],
        },
    }


def test_only_the_given_tables_are_searched():
    assert matching("test", ["Casablanca"], ["customers"]) == {}


def test_listed_columns_get_no_matching_values():
    # departements.code lists its values in the table document already
    assert matching("test", ["IT"], ["departements"]) == {}


def test_a_weaker_match_than_a_listed_column_is_dropped():
    # 'Mexico' is a country (100, listed): the city 'México D.F.' (90) is another value
    assert matching("test", ["Mexico"], ["customers"]) == {}
    # A term closer to the city than to the country keeps the city
    assert matching("test", ["Mexico D.F."], ["customers"]) == {"customers": {"city": ["México D.F."]}}


def test_each_term_is_matched_on_its_own():
    assert matching("test", ["Mexico", "Sao Paulo"], ["customers"]) == {"customers": {"city": ["Sao Paulo"]}}


# ── Values a query filters on ────────────────────────────────────────────────

def unknown(filters, tables):
    return [(u.table, u.column, u.value, u.closest) for u in find_unknown_values("test", filters, tables)]


def test_held_values_are_not_reported():
    assert unknown([("customers", "city", "Sao Paulo"), ("customers", "country", "Mexico")], ["customers"]) == []


def test_an_absent_value_comes_with_the_closest_ones():
    assert unknown([("customers", "city", "Sao Paolo")], ["customers"]) == [
        ("customers", "city", "Sao Paolo", ["Sao Paulo"]),
    ]


def test_the_check_is_exact_case_included():
    # Hive compares strings case-sensitively: 'usa' returns no row
    assert unknown([("customers", "country", "usa")], ["customers"]) == [
        ("customers", "country", "usa", ["USA"]),
    ]


def test_an_absent_value_may_have_no_close_one():
    assert unknown([("customers", "city", "Atlantis")], ["customers"]) == [("customers", "city", "Atlantis", [])]


def test_unqualified_columns_are_resolved_among_the_tables_of_the_query():
    assert unknown([("", "city", "Paris")], ["customers", "departements"])[0][:2] == ("customers", "city")
    assert unknown([("", "city", "Paris")], ["agences"]) == []


def test_an_ambiguous_unqualified_column_is_not_checked(index):
    write_value_index("test", {"a": {"city": {"Paris": 1}}, "b": {"city": {"Lyon": 1}}}, inline=set())
    assert unknown([("", "city", "Atlantis")], ["a", "b"]) == []
    assert unknown([("", "city", "Atlantis")], ["a"])[0][:2] == ("a", "city")


def test_columns_and_tables_not_indexed_are_not_checked():
    assert unknown([("customers", "fax", "x"), ("orders", "city", "x")], ["customers", "orders"]) == []
    assert unknown([("customers", "city", "Atlantis")], ["agences"]) == []


def not_found(terms, tables):
    return find_matching_values("test", terms, tables).not_found


def test_terms_close_to_no_value_are_reported():
    assert not_found(["Atlantis"], ["customers"]) is True
    # A term found in a listed column: found, nothing to add
    assert not_found(["Mexico"], ["customers"]) is False
    # One term found is enough: the others may be spellings of the same value
    assert not_found(["Londres", "London"], ["customers"]) is False


def test_nothing_is_reported_when_nothing_was_searched():
    assert not_found(["Atlantis"], ["orders"]) is False
    assert not_found([], ["customers"]) is False


def test_every_close_value_is_kept_up_to_top_k_then_counted(index, monkeypatch):
    cities = {f"Santa City {i:02d}": 1 for i in range(25)}
    write_value_index("test", {"t": {"city": cities}}, inline=set())

    found = find_matching_values("test", ["Santa City"], ["t"])
    assert len(found.values["t"]["city"]) == 20 and found.more == {"t": {"city": 5}}

    monkeypatch.setattr(value_index.cfg.value_search, "top_k", 30)
    found = find_matching_values("test", ["Santa City"], ["t"])
    assert len(found.values["t"]["city"]) == 25 and found.more == {}
