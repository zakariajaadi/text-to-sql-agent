"""Type of the tables in the studio (offline/studio/table_types.py): read from the profile, the columns and the
relations, without an LLM; a type corrected by hand wins."""

import pytest

from offline.studio.state import edit_semantics
from offline.studio.table_types import table_types

COLUMNS = {
    "operation": {"id_operation": "BIGINT", "id_compte": "BIGINT", "id_devise": "STRING", "montant": "DECIMAL(18,2)",
                  "date_operation": "STRING", "date_technique": "STRING"},
    "ligne":     {"id_operation": "BIGINT", "id_produit": "BIGINT", "quantite": "INT", "prix": "DOUBLE"},
    "compte":    {"id_compte": "BIGINT", "id_client": "BIGINT", "libelle": "STRING", "ouverture": "DATE", "agence": "STRING"},
    "client":    {"id_client": "BIGINT", "nom": "STRING", "segment": "STRING", "pays": "STRING", "telephone": "STRING"},
    "produit":   {"id_produit": "BIGINT", "id_famille": "INT", "libelle": "STRING", "prix": "DOUBLE", "stock": "INT"},
    "famille":   {"id_famille": "INT", "libelle": "STRING"},
    "devise":    {"id_devise": "STRING", "libelle": "STRING", "id_zone": "INT"},
    "zone":      {"id_zone": "INT", "libelle": "STRING"},
    "client_produit": {"id_client": "BIGINT", "id_produit": "BIGINT"},
    "client_hist":    {"id_client": "BIGINT", "nom": "STRING", "date_technique": "STRING"},
    "vide":      {"id_vide": "INT", "valeur": "STRING"},
    "notes":     {"texte": "STRING"},
}
PROFILE = {
    "operation": {"rows": 50_000, "keys": {"id_operation": {"non_null": 50_000, "distinct": 50_000}}},
    "ligne":     {"rows": 120_000, "keys": {}},
    "compte":    {"rows": 3_000, "keys": {"id_compte": {"non_null": 3_000, "distinct": 3_000}}},
    "client":    {"rows": 2_000, "keys": {"id_client": {"non_null": 2_000, "distinct": 2_000}}},
    "produit":   {"rows": 300, "keys": {"id_produit": {"non_null": 300, "distinct": 300}}},
    "famille":   {"rows": 12, "keys": {"id_famille": {"non_null": 12, "distinct": 12}}},
    "devise":    {"rows": 160, "keys": {"id_devise": {"non_null": 160, "distinct": 160}}},
    "zone":      {"rows": 5, "keys": {"id_zone": {"non_null": 5, "distinct": 5}}},
    "client_produit": {"rows": 0, "keys": {}},
    "client_hist":    {"rows": 9_000, "keys": {"id_client": {"non_null": 9_000, "distinct": 2_000}}},
    "vide":      {"rows": 0, "keys": {}},
    "notes":     {"rows": 40, "keys": {}},
}
SEMANTICS = {
    # a date stored as text, formatted in tables.yaml; the partition of its read mode is no date of an event
    "operation":   {"formats": {"date_operation": "date dd/MM/yyyy"}, "read_mode": "all", "partition_column": "date_technique"},
    "client_hist": {"read_mode": "last", "partition_column": "date_technique"},
}
RELATIONS = [
    ("operation.id_compte", "compte.id_compte"), ("operation.id_devise", "devise.id_devise"),
    ("ligne.id_operation", "operation.id_operation"), ("ligne.id_produit", "produit.id_produit"),
    ("compte.id_client", "client.id_client"), ("produit.id_famille", "famille.id_famille"),
    ("devise.id_zone", "zone.id_zone"),
    ("client_produit.id_client", "client.id_client"), ("client_produit.id_produit", "produit.id_produit"),
    ("client_hist.id_client", "client.id_client"),
]


def detect(semantics=SEMANTICS, columns=COLUMNS):
    return table_types(list(COLUMNS), columns, PROFILE, semantics, RELATIONS)


def test_each_table_has_the_type_its_data_says():
    types = {t: v["type"] for t, v in detect().items()}
    assert types == {
        "operation": "events",         # dated (a text date in tables.yaml), pointing to other tables
        "ligne": "events",             # measured, pointing to two tables, pointed to by none
        "compte": "entity",            # dated (its opening) and pointing to client, but about one per client
        "client": "entity",            # a unique key, pointed to
        "produit": "entity",           # measured and pointing to a reference, but pointed to: no event
        "famille": "reference",        # pointed to, undated, few rows and columns
        "devise": "reference",         # pointing to another reference
        "zone": "reference",
        "client_produit": "bridge",    # empty, but its columns point to two tables
        "client_hist": "entity",       # the snapshots of an entity
        "vide": None,                  # empty: nothing to measure
        "notes": None,                 # no unique key, no relation
    }


def test_each_type_says_why():
    types = detect()
    assert types["operation"]["reason"] == "1 date, 1 mesure, vise 2 tables (compte, devise), 50 000 lignes"
    assert types["famille"]["reason"] == "visée par produit, 12 lignes, 2 colonnes, pas de date"
    assert types["client_produit"]["reason"] == "2 colonnes, dont 2 vers d'autres tables (client, produit)"
    assert types["client"]["reason"] == "une ligne par id_client, visée par client_hist, client_produit, compte, 5 colonnes"
    assert types["compte"]["reason"] == "une ligne par id_compte, visée par operation, 5 colonnes"
    assert types["client_hist"]["reason"] == "garde les versions de chaque ligne (mode LAST)"


def test_a_type_corrected_by_hand_wins_over_the_detection():
    semantics = {**SEMANTICS, "produit": {"table_type": "reference"}, "vide": {"table_type": "reference"}, "zone": {"table_type": "x"}}
    types = detect(semantics)
    assert types["produit"] == {"type": "reference", "detected": "entity", "edited": True,
                                "reason": "une ligne par id_produit, visée par client_produit, ligne, 5 colonnes"}
    assert (types["vide"]["type"], types["vide"]["detected"]) == ("reference", None)
    assert types["zone"]["edited"] is False and types["zone"]["type"] == "reference"   # an unknown type is ignored


def test_without_the_columns_of_the_database_the_profile_tells_their_names():
    types = detect(columns={})
    # no type known: no date nor measure, the keys and relations still say the entities and references
    assert types["client"]["type"] == "entity" and types["famille"]["type"] == "reference"


def test_the_type_is_chosen_then_given_back_to_the_detection():
    entry = edit_semantics({"description": "Accounts."}, "table_type", "entity")
    assert entry == {"description": "Accounts.", "table_type": "entity"}
    assert edit_semantics(entry, "table_type", None) == {"description": "Accounts."}
    with pytest.raises(ValueError):
        edit_semantics(entry, "table_type", "fact")


def test_partitions_each_holding_a_snapshot_are_the_versions_of_an_entity():
    profile = {**PROFILE, "client_hist": {**PROFILE["client_hist"], "partitions": 4, "partitioned_by": ["date_technique"]}}
    semantics = {**SEMANTICS, "client_hist": {}}   # no read mode yet: the data says it
    types = table_types(list(COLUMNS), COLUMNS, profile, semantics, RELATIONS)
    assert (types["client_hist"]["type"], types["client_hist"]["reason"]) == (
        "entity", "une photo complète de ses lignes par partition de date_technique")
