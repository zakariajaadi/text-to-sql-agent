"""Read mode of the tables in the studio (offline/studio/reading.py) and the partitions the profile reads
(offline/semantic_gen/profiling.py): a mode proposed from the data, chosen by a human, checked for the agent."""

import pytest

from offline.semantic_gen.profiling import partition_columns, partition_count
from offline.studio.reading import edit_read_mode, reading, suggest

# A snapshot of 2 000 clients a day, 4 days kept; operations added once each, over 30 days
SNAPSHOT   = {"rows": 8_000, "partitions": 4, "partitioned_by": ["date_technique"],
              "keys": {"id_client": {"non_null": 8_000, "distinct": 2_000}, "id_agence": {"non_null": 8_000, "distinct": 40}}}
INCREMENTS = {"rows": 50_000, "partitions": 30, "partitioned_by": ["date_technique"],
              "keys": {"id_operation": {"non_null": 50_000, "distinct": 50_000}, "id_compte": {"non_null": 50_000, "distinct": 3_000}}}
COLUMNS    = ["id_client", "nom", "date_maj", "date_technique"]


def test_the_data_proposes_the_mode_of_a_partitioned_table():
    assert suggest(SNAPSHOT) == ("last", "id_client revient dans 4 partitions sur 4 en moyenne : chaque partition est une photo complète")
    assert suggest(INCREMENTS) == ("all", "chaque id_operation n'apparaît qu'une fois sur les 30 partitions : chacune ajoute des lignes")
    # a key coming back in a few partitions only: neither one, a human decides
    mixed = {**INCREMENTS, "keys": {"id_operation": {"non_null": 50_000, "distinct": 35_000}}}
    assert suggest(mixed) == (None, "id_operation revient 1,4 fois en moyenne sur 30 partitions : ni une photo par partition, ni des ajouts nets")
    assert suggest({**SNAPSHOT, "partitions": 1})[0] is None
    assert suggest({**SNAPSHOT, "partitioned_by": ["annee", "mois"]})[0] is None
    assert suggest({**SNAPSHOT, "keys": {}})[0] is None
    assert suggest({"rows": 10, "partitions": 0}) == (None, "")


def test_a_partitioned_table_without_a_mode_warns_unless_its_rows_are_stored_once():
    assert reading({}, SNAPSHOT)["warn"] is True
    assert reading({}, INCREMENTS)["warn"] is False   # read as it is, each row once: nothing counted twice
    assert reading({"read_mode": "last", "partition_column": "date_technique"}, SNAPSHOT)["warn"] is False
    assert reading({}, {"rows": 10, "partitions": 0})["warn"] is False


def test_what_the_agent_lacks_to_read_the_table():
    assert reading({"read_mode": "last"}, SNAPSHOT, COLUMNS)["problems"] == ["il manque la colonne de partition"]
    live = {"read_mode": "live", "key_columns": ["id_client"], "version_column": "time"}
    assert reading(live, {}, COLUMNS)["problems"] == ["time n'est pas une colonne de la table"]
    assert reading({"read_mode": "first"}, {}, COLUMNS)["problems"] == ["mode inconnu : first"]


def test_partitions_not_read_by_an_older_profile():
    assert reading({}, {"rows": 10}, partitioned_engine=True)["unread"] is True
    assert reading({}, {"rows": 10, "partitions": 0}, partitioned_engine=True)["unread"] is False
    assert reading({}, {}, partitioned_engine=True)["unread"] is False   # not profiled at all


def test_the_mode_is_chosen_with_its_columns_then_removed():
    entry = {"description": "Clients.", "read_mode": "live", "key_columns": ["id_client"], "version_column": "date_maj"}
    last = edit_read_mode(entry, "last", "date_technique", columns=COLUMNS)
    assert last == {"description": "Clients.", "read_mode": "last", "partition_column": "date_technique"}
    live = edit_read_mode(last, "live", key_columns=["id_client", "id_client"], version_column="date_maj", columns=COLUMNS)
    assert live == {"description": "Clients.", "read_mode": "live", "key_columns": ["id_client"], "version_column": "date_maj"}
    assert edit_read_mode(live, None) == {"description": "Clients."}


@pytest.mark.parametrize("mode, partition, keys, version, message", [
    ("last", None, [], None, "colonne de partition"),
    ("live", None, [], "date_maj", "Choisis la clé"),
    ("live", None, ["id_client"], None, "colonne de version"),
    ("live", None, ["date_maj"], "date_maj", "à la fois"),
    ("all", "dt", [], None, "Colonne inconnue : dt"),
    ("first", None, [], None, "inconnu"),
])
def test_a_mode_the_agent_could_not_read_is_refused(mode, partition, keys, version, message):
    with pytest.raises(ValueError, match=message):
        edit_read_mode({}, mode, partition, keys, version, COLUMNS)


def test_the_partition_columns_read_from_describe_formatted():
    hive = [("customerid", "int", ""), ("time", "timestamp", ""), ("", None, None), ("# Partition Information", None, None),
            ("# col_name", "data_type", "comment"), ("date_technique", "date", ""), ("", None, None),
            ("# Detailed Table Information", None, None), ("Database:", "chinook", None)]
    impala = [("# col_name            ", "data_type", "comment"), ("", None, None), ("id", "int", None), ("", None, None),
              ("# Partition Information", None, None), ("# col_name            ", "data_type", "comment"), ("", None, None),
              ("annee", "int", None), ("mois", "int", None), ("", None, None), ("# Detailed Table Information", None, None)]
    assert partition_columns(hive) == ["date_technique"]
    assert partition_columns(impala) == ["annee", "mois"]
    assert partition_columns(hive[:3] + hive[6:]) == []
    assert partition_count([("date_technique=2026-09-25",), ("date_technique=2026-09-26",)]) == 2
    assert partition_count([("2026", "1", 120), ("2026", "2", 98), ("Total", "", 218)]) == 2   # Impala adds a total
