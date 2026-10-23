"""FK expansion (lib/fk_graph.py): the tables on a shortest join path between selected tables are added."""

from lib.fk_graph import expand_fk


def graph(*edges: tuple[str, str]) -> dict[str, list[str]]:
    """Bidirectional FK graph, as resources.build_fk_graph builds it."""
    g: dict[str, list[str]] = {}
    for a, b in edges:
        g.setdefault(a, []).append(b)
        g.setdefault(b, []).append(a)
    return g


# Two id families bridged by a mapping table, a hub with satellites
BANK = graph(
    ("client_master", "client"), ("client_master", "personne"), ("reclamations", "client"),
    ("impaye", "personne"), ("incident", "personne"), ("gestionnaire", "personne"),
)


def test_two_tables_in_between_are_added():
    assert expand_fk(["client", "impaye"], BANK) == ["client", "impaye", "client_master", "personne"]


def test_one_bridge_is_added_as_before():
    assert expand_fk(["impaye", "incident"], BANK) == ["impaye", "incident", "personne"]


def test_every_tied_bridge_is_added():
    g = graph(("order", "line"), ("line", "product"), ("order", "promo"), ("promo", "product"))
    assert expand_fk(["order", "product"], g) == ["order", "product", "line", "promo"]


def test_tables_joined_directly_or_too_far_add_nothing():
    assert expand_fk(["client", "client_master"], BANK) == ["client", "client_master"]
    # reclamations → client → client_master → personne → impaye: 4 joins
    assert expand_fk(["reclamations", "impaye"], BANK) == ["reclamations", "impaye"]
    assert expand_fk(["reclamations", "impaye"], BANK, max_joins=4) == [
        "reclamations", "impaye", "client", "client_master", "personne"]


def test_unconnected_tables_add_nothing():
    assert expand_fk(["client", "island"], {**BANK, "island": []}) == ["client", "island"]
