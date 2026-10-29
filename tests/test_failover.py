"""
Failover across Impala coordinators: the IMPALA_COORDINATORS list in the config (src/config.py),
the in-process name resolution and the startup failover (src/lib/database.py), and the query retry
moving to the next coordinator instead of the same host (agent/nodes.py _run_with_reconnect).
"""

import socket
from types import SimpleNamespace

import pytest

from agent import nodes
from config import DatabaseConfig
from lib import database


# --- config: the coordinator list -------------------------------------------------------------

def test_coordinators_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("IMPALA_HOST", "dn2.example")
    monkeypatch.setenv("IMPALA_COORDINATORS", "dn1.example=10.0.0.1, dn2.example=10.0.0.2")
    db = DatabaseConfig(name="Sales", dialect="impala")
    assert db.coordinators == ["dn1.example", "dn2.example"]
    assert db.coordinator_ips == {"dn1.example": "10.0.0.1", "dn2.example": "10.0.0.2"}
    assert db.current_host == "dn2.example"


def test_the_env_host_joins_a_list_that_misses_it(monkeypatch):
    monkeypatch.setenv("IMPALA_HOST", "lb.example")
    monkeypatch.setenv("IMPALA_COORDINATORS", "dn1.example=10.0.0.1")
    assert DatabaseConfig(name="Sales", dialect="impala").coordinators == ["lb.example", "dn1.example"]


def test_a_single_coordinator_without_the_variable(monkeypatch):
    monkeypatch.delenv("IMPALA_COORDINATORS", raising=False)
    monkeypatch.setenv("IMPALA_HOST", "dn1.example")
    db = DatabaseConfig(name="Sales", dialect="impala")
    assert db.coordinators == ["dn1.example"]
    assert db.coordinator_ips == {}


def test_switch_host_rotates_and_rewrites_the_uri(monkeypatch):
    monkeypatch.setenv("IMPALA_HOST", "dn1.example")
    monkeypatch.setenv("IMPALA_COORDINATORS", "dn1.example=10.0.0.1,dn2.example=10.0.0.2")
    db = DatabaseConfig(name="Sales", dialect="impala")
    assert db.switch_host() == "dn2.example"
    assert db.uri == "impala4://dn2.example:21050/sales"
    assert db.switch_host() == "dn1.example"   # wraps around to the preferred host


# --- lib/database: name resolution and startup ------------------------------------------------

def test_resolve_hosts_locally_maps_listed_names_only():
    original = socket.getaddrinfo
    try:
        database.resolve_hosts_locally({"dn1.internal": "127.0.0.1"})
        assert socket.getaddrinfo("dn1.internal", 80)[0][4][0] == "127.0.0.1"
        assert socket.getaddrinfo("localhost", 80)   # unlisted names still resolve
    finally:
        socket.getaddrinfo = original


def _config(coordinators):
    cfg = SimpleNamespace(name="Sales", coordinators=list(coordinators), current_host=coordinators[0] if coordinators else "")

    def switch_host():
        index = (cfg.coordinators.index(cfg.current_host) + 1) % len(cfg.coordinators)
        cfg.current_host = cfg.coordinators[index]
        return cfg.current_host

    cfg.switch_host = switch_host
    return cfg


def test_startup_moves_to_the_next_coordinator(monkeypatch):
    db_cfg = _config(["dn1", "dn2"])
    builds = []

    def build(cfg):
        builds.append(cfg.current_host)
        if cfg.current_host == "dn1":
            raise RuntimeError("unreachable at startup")
        return "database"

    monkeypatch.setattr(database, "build_database", build)
    assert database.build_database_failing_over(db_cfg) == "database"
    assert builds == ["dn1", "dn2"]


def test_startup_raises_once_every_coordinator_failed(monkeypatch):
    db_cfg = _config(["dn1", "dn2"])
    monkeypatch.setattr(database, "build_database", lambda cfg: (_ for _ in ()).throw(RuntimeError("unreachable")))
    with pytest.raises(RuntimeError):
        database.build_database_failing_over(db_cfg)


# --- agent: the query retry hops coordinators -------------------------------------------------

@pytest.fixture
def agent_db(monkeypatch):
    """Three coordinators, a registry entry, no sleep: tests drive run_capped_query and switch_coordinator."""
    monkeypatch.setattr(nodes, "cfg", SimpleNamespace(get_db=lambda name: SimpleNamespace(coordinators=["a", "b", "c"])))
    monkeypatch.setattr(nodes, "RECONNECT_DELAY_S", 0)
    monkeypatch.setitem(nodes.db_registry, "bank", SimpleNamespace(_engine=SimpleNamespace(dispose=lambda: None)))


def test_failover_switches_then_reruns_the_query(agent_db, monkeypatch):
    calls = {"runs": 0, "switches": 0}

    def run(db_name, query, max_rows, cancel_key=None):
        calls["runs"] += 1
        if calls["runs"] == 1:
            raise TimeoutError("read timeout")
        return "rows"

    monkeypatch.setattr(nodes, "run_capped_query", run)
    monkeypatch.setattr(nodes, "switch_coordinator", lambda db_name: calls.update(switches=calls["switches"] + 1) or "b")
    assert nodes._run_with_reconnect("bank", "SELECT 1", 10) == "rows"
    assert calls == {"runs": 2, "switches": 1}


def test_failover_skips_a_dead_next_coordinator(agent_db, monkeypatch):
    switches = []

    def switch(db_name):
        switches.append(db_name)
        if len(switches) == 1:
            raise RuntimeError("unreachable")   # the next coordinator is down too: move on
        return "c"

    monkeypatch.setattr(nodes, "run_capped_query",
                        lambda *args, **kwargs: "rows" if len(switches) == 2 else (_ for _ in ()).throw(TimeoutError()))
    monkeypatch.setattr(nodes, "switch_coordinator", switch)
    assert nodes._run_with_reconnect("bank", "SELECT 1", 10) == "rows"
    assert len(switches) == 2


def test_failover_raises_once_every_coordinator_failed(agent_db, monkeypatch):
    monkeypatch.setattr(nodes, "run_capped_query", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("read timeout")))
    monkeypatch.setattr(nodes, "switch_coordinator", lambda db_name: "next")
    with pytest.raises(TimeoutError):
        nodes._run_with_reconnect("bank", "SELECT 1", 10)


def test_a_single_coordinator_keeps_the_same_host_retry(agent_db, monkeypatch):
    monkeypatch.setattr(nodes, "cfg", SimpleNamespace(get_db=lambda name: SimpleNamespace(coordinators=[])))
    runs = []

    def run(db_name, query, max_rows, cancel_key=None):
        runs.append(db_name)
        if len(runs) == 1:
            raise TimeoutError("timed out")
        return "rows"

    monkeypatch.setattr(nodes, "run_capped_query", run)
    monkeypatch.setattr(nodes, "switch_coordinator", lambda db_name: pytest.fail("no coordinator to switch to"))
    assert nodes._run_with_reconnect("bank", "SELECT 1", 10) == "rows"
    assert runs == ["bank", "bank"]


def test_a_query_error_is_not_retried(agent_db, monkeypatch):
    monkeypatch.setattr(nodes, "run_capped_query", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad sql")))
    with pytest.raises(ValueError):
        nodes._run_with_reconnect("bank", "SELECT 1", 10)
