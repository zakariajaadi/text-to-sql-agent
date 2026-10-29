"""offline.doctor: coordinator list parsing and in-process name resolution."""

import socket

from offline.doctor import coordinators, resolve_locally


def test_coordinators_parses_pairs_and_skips_malformed(monkeypatch):
    monkeypatch.setenv("IMPALA_COORDINATORS", "dn1.example.com=10.0.0.1, dn2.example.com=10.0.0.2, broken")
    assert coordinators() == [("dn1.example.com", "10.0.0.1"), ("dn2.example.com", "10.0.0.2")]


def test_coordinators_defaults_to_the_env_host(monkeypatch):
    monkeypatch.delenv("IMPALA_COORDINATORS", raising=False)
    monkeypatch.setenv("IMPALA_HOST", "dn1.example.com")
    monkeypatch.setenv("IMPALA_IP", "10.0.0.1")
    assert coordinators() == [("dn1.example.com", "10.0.0.1")]


def test_coordinators_empty_without_env(monkeypatch):
    for var in ("IMPALA_COORDINATORS", "IMPALA_HOST", "IMPALA_IP"):
        monkeypatch.delenv(var, raising=False)
    assert coordinators() == []


def test_resolve_locally_maps_listed_names_only():
    original = socket.getaddrinfo
    try:
        resolve_locally({"dn1.internal": "127.0.0.1"})
        assert socket.getaddrinfo("dn1.internal", 80)[0][4][0] == "127.0.0.1"
        assert socket.getaddrinfo("localhost", 80)  # unlisted names still resolve
    finally:
        socket.getaddrinfo = original
