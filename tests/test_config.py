"""Database settings (src/config.py): engine URI and driver options, local config overrides."""

import shutil

import pytest

import config
from config import DatabaseConfig


def test_hive_uri_uses_the_lowercased_name(monkeypatch):
    monkeypatch.setenv("HIVE_HOST", "hive")
    db = DatabaseConfig(name="Chinook", dialect="hive")
    assert db.uri == "hive://hive@hive:10000/chinook"
    assert db.connect_args == {}


def test_impala_defaults_to_kerberos(monkeypatch):
    for var in ("IMPALA_PORT", "IMPALA_AUTH_MECHANISM", "IMPALA_KERBEROS_SERVICE", "IMPALA_USE_SSL",
                "IMPALA_CONNECT_TIMEOUT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("IMPALA_HOST", "impala-coordinator.example")
    db = DatabaseConfig(name="Sales", dialect="impala")
    assert db.uri == "impala4://impala-coordinator.example:21050/sales"
    assert db.connect_args == {"auth_mechanism": "GSSAPI", "kerberos_service_name": "impala", "use_ssl": False,
                               "timeout": 20, "retries": 1}


def test_impala_driver_options_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("IMPALA_AUTH_MECHANISM", "NOSASL")
    monkeypatch.setenv("IMPALA_USE_SSL", "True")
    db = DatabaseConfig(name="Sales", dialect="impala")
    assert db.connect_args["auth_mechanism"] == "NOSASL"
    assert db.connect_args["use_ssl"] is True


def test_impala_queries_are_limited_by_default(monkeypatch):
    for var in ("IMPALA_EXEC_TIME_LIMIT_S", "IMPALA_MEM_LIMIT", "IMPALA_AUTH_MECHANISM"):
        monkeypatch.delenv(var, raising=False)
    db = DatabaseConfig(name="Sales", dialect="impala")
    assert db.query_options == {"EXEC_TIME_LIMIT_S": "120", "MEM_LIMIT": "2g"}
    # Sent with every query of the engine, next to the driver options
    args = db.engine_args()
    assert args["execution_options"] == {"cursor_configuration": {"EXEC_TIME_LIMIT_S": "120", "MEM_LIMIT": "2g"}}
    assert args["connect_args"]["auth_mechanism"] == "GSSAPI"


def test_impala_limits_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("IMPALA_EXEC_TIME_LIMIT_S", "30")
    monkeypatch.setenv("IMPALA_MEM_LIMIT", "")      # empty: no memory limit
    assert DatabaseConfig(name="Sales", dialect="impala").query_options == {"EXEC_TIME_LIMIT_S": "30"}


def test_hive_has_no_engine_args():
    assert DatabaseConfig(name="Chinook", dialect="hive").engine_args() == {}


def test_unknown_dialect_is_rejected():
    with pytest.raises(ValueError, match="Unsupported dialect"):
        DatabaseConfig(name="Sales", dialect="oracle")


def test_local_config_replaces_the_databases(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    shutil.copy(config.BASE_DIR / "config" / "config.yaml", tmp_path / "config" / "config.yaml")
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)

    # No local file: the databases of config.yaml
    assert [db.name for db in config.load_config().databases] == ["Chinook", "Northwind"]

    (tmp_path / "config" / "config.local.yaml").write_text(
        "databases:\n"
        "  - {name: Sales, dialect: impala, sample_rows: 0}\n"
    )
    databases = config.load_config().databases
    assert [(db.name, db.dialect) for db in databases] == [("Sales", "impala")]
    assert databases[0].sample_rows == 0

    # The tables are chosen in the studio (semantic_layer/<db>/scope.yaml): a list left in the config stops the start
    (tmp_path / "config" / "config.local.yaml").write_text(
        "databases:\n"
        "  - {name: Sales, dialect: impala, include_tables: [orders]}\n"
    )
    with pytest.raises(TypeError, match="include_tables"):
        config.load_config()
