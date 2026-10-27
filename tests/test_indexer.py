"""Offline indexer (offline/indexer/): Chroma refresh order, retries of the categorical values queries, and the
incremental index (values kept for a table whose data did not change, documents embedded when their text did)."""

import json
import sys
import types

import pytest
from langchain_core.documents import Document
from loguru import logger

# schema_indexer imports the runtime resources (API keys, databases, Chroma index): stubbed here
sys.modules.setdefault("resources", types.SimpleNamespace(db_registry={}, vector_store=None))

from offline.indexer import categorical_values, schema_indexer, value_cache  # noqa: E402
from offline.indexer.categorical_values import (  # noqa: E402
    CategoricalValuesError,
    fetch_categorical_columns_values,
)
from offline.indexer.data_version import data_version, files_version, table_parameters  # noqa: E402
from offline.indexer.schema_indexer import SchemaIndexer  # noqa: E402


class FakeStore:
    """The few Chroma calls of SchemaIndexer.index(), over a dict {id: Document}."""

    def __init__(self, documents: dict[str, Document], fail_on_add: bool = False):
        self.documents   = dict(documents)
        self.fail_on_add = fail_on_add

    def add_documents(self, documents, ids):
        if self.fail_on_add:
            raise RuntimeError("402 RESOURCE_EXHAUSTED")
        self.embedded = list(ids)
        self.documents.update(zip(ids, [Document(page_content=d.page_content, metadata=dict(d.metadata)) for d in documents]))

    def get(self, ids=None, where=None, include=()):
        if ids is not None:
            found = [i for i in ids if i in self.documents]
            return {"ids": found, "documents": [self.documents[i].page_content for i in found],
                    "metadatas": [dict(self.documents[i].metadata) for i in found]}
        db_name = where["db_name"]["$eq"]
        return {"ids": [i for i, doc in self.documents.items() if doc.metadata["db_name"] == db_name]}

    def delete(self, ids):
        for i in ids:
            del self.documents[i]


class FakeDB:
    """A database that only lists its tables (the GROUP BY results come from the column_values fixture)."""

    def __init__(self, tables: list[str]):
        self.tables = tables

    def get_usable_table_names(self):
        return self.tables


def doc(db_name: str, table: str, content: str = "") -> Document:
    return Document(page_content=content, metadata={"db_name": db_name, "table_name": table})


@pytest.fixture
def indexer(monkeypatch):
    """A SchemaIndexer of the database "test", on a fake store holding its previous index and another database's."""
    store = FakeStore({
        "test_customers": doc("test", "customers", "old"),
        "test_orders":    doc("test", "orders", "old"),
        "test_legacy":    doc("test", "legacy", "old"),
        "other_orders":   doc("other", "orders", "old"),
    })
    monkeypatch.setattr(schema_indexer, "db_registry", {"test": FakeDB(["customers", "orders"])})
    monkeypatch.setattr(schema_indexer, "vector_store", types.SimpleNamespace(get_store=lambda: store))
    return SchemaIndexer("test")


@pytest.fixture
def column_values(monkeypatch):
    """
    Fake GROUP BY results, set per test as {(table, column): [result of each call]}:
    a dict is returned, an exception is raised. Retries do not wait.
    """
    results = {}

    def column_value_counts(db, table, column, limit):
        result = results[(table, column)].pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(categorical_values, "_column_value_counts", column_value_counts)
    monkeypatch.setattr(categorical_values, "RETRY_DELAY_SECONDS", 0)
    return results


# =============================================================================
# SchemaIndexer.index
# =============================================================================

def test_index_replaces_the_documents_and_removes_the_dropped_tables(indexer):
    indexer.index([doc("test", "customers", "new"), doc("test", "orders", "new")])

    documents = indexer.vsm.get_store().documents
    assert sorted(documents) == ["other_orders", "test_customers", "test_orders"]
    assert documents["test_customers"].page_content == "new"
    assert documents["other_orders"].page_content == "old"


def test_a_failed_embedding_leaves_the_previous_index_intact(indexer):
    store = indexer.vsm.get_store()
    store.fail_on_add = True

    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        indexer.index([doc("test", "customers", "new")])

    assert sorted(store.documents) == ["other_orders", "test_customers", "test_legacy", "test_orders"]
    assert store.documents["test_customers"].page_content == "old"


def test_the_trace_holds_what_went_into_each_document_never_a_value(indexer, monkeypatch):
    monkeypatch.setattr(schema_indexer, "cfg", types.SimpleNamespace(get_db=lambda name: types.SimpleNamespace(sample_rows=0)))
    customers = "CREATE TABLE `customers` (\n\t`id` INT,  -- Id.\n\t`country` STRING  -- Country. Values: 'USA'\n)"
    semantics = {"customers": {"description": "Customers.", "categorical_columns": ["country", "city"]}}
    values    = {"customers": {"country": {"USA": 13}, "city": {f"city {i}": 1 for i in range(40)}}}

    trace = indexer.trace([doc("test", "customers", customers), doc("test", "orders", "CREATE TABLE `orders` (\n\t`id` INT\n)")],
                          values, semantics, [("orders.customerid", "customers.id"), ("a.x", "b.y")])
    assert trace["sample_rows"] == 0 and trace["built_at"] > 0
    assert trace["tables"]["customers"] == {
        "semantics": semantics["customers"], "relations": [["orders.customerid", "customers.id"]],
        "columns": {"id": "INT", "country": "STRING"}, "chars": len(customers),
        "values": {"country": {"count": 1, "inline": True}, "city": {"count": 40, "inline": False}},
    }
    assert trace["tables"]["orders"]["semantics"] is None


# =============================================================================
# Categorical values
# =============================================================================

def test_a_transient_failure_is_retried(column_values):
    column_values[("customers", "region")] = [RuntimeError("Tez session lost"), {"WA": 3, "SP": 2}]

    values = fetch_categorical_columns_values(FakeDB([]), "test", "customers", ["region"])
    assert values == {"region": {"WA": 3, "SP": 2}}


def test_a_column_failing_after_its_retries_fails_with_every_failed_column(column_values):
    error = RuntimeError("Tez session lost")
    column_values[("customers", "region")]       = [error] * (categorical_values.QUERY_RETRIES + 1)
    column_values[("customers", "country")]      = [{"USA": 13}]
    column_values[("customers", "contacttitle")] = [error] * (categorical_values.QUERY_RETRIES + 1)

    with pytest.raises(CategoricalValuesError) as e:
        fetch_categorical_columns_values(FakeDB([]), "test", "customers", ["region", "country", "contacttitle"])
    assert e.value.columns == ["customers.region", "customers.contacttitle"]


def test_fetch_values_reports_the_failed_columns_of_every_table(indexer, column_values, monkeypatch):
    monkeypatch.setattr(schema_indexer, "load_semantics", lambda db_name: {
        "customers": {"categorical_columns": ["region", "country"]},
        "orders":    {"categorical_columns": ["status"]},
    })
    error = RuntimeError("Tez session lost")
    column_values[("customers", "region")]  = [error] * (categorical_values.QUERY_RETRIES + 1)
    column_values[("customers", "country")] = [{"USA": 13}]
    column_values[("orders", "status")]     = [error] * (categorical_values.QUERY_RETRIES + 1)

    with pytest.raises(CategoricalValuesError) as e:
        indexer.fetch_values()
    assert e.value.columns == ["customers.region", "orders.status"]
    assert "customers.region, orders.status" in str(e.value)


# =============================================================================
# Incremental index: the values of the tables whose data did not change, the documents whose text did
# =============================================================================

def test_only_the_documents_whose_text_changed_are_embedded_again(indexer):
    store = indexer.vsm.get_store()
    indexer.index([doc("test", "customers", "v1"), doc("test", "orders", "v1")])
    assert store.embedded == ["test_customers", "test_orders"]   # "old", without the embedding model: embedded again

    store.embedded = []
    indexer.index([doc("test", "customers", "v1"), doc("test", "orders", "v2")])
    assert store.embedded == ["test_orders"]
    assert store.documents["test_orders"].page_content == "v2"

    indexer.index([doc("test", "customers", "v1"), doc("test", "orders", "v2")], full=True)
    assert store.embedded == ["test_customers", "test_orders"]


@pytest.fixture
def versions(indexer, column_values, monkeypatch, tmp_path):
    """The data version of each table, set per test ({table: version}); the cache of the values in tmp_path."""
    current = {}
    monkeypatch.setattr(schema_indexer, "data_version", lambda db, table: current.get(table))
    monkeypatch.setattr(value_cache, "value_cache_path", lambda db_name: tmp_path / f"{db_name}.values.json")
    monkeypatch.setattr(schema_indexer, "load_semantics", lambda db_name: {
        "customers": {"categorical_columns": ["country", "city"]},
        "orders":    {"categorical_columns": ["status"]},
    })
    return current


def index_values(indexer) -> dict:
    """fetch_values, then the cache of the run written, as run() does."""
    values = indexer.fetch_values()
    value_cache.write_value_cache("test", indexer.cache)
    return values


def test_the_values_of_a_table_whose_data_did_not_change_are_kept(indexer, column_values, versions, monkeypatch):
    versions.update(customers="iceberg:1", orders="files:a")
    column_values.update({("customers", "country"): [{"USA": 13}], ("customers", "city"): [{"Paris": 2}],
                          ("orders", "status"): [{"shipped": 5}]})
    first = index_values(indexer)

    # nothing changed: not one GROUP BY (column_values has no result left), the same values
    assert index_values(indexer) == first

    # the data of orders changed (a load): only orders is read again
    versions["orders"] = "files:b"
    column_values[("orders", "status")] = [{"shipped": 6}]
    assert index_values(indexer)["orders"] == {"status": {"shipped": 6}}

    # a column new to the categorical ones is read, the others kept
    monkeypatch.setattr(schema_indexer, "load_semantics", lambda db_name: {
        "customers": {"categorical_columns": ["country", "city", "segment"]},
        "orders":    {"categorical_columns": ["status"]},
    })
    column_values[("customers", "segment")] = [{"B2B": 4}]
    assert index_values(indexer)["customers"] == {"country": {"USA": 13}, "city": {"Paris": 2}, "segment": {"B2B": 4}}


def test_without_a_version_or_with_full_every_value_is_read(indexer, column_values, versions):
    versions.update(customers="iceberg:1")   # orders: no version (Hive, Kudu, the least doubt)
    column_values.update({("customers", "country"): [{"USA": 13}] * 3, ("customers", "city"): [{"Paris": 2}] * 3,
                          ("orders", "status"): [{"shipped": 5}] * 3})
    index_values(indexer)
    index_values(indexer)
    assert column_values[("orders", "status")] == [{"shipped": 5}]        # read at each run
    assert column_values[("customers", "country")] == [{"USA": 13}] * 2   # read once, then kept

    indexer.fetch_values(full=True)
    assert column_values[("customers", "country")] == [{"USA": 13}]       # read again


def test_another_max_values_reads_every_value_again(indexer, column_values, versions, tmp_path):
    versions.update(customers="iceberg:1", orders="iceberg:2")
    column_values.update({("customers", "country"): [{"USA": 13}] * 2, ("customers", "city"): [{"Paris": 2}] * 2,
                          ("orders", "status"): [{"shipped": 5}] * 2})
    index_values(indexer)
    cache = json.loads((tmp_path / "test.values.json").read_text())
    cache["max_values"] += 1
    (tmp_path / "test.values.json").write_text(json.dumps(cache))
    index_values(indexer)
    assert column_values[("orders", "status")] == []


# =============================================================================
# The version of the data of a table, from the metastore
# =============================================================================

class FakeEngine:
    """An engine answering DESCRIBE FORMATTED and SHOW FILES with the rows given."""

    def __init__(self, dialect: str, describe: list, files: list | Exception = ()):
        self.dialect = types.SimpleNamespace(name=dialect, identifier_preparer=types.SimpleNamespace(quote=lambda x: f"`{x}`"))
        self.describe, self.files = describe, files

    def connect(self):
        engine = self

        class Connection:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, query):
                rows = engine.describe if str(query).startswith("DESCRIBE") else engine.files
                if isinstance(rows, Exception):
                    raise rows
                return types.SimpleNamespace(fetchall=lambda: list(rows))

        return Connection()


# DESCRIBE FORMATTED of an Iceberg table of Impala, as Hue shows it (awb_silver.client, values made up)
ICEBERG = [("# col_name", "data_type", "comment"), ("id", "int", None), ("", None, None),
           ("# Detailed Table Information", None, None), ("Table Type:", "EXTERNAL_TABLE", None),
           ("Table Parameters:", None, None), ("", "EXTERNAL", "TRUE"),
           ("", "current-snapshot-id ", "5875541796799705853 "), ("", "table_type", "ICEBERG"), ("", None, None),
           ("# Storage Information", None, None), ("", "current-snapshot-id", "a storage field, not read")]
PARQUET = [("Table Parameters:", None, None), ("", "numFiles", "2"), ("# Storage Information", None, None)]
FILES   = [("hdfs://warehouse/t/part-0001.parq", "12.30KB", ""), ("hdfs://warehouse/t/part-0002.parq", "8.10KB", "")]


def database(dialect: str, describe: list, files=()):
    return types.SimpleNamespace(_engine=FakeEngine(dialect, describe, files))


def test_the_parameters_of_describe_formatted():
    assert table_parameters(ICEBERG) == {"EXTERNAL": "TRUE", "current-snapshot-id": "5875541796799705853",
                                         "table_type": "ICEBERG"}


def test_the_data_version_of_a_table():
    assert data_version(database("impala4", ICEBERG), "client") == "iceberg:5875541796799705853"
    # another table on files: its files and their sizes, in any order; a load changes them
    version = data_version(database("impala4", PARQUET, FILES), "t")
    assert version.startswith("files:") and version == data_version(database("impala", PARQUET, FILES[::-1]), "t")
    assert version != data_version(database("impala4", PARQUET, [FILES[0], ("hdfs://warehouse/t/part-0003.parq", "9.00KB", "")]), "t")
    # Kudu, another engine, an error: None, the values are read again
    kudu = [("Table Parameters:", None, None), ("", "storage_handler", "org.apache.hadoop.hive.kudu.KuduStorageHandler")]
    assert data_version(database("impala4", kudu, FILES), "t") is None
    assert data_version(database("hive", ICEBERG), "client") is None
    assert data_version(database("impala4", PARQUET, RuntimeError("AuthorizationException")), "t") is None
    assert files_version([]) == files_version([])


def test_a_view_or_a_storage_handler_has_no_version_and_no_warning():
    # awb_silver.connaissance_client_automobile: a view over another database, SHOW FILES refused on it
    view  = [("# Detailed Table Information", None, None), ("Table Type:", "VIRTUAL_VIEW      ", None),
             ("Table Parameters:", None, None), ("", "transient_lastDdlTime", "1790610338"), ("", None, None),
             ("# Storage Information", None, None)]
    hbase = [("Table Type:", "EXTERNAL_TABLE", None), ("Table Parameters:", None, None),
             ("", "storage_handler", "org.apache.hadoop.hive.hbase.HBaseStorageHandler")]
    refused = RuntimeError("AnalysisException: SHOW FILES not applicable to a non hdfs table")
    warnings = []
    sink = logger.add(warnings.append, level="WARNING", format="{message}")
    try:
        assert data_version(database("impala4", view, refused), "connaissance_client_automobile") is None
        assert data_version(database("impala4", hbase, refused), "t") is None
        assert warnings == []
        # a real error still says so
        assert data_version(database("impala4", PARQUET, refused), "t") is None
        assert len(warnings) == 1 and "non hdfs table" in warnings[0]
    finally:
        logger.remove(sink)
