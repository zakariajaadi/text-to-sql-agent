"""Offline indexer (offline/indexer/): Chroma refresh order, and retries of the categorical values queries."""

import sys
import types

import pytest
from langchain_core.documents import Document

# schema_indexer imports the runtime resources (API keys, databases, Chroma index): stubbed here
sys.modules.setdefault("resources", types.SimpleNamespace(db_registry={}, vector_store=None))

from offline.indexer import categorical_values, schema_indexer  # noqa: E402
from offline.indexer.categorical_values import (  # noqa: E402
    CategoricalValuesError,
    fetch_categorical_columns_values,
)
from offline.indexer.schema_indexer import SchemaIndexer  # noqa: E402


class FakeStore:
    """The few Chroma calls of SchemaIndexer.index(), over a dict {id: Document}."""

    def __init__(self, documents: dict[str, Document], fail_on_add: bool = False):
        self.documents   = dict(documents)
        self.fail_on_add = fail_on_add

    def add_documents(self, documents, ids):
        if self.fail_on_add:
            raise RuntimeError("402 RESOURCE_EXHAUSTED")
        self.documents.update(zip(ids, documents))

    def get(self, where, include):
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
