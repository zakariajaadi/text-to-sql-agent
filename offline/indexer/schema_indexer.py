# offline/indexer/schema_indexer.py

"""
Builds the indexes read by the agent at query time:
    - the Chroma schema index (resources.get_schema_retriever)
    - the value index of the categorical columns (lib/value_index.py)

Each table becomes one document (DDL + description header + inline column comments)
indexed in a shared Chroma store. No LLM is involved: descriptions come from the reviewed
semantic_layer/<db>/tables.yaml, the values of its categorical columns are read from the
database, and undocumented tables are indexed with their raw DDL.

Run it last, once the generated YAML files are reviewed (see offline/cli.py).
Pipeline order: fetch_values → build_documents → index → index_values, then the trace of the run
(offline/indexer/trace.py), read by the studio.

Incremental unless full: the values of a table whose data has not changed since the last index are kept
(offline/indexer/value_cache.py, data_version.py), and a document whose text has not changed is not embedded
again. Both give the same index as reading and embedding everything again.
"""

import time
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from loguru import logger

from config import cfg
from lib.column_pruning import get_column_types
from lib.semantics import excluded_columns, load_fk_relations, load_semantics, render_table
from lib.value_index import write_value_index
from offline.indexer.categorical_values import (
    CategoricalValuesError,
    fetch_categorical_columns_values,
    fits_in_prompt,
)
from offline.indexer.data_version import data_version
from offline.indexer.trace import write_trace
from offline.indexer.value_cache import kept_columns, load_value_cache, write_value_cache
from resources import db_registry, vector_store


class SchemaIndexer:
    """
    Builds the schema index of a single database.

    Each table becomes one Langchain Document in the shared Chroma store, tagged with
    its database name in metadata. The retriever can then filter by database
    at query time without requiring one vector store per database.

    Attributes:
        db:      SQLDatabase connection for the target database.
        db_name: Name of the target database.
        vsm:     VectorStoreManager pointing to the shared Chroma store.
    """

    def __init__(self, db_name: str):
        """
        Args:
            db_name: Name of the database to index. Must match an entry
                     in config.yaml databases[].name.
        """
        self.db      = db_registry[db_name]
        self.db_name = db_name

        # Single shared Chroma store — same persist_directory for all databases
        self.vsm = vector_store
        # {table: {version, columns, values}} of this run, for the cache of the next one (fetch_values, run)
        self.cache: dict[str, dict] = {}

    def fetch_values(self, full: bool = False) -> dict[str, dict[str, dict[str, int]]]:
        """
        Values of the categorical columns of the documented tables: {table: {column: {value: row count}}}.
        Read from the database, or kept from the last index for a table whose data has the same version
        (data_version), a column new to the categorical ones read anyway; full: every value read again.

        Table after table: each GROUP BY runs a Hive job, and concurrent jobs fail on a local Tez.

        Raises CategoricalValuesError, once every table is read, when the query of some columns
        still failed after its retries: the run stops before writing any index, rather than
        indexing documents and a value index silently missing those value lists.
        """
        semantics = load_semantics(self.db_name)
        previous  = {} if full else load_value_cache(self.db_name)

        values, failed, kept_tables, self.cache = {}, [], 0, {}
        for table in [t for t in self.db.get_usable_table_names() if t in semantics]:
            # An excluded column is never read, even when declared categorical
            excluded = excluded_columns(semantics[table])
            columns  = [c for c in semantics[table].get("categorical_columns") or [] if c not in excluded]
            version  = data_version(self.db, table) if columns else None
            kept     = [] if full else kept_columns(previous.get(table), version, columns)
            to_read  = [c for c in columns if c not in kept]
            if kept and not to_read:
                logger.info(f"[{self.db_name}] Values kept: {table} ({len(kept)} column(s), same data since the last index)")
                kept_tables += 1
            else:
                logger.info(f"[{self.db_name}] Fetching values: {table} ({len(to_read)} column(s)"
                            + (f", {len(kept)} kept" if kept else "") + ")")
            try:
                read = fetch_categorical_columns_values(self.db, self.db_name, table, to_read) if to_read else {}
            except CategoricalValuesError as e:
                # Keep reading the other tables: every failed column is reported at once
                failed += e.columns
                continue
            # The values kept (a column without a value list last time has none), in the order of the columns
            found = {**{c: previous[table]["values"][c] for c in kept if c in previous[table]["values"]}, **read}
            values[table] = {c: found[c] for c in columns if c in found}
            if version:
                self.cache[table] = {"version": version, "columns": columns, "values": values[table]}

        if failed:
            raise CategoricalValuesError(self.db_name, failed)
        if kept_tables:
            logger.info(f"[{self.db_name}] Values kept for {kept_tables} table(s): their data has not changed since the last index")
        return values

    def build_documents(self, values: dict[str, dict[str, dict[str, int]]]) -> list[Document]:
        """
        Build one Document per table, ready to be embedded into Chroma — no LLM.

        Tables documented in `semantic_layer/<db>/tables.yaml` are rendered by `render_table()`
        (description header + inline column comments, with the values of the categorical
        columns that fit in the prompt). Undocumented tables are indexed with their raw DDL,
        and a warning suggests running `semantics` for them.

        values: the categorical values, from fetch_values
        """
        tables       = self.db.get_usable_table_names()
        semantics    = load_semantics(self.db_name)
        fk_relations = load_fk_relations(self.db_name)
        dialect      = cfg.get_db(self.db_name).dialect

        # DDL + sample rows: one database round-trip per table. Tables are independent: fetched
        # in parallel (max_workers stays under the SQLAlchemy pool size: 5 + 10 overflow)
        with ThreadPoolExecutor(max_workers=8) as executor:
            ddls = dict(zip(tables, executor.map(
                lambda table: self.db.get_table_info(table_names=[table]).strip(), tables
            )))

        # Column types of every table: a join key stored as text on one side is cast in "-- Joins"
        column_types = {
            f"{table}.{column}": column_type
            for table, ddl in ddls.items()
            for column, column_type in get_column_types(ddl).items()
        }

        def build_document(table: str) -> Document:
            ddl = ddls[table]

            if table in semantics:
                # Long value lists are left to the value index, searched at query time
                inline  = {
                    column: sorted(counts)
                    for column, counts in values.get(table, {}).items()
                    if fits_in_prompt(counts)
                }
                content = render_table(table, ddl, semantics[table], fk_relations, inline,
                                       column_types, dialect)
                logger.info(f"[{self.db_name}] Built: {table}")
            else:
                content = ddl
                logger.warning(f"[{self.db_name}] Built: {table} (raw DDL, no semantics — run `make semantics`)")

            return Document(
                page_content=content,
                metadata={"table_name": table, "db_name": self.db_name},
            )

        return [build_document(table) for table in tables]

    def index(self, documents: list[Document], full: bool = False):
        """
        Embed documents into the shared Chroma store with deterministic IDs.

        Write first: the documents whose text (or embedding model) changed since the last index are upserted,
        every one with full, then the other documents of this database are deleted, so tables dropped or
        renamed since the last run do not linger in the index. A document left as it was keeps its embedding:
        the same text gives the same one. A failure while embedding (API quota, network) leaves the previous
        index of the database intact.
        IDs are built as "<db_name>_<table_name>".
        """
        store = self.vsm.get_store()

        # Build deterministic IDs; the embedding model in the metadata: another one embeds every document again
        ids = [
            f"{doc.metadata['db_name']}_{doc.metadata['table_name']}"
            for doc in documents
        ]
        for doc in documents:
            doc.metadata["embedding_model"] = cfg.retrieval.embedding_model

        # Index the documents that changed: add_documents upserts, an existing ID is replaced
        before = {}
        if ids and not full:
            got    = store.get(ids=ids, include=["documents", "metadatas"])
            before = {i: (text, metadata) for i, text, metadata in zip(got["ids"], got["documents"], got["metadatas"])}
        changed = [(i, doc) for i, doc in zip(ids, documents) if before.get(i) != (doc.page_content, doc.metadata)]
        if changed:
            store.add_documents([doc for _, doc in changed], ids=[i for i, _ in changed])
        if len(changed) < len(documents):
            logger.info(f"[{self.db_name}] Embedded {len(changed)} of {len(documents)} schemas, "
                        f"the {len(documents) - len(changed)} others unchanged since the last index")

        # Only then purge the documents of the tables that are gone
        indexed = store.get(where={"db_name": {"$eq": self.db_name}}, include=[])["ids"]
        stale   = sorted(set(indexed) - set(ids))
        if stale:
            store.delete(ids=stale)
            logger.info(f"[{self.db_name}] Removed {len(stale)} stale schema(s): {', '.join(stale)}")

        logger.info(f"[{self.db_name}] Indexed {len(documents)} schemas to Chroma.")

    def index_values(self, values: dict[str, dict[str, dict[str, int]]]):
        """Write the value index of this database (lib/value_index.py), every categorical column included."""
        inline = {
            (table, column)
            for table, columns in values.items()
            for column, counts in columns.items()
            if fits_in_prompt(counts)
        }
        write_value_index(self.db_name, values, inline)

    def trace(self, documents: list[Document], values: dict[str, dict[str, dict[str, int]]],
              semantics: dict, fk_relations: list[tuple[str, str]]) -> dict:
        """
        What went into each document (offline/indexer/trace.py): the semantic layer read when the run
        started, the columns the document shows, the number of values indexed. Never a value.

        semantics, fk_relations: read before the documents are built. A file corrected during the run
        then shows as changed since the index, never the reverse.
        """
        tables = {}
        for document in documents:
            table = document.metadata["table_name"]
            tables[table] = {
                "semantics": semantics.get(table),
                "relations": [[source, target] for source, target in fk_relations
                              if table in (source.split(".")[0], target.split(".")[0])],
                "columns":   get_column_types(document.page_content),
                "chars":     len(document.page_content),
                "values":    {column: {"count": len(counts), "inline": fits_in_prompt(counts)}
                              for column, counts in values.get(table, {}).items()},
            }
        return {"built_at": time.time(), "sample_rows": cfg.get_db(self.db_name).sample_rows, "tables": tables}

    def run(self, full: bool = False):
        """
        Run the pipeline: fetch_values → build_documents → index → index_values, then write its trace and the
        values read with the version of their data, for the next run. full: every value read, every document embedded.
        """
        semantics, fk_relations = load_semantics(self.db_name), load_fk_relations(self.db_name)
        values    = self.fetch_values(full)
        documents = self.build_documents(values)
        self.index(documents, full)
        self.index_values(values)
        write_trace(self.db_name, self.trace(documents, values, semantics, fk_relations))
        write_value_cache(self.db_name, self.cache)
