# offline/indexer/schema_indexer.py

"""
Builds the Chroma schema index read by the agent at query time (resources.get_schema_retriever).

Each table becomes one document (DDL + description header + inline column comments)
indexed in a shared Chroma store. No LLM is involved: descriptions come from the reviewed
semantic_layer/<db>/tables.yaml, the values of its categorical columns are read from the
database, and undocumented tables are indexed with their raw DDL.

Run it last, once the generated YAML files are reviewed (see offline/cli.py).
Pipeline order: build_documents → index
"""

from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from loguru import logger

from lib.semantics import load_fk_relations, load_semantics, render_table
from offline.indexer.categorical_values import fetch_categorical_columns_values
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

    def build_documents(self) -> list[Document]:
        """
        Build one Document per table, ready to be embedded into Chroma — no LLM.

        Tables documented in `semantic_layer/<db>/tables.yaml` are rendered by `render_table()`
        (description header + inline column comments, with the values of the categorical
        columns read from the database). Undocumented tables are indexed with their raw DDL,
        and a warning suggests running `semantics` for them.
        """
        tables       = self.db.get_usable_table_names()
        semantics    = load_semantics(self.db_name)
        fk_relations = load_fk_relations(self.db_name)

        # Values of the categorical columns, table after table: each GROUP BY runs a Hive
        # job, and concurrent jobs fail on a local Tez
        values = {}
        for table in [t for t in tables if t in semantics]:
            columns = semantics[table].get("categorical_columns") or []
            logger.info(f"[{self.db_name}] Fetching values: {table} ({len(columns)} column(s))")
            values[table] = fetch_categorical_columns_values(self.db, self.db_name, table, columns)

        def build_document(table: str) -> Document:
            # DDL + sample rows: one database round-trip per table
            ddl = self.db.get_table_info(table_names=[table]).strip()

            if table in semantics:
                content = render_table(table, ddl, semantics[table], fk_relations, values[table])
                logger.info(f"[{self.db_name}] Built: {table}")
            else:
                content = ddl
                logger.warning(f"[{self.db_name}] Built: {table} (raw DDL, no semantics — run `make semantics`)")

            return Document(
                page_content=content,
                metadata={"table_name": table, "db_name": self.db_name},
            )

        # Tables are independent: fetch their DDL in parallel
        # (max_workers stays under the SQLAlchemy pool size: 5 + 10 overflow)
        with ThreadPoolExecutor(max_workers=8) as executor:
            documents = list(executor.map(build_document, tables))

        return documents

    def index(self, documents: list[Document]):
        """
        Embed documents into the shared Chroma store with deterministic IDs.

        Full refresh per database: every document of this database is deleted first,
        so tables dropped or renamed since the last run do not linger in the index.
        IDs are built as "<db_name>_<table_name>".
        """
        store = self.vsm.get_store()

        # Build deterministic IDs
        ids = [
            f"{doc.metadata['db_name']}_{doc.metadata['table_name']}"
            for doc in documents
        ]

        # Purge this database's documents, then index the current ones
        store.delete(where={"db_name": {"$eq": self.db_name}})
        store.add_documents(documents, ids=ids)

        logger.info(f"[{self.db_name}] Indexed {len(documents)} schemas to Chroma.")

    def run(self):
        """Run the full pipeline: build_documents → index."""
        documents = self.build_documents()
        self.index(documents)
