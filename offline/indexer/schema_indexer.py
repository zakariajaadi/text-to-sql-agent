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
Pipeline order: fetch_values → build_documents → index → index_values
"""

from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from loguru import logger

from config import cfg
from lib.column_pruning import get_column_types
from lib.semantics import load_fk_relations, load_semantics, render_table
from lib.value_index import write_value_index
from offline.indexer.categorical_values import (
    fetch_categorical_columns_values,
    fits_in_prompt,
)
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

    def fetch_values(self) -> dict[str, dict[str, dict[str, int]]]:
        """
        Values of the categorical columns of the documented tables, read from the database:
        {table: {column: {value: row count}}}.

        Table after table: each GROUP BY runs a Hive job, and concurrent jobs fail on a local Tez.
        """
        semantics = load_semantics(self.db_name)

        values = {}
        for table in [t for t in self.db.get_usable_table_names() if t in semantics]:
            columns = semantics[table].get("categorical_columns") or []
            logger.info(f"[{self.db_name}] Fetching values: {table} ({len(columns)} column(s))")
            values[table] = fetch_categorical_columns_values(self.db, self.db_name, table, columns)
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

    def index_values(self, values: dict[str, dict[str, dict[str, int]]]):
        """Write the value index of this database (lib/value_index.py), every categorical column included."""
        inline = {
            (table, column)
            for table, columns in values.items()
            for column, counts in columns.items()
            if fits_in_prompt(counts)
        }
        write_value_index(self.db_name, values, inline)

    def run(self):
        """Run the full pipeline: fetch_values → build_documents → index → index_values."""
        values = self.fetch_values()
        self.index(self.build_documents(values))
        self.index_values(values)
