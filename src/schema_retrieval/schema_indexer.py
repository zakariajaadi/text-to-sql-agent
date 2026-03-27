# src/schema_retrieval/schema_indexer.py

"""
Offline pipeline for schema indexing across multiple databases.

All schemas are stored in a single shared Chroma vector store.
Each document carries a 'db_name' metadata field that enables
per-database filtering at retrieval time.

Two-step process:
    1. extract — DB → DDL files under data/schemas/<db_name>/
    2. index   — DDL files → Chroma (shared store, filtered by db_name)

CLI usage:
    # Index all databases defined in config.yaml
    python -m schema_retrieval.schema_indexer

    # Index a specific database only
    python -m schema_retrieval.schema_indexer --db Chinook

    # Run only the extract or index step
    python -m schema_retrieval.schema_indexer extract --db Northwind
    python -m schema_retrieval.schema_indexer index  --db Northwind
"""

import sys
from pathlib import Path

from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from loguru import logger

from config import cfg, embedding_model
from schema_retrieval.vector_store import VectorStoreManager


class SchemaIndexer:
    """
    Extracts and indexes the DDL schemas of a single database.

    Each table is extracted as a separate .sql file, then embedded
    into the shared Chroma store with its database name in metadata.
    This allows the retriever to filter by database at query time
    without maintaining separate vector stores per database.

    Attributes:
        db:          SQLDatabase connection for the target database.
        db_name:     Name of the target database (e.g. "Chinook").
        schemas_dir: Directory where extracted .sql files are saved.
        vsm:         VectorStoreManager pointing to the shared Chroma store.
    """

    def __init__(self, db_name: str):
        """
        Args:
            db_name: Name of the database to index. Must match an entry
                     in config.yaml databases[].name.
        """
        db_cfg = cfg.get_db(db_name)

        self.db          = SQLDatabase.from_uri(db_cfg.uri)
        self.db_name     = db_name
        self.schemas_dir = Path(cfg.retrieval.schemas_directory) / db_name

        # Single shared Chroma store — same persist_directory for all databases
        self.vsm = VectorStoreManager(
            embedding_model=embedding_model,
            persist_directory=cfg.retrieval.persist_directory,
        )

    def extract(self):
        """
        Extract DDL and sample rows for each table into individual .sql files.

        Files are saved under data/schemas/<db_name>/<table_name>.sql.
        The two-step design (extract then index) allows manual inspection
        and enrichment of schema files before embedding — useful for adding
        business descriptions or documenting common join patterns.
        """
        self.schemas_dir.mkdir(parents=True, exist_ok=True)

        tables = self.db.get_usable_table_names()
        for table in tables:
            # get_table_info returns DDL + 3 sample rows by default
            ddl      = self.db.get_table_info(table_names=[table]).strip()
            filepath = self.schemas_dir / f"{table}.sql"
            filepath.write_text(ddl, encoding="utf-8")
            logger.info(f"[{self.db_name}] Extracted: {filepath.name}")

        logger.info(f"[{self.db_name}] {len(tables)} schemas saved to {self.schemas_dir}")

    def index(self):
        """
        Read extracted .sql files and embed them into the shared Chroma store.

        Each document carries two metadata fields:
            - table_name: used downstream by rerank_tables to identify tables.
            - db_name:    used by the retriever to filter by database.

        Raises a warning and exits early if no .sql files are found —
        run extract() first.
        """
        files = sorted(self.schemas_dir.glob("*.sql"))
        if not files:
            logger.warning(f"[{self.db_name}] No .sql files found. Run 'extract' first.")
            return

        documents = [
            Document(
                page_content=f.read_text(encoding="utf-8"),
                metadata={
                    "table_name": f.stem,
                    "db_name":    self.db_name,
                },
            )
            for f in files
        ]

        self.vsm.get_store().add_documents(documents)
        logger.info(f"[{self.db_name}] Indexed {len(documents)} schemas to Chroma.")

    def run(self):
        """Run the full pipeline: extract then index."""
        self.extract()
        self.index()


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    """
    CLI entry point supporting optional --db flag and command selection.

    Examples:
        python -m schema_retrieval.schema_indexer              # all dbs, full run
        python -m schema_retrieval.schema_indexer --db Chinook # one db, full run
        python -m schema_retrieval.schema_indexer extract      # all dbs, extract only
        python -m schema_retrieval.schema_indexer index --db Northwind
    """
    args    = sys.argv[1:]
    command = "all"
    db_name = None

    if "--db" in args:
        idx     = args.index("--db")
        db_name = args[idx + 1]
        args    = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]

    if args:
        command = args[0]

    targets = (
        [cfg.get_db(db_name).name]
        if db_name
        else [db.name for db in cfg.databases]
    )

    for name in targets:
        indexer = SchemaIndexer(name)
        if command == "extract":
            indexer.extract()
        elif command == "index":
            indexer.index()
        else:
            indexer.run()


if __name__ == "__main__":
    main()