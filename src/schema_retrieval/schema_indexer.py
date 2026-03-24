"""
indexer.py

Offline pipeline for schema indexing. Two steps:
  1. extract: DB → DDL files (data/schemas/)
  2. index:   DDL files → Chroma vector store

"""

import sys
from pathlib import Path

from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from loguru import logger

from config import cfg, embeddings
from schema_retrieval.vector_store import VectorStoreManager


class SchemaIndexer:
    """
    Extracts table DDL from the database and indexes it into Chroma.

    Two-step process designed to allow manual inspection and enrichment
    of schema files between extraction and indexing (e.g., adding
    business descriptions, documenting common joins).
    """

    def __init__(self):
        self.db = SQLDatabase.from_uri(cfg.database.uri)
        self.schemas_dir = Path(cfg.schema_retrieval.schemas_directory)
        self.vsm = VectorStoreManager(
            embedding_model=embeddings,
            persist_directory=cfg.schema_retrieval.persist_directory,
        )

    def extract(self):
        """Extract DDL + sample rows from each table into individual .sql files."""
        self.schemas_dir.mkdir(parents=True, exist_ok=True)

        tables = self.db.get_usable_table_names()
        for table in tables:
            # get_table_info returns DDL + 3 sample rows by default
            ddl = self.db.get_table_info(table_names=[table]).strip()
            filepath = self.schemas_dir / f"{table}.sql"
            filepath.write_text(ddl, encoding="utf-8")
            logger.info(f"Extracted: {filepath.name}")

        logger.info(f"{len(tables)} schemas saved to {self.schemas_dir}")

    def index(self):
        """Read .sql files, wrap as Documents, and embed into Chroma."""
        files = sorted(self.schemas_dir.glob("*.sql"))
        if not files:
            logger.warning(f"No .sql files found in {self.schemas_dir}. Run 'extract' first.")
            return

        # One Document per table — table name stored in metadata for
        # downstream filtering after retrieval
        documents = [
            Document(
                page_content=f.read_text(encoding="utf-8"),
                metadata={"table_name": f.stem},
            )
            for f in files
        ]

        self.vsm.get_store().add_documents(documents)
        logger.info(f"Indexed {len(documents)} schemas to Chroma.")

    def run(self):
        """Full pipeline: extract then index."""
        self.extract()
        self.index()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    indexer = SchemaIndexer()
    command = sys.argv[1] if len(sys.argv) > 1 else "all"

    if command == "extract":
        indexer.extract()
    elif command == "index":
        indexer.index()
    else:
        indexer.run()


if __name__ == "__main__":
    main()