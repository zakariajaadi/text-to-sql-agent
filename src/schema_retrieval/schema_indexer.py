# src/schema_retrieval/schema_indexer.py

"""
Two-step process:
    1. index    — DB → enriched DDL via LLM → Chroma 
    2. describe — Chroma schemas → database-level description → db_descriptions.yaml

CLI usage:
    # Full pipeline (index + describe) for all databases
    python -m schema_retrieval.schema_indexer

    # Full pipeline for a specific database only
    python -m schema_retrieval.schema_indexer --db Chinook

    # Run describe step only
    python -m schema_retrieval.schema_indexer describe --db Northwind
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from omegaconf import OmegaConf

from config import BASE_DIR, cfg, embedding_model
from prompts import DB_DESCRIPTION_PROMPT, SCHEMA_DESCRIPTION_PROMPT
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

    def extract(self) -> list[Document]:
        """
        Extract and enrich DDL for each table in parallel.
        Returns documents ready for indexing — no disk write.
        """
        tables         = self.db.get_usable_table_names()
        all_tables_str = ", ".join(tables)

        def enrich_table(table: str) -> Document:
            # Get table DLL
            ddl      = self.db.get_table_info(table_names=[table]).strip()
            # Enrich table DDL with Business and technical info
            enriched = self._enrich_chain(ddl, all_tables_str)
            logger.info(f"[{self.db_name}] Extracted: {table}")
            # return enriched DDL as document
            return Document(
                page_content=enriched,
                metadata={"table_name": table, "db_name": self.db_name},
            )
        # Parallel execution of enrich table
        with ThreadPoolExecutor() as executor:
            documents = list(executor.map(enrich_table, tables))

        return documents


    def index(self, documents: list[Document]):
        """
        Embed documents into the shared Chroma store with deterministic IDs.

        Deduplication strategy: delete-then-add with deterministic IDs.
            - IDs are built as "<db_name>_<table_name>"
            - delete() is a no-op on unknown IDs, so the first run is safe.
        """
        store = self.vsm.get_store()

        # Build deterministic IDs
        ids = [
            f"{doc.metadata['db_name']}_{doc.metadata['table_name']}"
            for doc in documents
        ]

        # Index documents with dedup
        store.delete(ids=ids)
        store.add_documents(documents, ids=ids)

        logger.info(f"[{self.db_name}] Indexed {len(documents)} schemas to Chroma.")


    def describe(self):
        """Generate a database-level description from Chroma schemas."""
        from config import fast_model

        # Load schemas from Chroma
        store = self.vsm.get_store()
        docs  = store.get(where={"db_name": {"$eq": self.db_name}})
        all_schemas = "\n\n".join(docs["documents"])
        
        # Describe database
        prompt      = ChatPromptTemplate.from_messages([("human", DB_DESCRIPTION_PROMPT)])
        chain       = prompt | fast_model | StrOutputParser()
        description = chain.invoke({"db_name": self.db_name, "schemas": all_schemas})
        
        # Save description
        desc_path        = BASE_DIR / "config" / "db_descriptions.yaml"
        existing         = OmegaConf.load(desc_path) if desc_path.exists() else OmegaConf.create()
        existing[self.db_name] = description.strip()
        OmegaConf.save(existing, desc_path)

        logger.info(f"[{self.db_name}] Database description written to {desc_path.name}")


    def run(self):
        """Run the full pipeline: extract → index → describe."""
        documents = self.extract()
        self.index(documents)
        self.describe()

    # helpers 
    def _enrich_chain(self, ddl: str, all_tables: str) -> str:
        from config import fast_model
         
        # Build chain
        prompt = ChatPromptTemplate.from_messages([
            ("human", SCHEMA_DESCRIPTION_PROMPT),
        ])  
        chain = prompt | fast_model | StrOutputParser()
        
        # return enrichement 
        return chain.invoke({
            "db_name":    self.db_name,
            "all_tables": all_tables,
            "ddl":        ddl,
        })


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    """
    CLI entry point supporting optional --db flag and command selection.

    Examples:
        python -m schema_retrieval.schema_indexer              # all dbs, full pipeline
        python -m schema_retrieval.schema_indexer --db Chinook # one db, full pipeline
        python -m schema_retrieval.schema_indexer describe --db Northwind
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
        if command == "describe":
            indexer.describe()
        else:
            indexer.run()


if __name__ == "__main__":
    main()
