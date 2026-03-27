# src/schema_retrieval/retriever.py

from config import cfg, embedding_model
from schema_retrieval.vector_store import VectorStoreManager


def get_schema_retriever(db_name: str, k: int | None = None):
    """
    Return a retriever scoped to a specific database via metadata filtering.

    All schemas from all databases are stored in a single Chroma store.
    The db_name filter ensures only tables belonging to the target database
    are returned, preventing cross-database schema leakage.
    """
    vsm = VectorStoreManager(
        embedding_model=embedding_model,
        persist_directory=cfg.retrieval.persist_directory,
    )

    return vsm.as_retriever(
        k=k or cfg.retrieval.top_k,
        filter={"db_name": db_name},
    )