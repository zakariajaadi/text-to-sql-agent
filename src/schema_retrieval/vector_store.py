# src/schema_retrieval/vector_store.py

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


class VectorStoreManager:
    """
    Minimal wrapper around a persistent Chroma vector store.

    A single store instance is shared across all databases.
    Database-level isolation is achieved at query time via metadata
    filters (e.g. {"db_name": "Chinook"}), not via separate stores.

    The store is lazily initialized on first access to avoid unnecessary
    connections at import time.
    """

    def __init__(self, embedding_model: Embeddings, persist_directory: str = "data/chroma_db"):
        """
        Args:
            embedding_model:    LangChain-compatible embeddings instance.
            persist_directory:  Path to the Chroma persistence directory.
                                Shared across all databases.
        """
        self.embeddings        = embedding_model
        self.persist_directory = persist_directory
        self._store            = None

    def get_store(self) -> Chroma:
        """
        Return the Chroma store, initializing it on first call.

        Subsequent calls return the same instance (lazy singleton pattern).
        """
        if self._store is None:
            self._store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
        return self._store

    def as_retriever(self, k: int = 3, filter: dict | None = None):
        """
        Return a retriever with optional metadata filtering.
        """
        search_kwargs = {"k": k}
        if filter:
            search_kwargs["filter"] = filter
        return self.get_store().as_retriever(search_kwargs=search_kwargs)