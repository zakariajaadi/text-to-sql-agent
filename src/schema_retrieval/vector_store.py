"""
vector_store.py

Minimal Chroma vector store manager for schema indexing.
"""

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


class VectorStoreManager:

    def __init__(self, embedding_model: Embeddings, persist_directory: str = "data/chroma_db"):
        self.embeddings = embedding_model
        self.persist_directory = persist_directory
        self._store = None

    def get_store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
        return self._store

    def as_retriever(self, k: int = 3):
        return self.get_store().as_retriever(search_kwargs={"k": k})