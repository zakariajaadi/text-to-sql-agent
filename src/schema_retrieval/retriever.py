# src/schema_retrieval/retriever.py

from config import cfg, embedding_model
from schema_retrieval.vector_store import VectorStoreManager


def get_schema_retriever(k: int = 5):
    vsm = VectorStoreManager(
        embedding_model=embedding_model,
        persist_directory=cfg.retrieval.persist_directory,
    )
    return vsm.as_retriever(k=k or cfg.retrieval.top_k)