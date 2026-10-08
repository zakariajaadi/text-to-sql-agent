"""
Shared runtime resources, built once at import from the settings of config.py:
LLMs, embedding model, semantic cache, database connections, FK graphs and schema store.
Importing this module needs the API keys, the databases and the Chroma index.
"""

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_community.utilities import SQLDatabase
from langchain_redis import RedisSemanticCache
from loguru import logger
from omegaconf import OmegaConf
from sqlalchemy import inspect as sa_inspect

from config import SEMANTIC_LAYER_DIR, DatabaseConfig, LLMConfig, cfg
from schema_retrieval.vector_store import VectorStoreManager

# =============================================================================
# Builders
# =============================================================================

def build_chat_model(llm_cfg: LLMConfig):
    return init_chat_model(
        llm_cfg.model,
        temperature=llm_cfg.temperature
    )


def build_database(db_cfg: DatabaseConfig) -> SQLDatabase:
    """Build a langchain SQLAlchemy wrapper from the URI."""
    if db_cfg.dialect == "hive":
        return SQLDatabase.from_uri(
            db_cfg.uri,
            lazy_table_reflection=True,  # don't reflect the whole metastore at startup
        )
    return SQLDatabase.from_uri(db_cfg.uri)


def build_schema_store() -> dict[str, dict[str, str]]:
    """Load all enriched DDLs from Chroma into memory at startup."""
    try:
        vsm   = VectorStoreManager(embedding_model, cfg.retrieval.persist_directory)
        store = vsm.get_store()

        result = {}
        for db_cfg in cfg.databases:
            docs = store.get(where={"db_name": {"$eq": db_cfg.name}})
            result[db_cfg.name] = {
                meta["table_name"]: doc
                for meta, doc in zip(docs["metadatas"], docs["documents"])
            }
            if not result[db_cfg.name]:
                logger.warning(f"[schema_store] No schemas found for {db_cfg.name} — run indexer first.")
        return result

    except Exception as e:
        logger.error(f"[schema_store] Failed to load from Chroma: {e}")
        return {db_cfg.name: {} for db_cfg in cfg.databases}

# =============================================================================
# Load foreign key relations as a knowlege graph for a database
# =============================================================================

def load_fk_relations(db_name: str) -> list[tuple[str, str]]:
    """Return (from_column, to_column) pairs, as table.column, declared in semantic_layer/<db>/relationships.yaml."""
    path = SEMANTIC_LAYER_DIR / db_name / "relationships.yaml"
    if not path.exists():
        return []
    relations = OmegaConf.load(path).get("relations") or []
    return [(r["from"], r["to"]) for r in relations]


def build_fk_graph(db: SQLDatabase, db_name: str) -> dict[str, list[str]]:
    """
    Build a bidirectional FK graph.
    If A references B via FK, both graph[A] contains B and graph[B] contains A.
    Bidirectionality ensures fact tables are reachable from dimension tables and vice versa.

    FKs come from SQLAlchemy reflection. When the engine exposes none (Hive, Impala),
    they come from semantic_layer/<db>/relationships.yaml instead.
    """
    try:
        inspector = sa_inspect(db._engine)
        tables    = db.get_usable_table_names()

        edges = [
            (table, fk["referred_table"])
            for table in tables
            for fk in inspector.get_foreign_keys(table)
        ]
        if not edges:
            edges = [
                (source.split(".")[0], target.split(".")[0])
                for source, target in load_fk_relations(db_name)
            ]

        fk_graph: dict[str, set[str]] = {table: set() for table in tables}
        for source, target in edges:
            if source in fk_graph and target in fk_graph:
                fk_graph[source].add(target)
                fk_graph[target].add(source)
            else:
                logger.warning(f"[build_fk_graph] {db_name}: unknown table in {source} -> {target}")

        return {k: list(v) for k, v in fk_graph.items()}

    except Exception as e:
        logger.error(f"[build_fk_graph] Failed: {e}")
        return {}

# =============================================================================
# Init singletons
# =============================================================================

model           = build_chat_model(cfg.llm)
fast_model      = build_chat_model(cfg.llm_fast)
embedding_model = init_embeddings(cfg.retrieval.embedding_model)

logger.debug(f"Initialized LLM: {cfg.llm.model} | Fast LLM: {cfg.llm_fast.model}.")
logger.debug(f"Initialized Embedding Model: {cfg.retrieval.embedding_model}")

# Semantic cache — caches LLM responses for semantically similar questions
llm_cache: RedisSemanticCache | None = (
    RedisSemanticCache(
        embeddings=embedding_model,
        redis_url=cfg.cache.redis_url,
        distance_threshold=cfg.cache.distance_threshold,
        ttl=cfg.cache.ttl,
    )
    if cfg.cache.enabled
    else None
)

logger.debug(f"Semantic cache {'enabled — Redis: ' + cfg.cache.redis_url if llm_cache else 'disabled'}")

# Database registry
db_registry: dict[str, SQLDatabase] = {
    db_cfg.name: build_database(db_cfg)
    for db_cfg in cfg.databases
}

logger.debug(f"Initialized databases: {list(db_registry.keys())}.")

# =============================================================================
# Derived from config — computed once at startup
# =============================================================================

# Bidirectional FK graph — used by expand_fk in rerank_tables
fk_graphs: dict[str, dict[str, list[str]]] = {
    db_cfg.name: build_fk_graph(db_registry[db_cfg.name], db_cfg.name)
    for db_cfg in cfg.databases
}

# Enriched DDLs loaded once at startup — avoids disk I/O at query time
schema_store = build_schema_store()
