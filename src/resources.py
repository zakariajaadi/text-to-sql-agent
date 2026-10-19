"""
Shared runtime resources, built once at import from the settings of config.py:
LLMs, embedding model, vector store, semantic cache, database connections, FK graphs, schema store
(and its retriever) and metrics.
Importing this module needs the API keys, the databases and the Chroma index.
"""

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_community.utilities import SQLDatabase
from langchain_redis import RedisSemanticCache
from loguru import logger
from sqlalchemy import inspect as sa_inspect

from config import DatabaseConfig, LLMConfig, cfg
from lib.db_errors import is_unreachable
from lib.metrics import Metric, load_metrics
from lib.semantics import load_fk_relations
from lib.vector_store import VectorStoreManager

# =============================================================================
# Builders
# =============================================================================

def build_chat_model(llm_cfg: LLMConfig):
    return init_chat_model(
        llm_cfg.model,
        temperature=llm_cfg.temperature
    )


def build_database(db_cfg: DatabaseConfig) -> SQLDatabase:
    """
    Build a langchain SQLAlchemy wrapper from the URI.
    With include_tables, the agent, the indexer and the generators only see those tables
    (an unknown name stops the startup). sample_rows: rows under each DDL of get_table_info.
    """
    try:
        db = SQLDatabase.from_uri(
            db_cfg.uri,
            engine_args=db_cfg.engine_args() or None,
            include_tables=list(db_cfg.include_tables) or None,
            sample_rows_in_table_info=db_cfg.sample_rows,
            lazy_table_reflection=True,  # don't reflect the whole metastore at startup
        )
    except Exception as e:
        if is_unreachable(e):
            raise RuntimeError(
                f"The {db_cfg.dialect} database {db_cfg.name} is unreachable at startup ({type(e).__name__}: "
                f"{str(e)[:200]}). Check the network or VPN (in Docker: restart Docker Desktop), then start again."
            ) from e
        raise
    if db_cfg.dialect == "impala":
        quote_every_identifier(db._engine.dialect)
    return db


def quote_every_identifier(dialect) -> None:
    """
    Backtick every identifier SQLAlchemy writes for this dialect (sample rows, categorical values).
    impyla's reserved word list stops at Impala 2: a column named like an Impala 4 reserved word
    (position, ...) breaks the query unquoted. The agent's own queries are quoted by normalize_sql.
    """
    preparer = dialect.identifier_preparer
    preparer._requires_quotes = lambda value: True
    preparer._strings.clear()   # identifiers already formatted, cached unquoted


def build_schema_store() -> dict[str, dict[str, str]]:
    """Load all enriched DDLs from Chroma into memory at startup."""
    try:
        store = vector_store.get_store()

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
# Foreign key relations as a knowledge graph for a database
# =============================================================================

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

# Shared Chroma store of the indexed table documents — one store for all databases
vector_store = VectorStoreManager(embedding_model, cfg.retrieval.persist_directory)

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


def get_schema_retriever(db_name: str, k: int | None = None):
    """
    Return a retriever scoped to a specific database via metadata filtering.

    All schemas from all databases are stored in a single Chroma store.
    The db_name filter ensures only tables belonging to the target database
    are returned, preventing cross-database schema leakage.
    """
    return vector_store.as_retriever(
        k=k or cfg.retrieval.top_k,
        filter={"db_name": db_name},
    )


# Business metrics of each database, validated against the indexed schemas — an invalid one stops the app.
# Built on first access (`from resources import metrics_store`, through the module __getattr__ below):
# the indexer and the generators import this module too, and must keep working while metrics.yaml
# references a table that is not indexed yet — `make index` is what fixes it.
_metrics_store: dict[str, dict[str, Metric]] | None = None


def __getattr__(name: str):
    global _metrics_store
    if name != "metrics_store":
        raise AttributeError(f"module 'resources' has no attribute {name!r}")
    if _metrics_store is None:
        _metrics_store = {
            db_cfg.name: load_metrics(db_cfg.name, db_cfg.dialect, schema_store[db_cfg.name])
            for db_cfg in cfg.databases
        }
    return _metrics_store
