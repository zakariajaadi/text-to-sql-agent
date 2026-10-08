import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_community.utilities import SQLDatabase
from langchain_redis import RedisSemanticCache
from loguru import logger
from omegaconf import OmegaConf
from sqlalchemy import inspect as sa_inspect

from schema_retrieval.vector_store import VectorStoreManager

BASE_DIR = Path(__file__).resolve().parents[1]

if os.getenv("ENV", "dev") == "dev":
    load_dotenv()


# Set loggings level
logging.getLogger("chainlit").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("redisvl").setLevel(logging.WARNING)


# Configure loguru logging
logger.remove() 
logger.add(
    sys.stderr, 
    colorize=True,  # Force colors for docker
    level="DEBUG",    
    enqueue=True,
    filter=lambda record: "Translation file" not in record["message"],
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"   
)


# =============================================================================
# Config Data Classes
# =============================================================================

@dataclass
class LLMConfig:
    model: str
    temperature: float

    def build(self):
        return init_chat_model(
            self.model,
            temperature=self.temperature
        )


@dataclass
class DatabaseConfig:
    name: str
    dialect:str
    description: str
    max_rows: int = 100   # safety cap on the rows read from a query result (never a LIMIT in the SQL)
    tagline: str = ""                              # one-line presentation of the Chainlit welcome screen
    starters: list = field(default_factory=list)  # example questions of the Chainlit welcome screen: {icon, question}
    uri: str = ""
    #schema: str | None = None          # base Hive (ex: "sales_db")

    def __post_init__(self):
        
        if self.dialect == "hive":
            host = os.getenv("HIVE_HOST", "localhost")
            port = os.getenv("HIVE_PORT", "10000")
            user = os.getenv("HIVE_USER", "hive")
            self.uri = f"hive://{user}@{host}:{port}/{self.name.lower()}"

        else:
            raise ValueError(f"Unsupported dialect '{self.dialect}' for database {self.name} (expected: hive)")

    
    def build(self):
        """Build a langchain SQLAlchemy wrapper from the URI."""
        if self.dialect == "hive":
            return SQLDatabase.from_uri(
                self.uri,
                lazy_table_reflection=True,  # don't reflect the whole metastore at startup
            )
        return SQLDatabase.from_uri(self.uri)

@dataclass
class RetrievalConfig:
    embedding_model: str
    persist_directory: str
    schemas_directory: str
    top_k: int
    fk_neighbors: bool = False

    def __post_init__(self):
        self.persist_directory = str(BASE_DIR / self.persist_directory)
        self.schemas_directory = str(BASE_DIR / self.schemas_directory)

    def build(self):
        return init_embeddings(self.embedding_model)

@dataclass
class ColumnPruningConfig:
    enabled: bool
    min_columns: int   # tables with more columns are pruned by the LLM, narrower ones are kept whole

@dataclass
class LangSmithConfig:
    tracing: bool

@dataclass
class MemoryConfig:
    enabled: bool

@dataclass
class CacheConfig:
    enabled: bool
    redis_url: str
    distance_threshold: float
    ttl: int

@dataclass
class AppConfig:
    llm: LLMConfig
    llm_fast: LLMConfig
    databases: list[DatabaseConfig]
    retrieval: RetrievalConfig
    column_pruning: ColumnPruningConfig
    langsmith: LangSmithConfig
    memory: MemoryConfig
    cache: CacheConfig

    
    def get_db(self, name: str) -> DatabaseConfig:
        """Quick access to DatabaseConfig by name."""
        for db in self.databases:
            if db.name == name:
                return db
        raise ValueError(f"Database '{name}' not found in config.")

# =============================================================================
# Load config logic using OmegaConf
# =============================================================================

def load_config() -> AppConfig:
    config_path = BASE_DIR / "config" / "config.yaml"
    desc_path   = BASE_DIR / "config" / "db_descriptions.yaml"
    
    # Load config files
    cfg = OmegaConf.load(config_path)
    desc = OmegaConf.load(desc_path) if desc_path.exists() else OmegaConf.create()

    # Inject descriptions into config before resolving interpolations
    cfg  = OmegaConf.merge(cfg, {"db_descriptions": desc})
    OmegaConf.resolve(cfg)

    return AppConfig(
        llm=LLMConfig(**cfg.llm),
        llm_fast=LLMConfig(**cfg.llm_fast),
        databases=[DatabaseConfig(**db) for db in cfg.databases], 
        retrieval=RetrievalConfig(**cfg.retrieval),
        column_pruning=ColumnPruningConfig(**cfg.column_pruning),
        langsmith=LangSmithConfig(**cfg.langsmith),
        memory=MemoryConfig(**cfg.memory),
        cache=CacheConfig(**cfg.cache),

    )


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
    """Return (from_column, to_column) pairs, as table.column, declared in fk_relations.yaml."""
    path = BASE_DIR / "config" / "fk_relations.yaml"
    if not path.exists():
        return []
    relations = OmegaConf.load(path).get(db_name, {}).get("relations", [])
    return [(r["from"], r["to"]) for r in relations]


def build_fk_graph(db: SQLDatabase, db_name: str) -> dict[str, list[str]]:
    """
    Build a bidirectional FK graph.
    If A references B via FK, both graph[A] contains B and graph[B] contains A.
    Bidirectionality ensures fact tables are reachable from dimension tables and vice versa.

    FKs come from SQLAlchemy reflection. When the engine exposes none (Hive, Impala),
    they come from config/fk_relations.yaml instead.
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
# Init config and singletons
# =============================================================================

cfg             = load_config()
model           = cfg.llm.build()
fast_model      = cfg.llm_fast.build()
embedding_model = cfg.retrieval.build()

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
    db_cfg.name: db_cfg.build()
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

