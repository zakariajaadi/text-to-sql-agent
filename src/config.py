"""
Application settings: paths, environment, logging, and config/config.yaml as dataclasses.
Importing this module needs no database, vector store or API key: the shared runtime
objects (models, databases, schema store, FK graphs) are built in resources.py.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf

BASE_DIR = Path(__file__).resolve().parents[1]

# Reviewed description of each database, one folder per database:
# semantic_layer/<db>/tables.yaml, semantic_layer/<db>/relationships.yaml
SEMANTIC_LAYER_DIR = BASE_DIR / "semantic_layer"

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


@dataclass
class DatabaseConfig:
    name: str
    dialect:str
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

@dataclass
class ValueSearchConfig:
    directory: str     # one SQLite FTS5 index per database: <directory>/<db>.sqlite
    max_values: int    # a categorical column with more distinct values is not indexed
    min_score: int     # fuzzy match score (0-100) under which a value is not proposed
    top_k: int         # values of a column added to the schema; past them: "(+N more)", and a LIKE
    score_margin: int  # per term, matches more than this under its best one (in any column) are dropped

    def __post_init__(self):
        self.directory = str(BASE_DIR / self.directory)

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
    value_search: ValueSearchConfig
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

    # Load config file
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)

    return AppConfig(
        llm=LLMConfig(**cfg.llm),
        llm_fast=LLMConfig(**cfg.llm_fast),
        databases=[DatabaseConfig(**db) for db in cfg.databases], 
        retrieval=RetrievalConfig(**cfg.retrieval),
        value_search=ValueSearchConfig(**cfg.value_search),
        column_pruning=ColumnPruningConfig(**cfg.column_pruning),
        langsmith=LangSmithConfig(**cfg.langsmith),
        memory=MemoryConfig(**cfg.memory),
        cache=CacheConfig(**cfg.cache),

    )


cfg = load_config()
