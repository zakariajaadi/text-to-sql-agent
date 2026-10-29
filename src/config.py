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
from sqlalchemy.pool import NullPool

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
    sample_rows: int = 3   # rows shown under each DDL (indexed, sent to the LLM); 0 when the data must not leave
    user_language: str = "English"   # language users ask in: generated descriptions, metric synonyms, eval judge reasonings
    exclude_personal_data: bool = False  # make semantics excludes the columns the profile flags as personal data
    clarification: bool = True  # the planner may ask the user one question when two readings diverge (app only: the eval never asks)
    uri: str = ""
    connect_args: dict = field(default_factory=dict, init=False)  # driver options (authentication), from the environment
    query_options: dict = field(default_factory=dict, init=False) # options set on every query (limits), from the environment

    def __post_init__(self):

        # The engine database is the configured name, lowercased (e.g. "Chinook" -> chinook)
        if self.dialect == "hive":
            host = os.getenv("HIVE_HOST", "localhost")
            port = os.getenv("HIVE_PORT", "10000")
            user = os.getenv("HIVE_USER", "hive")
            self.uri = f"hive://{user}@{host}:{port}/{self.name.lower()}"

        elif self.dialect == "impala":
            # Impala >= 4 (impala4 dialect of impyla). On a Kerberos cluster (GSSAPI, the default), the
            # ticket comes from the environment: KRB5_CONFIG, and KRB5_CLIENT_KTNAME for a keytab
            host = os.getenv("IMPALA_HOST", "localhost")
            port = os.getenv("IMPALA_PORT", "21050")
            self.uri = f"impala4://{host}:{port}/{self.name.lower()}"
            self.connect_args = {
                "auth_mechanism":        os.getenv("IMPALA_AUTH_MECHANISM", "GSSAPI"),
                "kerberos_service_name": os.getenv("IMPALA_KERBEROS_SERVICE", "impala"),
                "use_ssl":               os.getenv("IMPALA_USE_SSL", "false").lower() == "true",
                # Seconds a network call may wait: an unreachable cluster fails fast instead of hanging.
                # Each call of a query is short (impyla polls a running query), so it never cuts one
                "timeout":               int(os.getenv("IMPALA_CONNECT_TIMEOUT", "20")),
                # impyla reopens a failed connection 3 times by default, each time waiting the timeout:
                # once is enough, sql_validator retries an unreachable database once itself
                "retries":               1,
            }
            # Impala cancels a query running longer, or needing more memory, than this (the agent
            # gets the error like any other): a wrong query never loads a shared cluster for long.
            # An empty value lifts the limit
            limits = {
                "EXEC_TIME_LIMIT_S": os.getenv("IMPALA_EXEC_TIME_LIMIT_S", "120"),
                "MEM_LIMIT":         os.getenv("IMPALA_MEM_LIMIT", "2g"),
            }
            self.query_options = {option: value for option, value in limits.items() if value}

        else:
            raise ValueError(f"Unsupported dialect '{self.dialect}' for database {self.name} (expected: hive, impala)")

    def engine_args(self) -> dict:
        """
        Arguments of the SQLAlchemy engine: driver options, the query options of impyla, sent with
        every query of the engine (agent, indexer, generators, eval), and the age of a pooled connection.
        """
        args = {}
        if self.connect_args:
            args["connect_args"] = dict(self.connect_args)
        if self.query_options:
            args["execution_options"] = {"cursor_configuration": dict(self.query_options)}
        if self.dialect == "impala":
            # No pool: a pooled connection left idle gets closed on the cluster side (impyla can't
            # reopen it over Kerberos — thrift_sasl: "Already open!"), and through Docker Desktop an
            # idle connection killed remotely is suspected of wedging vpnkit for the whole container.
            # Each query opens and closes its own connection; the Kerberos handshake costs ~0.2s
            args["poolclass"] = NullPool
        return args

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
class AppConfig:
    llm: LLMConfig
    llm_fast: LLMConfig
    databases: list[DatabaseConfig]
    retrieval: RetrievalConfig
    value_search: ValueSearchConfig
    column_pruning: ColumnPruningConfig
    langsmith: LangSmithConfig
    memory: MemoryConfig

    
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
    # Machine-specific overrides, kept out of git (e.g. the databases of a private cluster).
    # A list in it replaces the whole list of config.yaml: `databases` there is the full list
    local_path  = BASE_DIR / "config" / "config.local.yaml"

    # Load config file
    cfg = OmegaConf.load(config_path)
    if local_path.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(local_path))
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
    )


cfg = load_config()
