from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_community.utilities import SQLDatabase
from omegaconf import OmegaConf

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv()


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
    limit: int = 5
    uri: str = "" 

    def __post_init__(self):
        
        if self.dialect == "sqlite" :
            name = self.name if self.name.endswith(".db") else f"{self.name}.db"
            self.uri = f"sqlite:///{BASE_DIR / 'data' / name}"

    def build(self):
        return SQLDatabase.from_uri(self.uri)

@dataclass
class RetrievalConfig:
    embedding_model: str
    persist_directory: str
    schemas_directory: str
    top_k: int

    def __post_init__(self):
        self.persist_directory = str(BASE_DIR / self.persist_directory)
        self.schemas_directory = str(BASE_DIR / self.schemas_directory)

    def build(self):
        return init_embeddings(self.embedding_model)

@dataclass
class LangSmithConfig:
    tracing: bool


@dataclass
class AppConfig:
    llm: LLMConfig
    llm_fast: LLMConfig
    databases: list[DatabaseConfig]
    retrieval: RetrievalConfig
    langsmith: LangSmithConfig
    
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
    cfg = OmegaConf.load(config_path)

    return AppConfig(
        llm=LLMConfig(**cfg.llm),
        llm_fast=LLMConfig(**cfg.llm_fast),
        databases=[DatabaseConfig(**db) for db in cfg.databases], 
        retrieval=RetrievalConfig(**cfg.retrieval),
        langsmith=LangSmithConfig(**cfg.langsmith),
    )


# =============================================================================
# Init config and different singletons
# =============================================================================

# Config object
cfg = load_config()

# LLM 
model = cfg.llm.build()

# Fast LLM — for low-complexity nodes (rerank_tables, check_scope)
fast_model = cfg.llm_fast.build()

# Embedding model
embedding_model = cfg.retrieval.build()

# Registry for SQLDatabase instances 
# Used by tools.py to construct tools by db
db_registry: dict[str, SQLDatabase] = {
    db_cfg.name: db_cfg.build()
    for db_cfg in cfg.databases
}

# Derived from databases config — computed once at startup

# {db_name: [keywords]} for deterministic keyword routing
domain_rules: dict[str, list[str]] = {
    db.name: db.description.lower().split(", ")
    for db in cfg.databases
}

# {db_name: embedding_vector} for semantic similarity routing
db_embeddings: dict[str, list[float]] = {
    db.name: embedding_model.embed_query(db.description)
    for db in cfg.databases
}