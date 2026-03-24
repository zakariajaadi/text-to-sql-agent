from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_community.utilities import SQLDatabase
from omegaconf import OmegaConf

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv()


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

@dataclass
class LangSmithConfig:
    tracing: bool


@dataclass
class AppConfig:
    llm: LLMConfig
    database: DatabaseConfig
    retrieval: RetrievalConfig
    langsmith: LangSmithConfig



def load_config() -> AppConfig:
    config_path = BASE_DIR / "config" / "config.yaml"
    cfg = OmegaConf.load(config_path)

    return AppConfig(
        llm=LLMConfig(**cfg.llm),
        database=DatabaseConfig(**cfg.database),
        retrieval=RetrievalConfig(**cfg.retrieval),
        langsmith=LangSmithConfig(**cfg.langsmith),
    )


cfg = load_config()

model = init_chat_model(
    cfg.llm.model,
    temperature=cfg.llm.temperature
)

embedding_model = init_embeddings(cfg.retrieval.embedding_model)


db = SQLDatabase.from_uri(cfg.database.uri)
