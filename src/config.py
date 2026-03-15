from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import OmegaConf

from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase




BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv()


@dataclass
class LLMConfig:
    prefix: str
    model: str
    temperature: float


@dataclass
class DatabaseConfig:
    name: str
    dialect:str
    uri: str = "" 

    def __post_init__(self):
        
        if self.dialect == "sqlite" :
            self.uri = f"sqlite:///{BASE_DIR / 'data' / self.name}"


@dataclass
class AgentConfig:
    top_k: int
    dialect: str


@dataclass
class LangSmithConfig:
    tracing: bool


@dataclass
class AppConfig:
    llm: LLMConfig
    database: DatabaseConfig
    agent: AgentConfig
    langsmith: LangSmithConfig


def load_config() -> AppConfig:
    config_path = BASE_DIR / "config" / "config.yaml"
    cfg = OmegaConf.load(config_path)

    return AppConfig(
        llm=LLMConfig(**cfg.llm),
        database=DatabaseConfig(**cfg.database),
        agent=AgentConfig(**cfg.agent),
        langsmith=LangSmithConfig(**cfg.langsmith),
    )


cfg = load_config()

model = init_chat_model(
    f"{cfg.llm.prefix}:{cfg.llm.model}",
    temperature=cfg.llm.temperature
)

db = SQLDatabase.from_uri(cfg.database.uri)
