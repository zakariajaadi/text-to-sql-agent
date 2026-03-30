from loguru import logger 
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool


from config import db_registry, model



# =============================================================================
# Factory  — constructs the 3 tools per database
# =============================================================================

def _build_tools(llm: BaseChatModel, db: SQLDatabase) -> dict[str, BaseTool]:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools   = toolkit.get_tools()

    return {
        "list_tables": next(t for t in tools if t.name == "sql_db_list_tables"),
        "get_schema":  next(t for t in tools if t.name == "sql_db_schema"),
        "run_query":   next(t for t in tools if t.name == "sql_db_query"),
    }


# =============================================================================
# Registry — one entry per db, created on startup
# =============================================================================

_tools_registry: dict[str, dict[str, BaseTool]] = {
    db_name: _build_tools(llm=model, db=db_instance)
    for db_name, db_instance in db_registry.items()
}

logger.debug(f"Tools registry initialized for: {list(_tools_registry.keys())}")


# =============================================================================
# Accesseurs publics — utilisés par nodes.py
# =============================================================================

def get_run_query_tool(db_name: str) -> BaseTool:
    return _tools_registry[db_name]["run_query"]

def get_schema_tool_for(db_name: str) -> BaseTool:
    return _tools_registry[db_name]["get_schema"]

def get_list_tables_tool(db_name: str) -> BaseTool:
    return _tools_registry[db_name]["list_tables"]


# =============================================================================
# Vérification au démarrage
# =============================================================================

if __name__ == "__main__":

    for db_name, tools in _tools_registry.items():
        print(f"\n{'='*40}")
        print(f"Database: {db_name}")
        for key, tool in tools.items():
            print(f"  {key}: {tool.name}")

