from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.utilities.sql_database import truncate_word
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from loguru import logger
from sqlalchemy import text

from config import db_registry, model

# Factory  — constructs the 3 tools per database

def _build_tools(llm: BaseChatModel, db: SQLDatabase) -> dict[str, BaseTool]:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools   = toolkit.get_tools()

    return {
        "list_tables": next(t for t in tools if t.name == "sql_db_list_tables"),
        "get_schema":  next(t for t in tools if t.name == "sql_db_schema"),
        "run_query":   next(t for t in tools if t.name == "sql_db_query"),
    }


# Registry — one entry per db, created on startup

_tools_registry: dict[str, dict[str, BaseTool]] = {
    db_name: _build_tools(llm=model, db=db_instance)
    for db_name, db_instance in db_registry.items()
}

logger.debug(f"Tools registry initialized for: {list(_tools_registry.keys())}")


# Public accessors — used by nodes.py

def get_run_query_tool(db_name: str) -> BaseTool:
    return _tools_registry[db_name]["run_query"]

def get_schema_tool_for(db_name: str) -> BaseTool:
    return _tools_registry[db_name]["get_schema"]

def get_list_tables_tool(db_name: str) -> BaseTool:
    return _tools_registry[db_name]["list_tables"]


# Capped execution — used by the sql_validator node

def run_capped_query(db_name: str, query: str, max_rows: int) -> tuple[str, int, bool]:
    """
    Execute a query and read at most max_rows rows: the safety cap lives here, never
    in the SQL, so a LIMIT in the query only ever comes from the question itself.

    Returns the rows in the sql_db_query tool format (str of tuples, "" when empty,
    long values truncated), the number of rows read, and whether more were left unread.
    Raises SQLAlchemyError when the query fails.
    """
    db = db_registry[db_name]
    with db._engine.connect() as conn:
        rows = conn.execute(text(query)).fetchmany(max_rows + 1)

    truncated = len(rows) > max_rows
    content = [
        tuple(truncate_word(value, length=db._max_string_length) for value in row)
        for row in rows[:max_rows]
    ]
    return (str(content) if content else ""), len(content), truncated



if __name__ == "__main__":

    for db_name, tools in _tools_registry.items():
        print(f"\n{'='*40}")
        print(f"Database: {db_name}")
        for key, tool in tools.items():
            print(f"  {key}: {tool.name}")

