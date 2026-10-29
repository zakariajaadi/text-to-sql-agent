from datetime import date, time
from decimal import Decimal
from typing import NamedTuple

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.utilities.sql_database import truncate_word
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from loguru import logger
from sqlalchemy import text

from lib.query_cancel import cancel_event
from resources import db_registry, model

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

class QueryResult(NamedTuple):
    content:   str          # rows in the sql_db_query tool format, read by format_answer
    columns:   list[str]    # column names as returned by the driver
    rows:      list[list]   # JSON-safe values, read by the Chainlit result card
    truncated: bool         # the query returned more than max_rows rows


def run_capped_query(db_name: str, query: str, max_rows: int, cancel_key: str | None = None) -> QueryResult:
    """
    Execute a query and read at most max_rows rows: the safety cap lives here, never
    in the SQL, so a LIMIT in the query only ever comes from the question itself.

    cancel_key: the turn's key (app.py): the query is then polled and cancelled on the
    cluster when the user clicks Stop (lib/query_cancel.py), ending on QueryCancelled.
    content follows the sql_db_query tool format (str of tuples, "" when empty, long
    values truncated). Raises SQLAlchemyError when the query fails.
    """
    db = db_registry[db_name]
    with db._engine.connect() as conn:
        if cancel_key:
            conn = conn.execution_options(cancel_event=cancel_event(cancel_key))
        cursor  = conn.execute(text(query))
        columns = list(cursor.keys())
        rows    = cursor.fetchmany(max_rows + 1)

    truncated = len(rows) > max_rows
    rows      = rows[:max_rows]
    content   = [
        tuple(truncate_word(value, length=db._max_string_length) for value in row)
        for row in rows
    ]
    return QueryResult(
        content   = str(content) if content else "",
        columns   = columns,
        rows      = [[_to_json_safe(value) for value in row] for row in rows],
        truncated = truncated,
    )


def _to_json_safe(value):
    """Convert a driver value (Decimal, date, bytes...) to a JSON-serializable one."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, time)):  # datetime is a subclass of date
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)



if __name__ == "__main__":

    for db_name, tools in _tools_registry.items():
        print(f"\n{'='*40}")
        print(f"Database: {db_name}")
        for key, tool in tools.items():
            print(f"  {key}: {tool.name}")

