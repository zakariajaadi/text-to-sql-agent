from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseChatModel

from config import db, model


def get_tools(llm: BaseChatModel, db: SQLDatabase) -> dict:
    """
    Exposes 3 of the 4 SQLDatabaseToolkit tools. 

    SQLDatabase: a Python interface over SQLAlchemy — connects to the DB,
    lists tables, retrieves schemas and executes queries.

    SQLDatabaseToolkit: takes that SQLDatabase and encapsulates its features into
    LangChain tool objects with a name, description and input schema that the LLM
    can invoke via tool calling.
    """
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    return {
        "list_tables": next(t for t in tools if t.name == "sql_db_list_tables"),
        "get_schema":  next(t for t in tools if t.name == "sql_db_schema"),
        "run_query":   next(t for t in tools if t.name == "sql_db_query"),
    }

# ── Tools setup ───────────────────────────────────────────────────────────────

_tools = get_tools(llm=model,db=db)

list_tables_tool = _tools["list_tables"]
get_schema_tool  = _tools["get_schema"]
run_query_tool   = _tools["run_query"]



if __name__ == "__main__":
    from config import db, model
    tools = get_tools(model,db)
    print(tools["list_tables"].name)
    
    """for name, tool in tools.items():
        print(f"\n{'='*40}")
        print(f"Tool: {name}")
        print(f"Description: {tool.description}")
        pprint(tool.args_schema.model_json_schema()["properties"])"""