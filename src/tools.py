from typing import Dict

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool


from config import db, model



def get_tools(llm: BaseChatModel, db: SQLDatabase) -> Dict[str, BaseTool]:
    """
    Returns the 3 required tools (2 standard LangChain tools + 1 custom safe query tool)
    as a flat list ready for a LangChain Agent.
    """
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    toolkit_tools = toolkit.get_tools()
    
    # 1. Extract native tools 
    list_tables_tool = next(t for t in toolkit_tools if t.name == "sql_db_list_tables")
    get_schema_tool = next(t for t in toolkit_tools if t.name == "sql_db_schema")
    run_query_tool = next(t for t in toolkit_tools if t.name == "sql_db_query")

    
    # 2. Create personalized run query tool
    

    return {
        "list_tables": list_tables_tool,
        "get_schema":  get_schema_tool,
        "run_query":   run_query_tool,
    }


# =============================================================================
# Tools init 
# =============================================================================

_tools_dict = get_tools(llm=model,db=db)

list_tables_tool = _tools_dict["list_tables"]
get_schema_tool  = _tools_dict["get_schema"]
run_query_tool   = _tools_dict["run_query"]


if __name__ == "__main__":
    from pprint import pprint

    print(f"✅ Successfully initialized {len(_tools_dict)} tools.")

    for key, tool_obj in _tools_dict.items():
        print(f"\n{'='*40}")
        print(f"Dict Key: {key}")
        print(f"Tool Name: {tool_obj.name}")
        print(f"Description: {tool_obj.description}")
        try:
            pprint(tool_obj.args_schema.model_json_schema()["properties"])
        except Exception:
            print(f"Args: {tool_obj.args}")