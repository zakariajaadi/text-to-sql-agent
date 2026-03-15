from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState

from config import model, db
from guardrails import is_safe_query
from tools import get_tools
from prompts import GENERATE_QUERY_PROMPT, CHECK_QUERY_PROMPT

# ── Tools setup ───────────────────────────────────────────────────────────────
_tools = get_tools(llm=model,db=db)

list_tables_tool = _tools["list_tables"]
get_schema_tool  = _tools["get_schema"]
run_query_tool   = _tools["run_query"]

get_schema_node = ToolNode([get_schema_tool], name="get_schema")
run_query_node  = ToolNode([run_query_tool],  name="run_query")


# ── Nodes ─────────────────────────────────────────────────────────────────────
def list_tables(state: MessagesState) -> dict:
    """
    Deterministic node: retrieves all available tables without invoking the LLM.
    
    Manually constructs and executes the `sql_db_list_tables` tool call, bypassing
    the LLM entirely. 
    """

    tool_call = {
        "name": list_tables_tool.name,
        "args": {},
        "id": "list_tables_call",
        "type": "tool_call",
    }

    # Simulate the AIMessage a LLM would have sent to trigger the tool call (AI Message + tool call)
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])

    # Actually execute the tool, returns a ToolMessage with tool_call_id set
    tool_message = list_tables_tool.invoke(tool_call)

    # Simulate a Human-readable summary for downstream nodes (e.g. select_relevant_tables)
    response = AIMessage(f"Tables disponibles : {tool_message.content}")
    return {"messages": [tool_call_message, tool_message, response]}


def call_get_schema(state: MessagesState) -> dict:
    """
    LLM-driven node : selects the tables relevant to the user's question.
    
    Forces the LLM to make a tool call `sql_db_schema` on the tables it deems relevant.
    """
    llm_with_tools = model.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def generate_query(state: MessagesState) -> dict:
    """
    LLM-driven node — generates a SQL query or formulates the final answer.
    
    This node is invoked twice during a standard execution:
      
      - First pass: the LLM has the schema but no query results yet.
        It generates a SQL query and returns an AIMessage with a tool call
        to `sql_db_query`. The conditional edge routes to check_query.
      
      - Second pass: the LLM receives the query results from run_query.
        It formulates the final natural language answer with no tool call.
        The conditional edge routes to END.

    """
    llm_with_tools = model.bind_tools([run_query_tool])
    messages = [SystemMessage(content=GENERATE_QUERY_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def check_query(state: MessagesState) -> dict:
    """
    LLM-driven node — reviews and corrects the generated SQL query before execution.

    Extracts the raw SQL query from the last AIMessage's tool call args.
    """

    # Retrieve the SQL query from the last AIMessage in state from tool_calls
    tool_call = state["messages"][-1].tool_calls[0]
    query = tool_call["args"]["query"]
    
    # Guardrails
    if not is_safe_query(query):
        raise ValueError(f"Unsafe query detected: {query}")
    
    # No need for the full conversation history, only the sql query wrapped in human message
    messages = [
        SystemMessage(content=CHECK_QUERY_PROMPT),
        HumanMessage(content=query)
    ]
    
    # Force the LLM to always return a tool call to run_query (no free-text response allowed)
    llm_with_tools = model.bind_tools([run_query_tool], tool_choice="any")
    
    # Check the sql query
    response = llm_with_tools.invoke(messages)

    # Reuse the id from the previous message (generate_query's AIMessage)
    # so the ToolNode can correctly match this tool call to what's in the state
    response.id = state["messages"][-1].id
    return {"messages": [response]}