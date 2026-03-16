from typing import Literal
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from state import AgentState

from langgraph.graph import END

from config import model, db
from guardrails import is_safe_query
from tools import get_tools
from prompts import (GENERATE_QUERY_PROMPT, CHECK_QUERY_PROMPT,
                     CLASSIFY_QUESTION_PROMPT, PLAN_SQL_PROMPT,
                     FORMAT_ANSWER_PROMPT)
from loguru import logger
from utils import get_last_cycle, extract_schema_message
from models import QuestionComplexity
# ── Tools setup ───────────────────────────────────────────────────────────────
_tools = get_tools(llm=model,db=db)

list_tables_tool = _tools["list_tables"]
get_schema_tool  = _tools["get_schema"]
run_query_tool   = _tools["run_query"]

get_schema_node = ToolNode([get_schema_tool],
                           name="get_schema",
                           handle_tool_errors=True
)
run_query_node  = ToolNode([run_query_tool],
                            name="run_query",
                            handle_tool_errors=True
)


# ── Nodes ─────────────────────────────────────────────────────────────────────
def list_tables(state: AgentState) -> dict:
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

    # Actually execute the tool, returns a ToolMessage with tool_call_id set and response
    tool_message = list_tables_tool.invoke(tool_call)
    
    return {"messages": [tool_call_message, tool_message]}


def call_get_schema(state: AgentState) -> dict:
    """
    LLM-driven node : selects the tables relevant to the user's question.
    
    Forces the LLM to make a tool call `sql_db_schema` on the tables it deems relevant.
    """

    llm_with_tools = model.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}



def assess_question_complexity(state: AgentState) -> dict: 

    # LLM with structured output
    structured_llm= model.with_structured_output(QuestionComplexity)
    
    # Assess complexity
    messages=[SystemMessage(content=CLASSIFY_QUESTION_PROMPT),*state["messages"]]
    response= structured_llm.invoke(messages)
    logger.debug(f"query complexity = {response.question_complexity}")
    # return complexity
    return {"question_complexity":response.question_complexity}

def route_question_by_complexity(state: AgentState) -> Literal["generate_query","plan_query_generation",END]:
    question_complexity=state["question_complexity"]

    if question_complexity == "out_of_scope":
        return END
    
    elif question_complexity == "complex":
        return "plan_query_generation"
    else:
        return "generate_query"


def plan_query_generation(state: AgentState) -> dict: 

    messages=[SystemMessage(content=PLAN_SQL_PROMPT),*state["messages"]]
    response= model.invoke(messages)
    return {"messages":[response]}


def generate_query(state: AgentState) -> dict:
    """
    LLM-driven node — generates a SQL query

    """
    llm_with_tools = model.bind_tools([run_query_tool],tool_choice="any")
    messages = [SystemMessage(content=GENERATE_QUERY_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}



def check_query(state: AgentState) -> dict:
    """
    LLM-driven node — reviews and corrects the generated SQL query before execution.
    Injects the DB schema from state so the LLM can catch semantic errors (wrong FK, 
    wrong column names) in addition to syntactic ones.
    """

    # Retrieve the SQL query from the last AIMessage's tool call
    tool_call = state["messages"][-1].tool_calls[0]
    query = tool_call["args"]["query"]

    # Guardrail
    if not is_safe_query(query):
        raise ValueError(f"Unsafe query detected: {query}")

    # Extract the ToolMessage containing the schema from state
    schema_msg = extract_schema_message(state["messages"])
    schema_content = schema_msg.content if schema_msg else "Schema not available."


    messages = [
        SystemMessage(content=CHECK_QUERY_PROMPT.format(schema=schema_content)),
        HumanMessage(content=query)
    ]

    llm_with_tools = model.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke(messages)

    checked_query = response.tool_calls[0]["args"]["query"]

    corrected_message = AIMessage(
        content="",
        tool_calls=[{
            "name": run_query_tool.name,
            "args": {"query": checked_query},
            "id": tool_call["id"],
            "type": "tool_call",
        }]
    )

    return {"messages": [corrected_message]}




def format_answer(state: AgentState) -> dict:
    """
    LLM-driven node — formulates the final natural language answer
    from the SQL query results. No tools bound, pure text output.
    """
    messages = [SystemMessage(content=FORMAT_ANSWER_PROMPT)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

def route_after_run_query(state: AgentState) -> Literal["generate_query", "format_answer"]:
    """Route to generate_query on SQL error, format_answer on success."""
    
    last_message = state["messages"][-1]
    
    if isinstance(last_message, ToolMessage) and last_message.status == "error":
        logger.debug("SQL error detected, retrying query generation")
        return "generate_query"
    
    return "format_answer"