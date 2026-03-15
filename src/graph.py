from typing import Literal
from langgraph.graph import END, START, MessagesState, StateGraph

from nodes import (
    list_tables, call_get_schema, get_schema_node,
    generate_query, check_query, run_query_node
)


def should_continue(state: MessagesState) -> Literal["check_query", "__end__"]:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    return "check_query"


def build_graph():
    builder = StateGraph(MessagesState)

    builder.add_node(list_tables)
    builder.add_node(call_get_schema)
    builder.add_node(get_schema_node, "get_schema")
    builder.add_node(generate_query)
    builder.add_node(check_query)
    builder.add_node(run_query_node, "run_query")

    builder.add_edge(START,             "list_tables")
    builder.add_edge("list_tables",     "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema",      "generate_query")
    builder.add_conditional_edges("generate_query", should_continue)
    builder.add_edge("check_query",     "run_query")
    builder.add_edge("run_query",       "generate_query")

    return builder.compile()


if __name__=="__main__":
   agent = build_graph()
   
   from utils import save_graph_png
   save_graph_png(agent)
  