from typing import Literal
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from state import AgentState


from nodes import (
    list_tables, call_get_schema, get_schema_node,
    generate_query, check_query, run_query_node,
    assess_question_complexity, plan_query_generation, route_question_by_complexity
)


def should_continue(state: AgentState) -> Literal["check_query", END]:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    return "check_query"


def build_graph():
    checkpointer = MemorySaver()
    builder = StateGraph(AgentState)

    builder.add_node(list_tables)
    builder.add_node(call_get_schema)
    builder.add_node(get_schema_node, "get_schema")
    builder.add_node(assess_question_complexity)
    builder.add_node(plan_query_generation)
    builder.add_node(generate_query)
    builder.add_node(check_query)
    builder.add_node(run_query_node, "run_query")

    builder.add_edge(START,                       "list_tables")
    builder.add_edge("list_tables",               "call_get_schema")
    builder.add_edge("call_get_schema",           "get_schema")
    builder.add_edge("get_schema",                "assess_question_complexity")
    builder.add_conditional_edges("assess_question_complexity", route_question_by_complexity)
    builder.add_edge("plan_query_generation",     "generate_query")
    builder.add_conditional_edges("generate_query", should_continue)
    builder.add_edge("check_query",               "run_query")
    builder.add_edge("run_query",                 "generate_query")

    return builder.compile(checkpointer=checkpointer)



if __name__=="__main__":
   agent = build_graph()
   
   from utils import save_graph_png
   save_graph_png(agent)
  