from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from nodes import (
    format_answer,
    generate_query,
    get_schema_node,
    list_tables,
    plan_query_generation,
    question_complexity,
    relevant_tables,
    route_after_run_query,
    route_question_by_complexity,
    safe_run_query
)
from state import AgentState


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node(list_tables)
    builder.add_node(relevant_tables)
    builder.add_node(get_schema_node, "get_schema")
    builder.add_node(question_complexity)
    builder.add_node(plan_query_generation)
    builder.add_node(generate_query)
    builder.add_node("run_query",safe_run_query)
    builder.add_node(format_answer)

    builder.add_edge(START,                       "list_tables")
    builder.add_edge("list_tables",               "relevant_tables")
    builder.add_edge("relevant_tables",           "get_schema")
    builder.add_edge("get_schema",                "question_complexity")
    builder.add_conditional_edges("question_complexity", route_question_by_complexity)
    builder.add_edge("plan_query_generation",     "generate_query")
    builder.add_edge("generate_query",            "run_query")
    builder.add_conditional_edges("run_query", route_after_run_query, {
                                  "generate_query": "generate_query",  
                                  "format_answer": "format_answer"                               
                                  })
    builder.add_edge("format_answer",             END)

    return builder.compile(checkpointer=MemorySaver())

if __name__=="__main__":
   agent = build_graph()
   
   from utils import save_graph_png
   save_graph_png(agent)