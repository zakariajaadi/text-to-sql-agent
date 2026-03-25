from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from nodes import (
    check_scope,
    format_answer,
    rerank_tables,
    retrieve_schema,
    route_by_scope,
    sql_coder,
    sql_validator,
    route_after_validation
)
from state import AgentState


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node(retrieve_schema)
    builder.add_node(rerank_tables)
    builder.add_node(check_scope)
    builder.add_node(sql_coder)
    builder.add_node(sql_validator)
    builder.add_node(format_answer)

    builder.add_edge(START,              "retrieve_schema")
    builder.add_edge("retrieve_schema",  "rerank_tables")
    builder.add_edge("rerank_tables",    "check_scope")
    builder.add_conditional_edges("check_scope", route_by_scope)
    builder.add_edge("sql_coder",        "sql_validator")
    builder.add_conditional_edges("sql_validator", route_after_validation, {
                                  "sql_coder":    "sql_coder",
                                  "format_answer": "format_answer",
                                  })
    builder.add_edge("format_answer",    END)

    return builder.compile(checkpointer=MemorySaver())
  

if __name__=="__main__":
   agent = build_graph()
   
   from utils import save_graph_png
   save_graph_png(agent)