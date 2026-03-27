# src/graph.py

from langgraph.graph import END, START, StateGraph

from nodes import (
    format_answer,
    rerank_tables,
    retrieve_schema,
    route_after_validation,
    route_by_scope,
    route_database,
    sql_coder,
    sql_validator,
)
from state import AgentState


def build_graph():
    builder = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    builder.add_node(route_database)
    builder.add_node(retrieve_schema)
    builder.add_node(rerank_tables)
    builder.add_node(sql_coder)
    builder.add_node(sql_validator)
    builder.add_node(format_answer)

    # ── Edges ──────────────────────────────────────────────────────────────────
    builder.add_edge(START,             "route_database")
    builder.add_edge("route_database",  "retrieve_schema")
    builder.add_edge("retrieve_schema", "rerank_tables")
    builder.add_conditional_edges("rerank_tables", route_by_scope, {
        "in_scope":     "sql_coder",
        "out_of_scope": "format_answer",
    })
    builder.add_edge("sql_coder",       "sql_validator")
    builder.add_conditional_edges("sql_validator", route_after_validation, {
        "error":   "sql_coder",
        "success": "format_answer",
    })
    builder.add_edge("format_answer",   END)

    return builder.compile()


if __name__ == "__main__":
    agent = build_graph()

    from utils import save_graph_png
    save_graph_png(agent)