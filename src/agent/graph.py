# src/agent/graph.py

from pathlib import Path

from langchain_core.runnables.graph_mermaid import draw_mermaid_png
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    condense_question,
    format_answer,
    match_values,
    planner,
    prune_columns,
    rerank_tables,
    resolve_database,
    retrieve_schema,
    route_after_validation,
    route_by_scope,
    sql_coder,
    sql_validator,
)
from agent.state import AgentState
from config import cfg


def build_graph():
    builder = StateGraph(AgentState)

    # Nodes 
    builder.add_node(condense_question)
    builder.add_node(resolve_database)
    builder.add_node(retrieve_schema)
    builder.add_node(rerank_tables)
    builder.add_node(match_values)
    builder.add_node(prune_columns)
    builder.add_node(planner)
    builder.add_node(sql_coder)
    builder.add_node(sql_validator)
    builder.add_node(format_answer)

    # Edges
    builder.add_edge(START,                "condense_question")
    builder.add_edge("condense_question",  "resolve_database")
    builder.add_edge("resolve_database",   "retrieve_schema")
    builder.add_edge("retrieve_schema",    "rerank_tables")
    builder.add_conditional_edges("rerank_tables", route_by_scope, {
        "in_scope":     "match_values",
        "out_of_scope": "format_answer",
    })
    builder.add_edge("match_values",  "prune_columns")
    builder.add_edge("prune_columns", "planner")
    builder.add_edge("planner", "sql_coder")
    builder.add_edge("sql_coder",       "sql_validator")
    builder.add_conditional_edges("sql_validator", route_after_validation, {
        "error":   "sql_coder",
        "success": "format_answer",
    })
    builder.add_edge("format_answer",   END)

    # Set Checkpointer if memory is enabled
    checkpointer = None
    if cfg.memory.enabled:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


def save_graph_png(agent, orientation: str = "LR") -> Path:

    output_path = Path(__file__).resolve().parents[2] / "docs" / "graph.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mermaid_str = agent.get_graph().draw_mermaid().replace("graph TD", f"graph {orientation}")

    with open(output_path, "wb") as f:
        f.write(draw_mermaid_png(mermaid_syntax=mermaid_str))

    print(f"Graph PNG saved at: {output_path}")
    return output_path

if __name__ == "__main__":
    agent = build_graph()
    save_graph_png(agent, "TB")
    
    


