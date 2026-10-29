# src/agent/graph.py

from pathlib import Path

from langchain_core.runnables.graph_mermaid import draw_mermaid_png
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    ask_clarification,
    condense_question,
    format_answer,
    match_values,
    planner,
    prune_columns,
    report_unreachable,
    rerank_tables,
    resolve_database,
    retrieve_schema,
    route_after_coder,
    route_after_planner,
    route_after_validation,
    route_by_scope,
    sql_coder,
    sql_validator,
)
from agent.state import AgentState
from config import cfg

# Checkpoints kept per thread: about two turns of graph steps. Every turn resumes from the
# latest one only; the older ones are time-travel nothing in the app reads
MAX_CHECKPOINTS = 20


class TrimmedMemorySaver(MemorySaver):
    """
    MemorySaver keeping only the last MAX_CHECKPOINTS checkpoints of a thread. The graph writes
    one checkpoint per step and each carries the whole conversation: a session kept open all day
    would grow quadratically with the questions. The writes of a dropped checkpoint, and the
    channel blobs no remaining checkpoint references, are dropped with it.
    """

    def put(self, config, checkpoint, metadata, new_versions):
        result        = super().put(config, checkpoint, metadata, new_versions)
        thread_id     = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoints   = self.storage[thread_id][checkpoint_ns]
        if len(checkpoints) <= MAX_CHECKPOINTS:
            return result

        # Checkpoint ids are UUID6: sorted is chronological; the writes go with their checkpoint
        for checkpoint_id in sorted(checkpoints)[:-MAX_CHECKPOINTS]:
            del checkpoints[checkpoint_id]
            self.writes.pop((thread_id, checkpoint_ns, checkpoint_id), None)

        # Blobs hold the channel values: keep those a remaining checkpoint still references
        # (a channel not written for a while keeps its old version in every later checkpoint)
        referenced = {
            (channel, version)
            for serialized, _, _ in checkpoints.values()
            for channel, version in self.serde.loads_typed(serialized).get("channel_versions", {}).items()
        }
        for key in [k for k in self.blobs
                    if k[0] == thread_id and k[1] == checkpoint_ns and (k[2], k[3]) not in referenced]:
            del self.blobs[key]
        return result


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
    builder.add_node(report_unreachable)
    builder.add_node(ask_clarification)

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
    builder.add_conditional_edges("planner", route_after_planner, {
        "plan":    "sql_coder",
        "clarify": "ask_clarification",   # two plausible readings: one question to the user, no query
    })
    builder.add_conditional_edges("sql_coder", route_after_coder, {
        "query":         "sql_validator",
        "not_available": "format_answer",   # the information is in no column: no query
    })
    builder.add_conditional_edges("sql_validator", route_after_validation, {
        "error":       "sql_coder",
        "success":     "format_answer",
        "unreachable": "report_unreachable",
    })
    builder.add_edge("format_answer",      END)
    builder.add_edge("report_unreachable", END)
    builder.add_edge("ask_clarification",  END)

    # Set Checkpointer if memory is enabled — bounded: only the last checkpoints are kept
    checkpointer = TrimmedMemorySaver() if cfg.memory.enabled else None

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
    
    


