from pathlib import Path

import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tools import get_schema_tool, list_tables_tool, run_query_tool

# =============================================================================
# Graph
# Used by: graph.py
# =============================================================================

def save_graph_png(agent) -> Path:
    output_path = Path(__file__).resolve().parents[1] / "assets" / "graph.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        f.write(agent.get_graph().draw_mermaid_png())
    
    print(f"Graph PNG saved at: {output_path}")
    return output_path


# =============================================================================
# LangChain
# Used by: nodes.py (check_query, generate_query)
# =============================================================================


def get_last_cycle(messages: list) -> list:
    """Return technical messages from the last cycle (after the last user question)."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i + 1:]  # everything after the last user question
    return []


def extract_schema_message(messages: list) -> ToolMessage | None:
    """
    Scan state messages and return the ToolMessage produced by get_schema.
    Returns the last one found (in case of multi-turn conversations).
    """
    result = None
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == get_schema_tool.name:
            result = msg
    return result


# =============================================================================
# Chainlit UI helpers
# Used by: app.py (on_message, astream_events loop)
# =============================================================================


def normalize_chunk_content(content) -> str:
    """
    Normalize chunk content to handle different LLM provider formats.
    Gemini often returns a list of blocks/parts, while OpenAI returns a plain string.
    """
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return content if isinstance(content, str) else ""


async def process_node_output(event: dict, seen_msg_ids: set) -> None:
    """
    Extract and handle new messages from a LangGraph node output.
    Filters out already-processed messages using a set of IDs.
    """
    node_name = event["metadata"].get("langgraph_node")
    if not node_name:
        return

    output = event["data"].get("output", {})
    # ── Special case: assess_question_complexity returns structured data, not messages
    if node_name == "question_complexity" and "question_complexity" in output:
        complexity = output["question_complexity"]
        emoji = {"simple": "🟢", "complex": "🟠", "out_of_scope": "🔴"}.get(complexity, "⚪")
        async with cl.Step(name=f"{emoji} Complexity: {complexity}", type="run") as s:
            s.output = f"Question classified as **{complexity}**"
        return
    
    # Standard message-based processing
    if not isinstance(output, dict) or "messages" not in output:
        return

    for msg in output["messages"]:
        msg_id = id(msg)
        if msg_id not in seen_msg_ids:
            seen_msg_ids.add(msg_id)
            await handle_step(msg, node_name)


async def handle_step(msg, node_name: str = "") -> None:
    """
    Route a single message to the appropriate Chainlit UI step based on its type and origin node.

    Message flow:
      AIMessage(tool_calls)  → LLM is requesting a tool execution
      ToolMessage            → result returned by the executed tool
      AIMessage(no tools)    → plain LLM output (plan or final answer)
    """

    # ── Tool request: list available tables (deterministic, no LLM involved) ──
    if isinstance(msg, ToolMessage) and msg.name == list_tables_tool.name:
        async with cl.Step(name="📋 Available tables", type="tool") as s:
            s.output = msg.content

    # ── LLM tool call requests ─────────────────────────────────────────────────
    elif isinstance(msg, AIMessage) and msg.tool_calls:
        tool_name = msg.tool_calls[0]["name"]
        tool_args = msg.tool_calls[0]["args"]

        if tool_name == get_schema_tool.name:
            # Show which tables the LLM selected for schema retrieval
            async with cl.Step(name="🗂️ Fetching schema", type="tool") as s:
                s.input = str(tool_args)

        elif tool_name == run_query_tool.name:
            if node_name == "generate_query":
                # Raw SQL produced by the generation node (simple path: no verification follows)
                async with cl.Step(name="⚙️ Generated SQL", type="run") as s:
                    s.output = f"```sql\n{tool_args.get('query', '')}\n```"
            elif node_name == "check_query":
                # SQL after review and correction (complex path only)
                async with cl.Step(name="✅ Verified SQL", type="run") as s:
                    s.output = f"```sql\n{tool_args.get('query', '')}\n```"

    # ── Tool execution results ─────────────────────────────────────────────────
    elif isinstance(msg, ToolMessage):
        if msg.name == get_schema_tool.name:
            # Truncate schema to avoid flooding the UI
            async with cl.Step(name="🗂️ Schema retrieved", type="tool") as s:
                s.output = msg.content[:500]

        elif msg.name == run_query_tool.name:
            async with cl.Step(name="📊 Query results", type="tool") as s:
                s.output = msg.content

    # ── Plain LLM output: query plan or final answer ───────────────────────────
    elif isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        if node_name == "plan_query_generation":
            # Step-by-step plan produced before complex query generation
            async with cl.Step(name="📝 Query plan", type="run") as s:
                s.output = content

        elif node_name == "format_answer":
            # Fallback: display final answer if token streaming didn't fire
            if not cl.user_session.get("answer_streamed"):
                await cl.Message(content=content).send()


def get_node_step_title(node_name: str) -> str:
    """
    Return a human-readable step title for the given LangGraph node name.
    Returns an empty string if the node has no associated title.
    """
    return {
    # type="tool" — Chainlit prepends "Using", so no gerund
    "list_tables":           "📋 List tables",
    "run_query":             "📊 Run query",
    # type="run" — displayed as-is, gerund reads naturally
    "question_complexity":   "🧠 Assessing complexity...",
    "plan_query_generation": "📝 Planning query...",
    "generate_query":        "⚙️ Generating SQL...",
    "check_query":           "✅ Verifying SQL...",
    "format_answer":         "✍️ Formatting answer...",
}.get(node_name, "")

