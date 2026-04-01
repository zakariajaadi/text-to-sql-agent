"""
Chainlit UI helpers
Used by: app.py (on_message, astream_events loop)

"""
import chainlit as cl
import sqlparse
from langchain_core.messages import AIMessage, ToolMessage


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

    # ── route_database: show selected database ──
    if node_name == "route_database" and "selected_db" in output:
        selected = output["selected_db"]
        async with cl.Step(name=f"🗄️ Database: {selected}", type="run") as s:
            s.output = f"Routing to **{selected}**"
        return

    # ── retrieve_schema: show candidate tables ──
    if node_name == "retrieve_schema" and "candidate_schemas" in output:
        candidates = output["candidate_schemas"]
        tables = ", ".join(candidates.keys())
        async with cl.Step(name="🔍 Candidate tables", type="tool") as s:
            s.output = f"Retrieved {len(candidates)} tables: {tables}"
        return

    # ── rerank_tables: show selected tables + scope decision ──
    if node_name == "rerank_tables" and "retrieved_schema" in output:
        schema = output["retrieved_schema"]
        table_names = [
            line.split('"')[1]
            for line in schema.split("\n")
            if line.strip().startswith("CREATE TABLE")
        ]
        in_scope = output.get("is_in_scope", True)
        emoji = "🟢" if in_scope else "🔴"
        async with cl.Step(name=f"{emoji} Tables selected", type="run") as s:
            s.output = ", ".join(table_names)
        return

    # ── sql_coder: show generated SQL ──
    if node_name == "sql_coder" and "generated_query" in output:
        query = output["generated_query"]
        formatted = sqlparse.format(query, reindent=True, keyword_case="upper")
        async with cl.Step(name="⚙️ Generated SQL", type="run") as s:
            s.output = f"```sql\n{formatted}\n```"
        return

    # ── sql_validator: show feedback on failure ──
    if node_name == "sql_validator" and "validation_feedback" in output:
        feedback = output["validation_feedback"]
        if feedback:
            async with cl.Step(name="❌ Validation failed", type="tool") as s:
                s.output = feedback
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
    """Route a single message to the appropriate Chainlit UI step."""

    # ── sql_validator success: show query result ──
    if isinstance(msg, ToolMessage):
        async with cl.Step(name="📊 Query results", type="tool") as s:
            s.output = msg.content

    # ── Plain LLM output: final answer fallback ──
    elif isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        if node_name == "format_answer":
            if not cl.user_session.get("answer_streamed"):
                await cl.Message(content=content).send()


def get_node_step_title(node_name: str) -> str:
    return {
        "condense_question": "💬 Condensing question...",
        "route_database":    "🗄️ Routing database...",
        "retrieve_schema":   "🔍 Retrieving candidates...",
        "rerank_tables":     "🎯 Selecting tables...",
        "sql_coder":         "⚙️ Writing SQL...",
        "sql_validator":     "✅ Validating & executing...",
        "format_answer":     "✍️ Formatting answer...",
    }.get(node_name, "")