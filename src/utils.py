# src/utils.py

from pathlib import Path

import chainlit as cl
import sqlparse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables.graph_mermaid import draw_mermaid_png
from loguru import logger

from config import embedding_model
from tools import get_run_query_tool

# =============================================================================
# Graph
# Used by: graph.py
# =============================================================================


def save_graph_png(agent, orientation: str = "LR") -> Path:

    output_path = Path(__file__).resolve().parents[1] / "assets" / "graph.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mermaid_str = agent.get_graph().draw_mermaid().replace("graph TD", f"graph {orientation}")

    with open(output_path, "wb") as f:
        f.write(draw_mermaid_png(mermaid_syntax=mermaid_str))

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
            return messages[i + 1:]
    return []

def extract_sql(text: str) -> str:
    """Strip markdown code fences from LLM-generated SQL."""
    query = text.strip()
    if query.startswith("```"):
        query = query.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return query

def extract_schema_message(messages: list, db_name: str) -> ToolMessage | None:
    """
    Scan state messages and return the ToolMessage produced by get_schema for a given db.
    Returns the last one found (in case of multi-turn conversations).
    """
    schema_tool = get_run_query_tool(db_name)
    result = None
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == schema_tool.name:
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
    """
    Route a single message to the appropriate Chainlit UI step.
    """

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
        "route_database":   "🗄️ Routing database...",
        "retrieve_schema":  "🔍 Retrieving candidates...",
        "rerank_tables":    "🎯 Selecting tables...",
        "sql_coder":        "⚙️ Writing SQL...",
        "sql_validator":    "✅ Validating & executing...",
        "format_answer":    "✍️ Formatting answer...",
    }.get(node_name, "")

# =============================================================================
# Database routing
# Used by: nodes.py (route_database)
# =============================================================================

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x ** 2 for x in a) ** 0.5
    nb  = sum(x ** 2 for x in b) ** 0.5
    return dot / (na * nb + 1e-9)


def select_database(question: str, domain_rules: dict, db_embeddings: dict) -> str:
    """
    Select the target database for a given question.

    Two-pass strategy (no LLM involved):
        Pass 1 — keyword rules (<1ms):
            Match question tokens against domain_rules keywords.
            Returns immediately on first match.
        Pass 2 — embedding similarity (~30ms):
            Embed the question and compute cosine similarity against
            pre-computed db_embeddings. Returns the highest-scoring database.

    Args:
        question:      Raw user question.
        domain_rules:  {db_name: [keywords]} built from config descriptions.
        db_embeddings: {db_name: embedding_vector} pre-computed at startup.

    Returns:
        The name of the selected database.
    """
    q = question.lower()

    # Pass 1 — keyword rules
    for db_name, keywords in domain_rules.items():
        if any(kw in q for kw in keywords):
            logger.debug(f"[select_database] Rules match → {db_name}")
            return db_name

    # Pass 2 — embedding similarity (fallback for ambiguous questions)
    q_emb  = embedding_model.embed_query(question)
    scores = {
        name: _cosine_similarity(q_emb, emb)
        for name, emb in db_embeddings.items()
    }
    best = max(scores, key=scores.get)
    logger.debug(f"[select_database] Embedding match → {best} (scores={scores})")
    return best