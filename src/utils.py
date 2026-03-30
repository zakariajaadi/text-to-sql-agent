# src/utils.py

from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict

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

def get_question(state: dict) -> str:
    """Return the standalone question (post-condensation) or fall back to last HumanMessage."""
    return state.get("standalone_question") or next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
    )

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
        "condense_question": "💬 Condensing question...",
        "route_database":    "🗄️ Routing database...",
        "retrieve_schema":   "🔍 Retrieving candidates...",
        "rerank_tables":     "🎯 Selecting tables...",
        "sql_coder":         "⚙️ Writing SQL...",
        "sql_validator":     "✅ Validating & executing...",
        "format_answer":     "✍️ Formatting answer...",
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
        Pass 1 — keyword scoring (<1ms):
            Count matching keywords per database.
            Returns the highest-scoring database if there is a clear winner.
            Falls through to pass 2 on tie.
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

    # Pass 1 — keyword scoring
    scores = {
        db_name: sum(1 for kw in keywords if kw in q)
        for db_name, keywords in domain_rules.items()
    }

    best_score = max(scores.values())

    if best_score > 0:
        # Check for tie — if multiple databases have the same score, fall through
        winners = [db for db, score in scores.items() if score == best_score]
        if len(winners) == 1:
            logger.debug(f"[select_database] Keyword match → {winners[0]} (scores={scores})")
            return winners[0]

    # Pass 2 — embedding similarity (tie or no keyword match)
    q_emb  = embedding_model.embed_query(question)
    emb_scores = {
        name: _cosine_similarity(q_emb, emb)
        for name, emb in db_embeddings.items()
    }
    best = max(emb_scores, key=emb_scores.get)
    logger.debug(f"[select_database] Embedding match → {best} (scores={emb_scores})")
    return best

# =============================================================================
# FK expnading
# Used by: nodes.py (rerank_tables)
# =============================================================================

def expand_fk(selected: list[str], fk_graph: dict[str, list[str]]) -> list[str]:
    """
    Add bridge tables that connect pairs of already-selected tables via FK.

    LLMs systematically miss fact and bridge tables not explicitly named in the
    question. Two-pass expansion recovers them without flooding the context
    with all FK-linked tables:
        Pass 1 — direct neighbors of each selected table.
        Pass 2 — bridge tables shared between two selected tables.
    """
    expanded = set(selected)

    # Pass 1 — direct FK neighbors
    for table in selected:
        for neighbor in fk_graph.get(table, []):
            expanded.add(neighbor)

    # Pass 2 — bridge tables (common FK neighbors between pairs)
    for i, table_a in enumerate(selected):
        for table_b in selected[i + 1:]:
            neighbors_a = set(fk_graph.get(table_a, []))
            neighbors_b = set(fk_graph.get(table_b, []))
            expanded.update(neighbors_a & neighbors_b)

    return list(expanded)


# =============================================================================
# LangGraph Node Decorator: Semantic Caching Middleware
# Used by: src/nodes.py (sql_coder, format_answer)
# =============================================================================
def extract_text_content(val: Any) -> str:
    """
    Recursively extracts plain text from various LangChain/LLM output formats.
    Handles:
      - Lists of messages (takes the last one).
      - LangChain Message objects (AIMessage, ToolMessage).
      - Gemini-style content blocks (list of dicts with 'text').
      - Standard dictionaries or raw strings.
    """
    if not val:
        return ""

    # 1. If it's a list (like state['messages']), recurse on the last element
    if isinstance(val, list):
        return extract_text_content(val[-1])
    
    # 2. Extract content from Message objects (if present)
    content = getattr(val, "content", val)
    
    # 3. Handle Gemini-style content blocks: [{'text': '...'}, {'text': '...'}]
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content 
            if isinstance(block, dict)
        ).strip()
    
    # 4. Handle dictionary returns (e.g., {'text': '...'})
    if isinstance(content, dict):
        return content.get("text", str(content))
    
    # 5. Fallback to string representation
    return str(content)

def cached_node(key_fn: Callable[[Dict], str], 
                return_key: str, 
                cache_only_first_attempt: bool = True):
    """
    A universal decorator for LangGraph nodes to handle selective semantic caching.
    Automatically handles conversion between raw strings and LangChain Message objects.
    
    Args:
        key_fn: function(state) -> str, builds the unique cache key.
        return_key: The state key to read from/write to (e.g., 'generated_query' or 'messages').
        cache_only_first_attempt: If True, skips caching if 'validation_feedback' is present.
    """
    def decorator(fn: Callable):
        @wraps(fn)
        def wrapper(state: Dict, *args, **kwargs):
            # Import inside wrapper to avoid circular imports
            from config import llm_cache

            # Bypass caching if cache is not initialized or disabled
            if llm_cache is None:
                return fn(state, *args, **kwargs)

            # 1. Generate unique key for this specific node execution
            cache_key = key_fn(state)
            
            # 2. Semantic Lookup in Redis (using function name as namespace)
            cached = llm_cache.lookup(cache_key, llm_string=fn.__name__)
            
            if cached and len(cached) > 0:
                logger.debug(f"[{fn.__name__}] Semantic Cache Hit: {cache_key}")
                cached_text = cached[0].text
                
                # Smart Return: Wrap in AIMessage list if the state expects 'messages'
                if return_key == "messages":
                    from langchain_core.messages import AIMessage
                    return {return_key: [AIMessage(content=cached_text)]}
                
                # Otherwise return as raw data (string/query)
                return {return_key: cached_text}

            # 3. Cache Miss: Execute the actual node function
            result = fn(state, *args, **kwargs)
            
            # 4. Extract Value for Storage using our robust utility
            val_to_store = extract_text_content(result.get(return_key, ""))

            # 5. Selective Cache Update
            # We avoid caching "corrected" attempts (retries) to prevent storing 
            # edge-case fixes as the primary semantic answer for future queries.
            is_retry = bool(state.get("validation_feedback"))
            should_cache = not (cache_only_first_attempt and is_retry)
            
            if should_cache and val_to_store:
                from langchain_core.outputs import Generation
                llm_cache.update(
                    cache_key, 
                    llm_string=fn.__name__, 
                    return_val=[Generation(text=val_to_store)]
                )
                logger.debug(f"[{fn.__name__}] Cache Updated: {cache_key}")

            return result
        return wrapper
    return decorator

