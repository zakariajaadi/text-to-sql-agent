"""
Chainlit UI helpers
Used by: app.py (on_message, astream_events loop)
"""

import chainlit as cl
import sqlparse
from langchain_core.messages import AIMessage, ToolMessage

from lib.column_pruning import get_column_names


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

    output = event["data"].get("output")

    # Guard: output must be a dict to inspect node-specific keys
    if not isinstance(output, dict):
        return

    # ── condense_question: show rewritten question (only when rewritten) ──
    if node_name == "condense_question" and "standalone_question" in output:
        original = next(
            (m.content for m in reversed(event["data"].get("input", {}).get("messages", []))
             if hasattr(m, "content")),
            ""
        )
        condensed = output["standalone_question"]
        # Only show step if the question was actually rewritten
        if original and condensed and original.strip().lower() != condensed.strip().lower():
            async with cl.Step(name="💬 Question rewritten", type="run") as s:
                s.output = f"**Original:** {original}\n**Rewritten:** {condensed}"
        return

    # ── resolve_database: title-only, no body ──
    if node_name == "resolve_database" and "selected_db" in output:
        async with cl.Step(name=f"🗄️ Database: {output['selected_db']}", type="run"):
            pass
        return

    # ── retrieve_schema: skip (merged into rerank_tables display) ──
    if node_name == "retrieve_schema":
        return

    # ── rerank_tables: show final selected tables with count ──
    if node_name == "rerank_tables" and "selected_schemas" in output:
        table_names = list(output["selected_schemas"])
        in_scope = output.get("is_in_scope", True)
        emoji = "🟢" if in_scope else "🔴"
        async with cl.Step(name=f"{emoji} {len(table_names)} tables selected", type="run") as s:
            s.output = ", ".join(table_names)
        return
    
    # ── prune_columns: fixed title, column count before → after in the body ──
    if node_name == "prune_columns" and "retrieved_schema" in output:
        selected_schemas = event["data"].get("input", {}).get("selected_schemas", {})
        columns_before   = sum(len(get_column_names(ddl)) for ddl in selected_schemas.values())
        columns_after    = len(get_column_names(output["retrieved_schema"]))

        async with cl.Step(name="✂️ Column pruning", type="run") as s:
            if columns_after < columns_before:
                s.output = f"{columns_before} → {columns_after} columns"
            else:
                s.output = f"No column pruned ({columns_before} columns)"
        return

    # ── planner: skip (internal, no UI step) ──
    #if node_name == "planner":
    #    return
    
    # ── planner: show plan during debugging ──
    if node_name == "planner" and "query_plan" in output:
        async with cl.Step(name="📐 Query plan", type="run") as s:
            s.output = output["query_plan"]
        return
        
    # ── sql_coder: show generated SQL (strip step comments) ──
    if node_name == "sql_coder" and "generated_query" in output:
        query = output["generated_query"]
        formatted = sqlparse.format(query, reindent=True, keyword_case="upper")
        async with cl.Step(name="⚙️ Generated SQL", type="run") as s:
            s.output = f"```sql\n{formatted}\n```"
        return

    # ── sql_validator: show failure or results ──
    if node_name == "sql_validator":
        feedback = output.get("validation_feedback", "")
        if feedback:
            async with cl.Step(name="❌ Validation failed", type="tool") as s:
                s.output = feedback
            return

        messages = output.get("messages", [])
        query = ""
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tc = msg.tool_calls[0]
                query = tc["args"].get("query", "") if isinstance(tc, dict) else tc.args.get("query", "")
                break

        for msg in messages:
            if isinstance(msg, ToolMessage):
                seen_msg_ids.add(id(msg))
                async with cl.Step(name="📊 Query results", type="tool") as s:
                    s.output = _format_query_result(msg.content, query=query)
        return

    # ── Standard message-based processing (format_answer) ──
    if not isinstance(output, dict) or "messages" not in output:
        return

    for msg in output["messages"]:
        msg_id = id(msg)
        if msg_id not in seen_msg_ids:
            seen_msg_ids.add(msg_id)
            await handle_step(msg, node_name)


async def handle_step(msg, node_name: str = "") -> None:
    """Route a single message to the appropriate Chainlit UI step."""

    if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        if node_name == "format_answer":
            if not cl.user_session.get("answer_streamed"):
                await cl.Message(content=content).send()


def get_node_step_title(node_name: str) -> str:
    return {
        "condense_question": "💬 Condensing question...",
        "resolve_database":  "🗄️ Resolving database...",
        "retrieve_schema":   "🔍 Retrieving candidates...",
        "rerank_tables":     "🎯 Selecting tables...",
        "prune_columns":     "✂️ Pruning columns...",
        "planner":           "📐 Planning query...",
        "sql_coder":         "⚙️ Writing SQL...",
        "sql_validator":     "✅ Validating & executing...",
        "format_answer":     "✍️ Formatting answer...",
    }.get(node_name, "")


# ── Private helpers ───────────────────────────────────────────────────────────


# def _strip_step_comments(sql: str) -> str:
#     """Remove '-- Step N:' comments, keep '-- Assumption:' comments."""
#     return "\n".join(
#         line for line in sql.splitlines()
#         if not re.match(r"^\s*--\s*Step\s+\d+", line, re.IGNORECASE)
#     )


def _format_query_result(raw: str, max_rows: int = 20, query: str = "") -> str:
    """
    Convert raw SQL tool output to a markdown table via pandas.
    Extracts column names from the SQL SELECT clause when available.
    """
    import ast

    import pandas as pd

    try:
        rows = ast.literal_eval(raw.strip())
    except (ValueError, SyntaxError):
        return raw

    if not isinstance(rows, list) or not rows:
        return raw

    if not isinstance(rows[0], (tuple, list)):
        return raw

    df = pd.DataFrame(rows[:max_rows])

    col_names = _extract_column_names(query) if query else []
    if col_names and len(col_names) == len(df.columns):
        df.columns = col_names
    else:
        df.columns = [f"Col {i+1}" for i in range(len(df.columns))]

    table = df.to_markdown(index=False)

    if len(rows) > max_rows:
        table += f"\n\n*... {len(rows) - max_rows} more rows*"

    return table


def _extract_column_names(query: str) -> list[str]:
    """
    Extract column names/aliases from the SELECT clause using sqlparse.
    Handles: SELECT col, SELECT col AS alias, SELECT COUNT(*) AS alias.
    """
    parsed = sqlparse.parse(query)
    if not parsed:
        return []

    stmt = parsed[0]
    select_items = []
    in_select = False

    for token in stmt.tokens:
        if token.ttype is sqlparse.tokens.DML and token.normalized == "SELECT":
            in_select = True
            continue
        if in_select:
            if token.ttype is sqlparse.tokens.Keyword and token.normalized in ("FROM", "INTO"):
                break
            if isinstance(token, sqlparse.sql.IdentifierList):
                for ident in token.get_identifiers():
                    select_items.append(_get_col_display_name(ident))
            elif isinstance(token, sqlparse.sql.Identifier):
                select_items.append(_get_col_display_name(token))
            elif isinstance(token, sqlparse.sql.Function):
                select_items.append(str(token))

    return select_items


def _get_col_display_name(ident) -> str:
    """Return alias if present, otherwise the column name."""
    alias = ident.get_alias()
    if alias:
        return alias
    name = ident.get_real_name()
    return name if name else str(ident)