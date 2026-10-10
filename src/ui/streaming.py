"""
Chainlit UI helpers
Used by: app.py (on_message, astream_events loop)
"""

import asyncio
import re

import chainlit as cl
import sqlparse
from langchain_core.messages import AIMessage

from config import cfg
from lib.column_pruning import get_column_names
from lib.messages import get_question
from lib.sql_tools import get_query_tables
from resources import schema_store
from ui.follow_ups import suggest_follow_ups
from ui.result_card import build_result_card_props

FOLLOW_UPS_TIMEOUT_S = 10  # past this, the answer simply comes without suggestions

# Name of the step shown when the database rejects a query, renamed once the retry loop ends
RETRY_STEP_NAMES = {
    "running": "🔁 Query rejected, fixing it...",
    "fixed":   "🔁 Query fixed automatically",
    "failed":  "❌ Query could not be fixed",
}


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


async def process_node_output(event: dict, seen_msg_ids: set, duration: float | None = None) -> None:
    """
    Extract and handle new messages from a LangGraph node output.
    Filters out already-processed messages using a set of IDs.
    duration: seconds the node took (None for nested runs), appended to the step name.
    """
    node_name = event["metadata"].get("langgraph_node")
    if not node_name:
        return
    took = _took(duration)

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
            async with cl.Step(name=f"💬 Question rewritten{took}", type="run") as s:
                s.output = f"**Original:** {original}\n**Rewritten:** {condensed}"
        return

    # ── retrieve_schema: skip (merged into rerank_tables display) ──
    if node_name == "retrieve_schema":
        if duration is not None:
            cl.user_session.set("retrieve_duration", duration)
        return

    # ── rerank_tables: show the candidate tables passed to planner/sql_coder (the SQL may use fewer) ──
    if node_name == "rerank_tables" and "selected_schemas" in output:
        table_names = list(output["selected_schemas"])
        in_scope = output.get("is_in_scope", True)
        emoji = "🟢" if in_scope else "🔴"
        # The step covers both retrieve_schema (vector search) and rerank_tables (LLM)
        if duration is not None:
            took = _took(duration + (cl.user_session.get("retrieve_duration") or 0))
        tables = f"{len(table_names)} candidate table{'s' if len(table_names) != 1 else ''}"
        async with cl.Step(name=f"{emoji} {tables}{took}", type="run") as s:
            s.output = ", ".join(table_names)
        return
    
    # ── prune_columns: fixed title, column count before → after in the body ──
    if node_name == "prune_columns" and "retrieved_schema" in output:
        selected_schemas = event["data"].get("input", {}).get("selected_schemas", {})
        columns_before   = sum(len(get_column_names(ddl)) for ddl in selected_schemas.values())
        columns_after    = len(get_column_names(output["retrieved_schema"]))

        async with cl.Step(name=f"✂️ Column pruning{took}", type="run") as s:
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
        # Tables the plan uses beyond the candidates, loaded by planner
        state      = event["data"].get("input", {})
        candidates = state.get("selected_schemas", {})
        loaded     = [t for t in output.get("selected_schemas", {}) if t not in candidates]
        metrics    = state.get("selected_metrics") or []   # business metrics picked by rerank_tables
        async with cl.Step(name=f"📐 Query plan{took}", type="run") as s:
            s.output = output["query_plan"]
            if metrics:
                s.output += f"\n\n**Metrics:** {', '.join(metrics)}"
            if loaded:
                s.output += f"\n\n**Loaded for the plan:** {', '.join(loaded)}"
        return
        
    # ── sql_coder: show generated SQL (strip step comments) ──
    if node_name == "sql_coder" and "generated_query" in output:
        query = output["generated_query"]
        formatted = sqlparse.format(query, reindent=True, keyword_case="upper")
        # After a rejected query, sql_coder rewrites it from the database error
        name = "⚙️ Corrected SQL" if cl.user_session.get("sql_retries") else "⚙️ Generated SQL"
        async with cl.Step(name=f"{name}{took}", type="run") as s:
            s.output = f"```sql\n{formatted}\n```"
        return

    # ── sql_validator: show failure, or keep the result for the result card ──
    if node_name == "sql_validator":
        feedback = output.get("validation_feedback", "")
        if feedback:
            # Show the database error only, never the instructions written for sql_coder
            cl.user_session.set("sql_retries", cl.user_session.get("sql_retries", 0) + 1)
            async with cl.Step(name=RETRY_STEP_NAMES["running"], type="tool") as s:
                s.output = (
                    f"The database rejected the query:\n```\n{_short_db_error(output.get('validation_error', ''))}\n```\n"
                    "The agent reads this error and rewrites the query."
                )
            cl.user_session.get("retry_steps").append(s)
            return

        if "result_rows" in output:
            await rename_retry_steps("fixed")
            rows  = output["result_rows"]
            state = event["data"].get("input", {})
            cl.user_session.set("query_result", {
                "columns":   output["result_columns"],
                "rows":      rows,
                "truncated": output.get("result_truncated", False),
                "sql":       state.get("generated_query", ""),
                "tables":    _tables_in_sql(state.get("generated_query", ""), state["selected_db"]),
            })
            # Suggest follow-up questions while format_answer writes the answer
            cl.user_session.set("follow_ups_task", asyncio.create_task(suggest_follow_ups(
                question = get_question(state),
                schema   = state.get("retrieved_schema", ""),
                sql      = state.get("generated_query", ""),
                columns  = output["result_columns"],
                rows     = rows,
            )))
            more = "+" if output.get("result_truncated") else ""
            label = "row" if len(rows) == 1 and not more else "rows"
            async with cl.Step(name=f"📊 {len(rows)}{more} {label} returned{took}", type="tool"):
                pass
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
            if not cl.user_session.get("answer_msg"):
                answer_msg = cl.Message(content=content)
                await answer_msg.send()
                cl.user_session.set("answer_msg", answer_msg)


async def attach_result_card(answer_msg: cl.Message) -> None:
    """Attach the ResultCard element (chart / table / SQL) of this turn's query to the answer."""
    result = cl.user_session.get("query_result")
    if not result or not result["rows"]:
        return

    props = build_result_card_props(
        columns   = result["columns"],
        rows      = result["rows"],
        sql       = result["sql"],
        truncated = result["truncated"],
        tables    = result["tables"],
        retries   = cl.user_session.get("sql_retries", 0),
        db_name   = cl.user_session.get("chat_profile") or "",
    )
    answer_msg.elements = [cl.CustomElement(name="ResultCard", props=props, display="inline")]


async def rename_retry_steps(outcome: str) -> None:
    """Rename this turn's rejected-query steps once the retry loop ends ("fixed" or "failed")."""
    for step in cl.user_session.get("retry_steps") or []:
        step.name = RETRY_STEP_NAMES[outcome]
        await step.update()


async def send_follow_ups(answer_msg: cl.Message) -> None:
    """
    Send the FollowUps element (clickable next questions) under an answer already sent.
    Sent on its own once the suggestions are ready, so it never delays the answer and card.
    """
    task = cl.user_session.get("follow_ups_task")
    if not task:
        return
    try:
        questions = await asyncio.wait_for(task, timeout=FOLLOW_UPS_TIMEOUT_S)
    except asyncio.TimeoutError:
        return
    if not questions:
        return

    element = cl.CustomElement(name="FollowUps", props={"questions": questions}, display="inline")
    answer_msg.elements.append(element)
    await element.send(for_id=answer_msg.id)


def get_agent_step_summary(elapsed: float) -> str:
    """Final name of the agent step: tables the answer comes from and total time."""
    result = cl.user_session.get("query_result")
    parts  = ["SQL Agent"]
    if result and result["tables"]:
        n = len(result["tables"])
        parts.append(f"{n} table{'s' if n > 1 else ''}")
    parts.append(f"{elapsed:.1f} s")
    return " · ".join(parts)


def get_node_step_title(node_name: str) -> str:
    return {
        "condense_question": "💬 Condensing question...",
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


def _short_db_error(error: str, max_len: int = 240) -> str:
    """
    Keep the readable part of a database error. A Hive error comes wrapped by pyhive and
    SQLAlchemy: "(pyhive.exc.OperationalError) TExecuteStatementResp(... errorMessage="Error
    while compiling statement: FAILED: SemanticException [Error 10004]: Line 1:7 Invalid
    table alias or column reference 'x'") [SQL: ...] (Background on this error at: ...)".
    """
    match = re.search(r"""errorMessage=(['"])(.*?)\1[,)]""", error, re.DOTALL)
    message = match.group(2) if match else error.split("\n[SQL:")[0]
    message = message.replace("\\n", "\n")  # errorMessage comes from a repr: newlines are escaped
    # A wrapped error ends with its cause: "... dependencies.\nUnderlying error: org.apache...
    # SemanticException: Line 4:6 Expression not in GROUP BY key 'x'". The cause is the useful part.
    if "Underlying error:" in message:
        message = message.split("Underlying error:")[-1].strip()
        message = re.sub(r"^[\w.]+\.\w+Exception: ", "", message)  # org.apache...SemanticException:
    message = re.sub(r"^Error while compiling statement: FAILED: ", "", message)
    message = re.sub(r"^\w+Exception( \[Error \d+\])?:? ", "", message)   # SemanticException [Error 10004]:
    message = re.sub(r"^[Ll]ine \d+:\d+ ", "", message).strip()            # Line 1:7
    message = message[:1].upper() + message[1:]
    return message if len(message) <= max_len else message[:max_len].rstrip() + "…"


def _took(duration: float | None) -> str:
    """Step name suffix with the node duration (" · 1.2 s"), empty when unknown or negligible."""
    return f" · {duration:.1f} s" if duration is not None and duration >= 0.1 else ""


def _tables_in_sql(sql: str, db_name: str) -> list[str]:
    """
    Tables of the database the query actually reads, parsed by sqlglot.
    Matched against every table of the database, not only the selected ones: the SQL may read
    a table planner loaded, or one named by a "-- Joins:" hint.
    """
    known = {table.lower(): table for table in schema_store[db_name]}
    names = get_query_tables(sql, cfg.get_db(db_name).dialect)
    return [known[name.lower()] for name in names if name.lower() in known]
