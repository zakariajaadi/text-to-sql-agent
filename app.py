import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from graph import build_graph
from langgraph.errors import GraphRecursionError
from nodes import list_tables_tool, get_schema_tool, run_query_tool
from loguru import logger

import uuid
from config import model
from condense_question import condense_question_chain



@cl.on_chat_start
async def on_chat_start():
    agent = build_graph()
    condense_chain = condense_question_chain(model)
    
    cl.user_session.set("agent", agent)
    cl.user_session.set("condense_chain", condense_chain)
    cl.user_session.set("human_history", [])
    
    await cl.Message(content="👋 Posez-moi une question sur la base de données.").send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Main entry point for incoming chat messages.
    Orchestrates the LangGraph execution with real-time streaming and visual steps.
    """
    agent = cl.user_session.get("agent")
    condense_chain = cl.user_session.get("condense_chain")
    seen_msg_ids = set()
    response_msg = None

    # Reset the streaming flag for this new message
    cl.user_session.set("answer_streamed", False)

    # Condense follow-up question into a standalone question
    history = cl.user_session.get("human_history", [])
    standalone_question = (
        await condense_chain.ainvoke({
            "question": message.content,
            "chat_history": history
        })
        if history
        else message.content
    )

    logger.debug(f"standalone_question: {standalone_question}")


    # Update history with the original question (before condensation)
    history.append(HumanMessage(content=message.content))
    cl.user_session.set("human_history", history)

    try:
        # Fresh thread_id at each invocation — no state accumulation between questions
        config = {
            "recursion_limit": 15,
            "configurable": {"thread_id": str(uuid.uuid4())}
        }

        async with cl.Step(name="🔍 SQL Agent", type="run"):
            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=standalone_question)]},  # condensed question
                config=config,
                version="v2"
            ):
                kind = event["event"]

                # --- 1. Intermediate Node Completion ---
                if kind == "on_chain_end":
                    await _process_node_output(event, seen_msg_ids)

                # --- 2. Live Token Streaming ---
                elif kind == "on_chat_model_stream":
                    node_name = event["metadata"].get("langgraph_node", "")

                    # Only stream the final answer from format_answer
                    if node_name != "format_answer":
                        continue

                    chunk = event["data"]["chunk"]
                    content = _normalize_chunk_content(chunk.content)

                    if content and not chunk.tool_call_chunks:
                        if not response_msg:
                            response_msg = cl.Message(content="")
                            await response_msg.send()
                            cl.user_session.set("answer_streamed", True)

                        await response_msg.stream_token(content)

    except ValueError as e:
        await cl.Message(content=f"⚠️ Unauthorized Query: {e}").send()

    except GraphRecursionError:
        await cl.Message(content="⚠️ Reach limit of attempts. The query is too complex or failed to converge.").send()

    except Exception as e:
        logger.error(f"Unexpected error: {repr(e)}")
        await cl.Message(content="Sorry, I encountered a technical error.").send()

    if response_msg:
        await response_msg.update()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_chunk_content(content) -> str:
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


async def _process_node_output(event: dict, seen_msg_ids: set) -> None:
    """
    Extract and handle new messages from a LangGraph node output.
    Filters out already-processed messages using a set of IDs.
    """
    node_name = event["metadata"].get("langgraph_node")
    if not node_name:
        return

    output = event["data"].get("output", {})
    # ── Special case: assess_question_complexity returns structured data, not messages
    if node_name == "assess_question_complexity" and "question_complexity" in output:
        complexity = output["question_complexity"]
        emoji = {"simple": "🟢", "complex": "🟠", "out_of_scope": "🔴"}.get(complexity, "⚪")
        async with cl.Step(name=f"{emoji} Complexity: {complexity}", type="tool") as s:
            s.output = f"Question classified as **{complexity}**"
        return
    
    # Standard message-based processing
    if not isinstance(output, dict) or "messages" not in output:
        return

    for msg in output["messages"]:
        msg_id = id(msg)
        if msg_id not in seen_msg_ids:
            seen_msg_ids.add(msg_id)
            await _handle_step(msg, node_name)


async def _handle_step(msg, node_name: str = "") -> None:
    """
    Route a single message to the appropriate Chainlit UI step.
    """

    # ── ToolMessage: list_tables result ───────────────────────────────────────
    if isinstance(msg, ToolMessage) and msg.name == list_tables_tool.name:
        async with cl.Step(name="📋 Listing tables", type="tool") as s:
            s.output = msg.content

    # ── AIMessage with tool_calls: LLM requesting a tool ──────────────────────
    elif isinstance(msg, AIMessage) and msg.tool_calls:
        tool_name = msg.tool_calls[0]["name"]
        tool_args = msg.tool_calls[0]["args"]

        if tool_name == get_schema_tool.name:
            async with cl.Step(name="🗂️ Fetching schema", type="tool") as s:
                s.input = str(tool_args)

        elif tool_name == run_query_tool.name:
            if node_name == "check_query":
                async with cl.Step(name="⚙️ Generated SQL", type="tool") as s:
                    s.output = f"```sql\n{tool_args.get('query', '')}\n```"

    # ── ToolMessage: tool execution results ───────────────────────────────────
    elif isinstance(msg, ToolMessage):
        if msg.name == get_schema_tool.name:
            async with cl.Step(name="🗂️ Schema retrieved", type="tool") as s:
                s.output = msg.content[:500]

        elif msg.name == run_query_tool.name:
            async with cl.Step(name="📊 Query results", type="tool") as s:
                s.output = msg.content

    # ── AIMessage without tool_calls: plan or final answer ────────────────────
    elif isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:

        if node_name == "plan_query_generation":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            async with cl.Step(name="📝 Query plan", type="tool") as s:
                s.output = content

        elif node_name == "format_answer":
            # Fallback: if streaming didn't capture the answer, display it here
            if not cl.user_session.get("answer_streamed"):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                await cl.Message(content=content).send()