import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import uuid

import chainlit as cl
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from loguru import logger

from graph import build_graph
from utils import get_node_step_title, normalize_chunk_content, process_node_output


@cl.on_chat_start
async def on_chat_start():
    agent = build_graph()

    cl.user_session.set("agent", agent)
    cl.user_session.set("thread_id", str(uuid.uuid4()))

    await cl.Message(content="👋 Posez-moi une question sur la base de données.").send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Main entry point for incoming chat messages.
    Orchestrates the LangGraph execution with real-time streaming and visual steps.
    """
    agent = cl.user_session.get("agent")
    seen_msg_ids = set()
    response_msg = None

    # Reset the streaming flag for this new message
    cl.user_session.set("answer_streamed", False)

    try:
        # thread_id per session — LangGraph accumulates state across turns
        config = {
            "recursion_limit": 15,
            "configurable": {"thread_id": cl.user_session.get("thread_id")}
        }

        async with cl.Step(name="🤖 SQL Agent", type="run") as agent_step:
            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=message.content)]},
                config=config,
                version="v2"
            ):
                kind = event["event"]

                # --- 1. Intermediate Node Completion ---
                if kind == "on_chain_end":
                    await process_node_output(event, seen_msg_ids)
                    # Update the step header to reflect the current node
                    title = get_node_step_title(event["metadata"].get("langgraph_node", ""))
                    if title:
                        agent_step.name = title
                        await agent_step.update()

                # --- 2. Live Token Streaming ---
                elif kind == "on_chat_model_stream":
                    node_name = event["metadata"].get("langgraph_node", "")

                    # Only stream the final answer from format_answer
                    if node_name != "format_answer":
                        continue

                    chunk = event["data"]["chunk"]
                    content = normalize_chunk_content(chunk.content)

                    if content and not chunk.tool_call_chunks:
                        if not response_msg:
                            response_msg = cl.Message(content="")
                            await response_msg.send()
                            cl.user_session.set("answer_streamed", True)

                        await response_msg.stream_token(content)

            # Reset title to neutral once all nodes have completed
            agent_step.name = "🤖 SQL Agent"
            await agent_step.update()

    except ValueError as e:
        await cl.Message(content=f"⚠️ Unauthorized Query: {e}").send()

    except GraphRecursionError:
        await cl.Message(content="⚠️ Reach limit of attempts. The query is too complex or failed to converge.").send()

    except Exception as e:
        logger.error(f"Unexpected error: {repr(e)}")
        await cl.Message(content="Sorry, I encountered a technical error.").send()

    if response_msg:
        await response_msg.update()
