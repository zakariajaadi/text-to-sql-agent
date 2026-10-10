import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import time
import uuid
from pathlib import Path

import chainlit as cl
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from loguru import logger

from agent.graph import build_graph
from ui.streaming import (
    attach_result_card,
    get_agent_step_summary,
    get_node_step_title,
    normalize_chunk_content,
    process_node_output,
    rename_retry_steps,
    send_follow_ups,
)
from langsmith import Client
from config import cfg 


ls_client = Client()

PUBLIC_ICONS = Path(__file__).parent / "public" / "icons"

@cl.on_chat_start
async def on_chat_start():
    agent = build_graph()

    cl.user_session.set("agent", agent)
    if cfg.memory.enabled:
        cl.user_session.set("thread_id", str(uuid.uuid4()))
    # No welcome message: Chainlit only shows the starters while the thread is empty


@cl.set_chat_profiles
async def chat_profiles():
    """
    One profile per database: the user picks the database, switching starts a new conversation.
    Each profile shows its example questions (starters) on the welcome screen.
    """
    return [
        cl.ChatProfile(
            name=db.name,
            markdown_description=db.tagline or db.name,
            icon=_public_icon("database"),
            starters=[
                cl.Starter(label=s["question"], message=s["question"], icon=_public_icon(s.get("icon")))
                for s in db.starters
            ],
        )
        for db in cfg.databases
    ]


def _public_icon(name: str | None) -> str | None:
    """URL of public/icons/<name>.svg, or None when there is no such icon."""
    if name and (PUBLIC_ICONS / f"{name}.svg").exists():
        return f"/public/icons/{name}.svg"
    return None

@cl.on_message
async def on_message(message: cl.Message):
    """
    Main entry point for incoming chat messages.
    Orchestrates the LangGraph execution with real-time streaming and visual steps.
    """
    agent = cl.user_session.get("agent")
    seen_msg_ids = set()
    run_id= None
    turn_start  = time.perf_counter()
    node_starts = {}  # run_id of a running graph node → start time, to time its step

    # Reset the per-turn UI state: answer message (streamed or not), query result, SQL retries
    # and their steps, follow-up questions being suggested
    cl.user_session.set("answer_msg", None)
    cl.user_session.set("query_result", None)
    cl.user_session.set("sql_retries", 0)
    cl.user_session.set("retry_steps", [])
    cl.user_session.set("retrieve_duration", None)
    cl.user_session.set("follow_ups_task", None)

    try:
        
        config = {"recursion_limit": 15}
        if cfg.memory.enabled:
            # thread_id per session — LangGraph accumulates state across turns
            config["configurable"] = {"thread_id": cl.user_session.get("thread_id")}



        async with cl.Step(name="🤖 SQL Agent", type="run") as agent_step:
            async for event in agent.astream_events(
                {
                    "messages":    [HumanMessage(content=message.content)],
                    "selected_db": cl.user_session.get("chat_profile"),
                },
                config=config,
                version="v2"
            ):
                kind = event["event"]
                
                # Capture run_id from the root LangSmith run
                if run_id is None and "run_id" in event:
                    run_id = event["run_id"]


                # A graph node run is the chain named after its node (nested runs carry the
                # node name in their metadata only)
                node_name = event["metadata"].get("langgraph_node", "")
                is_node   = bool(node_name) and event.get("name") == node_name

                # --- 1. Node start: show it in the step header, start its timer ---
                if kind == "on_chain_start" and is_node:
                    node_starts[event["run_id"]] = time.perf_counter()
                    title = get_node_step_title(node_name)
                    if title:
                        agent_step.name = title
                        await agent_step.update()

                # --- 2. Intermediate Node Completion ---
                elif kind == "on_chain_end":
                    start    = node_starts.pop(event["run_id"], None) if is_node else None
                    duration = time.perf_counter() - start if start is not None else None
                    await process_node_output(event, seen_msg_ids, duration)

                # --- 3. Live Token Streaming ---
                elif kind == "on_chat_model_stream":
                    node_name = event["metadata"].get("langgraph_node", "")

                    # Only stream the final answer from format_answer
                    if node_name != "format_answer":
                        continue

                    chunk = event["data"]["chunk"]
                    content = normalize_chunk_content(chunk.content)

                    if content and not chunk.tool_call_chunks:
                        response_msg = cl.user_session.get("answer_msg")
                        if not response_msg:
                            response_msg = cl.Message(content="")
                            await response_msg.send()
                            cl.user_session.set("answer_msg", response_msg)

                        await response_msg.stream_token(content)

            # Once all nodes have completed: tables the answer comes from and total time
            agent_step.name = get_agent_step_summary(time.perf_counter() - turn_start)
            await agent_step.update()

    except GraphRecursionError:
        await rename_retry_steps("failed")
        await cl.Message(content="⚠️ Nombre maximal de tentatives atteint. La requête est trop complexe ou n'a pas abouti.").send()

    except Exception as e:
        logger.error(f"Unexpected error: {repr(e)}")
        await cl.Message(content="Désolé, une erreur technique est survenue.").send()
    
    # Attach the result card and feedback actions to the response message itself
    response_msg = cl.user_session.get("answer_msg")
    if response_msg:
        await attach_result_card(response_msg)
        if run_id:
            response_msg.actions = [
                cl.Action(name="thumbs_up",   payload={"run_id": str(run_id)}, label="👍"),
                cl.Action(name="thumbs_down", payload={"run_id": str(run_id)}, label="👎"),
            ]
        await response_msg.update()
        await send_follow_ups(response_msg)
        


@cl.action_callback("thumbs_up")
async def on_thumbs_up(action: cl.Action):
    run_id = action.payload.get("run_id")
    if cfg.langsmith.tracing and run_id:
        ls_client.create_feedback(run_id=run_id, key="user_feedback", score=1)
    await action.remove()


@cl.action_callback("thumbs_down")
async def on_thumbs_down(action: cl.Action):
    run_id = action.payload.get("run_id")
    if cfg.langsmith.tracing and run_id:
        ls_client.create_feedback(run_id=run_id, key="user_feedback", score=0)
    await action.remove()