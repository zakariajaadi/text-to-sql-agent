import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import asyncio
import uuid
from pathlib import Path

import chainlit as cl
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langsmith import Client
from loguru import logger

from agent.graph import build_graph
from config import cfg
from lib.query_cancel import discard_cancel, request_cancel
from ui.streaming import TurnRenderer

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
    Runs the LangGraph agent and renders its events live (ui/streaming.py).
    """
    agent = cl.user_session.get("agent")
    turn  = TurnRenderer(db_name=cl.user_session.get("chat_profile"))
    # The turn's cancel key: the Stop button sets its event, and the query the turn is
    # running gets cancelled on the cluster (lib/query_cancel.py)
    cancel_key = str(uuid.uuid4())

    config = {"recursion_limit": 15}
    if cfg.memory.enabled:
        # thread_id per session — LangGraph accumulates state across turns
        config["configurable"] = {"thread_id": cl.user_session.get("thread_id")}

    try:
        await turn.start()
        async for event in agent.astream_events(
            {
                "messages":    [HumanMessage(content=message.content)],
                "selected_db": cl.user_session.get("chat_profile"),
                # The planner may ask the user one question (ask_clarification); the eval never sets it
                "clarify":     cfg.get_db(cl.user_session.get("chat_profile")).clarification,
                "cancel_key":  cancel_key,
            },
            config=config,
            version="v2",
        ):
            await turn.handle(event)
        await turn.finish()

    except GraphRecursionError:
        await turn.finish("failed")
        await cl.Message(content="⚠️ Nombre maximal de tentatives atteint. La requête est trop complexe ou n'a pas abouti.").send()

    except asyncio.CancelledError:
        # Stop button: cancel the turn's running query on the cluster, close the steps, then
        # let the cancellation through
        request_cancel(cancel_key)
        await turn.finish("stopped")
        raise

    except Exception as e:
        logger.error(f"Unexpected error: {repr(e)}")
        await turn.finish("failed")
        await cl.Message(content="Désolé, une erreur technique est survenue.").send()

    finally:
        # The thread polling the query holds the event itself: forgetting the key loses nothing
        discard_cancel(cancel_key)

    await turn.finalize_answer()


async def record_vote(action: cl.Action, score: int):
    """Send the vote to LangSmith, then leave only the chosen thumb: one vote per answer."""
    run_id = action.payload.get("run_id")
    if cfg.langsmith.tracing and run_id:
        ls_client.create_feedback(run_id=run_id, key="user_feedback", score=score)
    await action.remove()
    # The frontend removes an action by its id alone
    if other_id := action.payload.get("other_id"):
        await cl.Action(id=other_id, name="", payload={}).remove()
    # The chosen thumb comes back without a callback; its "voted-" id styles it (filled, not clickable, custom.css)
    await cl.Action(id=f"voted-{uuid.uuid4()}", name="vote_cast", payload={}, icon=action.icon).send(for_id=action.forId)


@cl.action_callback("thumbs_up")
async def on_thumbs_up(action: cl.Action):
    await record_vote(action, score=1)


@cl.action_callback("thumbs_down")
async def on_thumbs_down(action: cl.Action):
    await record_vote(action, score=0)