import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import chainlit as cl
from langchain_core.messages import AIMessage, ToolMessage
from graph import build_graph

agent = build_graph()


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("agent", agent)
    await cl.Message(content="👋 Posez-moi une question sur la base de données.").send()


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    final_response = ""

    async with cl.Step(name="🔍 SQL Agent", type="run"):
        async for step in agent.astream(
            {"messages": [{"role": "user", "content": message.content}]},
            stream_mode="values"
        ):
            last_msg = step["messages"][-1]
            await _handle_step(last_msg)

            if isinstance(last_msg, AIMessage) and last_msg.content and not last_msg.tool_calls:
                final_response = last_msg.content

    if final_response:
        response_msg = cl.Message(content="")
        await response_msg.send()
        for char in final_response:
            await response_msg.stream_token(char)
        await response_msg.update()


# Helpers


async def _handle_step(msg):
    if isinstance(msg, AIMessage) and "Available tables" in (msg.content or ""):
        async with cl.Step(name="📋 Listing tables", type="tool") as s:
            s.output = msg.content

    elif isinstance(msg, AIMessage) and msg.tool_calls:
        tool_name = msg.tool_calls[0]["name"]
        tool_args = msg.tool_calls[0]["args"]

        if tool_name == "sql_db_schema":
            async with cl.Step(name="🗂️ Fetching schema", type="tool") as s:
                s.input = str(tool_args)

        elif tool_name == "sql_db_query":
            async with cl.Step(name="⚙️ Generated SQL", type="tool") as s:
                s.output = f"```sql\n{tool_args.get('query', '')}\n```"

    elif isinstance(msg, ToolMessage):
        if msg.name == "sql_db_schema":
            async with cl.Step(name="🗂️ Schema retrieved", type="tool") as s:
                s.output = msg.content[:500]

        elif msg.name == "sql_db_query":
            async with cl.Step(name="📊 Query results", type="tool") as s:
                s.output = msg.content