import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from graph import build_graph
from langgraph.errors import GraphRecursionError
from nodes import list_tables_tool,get_schema_tool,run_query_tool
from loguru import logger


@cl.on_chat_start
async def on_chat_start():
    agent = build_graph()  # one graph instance par session
    cl.user_session.set("agent", agent)
    await cl.Message(content="👋 Posez-moi une question sur la base de données.").send()



@cl.on_message
async def on_message(message: cl.Message):
    """
    Main entry point for incoming chat messages.
    Orchestrates the LangGraph execution with real-time streaming and visual steps.
    """
    agent = cl.user_session.get("agent")
    
    # Local set to track messages processed within this specific execution loop
    seen_msg_ids = set()

    response_msg = None
    try:
        # config
        config={"recursion_limit": 15,                                # Safety guard against infinite SQL loops
                "configurable": {"thread_id": cl.context.session.id}  # unique id per chainlit session
               }
               
        # Create a collapsible visual group in Chainlit for technical steps
        async with cl.Step(name="🔍 SQL Agent", type="run"):
            # Use v2 events for more granular control over the execution stream
            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=message.content)]},
                config=config, 
                version="v2"
            ):
                kind = event["event"]

                # --- 1. Intermediate Node Completion ---
                if kind == "on_chain_end":
                    # Process structured outputs (Tool calls, SQL results, etc.)
                    await _process_node_output(event, seen_msg_ids)

                # --- 2. Live Token Streaming ---
                elif kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    content = _normalize_chunk_content(chunk.content)

                    # Filter: Only stream final answer text. 
                    # If 'tool_call_chunks' is present, the model is generating SQL, not text.
                    if content and not chunk.tool_call_chunks:
                        if not response_msg:
                            # Initialize the UI message on the first valid text token
                            response_msg = cl.Message(content="")
                            await response_msg.send()
                        
                        await response_msg.stream_token(content)

    except ValueError as e:
        # Catch security guardrail violations (e.g., from is_safe_query)
        await cl.Message(content=f"⚠️ Unauthorized Query: {e}").send()
    
    except GraphRecursionError:
        # Handle cases where the agent keeps failing to generate valid SQL
        await cl.Message(content="⚠️ Reach limit of attempts. The query is too complex or failed to converge.").send()
    
    except Exception as e:
        # Fallback for unexpected system errors
        logger.error(f"Unexpected error: {repr(e)}")
        await cl.Message(content="Sorry, I encountered a technical error.").send()

    # Finalize the message stream in the UI
    if response_msg:
        await response_msg.update()


# Helpers

def _normalize_chunk_content(content) -> str:
    """
    Normalize chunk content to handle different LLM provider formats.
    
    Gemini often returns a list of blocks/parts, while OpenAI returns a plain string.
    This helper ensures the streaming logic receives a consistent string format.
    """
    if isinstance(content, list):
        # Extract text from complex block structures (common in Gemini/Anthropic)
        return " ".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return content if isinstance(content, str) else ""


async def _process_node_output(event: dict, seen_msg_ids: set) -> None:
    """
    Extract and handle new messages from a LangGraph node output.
    
    LangGraph emits the full state history at the end of each node. 
    This function filters out already-processed messages using a set of IDs
     to prevent duplicate logs/steps in the Chainlit UI.
    """
    # langgraph_node metadata identifies which specific node just finished
    node_name = event["metadata"].get("langgraph_node")
    if not node_name:
        return

    output = event["data"].get("output", {})
    # Ensure the output follows the MessagesState structure
    if not isinstance(output, dict) or "messages" not in output:
        return

    for msg in output["messages"]:
        # Use Python's object ID for reliable deduplication during the session
        msg_id = id(msg)
        if msg_id not in seen_msg_ids:
            seen_msg_ids.add(msg_id)
            # Route the new message to the UI handler (e.g., showing SQL or Schema)
            await _handle_step(msg)

async def _handle_step(msg):
   
    
    if isinstance(msg, ToolMessage) and msg.name == list_tables_tool.name:
        async with cl.Step(name="📋 Listing tables", type="tool") as s:
            s.output = msg.content

    elif isinstance(msg, AIMessage) and msg.tool_calls:
        tool_name = msg.tool_calls[0]["name"]
        tool_args = msg.tool_calls[0]["args"]

        if tool_name == get_schema_tool.name :
            async with cl.Step(name="🗂️ Fetching schema", type="tool") as s:
                s.input = str(tool_args)

        elif tool_name == run_query_tool.name :
            async with cl.Step(name="⚙️ Generated SQL", type="tool") as s:
                s.output = f"```sql\n{tool_args.get('query', '')}\n```"

    elif isinstance(msg, ToolMessage):
        if msg.name == get_schema_tool.name:
            async with cl.Step(name="🗂️ Schema retrieved", type="tool") as s:
                s.output = msg.content[:500]

        elif msg.name == run_query_tool.name:
            async with cl.Step(name="📊 Query results", type="tool") as s:
                s.output = msg.content