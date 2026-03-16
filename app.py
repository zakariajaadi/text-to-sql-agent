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
                    node_name = event["metadata"].get("langgraph_node", "")
                    
                    # Only stream the final answer from generate_query
                    # This filters out text from assess_question_complexity (structured output)
                    # and plan_query_generation (SQL plan), which are handled via cl.Step
                    if node_name != "generate_query":
                        continue

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
    
    The node_name is forwarded to _handle_step to allow node-aware rendering —
    for example, distinguishing a plan AIMessage from a final answer AIMessage.
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
            # Forward node_name so _handle_step can distinguish ambiguous message types
            await _handle_step(msg, node_name)


async def _handle_step(msg, node_name: str = "") -> None:
    """
    Route a single message to the appropriate Chainlit UI step.

    Each branch maps a message type + node origin to a visual step in the UI.
    The node_name parameter is essential for ambiguous cases — specifically,
    plain AIMessages without tool_calls, which can be either a SQL plan
    (from plan_query_generation) or the final answer (from generate_query).
    The final answer is already streamed token-by-token and must not be duplicated here.
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
            # Show which tables the LLM selected for schema retrieval
            async with cl.Step(name="🗂️ Fetching schema", type="tool") as s:
                s.input = str(tool_args)

        elif tool_name == run_query_tool.name:
            if node_name == "check_query":
                # Display the generated SQL query before execution
                async with cl.Step(name="⚙️ Generated SQL", type="tool") as s:
                    s.output = f"```sql\n{tool_args.get('query', '')}\n```"

    # ── ToolMessage: tool execution results ───────────────────────────────────
    elif isinstance(msg, ToolMessage):
        if msg.name == get_schema_tool.name:
            # Truncate schema output to avoid flooding the UI
            async with cl.Step(name="🗂️ Schema retrieved", type="tool") as s:
                s.output = msg.content[:500]

        elif msg.name == run_query_tool.name:
            async with cl.Step(name="📊 Query results", type="tool") as s:
                s.output = msg.content

    # ── AIMessage without tool_calls: plan or final answer ────────────────────
    elif isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
        # Only render the SQL plan — the final answer is already streamed
        # token-by-token via on_chat_model_stream and must not be duplicated here
        if node_name == "plan_query_generation":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            async with cl.Step(name="📝 Query plan", type="tool") as s:
                s.output = content