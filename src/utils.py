from pathlib import Path
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


def save_graph_png(agent) -> Path:
    output_path = Path(__file__).resolve().parents[1] / "assets" / "graph.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        f.write(agent.get_graph().draw_mermaid_png())
    
    print(f"Graph PNG saved at: {output_path}")
    return output_path


def get_last_cycle(messages: list) -> list:
    """Return technical messages from the last cycle (after the last user question)."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i + 1:]  # everything after the last user question
    return []

"""
def generate_query(state: AgentState) -> dict:
    

    # ---- Contexte Management ---- #
   
    messages = state["messages"]
    
    # 1. Keep all user questions (business context)
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    
    # 2. Retrieve only the last technical cycle (schema or SQL error)
    last_technical_cycle = get_last_cycle(messages)
    
    # 3. context assembly : prompt + previous question + last tech cycle + current question
    messages = (
        [SystemMessage(content=GENERATE_QUERY_PROMPT)]
        + human_messages[:-1]       # previous questions
        + last_technical_cycle      # last cycle with only the fresh db schema
        + [human_messages[-1]]      # current question
    )
    
    # ---- Invoke model ---- #
    response = model.bind_tools([run_query_tool]).invoke(messages)
    
    return {"messages": [response]}
"""