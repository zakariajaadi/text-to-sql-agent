from pathlib import Path

def save_graph_png(agent) -> Path:
    output_path = Path(__file__).resolve().parents[1] / "assets" / "graph.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        f.write(agent.get_graph().draw_mermaid_png())
    
    print(f"Graph PNG saved at: {output_path}")
    return output_path