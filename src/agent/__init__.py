# src/agent/__init__.py

# Keep this file empty: importing a light submodule (e.g. agent.prompts or agent.schemas
# from ui/follow_ups.py) runs this file first. Re-exporting anything here (e.g. build_graph)
# would load agent.graph → agent.nodes → lib and resources on every
# `import agent.x`, and would create a cycle as soon as a lib module imports agent.
