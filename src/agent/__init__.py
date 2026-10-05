# src/agent/__init__.py

# Keep this file empty: lib/messages.py imports agent.tools, and agent.nodes imports lib.
# Re-exporting anything here (e.g. build_graph) would close the cycle
# lib → agent/__init__ → agent.graph → agent.nodes → lib and break at import time.
