"""
The runtime resources (resources.py: LLM clients and API keys, databases, Chroma index) are never
built by the tests: one stub stands for them, before any test module imports the agent or the
generators. A test sets the attributes it needs with monkeypatch.
"""

import sys
import types
from collections import defaultdict
from unittest.mock import MagicMock

sys.modules.setdefault("resources", types.SimpleNamespace(
    db_registry={},
    vector_store=None,
    model=MagicMock(),
    fast_model=MagicMock(),
    rank_tables=MagicMock(),
    fk_graphs=defaultdict(dict),
    schema_store=defaultdict(dict),
    metrics_store=defaultdict(dict),
    switch_coordinator=MagicMock(),
))
