# offline/query_cancel.py

"""
Stopping the CLI cancels its running query on the cluster: offline/studio stops a job with SIGTERM,
Ctrl+C sends SIGINT. The machinery lives in lib/query_cancel.py, shared with the app (whose Stop
button cancels the turn's query the same way); this module keeps the import path of the CLI.
"""

from lib.query_cancel import (  # noqa: F401
    DRIVERS,
    POLL_SECONDS,
    QueryCancelled,
    install,
)
