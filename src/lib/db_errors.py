"""
Database errors: telling an unreachable database (network, VPN, cluster down, Kerberos) from a
query the database rejected. Used by: agent/nodes.py (sql_validator), resources.py (startup).

An unreachable database is not the query's fault: the agent must not rewrite a correct query
for it, and the user must be told to retry later rather than read a SQL error.
"""

import socket

# Exception classes of the drivers raised when the server can't be reached or authenticated,
# by name: the drivers (thrift, kerberos) are optional imports
UNREACHABLE_CLASSES = {"TTransportException", "GSSError", "KrbError"}

# Messages of an unreachable server, when the driver wraps it in a generic error
UNREACHABLE_MESSAGES = (
    "could not connect", "connection refused", "connection reset", "timed out",
    "network is unreachable", "no route to host", "name or service not known",
    "cannot contact any kdc", "tsocket read 0 bytes",
    # HiveServer2 closed the session of a pooled connection (idle timeout): a fresh one works
    "invalid sessionhandle", "invalid operationhandle",
)


def is_unreachable(error: BaseException) -> bool:
    """
    True when the error, or one it wraps (SQLAlchemy's .orig, __cause__, __context__), says the
    database could not be reached: network errors, timeouts, transport and Kerberos failures.
    False for an error the database returned about the query (syntax, unknown column, limits).
    """
    seen = set()
    stack = [error]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))

        if isinstance(e, (ConnectionError, TimeoutError, socket.timeout, socket.gaierror)):
            return True
        if type(e).__name__ in UNREACHABLE_CLASSES:
            return True
        if getattr(e, "connection_invalidated", False):   # SQLAlchemy: the connection dropped
            return True
        message = str(e).lower()
        if any(text in message for text in UNREACHABLE_MESSAGES):
            return True

        stack += [getattr(e, "orig", None), e.__cause__, e.__context__]
    return False
