import sqlparse
import sqlparse.tokens as T
from sqlparse.tokens import DML, Keyword


def is_safe_query(query: str) -> bool:
    """
    Return True only if the query is a safe read-only statement.

    Two-pass validation:
        1. First meaningful token must be SELECT or WITH (allowlist).
        2. No forbidden DML/DDL keyword anywhere in the token stream (blocklist).

    Handles CTEs, case-insensitive keywords, and leading comments correctly.
    """
    statements = sqlparse.parse(query.strip())

    if not statements or len(statements) > 1:
        return False

    stmt = statements[0]

    # Pass 1 — first meaningful token must be SELECT or WITH
    # Skips whitespace and comments to avoid bypass via leading comment tricks
    for token in stmt.flatten():
        if token.ttype in (T.Whitespace, T.Newline, T.Comment.Single, T.Comment.Multiline):
            continue
        if token.normalized.upper() not in ("SELECT", "WITH"):
            return False
        break

    # Pass 2 — no forbidden keyword anywhere in the statement
    # Covers injection attempts like: SELECT 1; DROP TABLE users
    forbidden = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "REPLACE"}
    for token in stmt.flatten():
        if token.ttype in (DML, Keyword):
            if token.normalized.upper() in forbidden:
                return False

    return True