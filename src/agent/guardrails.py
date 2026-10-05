import sqlparse
import sqlparse.tokens as T
from sqlparse.tokens import DML, Keyword

COMMENTS = (T.Comment.Single, T.Comment.Multiline)

# Hive functions that reach outside SQL: Java calls and server file reads
FORBIDDEN_FUNCTIONS = {"REFLECT", "REFLECT2", "JAVA_METHOD", "IN_FILE"}

def is_safe_query(query: str) -> bool:
    """
    Return True only if the query is a safe read-only statement.

    Three-pass validation:
        1. First meaningful token must be SELECT or WITH (allowlist).
        2. No forbidden DML/DDL keyword anywhere in the token stream (blocklist).
        3. No escape from SQL (Hive): script execution (... USING 'script'),
           Java calls (reflect, java_method) or server file reads (in_file).

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
                
    # Pass 3 — no escape from SQL (Hive): shell scripts, Java calls, server file reads
    # A SELECT can run TRANSFORM / MAP / REDUCE ... USING 'script' or call reflect().
    tokens = [t for t in stmt.flatten() if not t.is_whitespace and t.ttype not in COMMENTS]
    for i, token in enumerate(tokens):
        word = token.value.upper()
        if word == "TRANSFORM" or word in FORBIDDEN_FUNCTIONS:
            return False
        # USING followed by a string = script execution (JOIN ... USING (col) stays allowed)
        if word == "USING" and i + 1 < len(tokens) and tokens[i + 1].ttype in T.Literal.String:
            return False

    return True