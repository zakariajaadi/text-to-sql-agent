import sqlparse
from sqlparse.tokens import Keyword, DML


def is_safe_query(query: str) -> bool:
    """Return True only if the query is a single SELECT statement."""
    statements = sqlparse.parse(query.strip())

    # Reject empty or multi-statement queries
    if not statements or len(statements) > 1:
        return False

    # Walk tokens and check the first meaningful keyword is SELECT
    for token in statements[0].flatten():
        if token.ttype in (DML, Keyword):
            return token.normalized.upper() == "SELECT"

    return False