import sqlparse
from sqlparse.tokens import DML, Keyword


def is_safe_query(query: str) -> bool:
    """Return True only if the query is a single SELECT statement (including CTEs)."""
    statements = sqlparse.parse(query.strip())

    if not statements or len(statements) > 1:
        return False

    # Strip comments before checking
    cleaned = query.strip().upper()
    
    # Accept queries starting with WITH (CTEs) only if they contain SELECT
    if cleaned.startswith("WITH"):
        return "SELECT" in cleaned and not any(
            kw in cleaned for kw in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE")
        )

    for token in statements[0].flatten():
        if token.ttype in (DML, Keyword):
            return token.normalized.upper() == "SELECT"

    return False