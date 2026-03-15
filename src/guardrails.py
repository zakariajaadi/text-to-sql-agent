

def is_safe_query(query: str) -> bool:
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER", "CREATE"]
    query_upper = query.upper()
    return not any(keyword in query_upper for keyword in forbidden)