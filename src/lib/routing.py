"""
Database routing
Used by: nodes.py (route_database)

"""

from loguru import logger

from config import embedding_model


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x ** 2 for x in a) ** 0.5
    nb  = sum(x ** 2 for x in b) ** 0.5
    return dot / (na * nb + 1e-9)


def select_database(question: str, domain_rules: dict, db_embeddings: dict) -> str:
    """
    Select the target database for a given question.

    Two-pass strategy (no LLM involved):
        Pass 1 — keyword scoring (<1ms):
            Count matching keywords per database.
            Returns the highest-scoring database if there is a clear winner.
            Falls through to pass 2 on tie.
        Pass 2 — embedding similarity (~30ms):
            Embed the question and compute cosine similarity against
            pre-computed db_embeddings. Returns the highest-scoring database.

    Args:
        question:      Raw user question.
        domain_rules:  {db_name: [keywords]} built from config descriptions.
        db_embeddings: {db_name: embedding_vector} pre-computed at startup.

    Returns:
        The name of the selected database.
    """
    q = question.lower()

    # Pass 1 — keyword scoring
    scores = {
        db_name: sum(1 for kw in keywords if kw in q)
        for db_name, keywords in domain_rules.items()
    }

    best_score = max(scores.values())

    if best_score > 0:
        # Check for tie — if multiple databases have the same score, fall through
        winners = [db for db, score in scores.items() if score == best_score]
        if len(winners) == 1:
            logger.debug(f"[select_database] Keyword match → {winners[0]} (scores={scores})")
            return winners[0]

    # Pass 2 — embedding similarity (tie or no keyword match)
    q_emb  = embedding_model.embed_query(question)
    emb_scores = {
        name: _cosine_similarity(q_emb, emb)
        for name, emb in db_embeddings.items()
    }
    best = max(emb_scores, key=emb_scores.get)
    logger.debug(f"[select_database] Embedding match → {best} (scores={emb_scores})")
    return best