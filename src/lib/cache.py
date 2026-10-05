"""
LangGraph Node Decorator: Semantic Caching Middleware
Used by: src/agent/nodes.py (sql_coder, format_answer ...)

"""

from functools import wraps
from typing import Any, Callable, Dict

from loguru import logger


def extract_text_content(val: Any) -> str:
    """
    Recursively extracts plain text from various LangChain/LLM output formats.
    Handles:
      - Lists of messages (takes the last one).
      - LangChain Message objects (AIMessage, ToolMessage).
      - Gemini-style content blocks (list of dicts with 'text').
      - Standard dictionaries or raw strings.
    """
    if not val:
        return ""

    # 1. If it's a list (like state['messages']), recurse on the last element
    if isinstance(val, list):
        return extract_text_content(val[-1])

    # 2. Extract content from Message objects (if present)
    content = getattr(val, "content", val)

    # 3. Handle Gemini-style content blocks: [{'text': '...'}, {'text': '...'}]
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content
            if isinstance(block, dict)
        ).strip()

    # 4. Handle dictionary returns (e.g., {'text': '...'})
    if isinstance(content, dict):
        return content.get("text", str(content))

    # 5. Fallback to string representation
    return str(content)


def cached_node(key_fn: Callable[[Dict], str],
                return_key: str,
                cache_only_first_attempt: bool = True):
    """
    A universal decorator for LangGraph nodes to handle selective semantic caching.
    Automatically handles conversion between raw strings and LangChain Message objects.

    Args:
        key_fn: function(state) -> str, builds the unique cache key.
        return_key: The state key to read from/write to (e.g., 'generated_query' or 'messages').
        cache_only_first_attempt: If True, skips caching if 'validation_feedback' is present.
    """
    def decorator(fn: Callable):
        @wraps(fn)
        def wrapper(state: Dict, *args, **kwargs):
            # Import inside wrapper to avoid circular imports
            from config import llm_cache

            # Bypass caching if cache is not initialized or disabled
            if llm_cache is None:
                return fn(state, *args, **kwargs)

            # Retry: bypass the cache entirely — a lookup would return the result that just failed
            is_retry = bool(state.get("validation_feedback"))
            if cache_only_first_attempt and is_retry:
                return fn(state, *args, **kwargs)

            # 1. Generate unique key for this specific node execution
            cache_key = key_fn(state)

            # 2. Semantic Lookup in Redis (using function name as namespace)
            cached = llm_cache.lookup(cache_key, llm_string=fn.__name__)

            if cached and len(cached) > 0:
                logger.debug(f"[{fn.__name__}] Semantic Cache Hit: {cache_key}")
                cached_text = cached[0].text

                # Smart Return: Wrap in AIMessage list if the state expects 'messages'
                if return_key == "messages":
                    from langchain_core.messages import AIMessage
                    return {return_key: [AIMessage(content=cached_text)]}

                # Otherwise return as raw data (string/query)
                return {return_key: cached_text}

            # 3. Cache Miss: Execute the actual node function
            result = fn(state, *args, **kwargs)

            # 4. Extract Value for Storage using our robust utility
            val_to_store = extract_text_content(result.get(return_key, ""))

            # 5. Selective Cache Update
            is_retry = bool(state.get("validation_feedback"))
            should_cache = not (cache_only_first_attempt and is_retry)

            if should_cache and val_to_store:
                from langchain_core.outputs import Generation
                llm_cache.update(
                    cache_key,
                    llm_string=fn.__name__,
                    return_val=[Generation(text=val_to_store)]
                )
                logger.debug(f"[{fn.__name__}] Cache Updated: {cache_key}")

            return result
        return wrapper
    return decorator