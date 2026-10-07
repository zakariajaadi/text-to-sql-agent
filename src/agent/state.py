from langgraph.graph import MessagesState
from langgraph.managed import RemainingSteps


class AgentState(MessagesState):
     standalone_question: str            # self-contained rewrite of user question (output of condense_question)
     selected_db: str                    # target database name (output of route_database)
     is_in_scope: bool                   # whether the question can be answered (output of rerank_tables)
     candidate_schemas: dict[str, str]   # table_name → DDL from vector search (output of retrieve_schema)
     selected_schemas: dict[str, str]    # table_name → full DDL of the tables kept by rerank_tables, FK-expanded (output of rerank_tables)
     retrieved_schema: str               # DDL sent to planner and sql_coder, columns pruned (output of prune_columns)
     llm_selected_tables: list[str]      # tables selected by reranker LLM, before FK expansion, used by eval for precision/recall
     query_plan: str                     # execution plan from planner node
     generated_query: str                # SQL query (output of sql_coder)
     validation_feedback: str            # error message, empty on success (output of sql_validator)
     result_row_count: int               # rows read from the query result, at most max_rows (output of sql_validator)
     result_truncated: bool              # query returned more than max_rows rows (output of sql_validator)
     remaining_steps: RemainingSteps     # retry counter for sql_coder loop (managed by langgraph)

