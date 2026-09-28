from langgraph.graph import MessagesState
from langgraph.managed import RemainingSteps


class AgentState(MessagesState):
     standalone_question: str            # self-contained rewrite of user question (output of condense_question)
     selected_db: str                    # target database name (output of route_database)
     is_in_scope: bool                   # whether the question can be answered (output of rerank_tables)
     candidate_schemas: dict[str, str]   # table_name → DDL from vector search (output of retrieve_schema)
     retrieved_schema: str               # filtered DDL for sql_coder (output of rerank_tables)
     llm_selected_tables: list[str]      # tables selected by reranker LLM, before FK expansion, used by eval for precision/recall
     query_plan: str                     # execution plan from planner node
     generated_query: str                # SQL query (output of sql_coder)
     validation_feedback: str            # error message, empty on success (output of sql_validator)
     remaining_steps: RemainingSteps     # retry counter for sql_coder loop (managed by langgraph)

